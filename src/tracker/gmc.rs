// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Camera motion by sparse optical flow, `GMC(method="sparseOptFlow")` in Ultralytics.
//!
//! On a half-size grayscale frame, corners from `OpenCV` `goodFeaturesToTrack` are followed into
//! the next frame by `calcOpticalFlowPyrLK` and fitted with `estimateAffinePartial2D`, each
//! written here with the same parameters and arithmetic, down to the float rounding of the
//! `OpenCV` x86 SIMD code, so corners and flow match it bit for bit.

use ndarray::Array3;

use super::{Mat8, Track, Warp};
#[allow(clippy::wildcard_imports)]
use crate::parallel::*;

const IDENTITY: Warp = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
/// Most corners tracked, and the quality of the weakest relative to the strongest.
const MAX_CORNERS: usize = 400;
const QUALITY: f32 = 0.01;
/// Lucas-Kanade window side, pyramid levels above the frame, iterations and step tolerance.
const WIN: usize = 21;
const LEVELS: usize = 3;
const ITERS: usize = 30;
const EPS: f64 = 0.01;
/// Bits of the bilinear weights in Lucas-Kanade.
const W_BITS: i32 = 14;
/// Border kept around each image, wide enough for every window the flow reads.
const PAD: usize = WIN + 2;

/// `BORDER_REFLECT_101` index of `i` into `0..n`.
const fn reflect(mut i: isize, n: usize) -> usize {
    let n = n as isize;
    if n == 1 {
        return 0;
    }
    loop {
        if i < 0 {
            i = -i;
        } else if i >= n {
            i = 2 * n - 2 - i;
        } else {
            return i as usize;
        }
    }
}

/// `CV_DESCALE`: shift right by `n` with rounding.
const fn descale(x: i32, n: i32) -> i32 {
    (x + (1 << (n - 1))) >> n
}

/// An image with a border of `PAD` pixels around it, so the windows near its edge are read
/// without bound checks, like the padded pyramid levels of `OpenCV`.
#[derive(Debug, Clone)]
struct Padded<T> {
    w: usize,
    h: usize,
    stride: usize,
    px: Vec<T>,
}

impl<T: Copy + Default + Send + Sync> Padded<T> {
    /// A `w x h` image filled in place row by row by `fill(y, row)`, rows in parallel, then
    /// padded by reflecting it at the border (`BORDER_REFLECT_101`) or with zeros.
    fn build(w: usize, h: usize, reflected: bool, fill: impl Fn(usize, &mut [T]) + Sync) -> Self {
        let stride = w + 2 * PAD;
        let mut px = vec![T::default(); stride * (h + 2 * PAD)];
        px[PAD * stride..(PAD + h) * stride]
            .par_chunks_mut(stride)
            .enumerate()
            .for_each(|(y, row)| {
                let (left, rest) = row.split_at_mut(PAD);
                let (mid, right) = rest.split_at_mut(w);
                fill(y, mid);
                if reflected {
                    for x in 0..PAD {
                        left[x] = mid[reflect(x as isize - PAD as isize, w)];
                        right[x] = mid[reflect((w + x) as isize, w)];
                    }
                }
            });
        if reflected {
            for y in 0..PAD {
                let top = reflect(y as isize - PAD as isize, h);
                let bottom = reflect((h + y) as isize, h);
                for (dst, src) in [(y, top), (PAD + h + y, bottom)] {
                    px.copy_within((PAD + src) * stride..(PAD + src + 1) * stride, dst * stride);
                }
            }
        }
        Self { w, h, stride, px }
    }

    /// The `w` pixels of row `y` starting at column `x`, which may lie outside the image.
    fn row(&self, x: isize, y: isize, w: usize) -> &[T] {
        &self.px[self.idx(x, y)..][..w]
    }

    /// Index of pixel `(x, y)`, which may lie up to `PAD` pixels outside the image.
    const fn idx(&self, x: isize, y: isize) -> usize {
        (y + PAD as isize) as usize * self.stride + (x + PAD as isize) as usize
    }
}

type Gray = Padded<u8>;

impl Gray {
    /// `cv2.cvtColor(rgb, COLOR_RGB2GRAY)` (15-bit weights) then `cv2.resize` to half size,
    /// bilinear.
    fn half_from_rgb(img: &Array3<u8>) -> Self {
        let (h, w, _) = img.dim();
        let rgb = img.as_standard_layout();
        let rgb = rgb.as_slice().unwrap_or_default();
        let gray = |y: usize, x: usize| {
            let p = &rgb[(y * w + x) * 3..][..3];
            descale(
                i32::from(p[0]) * 9798 + i32::from(p[1]) * 19235 + i32::from(p[2]) * 3735,
                15,
            )
        };
        let (dw, dh) = (w / 2, h / 2);
        // Source index and 11-bit weights of each output pixel, like `cv::resize`.
        let taps = |n: usize, out: usize| -> Vec<(usize, usize, i32, i32)> {
            let scale = n as f64 / out as f64;
            (0..out)
                .map(|d| {
                    let f = ((d as f64 + 0.5) * scale - 0.5) as f32;
                    let (mut s, mut f) = (f.floor(), f - f.floor());
                    if s < 0.0 {
                        (s, f) = (0.0, 0.0);
                    }
                    if s as usize >= n - 1 {
                        (s, f) = ((n - 1) as f32, 0.0);
                    }
                    let s = s as usize;
                    let weight = |v: f32| (v * 2048.0).round_ties_even() as i32;
                    (s, (s + 1).min(n - 1), weight(1.0 - f), weight(f))
                })
                .collect()
        };
        let (xs, ys) = (taps(w, dw), taps(h, dh));
        Self::build(dw, dh, true, |y, out| {
            let (y0, y1, b0, b1) = ys[y];
            for (o, &(x0, x1, a0, a1)) in out.iter_mut().zip(&xs) {
                let row = |y: usize| gray(y, x0) * a0 + gray(y, x1) * a1;
                *o = ((b0 * row(y0) + b1 * row(y1) + (1 << 21)) >> 22).clamp(0, 255) as u8;
            }
        })
    }

    /// `cv2.pyrDown`: 5x5 Gaussian blur and every other pixel.
    fn down(&self) -> Self {
        const K: [i32; 5] = [1, 4, 6, 4, 1];
        let (w, h) = (self.w.div_ceil(2), self.h.div_ceil(2));
        // Horizontal pass over source rows -2..2h+2, then the vertical one.
        let mut across = vec![0i32; (2 * h + 3) * w];
        across.par_chunks_mut(w).enumerate().for_each(|(r, out)| {
            let src = self.row(-2, r as isize - 2, 2 * w + 3);
            for (x, o) in out.iter_mut().enumerate() {
                *o = (0..5).map(|i| K[i] * i32::from(src[2 * x + i])).sum();
            }
        });
        Self::build(w, h, true, |y, out| {
            let rows: [&[i32]; 5] = std::array::from_fn(|j| &across[(2 * y + j) * w..][..w]);
            for (x, o) in out.iter_mut().enumerate() {
                let sum: i32 = (0..5).map(|j| K[j] * rows[j][x]).sum();
                *o = ((sum + 128) >> 8) as u8;
            }
        })
    }

    /// `[dx, dy]` Scharr derivatives with reflected borders, like `calcScharrDeriv`, and zero
    /// outside the image.
    fn scharr(&self) -> Padded<[i16; 2]> {
        let w = self.w;
        Padded::build(w, self.h, false, |y, out| {
            let y = y as isize;
            let [a, b, c] = [-1, 0, 1].map(|dy| self.row(-1, y + dy, w + 2));
            // Vertical pass: smoothed and differenced columns.
            let col: Vec<(i32, i32)> = (0..w + 2)
                .map(|x| {
                    let (a, b, c) = (i32::from(a[x]), i32::from(b[x]), i32::from(c[x]));
                    ((a + c) * 3 + b * 10, c - a)
                })
                .collect();
            for (x, o) in out.iter_mut().enumerate() {
                let (l, m, r) = (col[x], col[x + 1], col[x + 2]);
                *o = [(r.0 - l.0) as i16, ((l.1 + r.1) * 3 + m.1 * 10) as i16];
            }
        })
    }

    /// `cv2.goodFeaturesToTrack(maxCorners=400, qualityLevel=0.01, minDistance=0, blockSize=3)`:
    /// local maxima of the smaller structure tensor eigenvalue, strongest first.
    // `sqrt(b * b + t * t)` rather than `hypot`, which rounds differently from `OpenCV`.
    #[allow(clippy::imprecise_flops)]
    fn corners(&self) -> Vec<[f32; 2]> {
        let (w, h) = (self.w, self.h);
        if w < 3 || h < 3 {
            return Vec::new();
        }
        // Sobel derivatives scaled by 1 / (4 * 3 * 255), with the scale folded into the
        // smoothing taps and fused multiply-adds like the `OpenCV` SIMD filters (exact in f64).
        let scale = 1.0f32 / (4.0 * 3.0 * 255.0);
        let fma = |a: f32, b: f32, c: f32| (f64::from(a) * f64::from(b) + f64::from(c)) as f32;
        let mut cov = vec![[0.0f32; 3]; w * h];
        cov.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
            let [a, b, c] = [-1, 0, 1].map(|dy| self.row(-1, y as isize + dy, w + 2));
            for (x, o) in out.iter_mut().enumerate() {
                let p = |r: &[u8], dx: usize| f32::from(r[x + dx]);
                let smooth = |r| fma(p(r, 2), scale, fma(p(r, 1), 2.0 * scale, p(r, 0) * scale));
                let gx = fma(
                    scale,
                    (p(a, 2) - p(a, 0)) + (p(c, 2) - p(c, 0)),
                    2.0 * scale * (p(b, 2) - p(b, 0)),
                );
                let gy = smooth(c) - smooth(a);
                *o = [gx * gx, gx * gy, gy * gy];
            }
        });
        // 3x3 sums with reflected borders (exact in f64 like `boxFilter`) and the smaller
        // eigenvalue.
        let mut eig = vec![0.0f32; w * h];
        eig.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
            let rows = [-1, 0, 1].map(|dy| &cov[reflect(y as isize + dy, h) * w..][..w]);
            for (x, o) in out.iter_mut().enumerate() {
                let cols = [-1, 0, 1].map(|dx| reflect(x as isize + dx, w));
                let mut s = [0.0f64; 3];
                for r in rows {
                    for &i in &cols {
                        (0..3).for_each(|k| s[k] += f64::from(r[i][k]));
                    }
                }
                let (a, b, c) = (s[0] as f32 * 0.5, s[1] as f32, s[2] as f32 * 0.5);
                *o = (a + c) - (b * b + (a - c) * (a - c)).sqrt();
            }
        });
        // Local maxima above the quality floor, away from the border.
        let floor = eig.iter().copied().fold(0.0, f32::max) * QUALITY;
        let eig = &eig;
        let mut found: Vec<(f32, usize)> = (1..h - 1)
            .into_par_iter()
            .flat_map_iter(|y| {
                (1..w - 1).filter_map(move |x| {
                    let v = eig[y * w + x];
                    let peak = || {
                        (y - 1..=y + 1).all(|j| eig[j * w + x - 1..][..3].iter().all(|&n| n <= v))
                    };
                    (v > floor && peak()).then_some((v, y * w + x))
                })
            })
            .collect();
        // Strongest first, and of equals the later pixel, like OpenCV's sort.
        found.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.cmp(&a.1)));
        found
            .into_iter()
            .take(MAX_CORNERS)
            .map(|(_, i)| [(i % w) as f32, (i / w) as f32])
            .collect()
    }
}

/// Bilinear 14-bit weights of the four pixels around a point `(a, b)` past a pixel.
fn weights(a: f32, b: f32) -> [i32; 4] {
    let w = |v: f32| (v * (1 << W_BITS) as f32).round_ties_even() as i32;
    let (w00, w01, w10) = (w((1.0 - a) * (1.0 - b)), w(a * (1.0 - b)), w((1.0 - a) * b));
    [w00, w01, w10, (1 << W_BITS) - w00 - w01 - w10]
}

/// A frame pyramid with the Scharr derivatives of each level.
#[derive(Debug, Clone)]
struct Pyramid(Vec<(Gray, Padded<[i16; 2]>)>);

impl Pyramid {
    /// Levels until one would be no larger than the window, like `buildOpticalFlowPyramid`.
    fn new(frame: Gray) -> Self {
        let mut levels = vec![frame];
        while levels.len() <= LEVELS {
            let last = &levels[levels.len() - 1];
            if last.w.div_ceil(2) <= WIN || last.h.div_ceil(2) <= WIN {
                break;
            }
            levels.push(last.down());
        }
        let derivs: Vec<_> = levels.iter().map(Gray::scharr).collect();
        Self(levels.into_iter().zip(derivs).collect())
    }

    /// `cv2.calcOpticalFlowPyrLK(prev, next, points, None)` with a 21x21 window, 3 levels and
    /// 30 iterations: where each point moved, or `None` where it was lost.
    fn flow(&self, next: &Self, points: &[[f32; 2]]) -> Vec<Option<[f32; 2]>> {
        let max_level = self.0.len().min(next.0.len()) - 1;
        points
            .into_par_iter()
            .map(|p| {
                let mut next_pt = [0.0f32; 2];
                for level in (0..=max_level).rev() {
                    let s = 1.0 / (1u32 << level) as f32;
                    let prev_pt = [p[0] * s, p[1] * s];
                    next_pt = if level == max_level {
                        prev_pt
                    } else {
                        [next_pt[0] * 2.0, next_pt[1] * 2.0]
                    };
                    match self.track(&next.0[level].0, level, prev_pt, next_pt) {
                        Some(pt) => next_pt = pt,
                        None if level == 0 => return None,
                        None => {}
                    }
                }
                Some(next_pt)
            })
            .collect()
    }

    /// Lucas-Kanade iterations at one level from `start`: the new point, or `None` when the
    /// window has too little texture or leaves the image (where above the bottom level the
    /// point keeps its estimate). Window sums run in the float lanes of the `OpenCV` SSE code,
    /// four over the first 16 columns and one over the rest, so they round the same.
    fn track(
        &self,
        j_img: &Gray,
        level: usize,
        prev_pt: [f32; 2],
        start: [f32; 2],
    ) -> Option<[f32; 2]> {
        let (img, deriv) = &self.0[level];
        let half = (WIN - 1) as f32 * 0.5;
        let win = WIN as isize;
        let out = |x: f32, y: f32, g: &Gray| {
            let (x, y) = (x.floor() as isize, y.floor() as isize);
            x < -win || x >= g.w as isize || y < -win || y >= g.h as isize
        };
        let reduce = |tail: f32, l: [f32; 4]| tail + ((l[0] + l[2]) + (l[1] + l[3]));
        let fs = 1.0 / (1u32 << 20) as f32;
        let s = img.stride;
        let prev_pt = [prev_pt[0] - half, prev_pt[1] - half];
        if out(prev_pt[0], prev_pt[1], img) {
            return None;
        }
        // Window of the previous frame and its gradients, bilinear in fixed point.
        let (ix, iy) = (prev_pt[0].floor(), prev_pt[1].floor());
        let iw = weights(prev_pt[0] - ix, prev_pt[1] - iy);
        let origin = img.idx(ix as isize, iy as isize);
        // The window and its x and y gradients, row by row.
        let mut patch = [[[0i32; WIN]; WIN]; 3];
        let (mut lanes, mut tail) = ([[0.0f32; 4]; 3], [0.0f32; 3]);
        for y in 0..WIN {
            let row = origin + y * s;
            let (r0, r1) = (&img.px[row..][..=WIN], &img.px[row + s..][..=WIN]);
            let (d0, d1) = (&deriv.px[row..][..=WIN], &deriv.px[row + s..][..=WIN]);
            for x in 0..WIN {
                let p = |a: u8, b: u8, c: u8, d: u8| {
                    let [a, b, c, d] = [a, b, c, d].map(i32::from);
                    a * iw[0] + b * iw[1] + c * iw[2] + d * iw[3]
                };
                patch[0][y][x] = descale(p(r0[x], r0[x + 1], r1[x], r1[x + 1]), W_BITS - 5);
                for c in 0..2 {
                    let g = |d: &[[i16; 2]], x: usize| i32::from(d[x][c]);
                    let v = g(d0, x) * iw[0] + g(d0, x + 1) * iw[1] + g(d1, x) * iw[2];
                    patch[c + 1][y][x] = descale(v + g(d1, x + 1) * iw[3], W_BITS);
                }
            }
            let [gx, gy] = [&patch[1][y], &patch[2][y]];
            for (c, (a, b)) in [(gx, gx), (gx, gy), (gy, gy)].into_iter().enumerate() {
                let v = std::array::from_fn(|x| a[x] * b[x]);
                add_row(&mut lanes[c], &mut tail[c], &v, false);
            }
        }
        let [a11, a12, a22] = [0, 1, 2].map(|c| reduce(tail[c], lanes[c]) * fs);
        let det = a11 * a22 - a12 * a12;
        let min_eig = (a22 + a11 - ((a11 - a22) * (a11 - a22) + 4.0 * a12 * a12).sqrt())
            / (2 * WIN * WIN) as f32;
        if min_eig < 1e-4 || det < f32::EPSILON {
            return None;
        }
        let inv = 1.0 / det;
        let mut pt = [start[0] - half, start[1] - half];
        let mut next_pt = start;
        let mut prev_delta = [0.0f32; 2];
        for iter in 0..ITERS {
            if out(pt[0], pt[1], j_img) {
                return (level > 0).then_some(next_pt);
            }
            let (jx, jy) = (pt[0].floor(), pt[1].floor());
            let jw = weights(pt[0] - jx, pt[1] - jy);
            let origin = j_img.idx(jx as isize, jy as isize);
            let (mut lanes, mut tail) = ([[0.0f32; 4]; 2], [0.0f32; 2]);
            for y in 0..WIN {
                let row = origin + y * s;
                let (r0, r1) = (&j_img.px[row..][..=WIN], &j_img.px[row + s..][..=WIN]);
                let i = &patch[0][y];
                let diff: [i32; WIN] = std::array::from_fn(|x| {
                    let [a, b, c, d] = [r0[x], r0[x + 1], r1[x], r1[x + 1]].map(i32::from);
                    descale(a * jw[0] + b * jw[1] + c * jw[2] + d * jw[3], W_BITS - 5) - i[x]
                });
                for c in 0..2 {
                    let g = &patch[c + 1][y];
                    let v = std::array::from_fn(|x| diff[x] * g[x]);
                    add_row(&mut lanes[c], &mut tail[c], &v, true);
                }
            }
            let [b1, b2] = [0, 1].map(|c| reduce(tail[c], lanes[c]) * fs);
            let delta = [(a12 * b2 - a22 * b1) * inv, (a12 * b1 - a11 * b2) * inv];
            pt = [pt[0] + delta[0], pt[1] + delta[1]];
            next_pt = [pt[0] + half, pt[1] + half];
            // The stopping tests compare in double, like `OpenCV`.
            let [dx, dy] = delta.map(f64::from);
            if dx * dx + dy * dy <= EPS * EPS {
                break;
            }
            let back = |c: usize| f64::from((delta[c] + prev_delta[c]).abs()) < EPS;
            if iter > 0 && back(0) && back(1) {
                next_pt = [next_pt[0] - delta[0] * 0.5, next_pt[1] - delta[1] * 0.5];
                break;
            }
            prev_delta = delta;
        }
        // A point that ends outside the image is lost too.
        (level > 0 || !out(next_pt[0] - half, next_pt[1] - half, j_img)).then_some(next_pt)
    }
}

/// Add a window row of integer terms into the float sums of the `OpenCV` SSE code: four lanes
/// over columns 0..16 (columns `m` and `m + 4` added as integers first when `paired`), then
/// one sum over the rest.
fn add_row(lanes: &mut [f32; 4], tail: &mut f32, v: &[i32; WIN], paired: bool) {
    for block in [0, 8] {
        if paired {
            for m in 0..4 {
                lanes[m] += (v[block + m] + v[block + m + 4]) as f32;
            }
        } else {
            for m in 0..8 {
                lanes[m % 4] += v[block + m] as f32;
            }
        }
    }
    for &x in &v[16..] {
        *tail += x as f32;
    }
}

/// Least-squares similarity `[a, b, tx, ty]` mapping `src[i]` to `dst[i]` over `idx`, as
/// `x' = a x - b y + tx` and `y' = b x + a y + ty`.
fn fit(src: &[[f32; 2]], dst: &[[f32; 2]], idx: &[usize]) -> Option<[f64; 4]> {
    let n = idx.len() as f64;
    let mean = |p: &[[f32; 2]], k: usize| idx.iter().map(|&i| f64::from(p[i][k])).sum::<f64>() / n;
    let (mx, my, mu, mv) = (mean(src, 0), mean(src, 1), mean(dst, 0), mean(dst, 1));
    let (mut sxx, mut sa, mut sb) = (0.0, 0.0, 0.0);
    for &i in idx {
        let (x, y) = (f64::from(src[i][0]) - mx, f64::from(src[i][1]) - my);
        let (u, v) = (f64::from(dst[i][0]) - mu, f64::from(dst[i][1]) - mv);
        sxx += x * x + y * y;
        sa += x * u + y * v;
        sb += x * v - y * u;
    }
    (sxx > 0.0).then(|| {
        let (a, b) = (sa / sxx, sb / sxx);
        [a, b, mu - a * mx + b * my, mv - b * mx - a * my]
    })
}

/// `cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)`: a similarity from random point
/// pairs (with the `OpenCV` random generator and its default 3 px threshold, 0.99 confidence
/// and 2000 iterations), refitted to the inliers of the best one.
fn similarity(src: &[[f32; 2]], dst: &[[f32; 2]]) -> Option<Warp> {
    let n = src.len();
    let mut state = u64::MAX;
    let mut uniform = || {
        state = (state & 0xFFFF_FFFF) * 4_164_903_690 + (state >> 32);
        (state & 0xFFFF_FFFF) as usize % n
    };
    let inliers = |m: &[f64; 4]| -> Vec<usize> {
        let h = [m[0], -m[1], m[2], m[1], m[0], m[3]].map(|v| v as f32);
        (0..n)
            .filter(|&i| {
                let [x, y] = src[i];
                let a = h[0] * x + h[1] * y + h[2] - dst[i][0];
                let b = h[3] * x + h[4] * y + h[5] - dst[i][1];
                a * a + b * b <= 9.0
            })
            .collect()
    };
    let (mut best, mut iters, mut i) = (Vec::new(), 2000, 0);
    while i < iters {
        i += 1;
        let a = uniform();
        let b = std::iter::repeat_with(&mut uniform).find(|&b| b != a)?;
        let Some(m) = fit(src, dst, &[a, b]) else {
            continue;
        };
        let good = inliers(&m);
        if good.len() > best.len().max(1) {
            // Fewer iterations once enough points agree: log(1 - confidence) / log(1 - inliers^2).
            let ratio = good.len() as f64 / n as f64;
            let denom = (1.0 - ratio * ratio).max(f64::MIN_POSITIVE).ln();
            let num = 0.01f64.ln();
            if denom < 0.0 && -num < f64::from(iters) * -denom {
                iters = (num / denom).round() as i32;
            }
            best = good;
        }
    }
    let [a, b, tx, ty] = fit(src, dst, &best)?;
    Some([[a, -b, tx], [b, a, ty]])
}

/// Camera motion between consecutive frames.
#[derive(Debug, Clone, Default)]
pub(super) struct Gmc {
    /// The previous frame and its corners.
    prev: Option<(Pyramid, Vec<[f32; 2]>)>,
}

impl Gmc {
    /// The motion from the previous frame to `img` (RGB, HWC), identity on the first frame or
    /// without enough tracked corners.
    pub(super) fn apply(&mut self, img: &Array3<u8>) -> Warp {
        let frame = Gray::half_from_rgb(img);
        let corners = frame.corners();
        let frame = Pyramid::new(frame);
        let mut warp = IDENTITY;
        if let Some((prev, points)) = &self.prev {
            let moved = prev.flow(&frame, points);
            let (src, dst): (Vec<[f32; 2]>, Vec<[f32; 2]>) = points
                .iter()
                .zip(moved)
                .filter_map(|(&p, m)| Some((p, m?)))
                .unzip();
            if src.len() > 4
                && let Some(mut h) = similarity(&src, &dst)
            {
                // Back to full-size pixels.
                h[0][2] *= 2.0;
                h[1][2] *= 2.0;
                warp = h;
            }
        }
        self.prev = Some((frame, corners));
        warp
    }
}

impl Track {
    /// Move the track with the camera.
    pub(super) fn warp(&mut self, h: &Warp) {
        let r = [[h[0][0], h[0][1]], [h[1][0], h[1][1]]];
        // The center and its velocity rotate. The aspect ratio and height of an `xyah` state
        // stay, and sizes follow the box that encloses the rotated one (or only scale, for an
        // oriented box), so they never flip sign.
        let diag = if self.kind.xyah() {
            Some(1.0)
        } else {
            self.angle.map(|_| r[0][0].hypot(r[1][0]))
        };
        let mut m: Mat8 = [[0.0; 8]; 8];
        for o in [0, 2, 4, 6] {
            for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                m[o + i][o + j] = match (o % 4, diag) {
                    (0, _) => r[i][j],
                    (_, Some(d)) if i == j => d,
                    (_, Some(_)) => 0.0,
                    (_, None) => r[i][j].abs(),
                };
            }
        }
        self.mean = std::array::from_fn(|i| (0..8).map(|k| m[i][k] * self.mean[k]).sum());
        self.mean[0] += h[0][2];
        self.mean[1] += h[1][2];
        let mc: Mat8 = std::array::from_fn(|i| {
            std::array::from_fn(|j| (0..8).map(|k| m[i][k] * self.cov[k][j]).sum())
        });
        self.cov = std::array::from_fn(|i| {
            std::array::from_fn(|j| (0..8).map(|k| mc[i][k] * m[j][k]).sum())
        });
        if self.kind.xyah() {
            if self.last_observation[0] >= 0.0 {
                let [x1, y1, x2, y2] = self.last_observation.map(f64::from);
                let [cx, cy] = [(x1 + x2) / 2.0, (y1 + y2) / 2.0];
                let (nx, ny) = (
                    r[0][0] * cx + r[0][1] * cy + h[0][2],
                    r[1][0] * cx + r[1][1] * cy + h[1][2],
                );
                let (w, h) = (x2 - x1, y2 - y1);
                self.last_observation =
                    [nx - w / 2.0, ny - h / 2.0, nx + w / 2.0, ny + h / 2.0].map(|v| v as f32);
            }
        } else if let Some(a) = self.angle.as_mut() {
            *a += r[1][0].atan2(r[0][0]) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A textured frame shifted right by `dx` pixels.
    fn frame(dx: usize) -> Array3<u8> {
        Array3::from_shape_fn((240, 320, 3), |(y, x, _)| {
            let x = x + 320 - dx;
            (((x / 7) * 37 + (y / 5) * 91 + (x * y) % 13) % 256) as u8
        })
    }

    #[test]
    fn test_gmc_follows_a_camera_pan() {
        let mut gmc = Gmc::default();
        assert_eq!(gmc.apply(&frame(0)), IDENTITY);
        let h = gmc.apply(&frame(6));
        assert!((h[0][2] - 6.0).abs() < 0.5 && h[1][2].abs() < 0.5, "{h:?}");
        assert!(
            (h[0][0] - 1.0).abs() < 0.01 && h[1][0].abs() < 0.01,
            "{h:?}"
        );
    }
}
