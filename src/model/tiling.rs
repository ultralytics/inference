// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Tiled prediction, the Rust port of the Ultralytics `tile` argument.

// Pixel coordinates and tile counts are far below the f32 and u32 limits.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp
)]

use image::{DynamicImage, GenericImageView};
use ndarray::{Array2, Array3, Axis};

use super::YOLOModel;
use crate::error::{InferenceError, Result};
use crate::inference::InferenceConfig;
use crate::results::{Boxes, Keypoints, Obb, Results, Speed};
use crate::task::Task;
use crate::utils::{nms_per_class, nms_rotated_per_class};

/// A tile window as `[x1, y1, x2, y2]` in source pixels.
type Window = [u32; 4];

/// Windows covering an `h` x `w` image with 20% overlap and the last window aligned to the edge.
fn tile_windows(h: u32, w: u32, size: u32) -> Vec<Window> {
    let starts = |len: u32| -> Vec<u32> {
        let last = len.saturating_sub(size);
        let step = ((8 * size + 5) / 10).max(1); // round(size * 0.8)
        (0..last).step_by(step as usize).chain([last]).collect()
    };
    let xs = starts(w);
    starts(h)
        .into_iter()
        .flat_map(|y| {
            xs.iter()
                .map(move |&x| [x, y, (x + size).min(w), (y + size).min(h)])
        })
        .collect()
}

/// Axis-aligned extent `[x1, y1, x2, y2]` of a box row: `xyxy`, or the corners of an `xywhr` box.
fn extent(row: &[f32], rotated: bool) -> [f32; 4] {
    if !rotated {
        return [row[0], row[1], row[2], row[3]];
    }
    let (sin, cos) = row[4].sin_cos();
    let (hw, hh) = (row[2] / 2.0, row[3] / 2.0);
    let (dx, dy) = (
        hw.mul_add(cos.abs(), hh * sin.abs()),
        hw.mul_add(sin.abs(), hh * cos.abs()),
    );
    [row[0] - dx, row[1] - dy, row[0] + dx, row[1] + dy]
}

/// `(box, score, class)` for the `idx` rows, whose first `K` columns are the box.
fn tuples<const K: usize>(rows: &[Vec<f32>], idx: &[usize]) -> Vec<([f32; K], f32, usize)> {
    idx.iter()
        .map(|&i| {
            (
                std::array::from_fn(|j| rows[i][j]),
                rows[i][K],
                rows[i][K + 1] as usize,
            )
        })
        .collect()
}

/// Per-class NMS over the `idx` rows, returning the kept indices best first.
fn nms(rows: &[Vec<f32>], idx: &[usize], rotated: bool, iou: f32, max_det: usize) -> Vec<usize> {
    let kept = if rotated {
        nms_rotated_per_class(&tuples::<5>(rows, idx), iou, max_det)
    } else {
        nms_per_class(&tuples::<4>(rows, idx), iou, max_det)
    };
    kept.into_iter().map(|k| idx[k]).collect()
}

/// Merge tile detections already offset to source coordinates, as `ops.merge_tiles` does.
///
/// A box touching an interior window edge is a cut fragment when another window sees it clear of
/// its own interior edges. Fragments are dropped before NMS and restored afterwards when no kept
/// box covers more than half of them. Returns the kept row indices, best first.
fn merge_rows(
    rows: &[Vec<f32>],
    wins: &[Window],
    all: &[Window],
    shape: (u32, u32),
    rotated: bool,
    iou: f32,
    max_det: usize,
) -> Vec<usize> {
    let border = [0.0, 0.0, shape.1 as f32, shape.0 as f32];
    let ext: Vec<[f32; 4]> = rows.iter().map(|r| extent(r, rotated)).collect();
    let win_f = |w: &Window| w.map(|v| v as f32);
    let mut fragment: Vec<bool> = ext
        .iter()
        .zip(wins)
        .map(|(e, w)| {
            let own = win_f(w);
            (0..4).any(|k| {
                let touches = if k < 2 {
                    e[k] <= own[k] + 2.0
                } else {
                    e[k] >= own[k] - 2.0
                };
                touches && own[k] != border[k]
            })
        })
        .collect();
    for (i, f) in fragment.iter_mut().enumerate() {
        if *f {
            let e = ext[i];
            *f = all.iter().map(win_f).any(|r| {
                (0..4).all(|k| {
                    let clear = if k < 2 {
                        r[k] + 2.0 < e[k]
                    } else {
                        r[k] - 2.0 > e[k]
                    };
                    clear || r[k] == border[k]
                })
            });
        }
    }
    let valid: Vec<usize> = (0..rows.len()).filter(|&i| !fragment[i]).collect();
    let mut keep = nms(rows, &valid, rotated, iou, max_det);
    let lone: Vec<usize> = (0..rows.len()).filter(|&i| fragment[i]).collect();
    if lone.is_empty() {
        return keep;
    }
    let area = |e: &[f32; 4]| (e[2] - e[0]).max(0.0) * (e[3] - e[1]).max(0.0);
    let uncovered = |&i: &usize| {
        !keep.iter().any(|&k| {
            let (a, b) = (ext[i], ext[k]);
            let inter = [
                a[0].max(b[0]),
                a[1].max(b[1]),
                a[2].min(b[2]),
                a[3].min(b[3]),
            ];
            area(&inter) > area(&a) / 2.0
        })
    };
    let restored: Vec<usize> = lone.iter().copied().filter(uncovered).collect();
    keep.extend(restored);
    nms(rows, &keep, rotated, iou, max_det)
}

/// Offset every tile's boxes or oriented boxes (and pose keypoints) to source coordinates, merge
/// them with [`merge_rows`] and store the result in `out`.
fn merge_detections(
    out: &mut Results,
    tiles: &[Results],
    windows: &[Window],
    rotated: bool,
    config: &InferenceConfig,
) {
    let shape = out.orig_shape;
    let data = |t: &Results| {
        if rotated {
            t.obb.as_ref().map(|o| o.data.clone())
        } else {
            t.boxes.as_ref().map(|b| b.data.clone())
        }
    };
    let (mut rows, mut wins, mut src) = (Vec::new(), Vec::new(), Vec::new());
    for (t, win) in tiles.iter().zip(windows) {
        let Some(d) = data(t) else { continue };
        for (i, r) in d.rows().into_iter().enumerate() {
            let mut row = r.to_vec();
            for k in 0..if rotated { 2 } else { 4 } {
                row[k] += win[k % 2] as f32;
            }
            rows.push(row);
            wins.push(*win);
            src.push((t, i));
        }
    }
    let keep = merge_rows(
        &rows,
        &wins,
        windows,
        shape,
        rotated,
        config.iou_threshold,
        config.max_det,
    );
    let cols = rows.first().map_or(if rotated { 7 } else { 6 }, Vec::len);
    let merged = Array2::from_shape_fn((keep.len(), cols), |(i, c)| rows[keep[i]][c]);
    if rotated {
        out.obb = Some(Obb::new(merged, shape));
    } else {
        out.boxes = Some(Boxes::new(merged, shape));
    }
    if let Some(k0) = &tiles[0].keypoints {
        let (nk, kd) = (k0.data.shape()[1], k0.data.shape()[2]);
        let mut k = Array3::<f32>::zeros((keep.len(), nk, kd));
        for (o, &i) in keep.iter().enumerate() {
            let (t, row) = src[i];
            let Some(kpts) = &t.keypoints else { continue };
            let mut dst = k.index_axis_mut(Axis(0), o);
            dst.assign(&kpts.data.index_axis(Axis(0), row));
            for j in 0..nk {
                dst[[j, 0]] += wins[i][0] as f32;
                dst[[j, 1]] += wins[i][1] as f32;
            }
        }
        out.keypoints = Some(Keypoints::new(k, shape));
    }
}

impl YOLOModel {
    /// Run inference on overlapping `tile` x `tile` windows of a large image and merge them back
    /// into one result in source coordinates, like the Ultralytics `tile` argument.
    ///
    /// Windows overlap by 20% and the last window in each direction is aligned to the image edge.
    /// Detection, pose and oriented boxes are offset to source coordinates and merged with NMS
    /// using the configured `IoU` and `max_det`.
    ///
    /// # Errors
    ///
    /// Returns an error if `tile` is `0`, the model is not a detect, pose or obb model, or
    /// inference fails.
    pub fn predict_tiled(
        &mut self,
        image: &DynamicImage,
        path: String,
        tile: u32,
    ) -> Result<Vec<Results>> {
        let task = self.metadata.task;
        if tile == 0 || !matches!(task, Task::Detect | Task::Pose | Task::Obb) {
            return Err(InferenceError::ConfigError(format!(
                "tiled prediction needs a tile size above 0 and a detect, pose or obb model, got {task:?}"
            )));
        }
        let (w, h) = image.dimensions();
        let windows = tile_windows(h, w, tile);
        if windows == [[0, 0, w, h]] {
            let mut results = self.predict_internal(&[image], &[path])?;
            return Ok(results.pop().unwrap_or_default());
        }

        let batch = self.config.batch.unwrap_or(1).max(1);
        let mut tiles: Vec<Results> = Vec::with_capacity(windows.len());
        for chunk in windows.chunks(batch) {
            let crops: Vec<_> = chunk
                .iter()
                .map(|&[x1, y1, x2, y2]| image.crop_imm(x1, y1, x2 - x1, y2 - y1))
                .collect();
            let refs: Vec<_> = crops.iter().collect();
            let paths = vec![path.clone(); crops.len()];
            for mut r in self.predict_internal(&refs, &paths)? {
                tiles.push(r.pop().ok_or_else(|| {
                    InferenceError::InferenceError("tile returned no result".into())
                })?);
            }
        }

        let mut out = tiles[0].clone();
        out.orig_img = crate::preprocessing::image_to_array(image);
        out.orig_shape = (h, w);
        out.path = path;
        let total = |f: fn(&Speed) -> Option<f64>| tiles.iter().map(|t| f(&t.speed)).sum();
        out.speed = Speed {
            preprocess: total(|s| s.preprocess),
            inference: total(|s| s.inference),
            postprocess: total(|s| s.postprocess),
        };

        merge_detections(&mut out, &tiles, &windows, task == Task::Obb, &self.config);
        Ok(vec![out])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_windows_match_the_documented_counts() {
        // Counts from the Ultralytics tiling guide: 1920x1080 at 640 and 3840x2160 at 1280 both run 8.
        assert_eq!(tile_windows(1080, 1920, 640).len(), 8);
        assert_eq!(tile_windows(2160, 3840, 1280).len(), 8);
        // The last window is aligned to the edge and the others step by round(0.8 * size).
        let w = tile_windows(1080, 1920, 640);
        assert_eq!(w[0], [0, 0, 640, 640]);
        assert_eq!(w[1], [512, 0, 1152, 640]);
        assert_eq!(w[3], [1280, 0, 1920, 640]);
        assert_eq!(w[7], [1280, 440, 1920, 1080]);
    }

    #[test]
    fn test_small_images_get_one_window() {
        assert_eq!(tile_windows(300, 500, 640), [[0, 0, 500, 300]]);
        assert_eq!(tile_windows(640, 640, 640), [[0, 0, 640, 640]]);
    }

    #[test]
    fn test_merge_drops_a_cut_fragment_another_window_sees_whole() {
        // Two 100x100 windows overlapping on x = 50..100 of a 150x100 image. A person cut by the
        // first window's right edge (x = 100) is seen whole by the second window.
        let windows = [[0, 0, 100, 100], [50, 0, 150, 100]];
        let cut = vec![60.0, 10.0, 100.0, 90.0, 0.9, 0.0];
        let whole = vec![60.0, 10.0, 110.0, 90.0, 0.8, 0.0];
        let rows = [cut, whole];
        let wins = [windows[0], windows[1]];
        let keep = merge_rows(&rows, &wins, &windows, (100, 150), false, 0.5, 300);
        assert_eq!(keep, [1]);
    }

    #[test]
    fn test_merge_keeps_a_cut_box_nothing_else_covers() {
        let windows = [[0, 0, 100, 100], [50, 0, 150, 100]];
        let cut = vec![60.0, 10.0, 100.0, 90.0, 0.9, 0.0];
        let keep = merge_rows(&[cut], &[windows[0]], &windows, (100, 150), false, 0.5, 300);
        assert_eq!(keep, [0]);
    }
}
