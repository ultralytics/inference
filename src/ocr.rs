// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Scene text OCR, the Rust port of Ultralytics `solutions.OCR`.
//!
//! A YOLO text detector finds words as boxes, oriented boxes or masks, [`TextRecognizer`] reads
//! each crop with `PaddleOCR`, and the texts come back in reading order.

// Pixel coordinates, crop sizes, point counts and class indices are far below the f32, u32 and
// isize limits. The float math follows OpenCV and Ultralytics step by step, fused or not, with
// their exact comparisons and loops, to round like them.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::while_float,
    clippy::suboptimal_flops,
    clippy::imprecise_flops,
    clippy::manual_midpoint
)]

use std::cmp::Ordering::Equal;
use std::path::Path;

use image::{DynamicImage, RgbImage, imageops};
use ndarray::{Array4, s};
use ort::session::Session;
use ort::value::TensorRef;

use crate::error::{InferenceError, Result};
use crate::model::YOLOModel;
use crate::preprocessing::{LetterboxGeometry, fused_zerocopy_preprocess};
use crate::results::Results;

/// `PaddleOCR` PP-OCRv6 recognizer, with its character dictionary in the ONNX metadata.
pub const DEFAULT_RECOGNIZER: &str = "PP-OCRv6_medium_rec.onnx";
/// `PaddleOCR` text-line orientation classifier, which flags upside-down crops.
pub const DEFAULT_ORIENTATION: &str = "PP-LCNet_x0_25_textline_ori.onnx";

/// Crops per recognizer and classifier run, as in Ultralytics.
const BATCH: usize = 16;
/// Recognizer input height, minimum padded width and maximum width.
const REC_H: u32 = 48;
const REC_MIN_W: u32 = 320;
const REC_MAX_W: u32 = 3200;
/// Orientation classifier input size and `ImageNet` normalization.
const ORI_H: u32 = 80;
const ORI_W: u32 = 160;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// The four corners of a box or oriented box in source pixels.
type Corners = [[f32; 2]; 4];

/// One detected text, as in Ultralytics `SolutionResults.texts`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Text {
    /// Recognized text, empty when nothing was read.
    pub text: String,
    /// Recognition confidence, separate from the detector confidence.
    pub confidence: f32,
    /// Detector confidence.
    pub detection_confidence: f32,
    /// Index of the detection before reading-order sorting.
    pub detection_index: usize,
    /// The box or oriented box corners, or the mask outline of a segmentation detector, in source
    /// pixels.
    pub polygon: Vec<[f32; 2]>,
}

/// Text recognition for the boxes, oriented boxes or masks of a YOLO text detector, with
/// `PaddleOCR` or a function of your own.
///
/// # Example
///
/// ```no_run
/// use ultralytics_inference::YOLOModel;
/// use ultralytics_inference::ocr::{DEFAULT_ORIENTATION, DEFAULT_RECOGNIZER, TextRecognizer};
///
/// let mut detector = YOLOModel::load("yolo26n-text.onnx")?;
/// let mut ocr = TextRecognizer::new(&detector, DEFAULT_RECOGNIZER, DEFAULT_ORIENTATION)?;
/// let image = image::open("sign.jpg")?;
/// let result = detector.predict_image(&image, "sign.jpg".into())?.remove(0);
/// for text in ocr.read(&image, &result)? {
///     println!("{} {:.2}", text.text, text.confidence);
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct TextRecognizer(Backend);

/// A function reading `(text, confidence)` from an RGB crop.
type Recognize = Box<dyn FnMut(&RgbImage) -> (String, f32) + Send>;

enum Backend {
    Paddle(Paddle),
    Custom(Recognize),
}

/// The `PaddleOCR` recognizer and text-line orientation classifier.
struct Paddle {
    recognizer: Session,
    orientation: Session,
    /// CTC classes: the blank, the dictionary, then a space.
    characters: Vec<char>,
}

impl TextRecognizer {
    /// Load a recognizer and an orientation classifier, each an ONNX path that is downloaded
    /// when missing and known, on the device of `detector`. The recognizer is
    /// [`DEFAULT_RECOGNIZER`] or a `PaddleOCR` PP-OCRv5 language model for other scripts, such as
    /// `korean_PP-OCRv5_mobile_rec.onnx`.
    ///
    /// # Errors
    ///
    /// Returns an error if a model fails to load or the recognizer has no `character` dictionary
    /// in its metadata.
    pub fn new(
        detector: &YOLOModel,
        recognizer: impl AsRef<Path>,
        orientation: impl AsRef<Path>,
    ) -> Result<Self> {
        let recognizer = detector.companion_session(recognizer.as_ref())?;
        let orientation = detector.companion_session(orientation.as_ref())?;
        let dictionary = recognizer
            .metadata()
            .ok()
            .and_then(|m| m.custom("character"))
            .ok_or_else(|| {
                InferenceError::ModelLoadError(
                    "the recognizer has no `character` dictionary in its ONNX metadata".into(),
                )
            })?;
        let characters = std::iter::once('\0')
            .chain(dictionary.split('\n').filter_map(|c| c.chars().next()))
            .chain([' '])
            .collect();
        Ok(Self(Backend::Paddle(Paddle {
            recognizer,
            orientation,
            characters,
        })))
    }

    /// Read texts with `recognize` instead of `PaddleOCR`, like Ultralytics
    /// `recognizer=<callable>`: it gets each non-empty RGB crop on its own and returns the text and
    /// its confidence, without the orientation check.
    ///
    /// # Example
    ///
    /// ```
    /// use ultralytics_inference::ocr::TextRecognizer;
    ///
    /// let ocr = TextRecognizer::custom(|crop| (format!("{}x{}", crop.width(), crop.height()), 1.0));
    /// ```
    #[must_use]
    pub fn custom(recognize: impl FnMut(&RgbImage) -> (String, f32) + Send + 'static) -> Self {
        Self(Backend::Custom(Box::new(recognize)))
    }

    /// Read the text in every box, oriented box or mask of a detector `result` on `image`, in
    /// reading order. A mask is read from the minimum-area rectangle around its outline, like
    /// Ultralytics. A detection whose crop is empty keeps an empty text.
    ///
    /// # Errors
    ///
    /// Returns an error if recognition fails.
    pub fn read(&mut self, image: &DynamicImage, result: &Results) -> Result<Vec<Text>> {
        let rgb = image.to_rgb8();
        let (mut texts, mut pending, mut crops) = (Vec::new(), Vec::new(), Vec::new());
        for (i, (polygon, corners, detection_confidence, oriented)) in
            detections(result).into_iter().enumerate()
        {
            let mut crop = if oriented {
                crop_obb(&rgb, corners)
            } else {
                crop_box(&rgb, corners)
            };
            // Tall crops are read sideways, like PaddleOCR, for steep text.
            if f64::from(crop.height()) >= 1.5 * f64::from(crop.width()) {
                crop = imageops::rotate270(&crop);
            }
            if !crop.is_empty() {
                pending.push(i);
                crops.push(crop);
            }
            texts.push(Text {
                detection_confidence,
                detection_index: i,
                polygon,
                ..Text::default()
            });
        }
        let reads = match &mut self.0 {
            Backend::Paddle(paddle) => paddle.recognize(&crops)?,
            Backend::Custom(recognize) => crops.iter().map(recognize).collect(),
        };
        for (i, (text, confidence)) in pending.into_iter().zip(reads) {
            texts[i].text = text;
            texts[i].confidence = confidence;
        }
        let polygons: Vec<&[[f32; 2]]> = texts.iter().map(|t| t.polygon.as_slice()).collect();
        Ok(reading_order(&polygons)
            .into_iter()
            .map(|i| texts[i].clone())
            .collect())
    }
}

impl Paddle {
    /// Read every crop, then read the crops the classifier finds upside down again turned by 180
    /// degrees and keep the more confident reading.
    fn recognize(&mut self, crops: &[RgbImage]) -> Result<Vec<(String, f32)>> {
        let mut reads = self.read_crops(crops)?;
        let flip = self.upside_down(crops)?;
        let turned: Vec<RgbImage> = flip
            .iter()
            .map(|&i| imageops::rotate180(&crops[i]))
            .collect();
        for (&i, read) in flip.iter().zip(self.read_crops(&turned)?) {
            if read.1 > reads[i].1 {
                reads[i] = read;
            }
        }
        Ok(reads)
    }

    /// Recognize crops in batches, in BGR scaled to [-1, 1] and padded to the widest crop of
    /// their batch, then CTC decode them.
    fn read_crops(&mut self, crops: &[RgbImage]) -> Result<Vec<(String, f32)>> {
        let mut reads = Vec::with_capacity(crops.len());
        for chunk in crops.chunks(BATCH) {
            let sizes: Vec<(u32, u32)> = chunk.iter().map(rec_width).collect();
            let batch = tensor(chunk, REC_H, &sizes, ([0.5; 3], [0.5; 3], true));
            let characters = &self.characters;
            run(&mut self.recognizer, &batch, |shape, probs| {
                let classes = shape[2] as usize;
                let rows = probs.chunks(shape[1] as usize * classes);
                reads.extend(rows.map(|row| ctc_decode(row, classes, characters)));
            })?;
        }
        Ok(reads)
    }

    /// Indices of the crops the orientation classifier calls upside down.
    fn upside_down(&mut self, crops: &[RgbImage]) -> Result<Vec<usize>> {
        let mut flip = Vec::new();
        for (b, chunk) in crops.chunks(BATCH).enumerate() {
            let sizes = &[(ORI_W, ORI_W); BATCH][..chunk.len()];
            let batch = tensor(chunk, ORI_H, sizes, (MEAN, STD, false));
            run(&mut self.orientation, &batch, |_, scores| {
                let turned = scores.chunks(2).enumerate().filter(|(_, s)| s[1] > s[0]);
                flip.extend(turned.map(|(i, _)| b * BATCH + i));
            })?;
        }
        Ok(flip)
    }
}

/// Greedy CTC decoding: the first best class per step, without repeats and blanks, scored by
/// the mean probability of the kept steps.
fn ctc_decode(probs: &[f32], classes: usize, characters: &[char]) -> (String, f32) {
    let (mut text, mut sum, mut kept, mut prev) = (String::new(), 0.0_f32, 0_usize, 0);
    for step in probs.chunks(classes) {
        let first_max = |m: (usize, f32), (k, &v): (usize, &f32)| if v > m.1 { (k, v) } else { m };
        let (best, p) = step.iter().enumerate().fold((0, f32::MIN), first_max);
        if best != 0 && best != prev {
            text.push(characters.get(best).copied().unwrap_or(' '));
            sum += p;
            kept += 1;
        }
        prev = best;
    }
    (text, if kept == 0 { 0.0 } else { sum / kept as f32 })
}

/// `(resized, padded)` recognizer widths of a crop: its width at the recognizer height, and
/// that or the minimum width when it is narrower, both capped at the maximum.
fn rec_width(crop: &RgbImage) -> (u32, u32) {
    let ratio = f64::from(crop.width()) / f64::from(crop.height());
    let padded = (f64::from(REC_H) * ratio.max(f64::from(REC_MIN_W) / f64::from(REC_H))) as u32;
    let resized = ((f64::from(REC_H) * ratio).ceil() as u32).min(padded);
    (resized.min(REC_MAX_W), padded.min(REC_MAX_W))
}

/// A batch of `crops`, each resized to `height` and its `(resized, padded)` width with the
/// `OpenCV` `INTER_LINEAR` resize of the YOLO preprocessing, normalized by `mean` and `std`, in
/// BGR when `bgr`, and zero-padded to the widest padded width.
fn tensor(
    crops: &[RgbImage],
    height: u32,
    sizes: &[(u32, u32)],
    (mean, std, bgr): ([f32; 3], [f32; 3], bool),
) -> Array4<f32> {
    let width = sizes.iter().map(|s| s.1).max().unwrap_or(0) as usize;
    let mut batch = Array4::zeros((crops.len(), 3, height as usize, width));
    for (i, (crop, &(w, _))) in crops.iter().zip(sizes).enumerate() {
        let geom = LetterboxGeometry {
            new_w: w,
            new_h: height,
            pad_left: 0,
            pad_top: 0,
        };
        let (cw, ch) = crop.dimensions();
        let size = (height as usize, w as usize);
        let t = fused_zerocopy_preprocess(crop.as_raw(), cw, ch, size, &geom);
        for c in 0..3 {
            let src = t.slice(s![0, if bgr { 2 - c } else { c }, .., ..]);
            batch
                .slice_mut(s![i, c, .., ..w as usize])
                .zip_mut_with(&src, |o, &v| *o = (v - mean[c]) / std[c]);
        }
    }
    batch
}

/// Run a batch and hand the output shape and values to `read`.
fn run(
    session: &mut Session,
    batch: &Array4<f32>,
    read: impl FnOnce(&[i64], &[f32]),
) -> Result<()> {
    let fail = |e: ort::Error| InferenceError::InferenceError(e.to_string());
    let input = TensorRef::from_array_view(batch.view()).map_err(fail)?;
    let outputs = session.run(ort::inputs![input]).map_err(fail)?;
    let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(fail)?;
    read(shape, data);
    Ok(())
}

/// `(polygon, corners, confidence, oriented)` of every detection, as Ultralytics `OCR` takes
/// them: a mask outline with the corners of its minimum-area rectangle, the oriented box
/// corners, or the box corners clockwise from the top left.
fn detections(result: &Results) -> Vec<(Vec<[f32; 2]>, Corners, f32, bool)> {
    let Some(boxes) = &result.boxes else {
        return result.obb.as_ref().map_or_else(Vec::new, |obb| {
            let (c, conf) = (obb.xyxyxyxy(), obb.conf());
            (0..obb.len())
                .map(|i| {
                    let corners: Corners = std::array::from_fn(|j| [c[[i, j, 0]], c[[i, j, 1]]]);
                    (corners.to_vec(), corners, conf[i], true)
                })
                .collect()
        });
    };
    let conf = boxes.conf();
    if let Some(masks) = &result.masks {
        return masks
            .xy()
            .into_iter()
            .zip(conf.iter().copied())
            .map(|(outline, conf)| {
                let polygon: Vec<[f32; 2]> =
                    outline.rows().into_iter().map(|p| [p[0], p[1]]).collect();
                let corners = min_area_corners(&polygon);
                (polygon, corners, conf, true)
            })
            .collect();
    }
    let b = boxes.xyxy();
    (0..boxes.len())
        .map(|i| {
            let [x1, y1, x2, y2] = [0, 1, 2, 3].map(|k| b[[i, k]]);
            let corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]];
            (corners.to_vec(), corners, conf[i], false)
        })
        .collect()
}

/// Ultralytics `xywhr2xyxyxyxy(xyxyxyxy2xywhr(polygon))`: the corners of the `cv2.minAreaRect`
/// of `points`, with `w` the longer side and the angle in `[-pi/4, 3pi/4)`, in f64 and then f32
/// as `save_one_box` casts them.
fn min_area_corners(points: &[[f32; 2]]) -> Corners {
    use std::f64::consts::PI;
    let ([cx, cy], [w, h], angle) = min_area_rect(points);
    let (mut w, mut h) = (f64::from(w), f64::from(h));
    let mut theta = f64::from(angle) / 180.0 * PI;
    if w < h {
        (w, h) = (h, w);
        theta += PI / 2.0;
    }
    while theta >= 3.0 * PI / 4.0 {
        theta -= PI;
    }
    while theta < -PI / 4.0 {
        theta += PI;
    }
    let (cos, sin) = (theta.cos(), theta.sin());
    let (v1, v2) = (
        [w / 2.0 * cos, w / 2.0 * sin],
        [-h / 2.0 * sin, h / 2.0 * cos],
    );
    let c = [f64::from(cx), f64::from(cy)];
    [(1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0)]
        .map(|(a, b)| std::array::from_fn(|k| (c[k] + a * v1[k] + b * v2[k]) as f32))
}

/// `OpenCV` 5 `cv::minAreaRect`: `(center, (width, height), angle in degrees)` of the smallest
/// rectangle around `points`, by rotating calipers over their convex hull, in the same f32 and
/// f64 steps.
fn min_area_rect(points: &[[f32; 2]]) -> ([f32; 2], [f32; 2], f32) {
    let hull = convex_hull(points);
    let mut angle = -std::f64::consts::FRAC_PI_2;
    let (mut center, mut size) = ([0.0; 2], [0.0_f32; 2]);
    match hull.len() {
        0 => {}
        1 => center = hull[0],
        2 => {
            let (p, q) = (hull[0], hull[1]);
            center = [(p[0] + q[0]) * 0.5, (p[1] + q[1]) * 0.5];
            let (dx, dy) = (f64::from(p[0] - q[0]), f64::from(p[1] - q[1]));
            size = [0.0, (dx * dx + dy * dy).sqrt() as f32];
            if dx == 0.0 {
                size.swap(0, 1);
            } else if dy < 0.0 {
                angle = dy.atan2(dx);
                size.swap(0, 1);
            } else if dy > 0.0 {
                angle = -dx.atan2(dy);
            }
        }
        _ => {
            let [corner, up, across] = rotating_calipers(&hull);
            center = [
                corner[0] + (up[0] + across[0]) * 0.5,
                corner[1] + (up[1] + across[1]) * 0.5,
            ];
            let len = |v: [f32; 2]| {
                let (vx, vy) = (f64::from(v[0]), f64::from(v[1]));
                (vx * f64::from(v[0]) + vy * f64::from(v[1])).sqrt() as f32
            };
            size = [len(across), len(up)];
            if up[0] == 0.0 && up[1] > 0.0 {
                size.swap(0, 1);
            } else {
                angle = -f64::from(up[0]).atan2(f64::from(up[1]));
            }
        }
    }
    (center, size, (angle * 180.0 / std::f64::consts::PI) as f32)
}

/// `OpenCV` `rotatingCalipers` in `CALIPERS_MINAREARECT` mode for a counterclockwise hull: the
/// rectangle corner and its two side vectors.
fn rotating_calipers(points: &[[f32; 2]]) -> [[f32; 2]; 3] {
    let n = points.len();
    let (mut vect, mut inv_len) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let ([mut left, mut bottom, mut right, mut top], mut pt0) = ([0; 4], points[0]);
    let (mut left_x, mut right_x, mut top_y, mut bottom_y) = (pt0[0], pt0[0], pt0[1], pt0[1]);
    for i in 0..n {
        if pt0[0] < left_x {
            (left_x, left) = (pt0[0], i);
        }
        if pt0[0] > right_x {
            (right_x, right) = (pt0[0], i);
        }
        if pt0[1] > top_y {
            (top_y, top) = (pt0[1], i);
        }
        if pt0[1] < bottom_y {
            (bottom_y, bottom) = (pt0[1], i);
        }
        let pt = points[(i + 1) % n];
        let (dx, dy) = (f64::from(pt[0] - pt0[0]), f64::from(pt[1] - pt0[1]));
        vect.push([dx as f32, dy as f32]);
        inv_len.push((1.0 / (dx * dx + dy * dy).sqrt()) as f32);
        pt0 = pt;
    }
    let mut seq = [bottom, right, top, left];
    let (mut min_area, mut best) = (f32::MAX, (0, 0.0, 0.0, 0.0, 0.0, 0));
    for _ in 0..n {
        // The caliper side with the smallest angle to its next hull edge, the first of equals.
        let rot = [
            vect[seq[0]],
            [vect[seq[1]][1], -vect[seq[1]][0]],
            [-vect[seq[2]][0], -vect[seq[2]][1]],
            [-vect[seq[3]][1], vect[seq[3]][0]],
        ];
        let right_of = |v: [f32; 2], w: [f32; 2]| v[1] * w[0] + -v[0] * w[1] < 0.0;
        let main = (1..4).fold(0, |m, i| if right_of(rot[i], rot[m]) { i } else { m });
        let p = seq[main];
        let (lx, ly) = (vect[p][0] * inv_len[p], vect[p][1] * inv_len[p]);
        let (base_a, base_b) = [(lx, ly), (ly, -lx), (-lx, -ly), (-ly, lx)][main];
        seq[main] = (seq[main] + 1) % n;
        let d = |i: usize, j: usize| [points[i][0] - points[j][0], points[i][1] - points[j][1]];
        let [dx, dy] = d(seq[1], seq[3]);
        let width = dx * base_a + dy * base_b;
        let [dx, dy] = d(seq[2], seq[0]);
        let height = -dx * base_b + dy * base_a;
        let area = width * height;
        if area <= min_area {
            min_area = area;
            best = (seq[3], base_a, width, base_b, height, seq[0]);
        }
    }
    let (left, a1, width, b1, height, bottom) = best;
    let (a2, b2) = (-b1, a1);
    let c1 = a1 * points[left][0] + points[left][1] * b1;
    let c2 = a2 * points[bottom][0] + points[bottom][1] * b2;
    let idet = 1.0 / (a1 * b2 - a2 * b1);
    [
        [(c1 * b2 - c2 * b1) * idet, (a1 * c2 - a2 * c1) * idet],
        [a1 * width, b1 * width],
        [a2 * height, b2 * height],
    ]
}

/// `OpenCV` 5 `cv::convexHull` of f32 points, counterclockwise and as points, with its Sklansky
/// scan over the points sorted by x, its tie handling and its starting point.
fn convex_hull(points: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let total = points.len();
    if total == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..total).collect();
    order.sort_by(|&a, &b| {
        let (p, q) = (points[a], points[b]);
        p[0].partial_cmp(&q[0])
            .unwrap_or(Equal)
            .then(p[1].partial_cmp(&q[1]).unwrap_or(Equal))
            .then(a.cmp(&b))
    });
    let pt = |i: isize| points[order[i as usize]];
    let (mut miny, mut maxy) = (0, 0);
    for i in 1..total as isize {
        let y = pt(i)[1];
        if pt(miny)[1] > y {
            miny = i;
        }
        if pt(maxy)[1] < y {
            maxy = i;
        }
    }
    let last = total as isize - 1;
    let mut hull: Vec<isize> = Vec::new();
    if pt(0) == pt(last) {
        hull.push(0);
    } else {
        // The upper half, its right part first when counterclockwise.
        let tr = sklansky(&pt, 0, maxy, -1, 1);
        let tl = sklansky(&pt, last, maxy, -1, -1);
        hull.extend(&tl[..tl.len() - 1]);
        hull.extend(tr[1..].iter().rev());
        let stop = if tr.len() > 2 {
            tr[1]
        } else if tl.len() > 2 {
            tl[tl.len() - 2]
        } else {
            -1
        };
        let (mut bl, mut br) = (
            sklansky(&pt, 0, miny, 1, -1),
            sklansky(&pt, last, miny, 1, 1),
        );
        if stop >= 0 {
            let check = if bl.len() > 2 {
                bl[1]
            } else if bl.len() + br.len() > 2 {
                br[2 - bl.len()]
            } else {
                -1
            };
            // Points all on one line: the lower half mirrors the upper one.
            if check == stop || (check >= 0 && pt(check) == pt(stop)) {
                bl.truncate(2);
                br.truncate(2);
            }
        }
        hull.extend(&bl[..bl.len() - 1]);
        hull.extend(br[1..].iter().rev());
    }
    let mut hull: Vec<usize> = hull.into_iter().map(|i| order[i as usize]).collect();
    rotate_to_monotone(&mut hull);
    hull.into_iter().map(|i| points[i]).collect()
}

/// `OpenCV`'s cyclic shift that starts hull indices `idx` at their smallest or largest value
/// when that makes them one ascending or descending run.
fn rotate_to_monotone(idx: &mut [usize]) {
    let n = idx.len();
    if n < 3 {
        return;
    }
    let (mut min_i, mut max_i, mut lt) = (0, 0, 0);
    for i in 1..n {
        lt += usize::from(idx[i - 1] < idx[i]);
        if lt > 1 && lt + 2 <= i {
            break;
        }
        if idx[i] < idx[min_i] {
            min_i = i;
        }
        if idx[i] > idx[max_i] {
            max_i = i;
        }
    }
    let gap = min_i.abs_diff(max_i);
    if (gap == 1 || gap == n - 1) && (lt <= 1 || lt + 2 >= n) {
        let ascending = (max_i + 1) % n == min_i;
        let start = if ascending { min_i } else { max_i };
        let monotone =
            (0..n - 1).all(|i| (idx[(start + i) % n] < idx[(start + i + 1) % n]) == ascending);
        if start > 0 && monotone {
            idx.rotate_left(start);
        }
    }
}

/// `OpenCV` `Sklansky_` over the sorted points `pt` from `start` to `end`: the hull chain
/// indices, the angle at each point tested with unit vectors normalized in f64 and stored in f32, and
/// an f64 cross product.
fn sklansky(
    pt: &impl Fn(isize) -> [f32; 2],
    start: isize,
    end: isize,
    nsign: i32,
    sign2: i32,
) -> Vec<isize> {
    let sign = |v: f64| i32::from(v > 0.0) - i32::from(v < 0.0);
    let unit = |v: [f32; 2]| {
        let (x, y) = (f64::from(v[0]), f64::from(v[1]));
        let norm = (x * x + y * y).sqrt();
        let k = if norm == 0.0 { 0.0 } else { 1.0 / norm };
        v.map(|c| (f64::from(c) * k) as f32)
    };
    let incr = if end > start { 1 } else { -1 };
    if start == end || pt(start) == pt(end) {
        return vec![start];
    }
    let (mut pprev, mut pcur, mut pnext) = (start, start + incr, start + 2 * incr);
    let mut stack = vec![pprev, pcur, pnext];
    let after = end + incr;
    while pnext != after {
        let (p0, p1, p2) = (pt(pprev), pt(pcur), pt(pnext));
        let by = p2[1] - p1[1];
        if sign(f64::from(by)) == nsign {
            pnext += incr;
            *stack.last_mut().expect("the stack holds three points") = pnext;
            continue;
        }
        let a = unit([p1[0] - p0[0], p1[1] - p0[1]]);
        let b = unit([p2[0] - p1[0], by]);
        let convexity = f64::from(a[1]) * f64::from(b[0]) - f64::from(a[0]) * f64::from(b[1]);
        let n = stack.len();
        if sign(convexity) == sign2 && (a[0] != 0.0 || a[1] != 0.0) {
            (pprev, pcur) = (pcur, pnext);
            pnext += incr;
            stack.push(pnext);
        } else if pprev == start {
            pcur = pnext;
            pnext += incr;
            (stack[1], stack[2]) = (pcur, pnext);
        } else {
            stack[n - 2] = pnext;
            (pcur, pprev) = (pprev, stack[n - 4]);
            stack.pop();
        }
    }
    stack.pop();
    stack
}

/// Crop an axis-aligned box like Ultralytics `save_one_box` with no gain or padding: `xyxy ->
/// xywh -> xyxy` in f32, truncated toward zero and clipped to the image.
fn crop_box(src: &RgbImage, p: Corners) -> RgbImage {
    let ([x1, y1], [x2, y2]) = (p[0], p[2]);
    let (cx, cy, w, h) = ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1);
    // `as u32` truncates and stops at 0, and `crop_imm` clips the rest to the image.
    let [left, top, right, bottom] =
        [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0].map(|v| v as u32);
    let (width, height) = (right.saturating_sub(left), bottom.saturating_sub(top));
    imageops::crop_imm(src, left, top, width, height).to_image()
}

/// Warp an oriented box upright into a crop, as Ultralytics `save_one_box` does: the edge
/// spanning furthest left to right becomes the top, and `cv2.warpAffine` samples bilinearly with
/// black outside the image. The sampling repeats `OpenCV` 5's float `warpAffine` on x86, whose
/// SIMD loop maps 16 columns at a time and whose scalar loop maps the remaining columns with a
/// differently fused multiply-add.
fn crop_obb(src: &RgbImage, corners: Corners) -> RgbImage {
    // `p[j] = corners[(j + end - 2) % 4]`, with `end` the corner ending the longest leftward run.
    let run = |i: usize| corners[(i + 3) % 4][0] - corners[i][0];
    let end = (1..4).fold(0, |best, i| if run(i) > run(best) { i } else { best });
    let [p0, p1, p2, _] = std::array::from_fn(|j| corners[(j + end + 2) % 4]);
    let side = |a: [f32; 2], b: [f32; 2]| {
        let (dx, dy) = (a[0] - b[0], a[1] - b[1]);
        ((dx * dx + dy * dy).sqrt().round_ties_even() as u32).max(1)
    };
    let (w, h) = (side(p0, corners[(end + 1) % 4]), side(p0, p1));
    // The inverse map from the crop to the source: (0, 0) -> p2, (w, 0) -> p1, (w, h) -> p0.
    let [p0, p1, p2] = [p0, p1, p2].map(|p| p.map(f64::from));
    let (wf, hf) = (f64::from(w), f64::from(h));
    let row = |k: usize| [(p1[k] - p2[k]) / wf, (p0[k] - p1[k]) / hf, p2[k]].map(|v| v as f32);
    let (mx, my) = (row(0), row(1));
    let (sw, sh) = (i64::from(src.width()), i64::from(src.height()));
    let pixel = |x: i64, y: i64| {
        if (0..sw).contains(&x) && (0..sh).contains(&y) {
            src.get_pixel(x as u32, y as u32).0.map(f32::from)
        } else {
            [0.0; 3]
        }
    };
    let simd = w - w % 16;
    RgbImage::from_fn(w, h, |x, y| {
        let (xf, yf) = (x as f32, y as f32);
        let map = |m: [f32; 3]| {
            if x < simd {
                m[0].mul_add(xf, yf * m[1] + m[2])
            } else {
                xf.mul_add(m[0], yf * m[1]) + m[2]
            }
        };
        let (sx, sy) = (map(mx), map(my));
        let (fx, fy) = (sx - sx.floor(), sy - sy.floor());
        let (ix, iy) = (sx.floor() as i64, sy.floor() as i64);
        let (p00, p01) = (pixel(ix, iy), pixel(ix + 1, iy));
        let (p10, p11) = (pixel(ix, iy + 1), pixel(ix + 1, iy + 1));
        image::Rgb(std::array::from_fn(|c| {
            let v0 = fx.mul_add(p01[c] - p00[c], p00[c]);
            let v1 = fx.mul_add(p11[c] - p10[c], p10[c]);
            fy.mul_add(v1 - v0, v0).round_ties_even().clamp(0.0, 255.0) as u8
        }))
    })
}

/// Detection indices in reading order: polygons are grouped into lines by their center height,
/// lines go top to bottom and each line left to right, as Ultralytics `OCR.reading_order` does.
fn reading_order(polygons: &[&[[f32; 2]]]) -> Vec<usize> {
    // (mean y, min x, height) per polygon, in f32 like the NumPy points.
    let keys: Vec<(f32, f32, f32)> = polygons
        .iter()
        .map(|p| {
            let ys: Vec<f32> = p.iter().map(|c| c[1]).collect();
            let (lo, hi) = ys
                .iter()
                .fold((f32::MAX, f32::MIN), |m, &y| (m.0.min(y), m.1.max(y)));
            let min_x = p.iter().map(|c| c[0]).fold(f32::INFINITY, f32::min);
            (pairwise_sum(&ys) / ys.len() as f32, min_x, hi - lo)
        })
        .collect();
    let mut sorted: Vec<usize> = (0..polygons.len()).collect();
    sorted.sort_by(|&a, &b| {
        (keys[a].0, keys[a].1)
            .partial_cmp(&(keys[b].0, keys[b].1))
            .unwrap_or(Equal)
    });
    // (mean center height, tallest text, members) per line.
    let mut lines: Vec<(f64, f64, Vec<usize>)> = Vec::new();
    for i in sorted {
        let (y, h) = (f64::from(keys[i].0), f64::from(keys[i].2));
        match lines.last_mut() {
            Some((line_y, line_h, members))
                if (y - *line_y).abs() <= h.max(*line_h).max(1.0) * 0.5 =>
            {
                let n = members.len() as f64;
                *line_y = (*line_y * n + y) / (n + 1.0);
                *line_h = line_h.max(h);
                members.push(i);
            }
            _ => lines.push((y, h, vec![i])),
        }
    }
    lines.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Equal));
    lines
        .into_iter()
        .flat_map(|(_, _, mut members)| {
            members.sort_by(|&a, &b| (keys[a].1, a).partial_cmp(&(keys[b].1, b)).unwrap_or(Equal));
            members
        })
        .collect()
}

/// `NumPy`'s float32 pairwise sum, which `mean` uses: 8 running sums in blocks of up to 128
/// values, halved recursively above that.
fn pairwise_sum(v: &[f32]) -> f32 {
    let n = v.len();
    if n < 8 {
        return v.iter().fold(0.0, |s, &x| s + x);
    }
    if n > 128 {
        let half = n / 2 - n / 2 % 8;
        return pairwise_sum(&v[..half]) + pairwise_sum(&v[half..]);
    }
    let mut r: [f32; 8] = std::array::from_fn(|j| v[j]);
    let full = n - n % 8;
    for chunk in v[8..full].chunks(8) {
        for (r, &x) in r.iter_mut().zip(chunk) {
            *r += x;
        }
    }
    let sum = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    v[full..].iter().fold(sum, |s, &x| s + x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ctc_decode_drops_blanks_and_repeats() {
        // Steps pick classes 1, 1, 0, 1, 2, 2: the repeat merges, the blank splits "a a".
        let characters = ['\0', 'a', 'b', ' '];
        let best = [1, 1, 0, 1, 2, 2];
        let probs: Vec<f32> = best
            .iter()
            .flat_map(|&k| (0..4).map(move |c| if c == k { 0.5 } else { 0.1 }))
            .collect();
        assert_eq!(ctc_decode(&probs, 4, &characters), ("aab".into(), 0.5));
        assert_eq!(
            ctc_decode(&[0.9, 0.1, 0.0, 0.0], 4, &characters),
            (String::new(), 0.0)
        );
    }

    #[test]
    fn test_rec_width_pads_to_the_minimum_and_caps_long_crops() {
        // 48 x 96 is 96 wide at height 48, padded to 320; 10 x 100 is 480 wide.
        assert_eq!(rec_width(&RgbImage::new(96, 48)), (96, 320));
        assert_eq!(rec_width(&RgbImage::new(100, 10)), (480, 480));
        assert_eq!(rec_width(&RgbImage::new(1000, 10)), (3200, 3200));
    }

    #[test]
    fn test_upright_oriented_box_crops_like_its_box_from_any_first_corner() {
        let src = RgbImage::from_fn(40, 30, |x, y| image::Rgb([x as u8, y as u8, 7]));
        // An unrotated 20 x 10 box at (10, 5) in Ultralytics `xywhr2xyxyxyxy` corner order.
        let corners = [[30.0, 15.0], [30.0, 5.0], [10.0, 5.0], [10.0, 15.0]];
        let boxed = crop_box(&src, [[10.0, 5.0], [30.0, 5.0], [30.0, 15.0], [10.0, 15.0]]);
        assert_eq!(boxed.dimensions(), (20, 10));
        assert_eq!(boxed.get_pixel(0, 0).0, [10, 5, 7]);
        for start in 0..4 {
            let rolled = std::array::from_fn(|i| corners[(i + start) % 4]);
            assert_eq!(crop_obb(&src, rolled), boxed);
        }
    }

    #[test]
    fn test_min_area_rect_matches_opencv() {
        // Grid points with a duplicate-free collinear run, quarter pixels, and a single point;
        // `cv2.minAreaRect` and Ultralytics `xywhr2xyxyxyxy(xyxyxyxy2xywhr(points))` results.
        let grid = [
            [0.0, 11.0],
            [3.0, 6.0],
            [11.0, 10.0],
            [3.0, 4.0],
            [9.0, 5.0],
            [2.0, 11.0],
        ];
        let rect = (
            [5.134_146, 7.207_317_4],
            [11.043_155, 6.846_755_5],
            -6.340_191_4,
        );
        assert_eq!(min_area_rect(&grid), rect);
        let quarters = [[24.25, 23.75], [13.0, 7.75], [35.5, 42.25], [34.0, 18.5]];
        assert_eq!(
            min_area_corners(&quarters),
            [
                [35.500_004, 42.25],
                [45.314_66, 35.849_14],
                [22.814_657, 1.349_139],
                [13.000_001, 7.750_001_4],
            ]
        );
        assert_eq!(
            min_area_rect(&[[3.0, 4.0]]),
            ([3.0, 4.0], [0.0, 0.0], -90.0)
        );
    }

    #[test]
    fn test_reading_order_goes_by_line_then_left_to_right() {
        let quad = |x: f32, y: f32| [[x, y], [x + 10.0, y], [x + 10.0, y + 8.0], [x, y + 8.0]];
        // Two lines: 0 and 2 share a line with 2 a little lower, 1 is the next line down.
        let polygons = [
            quad(50.0, 10.0),
            quad(5.0, 40.0),
            quad(0.0, 12.0),
            quad(30.0, 41.0),
        ];
        let polygons = polygons.each_ref().map(<[[f32; 2]; 4]>::as_slice);
        assert_eq!(reading_order(&polygons), [2, 0, 1, 3]);
        assert_eq!(reading_order(&[]), Vec::<usize>::new());
    }

    #[test]
    fn test_custom_recognizer_reads_each_box_crop_in_reading_order() {
        use crate::results::{Boxes, Speed};
        let mut result = Results::new(
            ndarray::Array3::zeros((40, 60, 3)),
            String::new(),
            std::sync::Arc::default(),
            Speed::default(),
            (40, 60),
        );
        // A box on the second line, one on the first, and one outside the image.
        let data = ndarray::array![
            [0.0, 20.0, 30.0, 30.0, 0.8, 0.0],
            [10.0, 0.0, 50.0, 10.0, 0.9, 0.0],
            [70.0, 0.0, 80.0, 10.0, 0.7, 0.0],
        ];
        result.boxes = Some(Boxes::new(data, (40, 60)));
        let mut ocr = TextRecognizer::custom(|c| (format!("{}x{}", c.width(), c.height()), 0.5));
        let image = DynamicImage::new_rgb8(60, 40);
        let texts = ocr.read(&image, &result).unwrap();
        let read: Vec<_> = texts
            .iter()
            .map(|t| (t.text.as_str(), t.detection_index))
            .collect();
        assert_eq!(read, [("40x10", 1), ("", 2), ("30x10", 0)]);
    }

    /// The bus photo through YOLO26n-Text and PP-OCRv6 reads its destination sign. Skips when the
    /// models cannot be downloaded, like the batch test.
    #[test]
    fn test_text_recognizer_reads_the_bus_sign() {
        let _guard = crate::model::SESSION_BUILD_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Ok(mut detector) = YOLOModel::load("yolo26n-text.onnx") else {
            return;
        };
        let Ok(mut ocr) = TextRecognizer::new(&detector, DEFAULT_RECOGNIZER, DEFAULT_ORIENTATION)
        else {
            return;
        };
        let image = image::open("assets/bus.jpg").unwrap();
        let result = detector
            .predict_image(&image, String::new())
            .unwrap()
            .remove(0);
        let texts = ocr.read(&image, &result).unwrap();
        assert_eq!(
            texts.len(),
            result.obb.as_ref().map_or(0, crate::results::Obb::len)
        );
        assert!(
            texts
                .iter()
                .any(|t| t.text == "MADRID" && t.confidence > 0.9)
        );
    }
}
