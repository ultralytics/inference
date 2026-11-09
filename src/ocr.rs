// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Scene text OCR, the Rust port of Ultralytics `solutions.OCR`.
//!
//! A YOLO text detector finds words as boxes or oriented boxes, [`TextRecognizer`] reads each
//! crop with `PaddleOCR`, and the texts come back in reading order.

// Pixel coordinates, crop sizes and class indices are far below the f32 and u32 limits. The
// float math follows OpenCV and Ultralytics step by step, fused or not, to round like them.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::suboptimal_flops,
    clippy::imprecise_flops,
    clippy::manual_midpoint
)]

use std::path::Path;

use image::{DynamicImage, RgbImage, imageops};
use ndarray::{Array4, Axis, s};
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
const ORI_SIZE: (u32, u32) = (80, 160);
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Polygon corners in source pixels.
pub type Polygon = [[f32; 2]; 4];

/// One detected text, as in Ultralytics `SolutionResults.texts`.
#[derive(Debug, Clone, PartialEq)]
pub struct Text {
    /// Recognized text, empty when nothing was read.
    pub text: String,
    /// Recognition confidence, separate from the detector confidence.
    pub confidence: f32,
    /// Detector confidence.
    pub detection_confidence: f32,
    /// Index of the detection before reading-order sorting.
    pub detection_index: usize,
    /// The box or oriented box corners in source pixels.
    pub polygon: Polygon,
}

/// `PaddleOCR` recognition for the boxes or oriented boxes of a YOLO text detector.
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
pub struct TextRecognizer {
    recognizer: Session,
    orientation: Session,
    /// CTC classes: the blank, the dictionary, then a space.
    characters: Vec<char>,
}

impl TextRecognizer {
    /// Load a recognizer and an orientation classifier, each an ONNX path that is downloaded
    /// when missing and known, on the device of `detector`.
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
        Ok(Self {
            recognizer,
            orientation,
            characters,
        })
    }

    /// Read the text in every box or oriented box of a detector `result` on `image`, in reading
    /// order. A detection whose crop is empty keeps an empty text.
    ///
    /// # Errors
    ///
    /// Returns an error if recognition fails.
    pub fn read(&mut self, image: &DynamicImage, result: &Results) -> Result<Vec<Text>> {
        let rgb = image.to_rgb8();
        let mut texts = Vec::new();
        let mut crops = Vec::new();
        let mut cropped = Vec::new();
        for (i, (polygon, conf, obb)) in detections(result).into_iter().enumerate() {
            let mut crop = if obb {
                crop_obb(&rgb, polygon)
            } else {
                crop_box(&rgb, polygon)
            };
            // Tall crops are read sideways, like PaddleOCR, for steep text.
            if crop.width() > 0 && f64::from(crop.height()) >= 1.5 * f64::from(crop.width()) {
                crop = imageops::rotate270(&crop);
            }
            if crop.width() > 0 && crop.height() > 0 {
                cropped.push(i);
                crops.push(crop);
            }
            texts.push(Text {
                text: String::new(),
                confidence: 0.0,
                detection_confidence: conf,
                detection_index: i,
                polygon,
            });
        }
        for (i, (text, confidence)) in cropped.into_iter().zip(self.recognize(&crops)?) {
            texts[i].text = text;
            texts[i].confidence = confidence;
        }
        let polygons: Vec<Polygon> = texts.iter().map(|t| t.polygon).collect();
        Ok(reading_order(&polygons)
            .into_iter()
            .map(|i| texts[i].clone())
            .collect())
    }

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

    /// Recognize crops in batches: each is resized to the recognizer height, normalized to
    /// [-1, 1] in BGR order and zero-padded to the widest crop of its batch, then CTC decoded.
    fn read_crops(&mut self, crops: &[RgbImage]) -> Result<Vec<(String, f32)>> {
        let mut reads = Vec::with_capacity(crops.len());
        for chunk in crops.chunks(BATCH) {
            let sizes: Vec<(u32, u32)> = chunk.iter().map(rec_width).collect();
            let width = sizes.iter().map(|s| s.1).max().unwrap_or(REC_MIN_W) as usize;
            let mut batch = Array4::<f32>::zeros((chunk.len(), 3, REC_H as usize, width));
            for (i, (crop, &(resized, _))) in chunk.iter().zip(&sizes).enumerate() {
                let t = resize(crop, (REC_H, resized));
                for c in 0..3 {
                    batch
                        .slice_mut(s![i, c, .., ..resized as usize])
                        .zip_mut_with(&t.slice(s![0, 2 - c, .., ..]), |o, &v| {
                            *o = (v - 0.5) / 0.5;
                        });
                }
            }
            let characters = &self.characters;
            run(&mut self.recognizer, &batch, |shape, probs| {
                let classes = shape[2] as usize;
                for row in probs.chunks(shape[1] as usize * classes) {
                    reads.push(ctc_decode(row, classes, characters));
                }
            })?;
        }
        Ok(reads)
    }

    /// Indices of the crops the orientation classifier calls upside down.
    fn upside_down(&mut self, crops: &[RgbImage]) -> Result<Vec<usize>> {
        let mut flip = Vec::new();
        for (b, chunk) in crops.chunks(BATCH).enumerate() {
            let (h, w) = (ORI_SIZE.0 as usize, ORI_SIZE.1 as usize);
            let mut batch = Array4::<f32>::zeros((chunk.len(), 3, h, w));
            for (i, crop) in chunk.iter().enumerate() {
                let t = resize(crop, ORI_SIZE);
                for c in 0..3 {
                    batch.slice_mut(s![i, c, .., ..]).zip_mut_with(
                        &t.slice(s![0, c, .., ..]),
                        |o, &v| {
                            *o = (v - MEAN[c]) / STD[c];
                        },
                    );
                }
            }
            run(&mut self.orientation, &batch, |_, scores| {
                let turned = scores.chunks(2).map(|s| s[1] > s[0]);
                flip.extend(
                    turned
                        .enumerate()
                        .filter(|t| t.1)
                        .map(|(i, _)| b * BATCH + i),
                );
            })?;
        }
        Ok(flip)
    }
}

/// Greedy CTC decoding: the best class per step, without repeats and blanks, scored by the
/// mean probability of the kept steps.
fn ctc_decode(probs: &[f32], classes: usize, characters: &[char]) -> (String, f32) {
    let (mut text, mut sum, mut kept, mut prev) = (String::new(), 0.0_f32, 0_usize, 0);
    for step in probs.chunks(classes) {
        let (best, p) =
            step.iter().enumerate().fold(
                (0, f32::MIN),
                |acc, (k, &v)| {
                    if v > acc.1 { (k, v) } else { acc }
                },
            );
        if best != 0 && best != prev {
            text.push(characters.get(best).copied().unwrap_or(' '));
            sum += p;
            kept += 1;
        }
        prev = best;
    }
    let confidence = if kept == 0 { 0.0 } else { sum / kept as f32 };
    (text, confidence)
}

/// `(height, width)` resize of `crop` to `size`, as an RGB NCHW tensor in [0, 1], with the
/// `OpenCV` `INTER_LINEAR` resize the YOLO preprocessing uses.
fn resize(crop: &RgbImage, (h, w): (u32, u32)) -> Array4<f32> {
    let geom = LetterboxGeometry {
        new_w: w,
        new_h: h,
        pad_left: 0,
        pad_top: 0,
    };
    let (cw, ch) = crop.dimensions();
    fused_zerocopy_preprocess(crop.as_raw(), cw, ch, (h as usize, w as usize), &geom)
}

/// `(resized, padded)` recognizer widths of a crop: its width at the recognizer height, and
/// that or the minimum width when it is narrower, capped at the maximum.
fn rec_width(crop: &RgbImage) -> (u32, u32) {
    let ratio = f64::from(crop.width()) / f64::from(crop.height());
    let padded = (f64::from(REC_H) * ratio.max(f64::from(REC_MIN_W) / f64::from(REC_H))) as u32;
    if padded > REC_MAX_W {
        return (REC_MAX_W, REC_MAX_W);
    }
    (
        (f64::from(REC_H) * ratio).ceil().min(f64::from(padded)) as u32,
        padded,
    )
}

/// Run a batch and hand the output shape and values to `read`.
fn run<R>(
    session: &mut Session,
    batch: &Array4<f32>,
    read: impl FnOnce(&[i64], &[f32]) -> R,
) -> Result<R> {
    let input = TensorRef::from_array_view(batch.view())
        .map_err(|e| InferenceError::InferenceError(format!("Failed to create input: {e}")))?;
    let outputs = session
        .run(ort::inputs![input])
        .map_err(|e| InferenceError::InferenceError(format!("Inference failed: {e}")))?;
    let (shape, data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| InferenceError::InferenceError(format!("Failed to read output: {e}")))?;
    Ok(read(shape, data))
}

/// `(corners, confidence, oriented)` of every detection: the oriented box corners, or the box
/// corners clockwise from the top left.
fn detections(result: &Results) -> Vec<(Polygon, f32, bool)> {
    if let Some(obb) = &result.obb {
        let corners = obb.xyxyxyxy();
        return corners
            .axis_iter(Axis(0))
            .zip(obb.conf())
            .map(|(c, &conf)| (std::array::from_fn(|i| [c[[i, 0]], c[[i, 1]]]), conf, true))
            .collect();
    }
    result.boxes.as_ref().map_or_else(Vec::new, |boxes| {
        boxes
            .xyxy()
            .rows()
            .into_iter()
            .zip(boxes.conf())
            .map(|(b, &conf)| {
                let polygon = [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]];
                (polygon, conf, false)
            })
            .collect()
    })
}

/// Crop an axis-aligned box, truncated to whole pixels and clipped to the image, as Ultralytics
/// `save_one_box` does with no gain or padding.
fn crop_box(src: &RgbImage, p: Polygon) -> RgbImage {
    let (x1, y1, x2, y2) = (p[0][0], p[0][1], p[2][0], p[2][1]);
    // `xyxy -> xywh -> xyxy` in f32, then truncated toward zero.
    let (cx, cy, w, h) = ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1);
    let clip = |v: f32, max: u32| (v.trunc().max(0.0) as u32).min(max);
    let (left, top) = (
        clip(cx - w / 2.0, src.width()),
        clip(cy - h / 2.0, src.height()),
    );
    let (right, bottom) = (
        clip(cx + w / 2.0, src.width()),
        clip(cy + h / 2.0, src.height()),
    );
    imageops::crop_imm(
        src,
        left,
        top,
        right.saturating_sub(left),
        bottom.saturating_sub(top),
    )
    .to_image()
}

/// Warp an oriented box upright into a crop, as Ultralytics `save_one_box` does: the edge
/// spanning furthest left to right becomes the top, and `cv2.warpAffine` samples bilinearly with
/// black outside the image. The sampling repeats `OpenCV` 5's float `warpAffine` on x86, whose
/// SIMD loop maps 16 columns at a time and whose scalar loop maps the remaining columns with a
/// differently fused multiply-add.
fn crop_obb(src: &RgbImage, corners: Polygon) -> RgbImage {
    // `p[j] = corners[(j + end - 2) % 4]`, with `end` the corner ending the longest leftward run.
    let run = |i: usize| corners[(i + 3) % 4][0] - corners[i][0];
    let end = (1..4).fold(0, |best, i| if run(i) > run(best) { i } else { best });
    let p = |j: usize| corners[(j + end + 2) % 4];
    let side = |a: [f32; 2], b: [f32; 2]| {
        let (dx, dy) = (a[0] - b[0], a[1] - b[1]);
        ((dx * dx + dy * dy).sqrt().round_ties_even() as u32).max(1)
    };
    let (w, h) = (side(p(0), p(3)), side(p(0), p(1)));
    // The inverse map from the crop to the source: (0, 0) -> p2, (w, 0) -> p1, (w, h) -> p0.
    let (p0, p1, p2) = (
        p(0).map(f64::from),
        p(1).map(f64::from),
        p(2).map(f64::from),
    );
    let (wf, hf) = (f64::from(w), f64::from(h));
    let m = [
        (p1[0] - p2[0]) / wf,
        (p0[0] - p1[0]) / hf,
        p2[0],
        (p1[1] - p2[1]) / wf,
        (p0[1] - p1[1]) / hf,
        p2[1],
    ]
    .map(|v| v as f32);
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
        let (sx, sy) = if x < simd {
            (
                m[0].mul_add(xf, yf * m[1] + m[2]),
                m[3].mul_add(xf, yf * m[4] + m[5]),
            )
        } else {
            (
                xf.mul_add(m[0], yf * m[1]) + m[2],
                xf.mul_add(m[3], yf * m[4]) + m[5],
            )
        };
        let (ix, iy) = (sx.floor(), sy.floor());
        let (fx, fy) = (sx - ix, sy - iy);
        let (ix, iy) = (ix as i64, iy as i64);
        let (p00, p01, p10, p11) = (
            pixel(ix, iy),
            pixel(ix + 1, iy),
            pixel(ix, iy + 1),
            pixel(ix + 1, iy + 1),
        );
        image::Rgb(std::array::from_fn(|c| {
            let v0 = fx.mul_add(p01[c] - p00[c], p00[c]);
            let v1 = fx.mul_add(p11[c] - p10[c], p10[c]);
            fy.mul_add(v1 - v0, v0).round_ties_even().clamp(0.0, 255.0) as u8
        }))
    })
}

/// Detection indices in reading order: polygons are grouped into lines by their center height,
/// lines go top to bottom and each line left to right, as Ultralytics `OCR.reading_order` does.
#[must_use]
pub fn reading_order(polygons: &[Polygon]) -> Vec<usize> {
    let mean_y = |p: &Polygon| f64::from((p[0][1] + p[1][1] + p[2][1] + p[3][1]) / 4.0);
    let min_x = |p: &Polygon| p.iter().map(|c| c[0]).fold(f32::INFINITY, f32::min);
    let height = |p: &Polygon| {
        let ys = p.map(|c| c[1]);
        f64::from(
            ys.iter().copied().fold(f32::MIN, f32::max)
                - ys.iter().copied().fold(f32::MAX, f32::min),
        )
    };
    let mut sorted: Vec<usize> = (0..polygons.len()).collect();
    sorted.sort_by(|&a, &b| {
        let key = |i: usize| (mean_y(&polygons[i]), min_x(&polygons[i]));
        key(a)
            .partial_cmp(&key(b))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // (mean center height, tallest text, members) per line.
    let mut lines: Vec<(f64, f64, Vec<usize>)> = Vec::new();
    for i in sorted {
        let (y, h) = (mean_y(&polygons[i]), height(&polygons[i]));
        match lines.last_mut() {
            Some(line) if (y - line.0).abs() <= h.max(line.1).max(1.0) * 0.5 => {
                let n = line.2.len() as f64;
                line.0 = (line.0 * n + y) / (n + 1.0);
                line.1 = line.1.max(h);
                line.2.push(i);
            }
            _ => lines.push((y, h, vec![i])),
        }
    }
    lines.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    lines
        .into_iter()
        .flat_map(|(_, _, mut members)| {
            members.sort_by(|&a, &b| {
                let key = |i: usize| (min_x(&polygons[i]), i);
                key(a)
                    .partial_cmp(&key(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            members
        })
        .collect()
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
    fn test_reading_order_goes_by_line_then_left_to_right() {
        let quad = |x: f32, y: f32| [[x, y], [x + 10.0, y], [x + 10.0, y + 8.0], [x, y + 8.0]];
        // Two lines: 0 and 2 share a line with 2 a little lower, 1 is the next line down.
        let polygons = [
            quad(50.0, 10.0),
            quad(5.0, 40.0),
            quad(0.0, 12.0),
            quad(30.0, 41.0),
        ];
        assert_eq!(reading_order(&polygons), [2, 0, 1, 3]);
        assert_eq!(reading_order(&[]), Vec::<usize>::new());
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
