// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Utility functions for the inference library

/// Calculate `IoU` (Intersection over Union) between two bounding boxes
///
/// # Arguments
///
/// * `box1` - First bounding box [x1, y1, x2, y2]
/// * `box2` - Second bounding box [x1, y1, x2, y2]
///
/// # Returns
///
/// `IoU` value between 0.0 and 1.0
#[must_use]
pub(crate) fn calculate_iou(box1: &[f32; 4], box2: &[f32; 4]) -> f32 {
    let x1 = box1[0].max(box2[0]);
    let y1 = box1[1].max(box2[1]);
    let x2 = box1[2].min(box2[2]);
    let y2 = box1[3].min(box2[3]);

    let intersection = ((x2 - x1).max(0.0)) * ((y2 - y1).max(0.0));

    let area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    let area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    let union = area1 + area2 - intersection;

    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

/// Calculate parameters for the covariance matrix of an oriented bounding box
fn get_covariance_params(w: f32, h: f32, angle: f32) -> (f32, f32, f32) {
    let a = w.powi(2) / 12.0;
    let b = h.powi(2) / 12.0;

    let cos = angle.cos();
    let sin = angle.sin();
    let cos2 = cos.powi(2);
    let sin2 = sin.powi(2);

    let a_val = a * cos2 + b * sin2;
    let b_val = a * sin2 + b * cos2;
    let c_val = (a - b) * cos * sin;

    (a_val, b_val, c_val)
}

/// Calculate `ProbIoU` (Probabilistic `IoU`) between two oriented bounding boxes
///
/// This metric uses the Hellinger distance between 2D Gaussian distributions
/// to estimate the `IoU` of rotated bounding boxes. It is used in Ultralytics
/// OBB models for NMS.
///
/// It provides a differentiable and robust overlap metric for oriented boxes
/// where standard Polygon `IoU` can be unstable or computationally expensive.
///
/// # Arguments
///
/// * `box1` - [cx, cy, w, h, angle]
/// * `box2` - [cx, cy, w, h, angle]
///
/// # Returns
///
/// `ProbIoU` value between 0.0 and 1.0
#[must_use]
pub fn calculate_probiou(box1: &[f32; 5], box2: &[f32; 5]) -> f32 {
    probiou_prepared(&ProbIouBox::new(box1), &ProbIouBox::new(box2))
}

/// A rotated box reduced to the terms [`probiou_prepared`] needs.
///
/// The covariance `(a, b, c)` and its determinant depend only on one box, but
/// [`calculate_probiou`] recomputed them for both boxes of every pair, putting four trig
/// calls inside an O(n^2) loop. Building this once per box moves that work to O(n).
#[derive(Clone, Copy)]
struct ProbIouBox {
    x: f32,
    y: f32,
    a: f32,
    b: f32,
    c: f32,
    /// `a * b - c^2`, clamped at zero for the `t3` denominator.
    det: f32,
}

impl ProbIouBox {
    /// Reduce every box once, keeping each score and class alongside it so the shared
    /// suppression loop can consume the result directly.
    fn prepare_all(boxes: &[([f32; 5], f32, usize)]) -> Vec<(Self, f32, usize)> {
        boxes
            .iter()
            .map(|(b, score, class)| (Self::new(b), *score, *class))
            .collect()
    }

    fn new(b: &[f32; 5]) -> Self {
        let (a, bb, c) = get_covariance_params(b[2], b[3], b[4]);
        Self {
            x: b[0],
            y: b[1],
            a,
            b: bb,
            c,
            det: a.mul_add(bb, -c.powi(2)).max(0.0),
        }
    }
}

/// [`calculate_probiou`] over two boxes whose covariance terms are already computed.
fn probiou_prepared(p1: &ProbIouBox, p2: &ProbIouBox) -> f32 {
    let eps = 1e-7;

    let (a1, b1, c1) = (p1.a, p1.b, p1.c);
    let (a2, b2, c2) = (p2.a, p2.b, p2.c);
    let (x1, y1, x2, y2) = (p1.x, p1.y, p2.x, p2.y);

    let denom = (a1 + a2).mul_add(b1 + b2, -(c1 + c2).powi(2) + eps);

    let t1 = ((a1 + a2).mul_add((y1 - y2).powi(2), (b1 + b2) * (x1 - x2).powi(2)) / denom) * 0.25;

    let t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / denom) * 0.5;

    let t3_num = (a1 + a2).mul_add(b1 + b2, -(c1 + c2).powi(2));
    let t3_den = 4.0f32.mul_add((p1.det * p2.det).sqrt(), eps);
    let t3 = (t3_num / t3_den + eps).ln() * 0.5;

    let bd = (t1 + t2 + t3).clamp(eps, 100.0);
    let hd = (1.0 - (-bd).exp() + eps).sqrt();

    1.0 - hd
}

/// Shared per-class NMS implementation used by axis-aligned and rotated boxes.
///
/// `overlap` computes the IoU-like suppression score for the task-specific box type.
/// The public wrappers keep task-specific signatures while sharing the score sorting,
/// class filtering, and suppression loop.
fn nms_by_class<T>(
    boxes: &[(T, f32, usize)],
    iou_threshold: f32,
    max_det: usize,
    overlap: impl Fn(&T, &T) -> f32,
) -> Vec<usize> {
    // `max_det == 0` is reachable through `with_max_det(0)` and `--max-det 0`, and the cap
    // below is only checked after a box is pushed, so it has to be rejected up front.
    if boxes.is_empty() || max_det == 0 {
        return vec![];
    }

    // Sort by score (descending), with any NaN last. `total_cmp` alone would rank a positive
    // NaN above every finite score, letting it suppress a valid overlapping box of the same
    // class; testing `is_nan` first processes every real score before the NaNs.
    let mut indices: Vec<usize> = (0..boxes.len()).collect();
    indices.sort_by(|&a, &b| {
        boxes[a]
            .1
            .is_nan()
            .cmp(&boxes[b].1.is_nan())
            .then_with(|| boxes[b].1.total_cmp(&boxes[a].1))
    });

    let mut keep = vec![];
    let mut suppressed = vec![false; boxes.len()];

    for (pos, &i) in indices.iter().enumerate() {
        if suppressed[i] {
            continue;
        }
        keep.push(i);
        // Callers cap the result at `max_det`, so suppression past that point only decides
        // the order of boxes that are dropped. Stopping here is not an approximation.
        if keep.len() >= max_det {
            break;
        }

        let class_i = boxes[i].2;

        for &j in &indices[pos + 1..] {
            if !suppressed[j] && boxes[j].2 == class_i {
                let iou = overlap(&boxes[i].0, &boxes[j].0);
                if iou > iou_threshold {
                    suppressed[j] = true;
                }
            }
        }
    }

    keep
}

/// Per-class Non-Maximum Suppression (NMS) for filtering overlapping detections
///
/// Only suppresses boxes within the same class, matching Ultralytics behavior.
///
/// # Arguments
///
/// * `boxes` - Vector of bounding boxes with scores and class IDs [(bbox, score, `class_id`)]
/// * `iou_threshold` - `IoU` threshold for suppression
///
/// # Returns
///
/// Indices of boxes to keep
#[must_use]
pub fn nms_per_class(boxes: &[([f32; 4], f32, usize)], iou_threshold: f32) -> Vec<usize> {
    nms_by_class(boxes, iou_threshold, usize::MAX, calculate_iou)
}

/// [`nms_per_class`], stopping once `max_det` boxes are kept.
///
/// Callers cap the result at `max_det` anyway, so suppressing past that point only orders
/// boxes that get discarded. On a full `8400`-prediction head that is the difference
/// between roughly 26 ms and 2 ms.
pub(crate) fn nms_per_class_capped(
    boxes: &[([f32; 4], f32, usize)],
    iou_threshold: f32,
    max_det: usize,
) -> Vec<usize> {
    nms_by_class(boxes, iou_threshold, max_det, calculate_iou)
}

/// Rotated Per-class Non-Maximum Suppression (NMS) using `ProbIoU`
///
/// This function performs NMS specifically for Oriented Bounding Boxes (OBB).
/// Instead of standard `IoU`, it uses `ProbIoU` (Hellinger distance) to determine overlap,
/// which correctly handles the rotation angle of the boxes.
///
/// # Arguments
///
/// * `boxes` - Vector of rotated bounding boxes: [cx, cy, w, h, angle], score, `class_id`
/// * `iou_threshold` - `IoU` threshold for suppression
///
/// # Returns
///
/// Indices of boxes to keep
#[must_use]
pub fn nms_rotated_per_class(boxes: &[([f32; 5], f32, usize)], iou_threshold: f32) -> Vec<usize> {
    nms_by_class(
        &ProbIouBox::prepare_all(boxes),
        iou_threshold,
        usize::MAX,
        probiou_prepared,
    )
}

/// [`nms_rotated_per_class`], stopping once `max_det` boxes are kept. See
/// [`nms_per_class_capped`] for why that is not an approximation.
pub(crate) fn nms_rotated_per_class_capped(
    boxes: &[([f32; 5], f32, usize)],
    iou_threshold: f32,
    max_det: usize,
) -> Vec<usize> {
    nms_by_class(
        &ProbIouBox::prepare_all(boxes),
        iou_threshold,
        max_det,
        probiou_prepared,
    )
}

/// Simple pluralization for common COCO class names.
///
/// # Arguments
///
/// * `word` - Singular noun to pluralize
///
/// # Returns
///
/// Plural form of the word
#[must_use]
pub(crate) fn pluralize(word: &str) -> String {
    match word {
        "person" => "persons".to_string(),
        "bus" => "buses".to_string(),
        "knife" => "knives".to_string(),
        "mouse" => "mice".to_string(),
        "sheep" => "sheep".to_string(),
        "skis" => "skis".to_string(),
        _ => {
            if word.ends_with('s')
                || word.ends_with('x')
                || word.ends_with("ch")
                || word.ends_with("sh")
            {
                format!("{word}es")
            } else if word.ends_with('y')
                && !word.ends_with("ey")
                && !word.ends_with("ay")
                && !word.ends_with("oy")
            {
                format!("{}ies", &word[..word.len() - 1])
            } else {
                format!("{word}s")
            }
        }
    }
}

use image::DynamicImage;
use ndarray::Array3;

use crate::error::{InferenceError, Result};

/// Convert an HWC u8 array to a `DynamicImage`.
///
/// # Arguments
///
/// * `arr` - Input array with shape (H, W, 3).
///
/// # Returns
///
/// * A `DynamicImage` containing the image data.
///
/// # Errors
///
/// Returns an error if dimensions are invalid or conversion fails.
pub fn array_to_image(arr: &Array3<u8>) -> Result<DynamicImage> {
    let shape = arr.shape();
    let height = u32::try_from(shape[0])
        .map_err(|_| InferenceError::ImageError("Image height exceeds u32::MAX".to_string()))?;
    let width = u32::try_from(shape[1])
        .map_err(|_| InferenceError::ImageError("Image width exceeds u32::MAX".to_string()))?;

    // HWC RGB is already row-major, so a flat copy matches the pixel order `RgbImage` expects.
    // Copy the backing slice in one memcpy rather than element by element: at 1080p that is
    // 0.22 ms instead of 3.0 ms. `as_slice` succeeds whenever the array is contiguous, which
    // it is for anything built by `Array3::from_shape_vec`; the iterator covers other layouts.
    let rgb_data: Vec<u8> = arr
        .as_slice()
        .map_or_else(|| arr.iter().copied().collect(), <[u8]>::to_vec);

    let img_buffer = image::RgbImage::from_raw(width, height, rgb_data).ok_or_else(|| {
        InferenceError::ImageError("Failed to create image from array".to_string())
    })?;

    Ok(DynamicImage::ImageRgb8(img_buffer))
}

/// Convert a center-form box `(cx, cy, w, h)` to corner form `[x1, y1, x2, y2]`.
#[inline]
pub(crate) fn xywh_to_xyxy(cx: f32, cy: f32, w: f32, h: f32) -> [f32; 4] {
    [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_iou() {
        // (box1, box2, expected): partial overlap 25 / (100 + 100 - 25), identical, disjoint.
        let cases = [
            ([0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0], 0.142_857),
            ([0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0], 1.0),
            ([0.0, 0.0, 5.0, 5.0], [10.0, 10.0, 20.0, 20.0], 0.0),
        ];
        for (box1, box2, expected) in cases {
            assert!((calculate_iou(&box1, &box2) - expected).abs() < 0.001);
        }
    }

    #[test]
    fn test_calculate_probiou_identical() {
        // eps terms in the formula prevent reaching exactly 1.0; expect near-perfect overlap
        let box1 = [5.0, 5.0, 4.0, 2.0, 0.0];
        assert!(calculate_probiou(&box1, &box1) > 0.999);
    }

    #[test]
    fn test_calculate_probiou_distant() {
        let box1 = [0.0, 0.0, 2.0, 2.0, 0.0];
        let box2 = [1000.0, 1000.0, 2.0, 2.0, 0.0];
        assert!(calculate_probiou(&box1, &box2) < 0.01);
    }

    #[test]
    fn test_nms_per_class() {
        // Overlapping boxes of different classes are kept; non-overlapping ones always are.
        let boxes = vec![
            ([0.0, 0.0, 10.0, 10.0], 0.9, 0),
            ([1.0, 1.0, 11.0, 11.0], 0.8, 1),
            ([100.0, 100.0, 110.0, 110.0], 0.95, 0),
        ];
        assert_eq!(nms_per_class(&boxes, 0.5).len(), 3);

        // Overlapping within one class: the lower score is suppressed.
        let boxes = vec![
            ([0.0, 0.0, 10.0, 10.0], 0.9, 0),
            ([1.0, 1.0, 11.0, 11.0], 0.8, 0),
        ];
        let keep = nms_per_class(&boxes, 0.5);
        assert_eq!(keep, vec![0]);

        // A zero cap keeps nothing, matching the `truncate(max_det)` this replaced.
        let boxes = vec![
            ([0.0, 0.0, 10.0, 10.0], 0.9, 0),
            ([50.0, 50.0, 60.0, 60.0], 0.8, 0),
        ];
        assert!(nms_per_class_capped(&boxes, 0.5, 0).is_empty());
        assert_eq!(nms_per_class_capped(&boxes, 0.5, 1), vec![0]);

        // A NaN score used to panic in the sort comparator.
        let boxes = vec![
            ([0.0, 0.0, 10.0, 10.0], f32::NAN, 0),
            ([100.0, 100.0, 110.0, 110.0], 0.9, 0),
        ];
        assert_eq!(nms_per_class(&boxes, 0.5).len(), 2);

        // A NaN-scored box overlapping a valid one of the same class must not outrank it and
        // suppress it: the real detection is processed first and survives.
        let boxes = vec![
            ([0.0, 0.0, 10.0, 10.0], f32::NAN, 0),
            ([1.0, 1.0, 11.0, 11.0], 0.9, 0),
        ];
        assert_eq!(nms_per_class(&boxes, 0.5), vec![1]);
    }

    #[test]
    fn test_nms_rotated_per_class() {
        // Same rotated box twice: kept across classes, suppressed within one.
        let across = vec![
            ([5.0, 5.0, 4.0, 2.0, 0.0], 0.9, 0),
            ([5.0, 5.0, 4.0, 2.0, 0.0], 0.8, 1),
        ];
        assert_eq!(nms_rotated_per_class(&across, 0.5).len(), 2);

        let within = vec![
            ([5.0, 5.0, 4.0, 2.0, 0.0], 0.9, 0),
            ([5.0, 5.0, 4.0, 2.0, 0.0], 0.8, 0),
        ];
        assert_eq!(nms_rotated_per_class(&within, 0.5), vec![0]);
        assert!(nms_rotated_per_class_capped(&within, 0.5, 0).is_empty());
    }

    #[test]
    fn test_pluralize() {
        assert_eq!(pluralize("person"), "persons");
        assert_eq!(pluralize("bus"), "buses");
        assert_eq!(pluralize("match"), "matches");
        assert_eq!(pluralize("box"), "boxes");
        assert_eq!(pluralize("car"), "cars");
        assert_eq!(pluralize("baby"), "babies");
        assert_eq!(pluralize("toy"), "toys");
    }

    /// A non-contiguous array takes the iterator fallback instead of the memcpy, so check it
    /// lays the pixels out identically; a wrong order there would be silent.
    #[test]
    fn test_array_to_image_non_contiguous_matches() {
        let hwc = Array3::from_shape_fn((2, 3, 3), |(y, x, c)| {
            u8::try_from(y * 30 + x * 3 + c).unwrap()
        });
        // Same image stored as (W, H, C) and permuted back: equal contents, other strides.
        let whc = Array3::from_shape_fn((3, 2, 3), |(x, y, c)| {
            u8::try_from(y * 30 + x * 3 + c).unwrap()
        });
        let permuted = whc.permuted_axes([1, 0, 2]);
        assert_eq!(permuted, hwc);
        assert!(permuted.as_slice().is_none(), "expected the fallback path");

        let fast = array_to_image(&hwc).unwrap().to_rgb8();
        let slow = array_to_image(&permuted).unwrap().to_rgb8();
        assert_eq!(fast.dimensions(), (3, 2));
        assert_eq!(fast.as_raw(), slow.as_raw());
    }

    #[test]
    fn test_array_to_image() {
        let data = vec![255, 0, 0, 0, 255, 0]; // 2 pixels: Red, Green
        let arr = Array3::from_shape_vec((1, 2, 3), data).unwrap();
        let img = array_to_image(&arr).unwrap();

        assert_eq!(img.width(), 2);
        assert_eq!(img.height(), 1);

        let rgb = img.to_rgb8();
        let p1 = rgb.get_pixel(0, 0);
        assert_eq!(p1[0], 255);
        assert_eq!(p1[1], 0);
        assert_eq!(p1[2], 0);

        let p2 = rgb.get_pixel(1, 0);
        assert_eq!(p2[0], 0);
        assert_eq!(p2[1], 255);
        assert_eq!(p2[2], 0);
    }
}
