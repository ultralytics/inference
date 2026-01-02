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
pub fn calculate_iou(box1: &[f32; 4], box2: &[f32; 4]) -> f32 {
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
#[must_use]
pub fn calculate_probiou(box1: &[f32; 5], box2: &[f32; 5]) -> f32 {
    let eps = 1e-7;

    let x1 = box1[0];
    let y1 = box1[1];
    let w1 = box1[2];
    let h1 = box1[3];
    let r1 = box1[4];

    let x2 = box2[0];
    let y2 = box2[1];
    let w2 = box2[2];
    let h2 = box2[3];
    let r2 = box2[4];

    let (a1, b1, c1) = get_covariance_params(w1, h1, r1);
    let (a2, b2, c2) = get_covariance_params(w2, h2, r2);

    let t1 = ((a1 + a2).mul_add((y1 - y2).powi(2), (b1 + b2) * (x1 - x2).powi(2))
        / (a1 + a2).mul_add(b1 + b2, -(c1 + c2).powi(2) + eps))
        * 0.25;

    let t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2))
        / (a1 + a2).mul_add(b1 + b2, -(c1 + c2).powi(2) + eps))
        * 0.5;

    let t3_num = (a1 + a2).mul_add(b1 + b2, -(c1 + c2).powi(2));
    let t3_den = 4.0f32.mul_add(
        ((a1.mul_add(b1, -c1.powi(2))).max(0.0) * (a2.mul_add(b2, -c2.powi(2))).max(0.0)).sqrt(),
        eps,
    );
    let t3 = (t3_num / t3_den + eps).ln() * 0.5;

    let bd = (t1 + t2 + t3).clamp(eps, 100.0);
    let hd = (1.0 - (-bd).exp() + eps).sqrt();

    1.0 - hd
}

/// Non-Maximum Suppression (NMS) for filtering overlapping detections
///
/// # Arguments
///
/// * `boxes` - Vector of bounding boxes with scores [(bbox, score)]
/// * `iou_threshold` - `IoU` threshold for suppression
///
/// # Returns
///
/// Indices of boxes to keep
///
/// # Panics
///
/// Panics if `partial_cmp` fails for floating point comparisons (e.g. NaN).
#[must_use]
pub fn nms(boxes: &[([f32; 4], f32)], iou_threshold: f32) -> Vec<usize> {
    if boxes.is_empty() {
        return vec![];
    }

    // Sort by score (descending)
    let mut indices: Vec<usize> = (0..boxes.len()).collect();
    indices.sort_by(|&a, &b| boxes[b].1.partial_cmp(&boxes[a].1).unwrap());

    let mut keep = vec![];
    let mut suppressed = vec![false; boxes.len()];

    for &i in &indices {
        if suppressed[i] {
            continue;
        }
        keep.push(i);

        for &j in &indices {
            if !suppressed[j] && i != j {
                let iou = calculate_iou(&boxes[i].0, &boxes[j].0);
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
///
/// # Panics
///
/// Panics if `partial_cmp` fails for floating point comparisons (e.g. NaN).
#[must_use]
pub fn nms_per_class(boxes: &[([f32; 4], f32, usize)], iou_threshold: f32) -> Vec<usize> {
    if boxes.is_empty() {
        return vec![];
    }

    // Sort by score (descending)
    let mut indices: Vec<usize> = (0..boxes.len()).collect();
    indices.sort_by(|&a, &b| boxes[b].1.partial_cmp(&boxes[a].1).unwrap());

    let mut keep = vec![];
    let mut suppressed = vec![false; boxes.len()];

    for &i in &indices {
        if suppressed[i] {
            continue;
        }
        keep.push(i);

        let class_i = boxes[i].2;

        for &j in &indices {
            if !suppressed[j] && i != j {
                // Only suppress boxes of the same class
                if boxes[j].2 == class_i {
                    let iou = calculate_iou(&boxes[i].0, &boxes[j].0);
                    if iou > iou_threshold {
                        suppressed[j] = true;
                    }
                }
            }
        }
    }

    keep
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
/// * `iou_threshold` - `IoU` threshold
/// # Panics
///
/// Panics if `partial_cmp` fails for floating point comparisons (e.g. NaN).
#[must_use]
pub fn nms_rotated_per_class(boxes: &[([f32; 5], f32, usize)], iou_threshold: f32) -> Vec<usize> {
    if boxes.is_empty() {
        return vec![];
    }

    // Sort by score (descending)
    let mut indices: Vec<usize> = (0..boxes.len()).collect();
    indices.sort_by(|&a, &b| boxes[b].1.partial_cmp(&boxes[a].1).unwrap());

    let mut keep = vec![];
    let mut suppressed = vec![false; boxes.len()];

    for &i in &indices {
        if suppressed[i] {
            continue;
        }
        keep.push(i);

        let class_i = boxes[i].2;

        for &j in &indices {
            if !suppressed[j] && i != j && boxes[j].2 == class_i {
                let iou = calculate_probiou(&boxes[i].0, &boxes[j].0);
                if iou > iou_threshold {
                    suppressed[j] = true;
                }
            }
        }
    }

    keep
}

/// Simple pluralization for common COCO class names.
#[must_use]
pub fn pluralize(word: &str) -> String {
    match word {
        "person" => "persons".to_string(),
        "bus" => "buses".to_string(),
        "knife" => "knives".to_string(),
        "mouse" => "mice".to_string(),
        "sheep" => "sheep".to_string(),
        "skis" => "skis".to_string(),
        _ => {
            if word.ends_with('s') || word.ends_with("ch") || word.ends_with("sh") {
                format!("{word}es")
            } else if word.ends_with('y') && !word.ends_with("ey") && !word.ends_with("ay") {
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

    let mut rgb_data = Vec::with_capacity((height * width * 3) as usize);
    for y in 0..height as usize {
        for x in 0..width as usize {
            rgb_data.push(arr[[y, x, 0]]);
            rgb_data.push(arr[[y, x, 1]]);
            rgb_data.push(arr[[y, x, 2]]);
        }
    }

    let img_buffer = image::RgbImage::from_raw(width, height, rgb_data).ok_or_else(|| {
        InferenceError::ImageError("Failed to create image from array".to_string())
    })?;

    Ok(DynamicImage::ImageRgb8(img_buffer))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_iou() {
        let box1 = [0.0, 0.0, 10.0, 10.0];
        let box2 = [5.0, 5.0, 15.0, 15.0];
        let iou = calculate_iou(&box1, &box2);
        assert!((iou - 0.142_857).abs() < 0.001); // 25 / (100 + 100 - 25)
    }

    #[test]
    fn test_nms() {
        let boxes = vec![
            ([0.0, 0.0, 10.0, 10.0], 0.9),
            ([1.0, 1.0, 11.0, 11.0], 0.8),
            ([100.0, 100.0, 110.0, 110.0], 0.95),
        ];
        let keep = nms(&boxes, 0.5);
        assert_eq!(keep.len(), 2); // Should keep boxes at indices 2 and 0
        assert!(keep.contains(&2));
        assert!(keep.contains(&0));
    }

    #[test]
    fn test_nms_per_class() {
        // Two overlapping boxes of different classes should both be kept
        let boxes = vec![
            ([0.0, 0.0, 10.0, 10.0], 0.9, 0),        // class 0
            ([1.0, 1.0, 11.0, 11.0], 0.8, 1),        // class 1 (different class)
            ([100.0, 100.0, 110.0, 110.0], 0.95, 0), // class 0, non-overlapping
        ];
        let keep = nms_per_class(&boxes, 0.5);
        // All 3 boxes should be kept (overlapping boxes are different classes)
        assert_eq!(keep.len(), 3);
    }

    #[test]
    fn test_nms_per_class_suppression() {
        // Two overlapping boxes of the same class - lower score suppressed
        let boxes = vec![
            ([0.0, 0.0, 10.0, 10.0], 0.9, 0), // class 0, higher score
            ([1.0, 1.0, 11.0, 11.0], 0.8, 0), // class 0, lower score (suppressed)
        ];
        let keep = nms_per_class(&boxes, 0.5);
        assert_eq!(keep.len(), 1);
        assert!(keep.contains(&0)); // Keep higher score box
    }
}
