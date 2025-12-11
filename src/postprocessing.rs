// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Post-processing for YOLO model outputs.
//!
//! This module handles task-specific post-processing of raw model outputs,
//! including NMS, coordinate transformation, and result construction.

use std::collections::HashMap;

use ndarray::{s, Array2, Array3, ArrayView2};

use crate::inference::InferenceConfig;
use crate::preprocessing::{clip_coords, scale_coords, PreprocessResult};
use crate::results::{Boxes, Keypoints, Masks, Obb, Probs, Results, Speed};
use crate::task::Task;
use crate::utils::nms_per_class;

/// Post-process raw model output based on task type.
///
/// # Arguments
///
/// * `output` - Raw model output tensor.
/// * `task` - The task type (detect, segment, pose, etc.).
/// * `preprocess` - Preprocessing result containing scale/padding info.
/// * `config` - Inference configuration.
/// * `names` - Class ID to name mapping.
/// * `orig_img` - Original image as HWC array.
/// * `path` - Source image path.
/// * `speed` - Timing information.
/// * `inference_shape` - The inference tensor shape (height, width) after letterboxing.
///
/// # Returns
///
/// Processed Results object.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn postprocess(
    output: &[f32],
    output_shape: &[usize],
    task: Task,
    preprocess: &PreprocessResult,
    config: &InferenceConfig,
    names: &HashMap<usize, String>,
    orig_img: Array3<u8>,
    path: String,
    speed: Speed,
    inference_shape: (u32, u32),
) -> Results {
    match task {
        Task::Detect => postprocess_detect(
            output,
            output_shape,
            preprocess,
            config,
            names,
            orig_img,
            path,
            speed,
            inference_shape,
        ),
        Task::Segment => postprocess_segment(
            output,
            output_shape,
            preprocess,
            config,
            names,
            orig_img,
            path,
            speed,
            inference_shape,
        ),
        Task::Pose => postprocess_pose(
            output,
            output_shape,
            preprocess,
            config,
            names,
            orig_img,
            path,
            speed,
            inference_shape,
        ),
        Task::Classify => {
            postprocess_classify(output, output_shape, names, orig_img, path, speed, inference_shape)
        }
        Task::Obb => postprocess_obb(
            output,
            output_shape,
            preprocess,
            config,
            names,
            orig_img,
            path,
            speed,
            inference_shape,
        ),
    }
}

/// Post-process detection model output.
///
/// YOLO detection models output shape is typically [1, 84, 8400] where:
/// - 84 = 4 (bbox) + 80 (classes for COCO)
/// - 8400 = number of predictions (varies by input size)
#[allow(clippy::too_many_arguments)]
fn postprocess_detect(
    output: &[f32],
    output_shape: &[usize],
    preprocess: &PreprocessResult,
    config: &InferenceConfig,
    names: &HashMap<usize, String>,
    orig_img: Array3<u8>,
    path: String,
    speed: Speed,
    inference_shape: (u32, u32),
) -> Results {
    let mut results = Results::new(orig_img, path, names.clone(), speed, inference_shape);

    // Parse output shape - handle both [1, 84, 8400] and [1, 8400, 84] formats
    let (num_classes, num_predictions, is_transposed) = parse_detect_shape(output_shape, names.len());

    if output.is_empty() || num_predictions == 0 {
        return results;
    }

    // Convert flat output to 2D array
    let output_2d = if is_transposed {
        // Shape is [1, num_preds, num_features] - already in correct format
        Array2::from_shape_vec((num_predictions, 4 + num_classes), output.to_vec())
            .unwrap_or_else(|_| Array2::zeros((0, 0)))
    } else {
        // Shape is [1, num_features, num_preds] - need to transpose
        let arr = Array2::from_shape_vec((4 + num_classes, num_predictions), output.to_vec())
            .unwrap_or_else(|_| Array2::zeros((0, 0)));
        arr.t().to_owned()
    };

    if output_2d.is_empty() {
        return results;
    }

    // Extract boxes and scores
    let boxes_data = extract_detect_boxes(
        output_2d.view(),
        num_classes,
        preprocess,
        config,
    );

    if !boxes_data.is_empty() {
        results.boxes = Some(Boxes::new(boxes_data, preprocess.orig_shape));
    }

    results
}

/// Parse detection output shape to determine format.
///
/// Derives class count from output shape when metadata is missing (expected_classes == 0).
/// YOLO outputs are either [1, num_features, num_preds] or [1, num_preds, num_features]
/// where num_features = 4 (bbox) + num_classes.
fn parse_detect_shape(shape: &[usize], expected_classes: usize) -> (usize, usize, bool) {
    match shape.len() {
        2 => {
            // [num_preds, num_features] or [num_features, num_preds]
            let (a, b) = (shape[0], shape[1]);
            // Handle edge case where either dimension is less than 4
            if a < 4 && b < 4 {
                return (expected_classes.max(1), 0, false);
            }
            // When metadata is missing, infer from shape:
            // The smaller dimension (if >= 5) is likely num_features, larger is num_preds
            if expected_classes == 0 {
                // No metadata - infer from shape
                let (num_features, num_preds, transposed) = if a < b { (a, b, false) } else { (b, a, true) };
                let inferred_classes = num_features.saturating_sub(4);
                return (inferred_classes.max(1), num_preds, transposed);
            }
            if a == 4 + expected_classes || (a >= 4 && a > b) {
                // [num_features, num_preds]
                (a.saturating_sub(4), b, false)
            } else {
                // [num_preds, num_features]
                (b.saturating_sub(4), a, true)
            }
        }
        3 => {
            // [batch, ...] - ignore batch dimension
            let (a, b) = (shape[1], shape[2]);
            // Handle edge case where num_predictions is 0 or very small
            if b == 0 || a < 4 {
                return (expected_classes.max(1), 0, false);
            }
            // When metadata is missing, infer from shape
            if expected_classes == 0 {
                // No metadata - infer from shape
                // Typically num_features < num_preds (e.g., 84 < 8400)
                let (num_features, num_preds, transposed) = if a < b { (a, b, false) } else { (b, a, true) };
                let inferred_classes = num_features.saturating_sub(4);
                return (inferred_classes.max(1), num_preds, transposed);
            }
            if a == 4 + expected_classes || (expected_classes > 0 && a < b) {
                // [1, num_features, num_preds]
                (a.saturating_sub(4), b, false)
            } else {
                // [1, num_preds, num_features]
                (b.saturating_sub(4), a, true)
            }
        }
        _ => (expected_classes.max(1), 0, false),
    }
}

/// Extract detection boxes from model output.
fn extract_detect_boxes(
    output: ArrayView2<f32>,
    _num_classes: usize,
    preprocess: &PreprocessResult,
    config: &InferenceConfig,
) -> Array2<f32> {
    let num_predictions = output.nrows();
    let mut candidates = Vec::new();

    for i in 0..num_predictions {
        // Get class scores (columns 4 onwards)
        let class_scores = output.slice(s![i, 4..]);

        // Find best class
        let (best_class, best_score) = class_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, &score)| (idx, score))
            .unwrap_or((0, 0.0));

        // Skip low confidence detections
        if best_score < config.confidence_threshold {
            continue;
        }

        // Extract box coordinates (xywh format from model)
        let cx = output[[i, 0]];
        let cy = output[[i, 1]];
        let w = output[[i, 2]];
        let h = output[[i, 3]];

        // Convert xywh to xyxy
        let x1 = cx - w / 2.0;
        let y1 = cy - h / 2.0;
        let x2 = cx + w / 2.0;
        let y2 = cy + h / 2.0;

        // Scale coordinates back to original image space
        let scaled = scale_coords(&[x1, y1, x2, y2], preprocess.scale, preprocess.padding);

        // Clip to image bounds
        let clipped = clip_coords(&scaled, preprocess.orig_shape);

        // Store candidate: [x1, y1, x2, y2, conf, class]
        candidates.push((
            [clipped[0], clipped[1], clipped[2], clipped[3]],
            best_score,
            best_class,
        ));
    }

    if candidates.is_empty() {
        return Array2::zeros((0, 6));
    }

    // Apply per-class NMS (only suppress boxes within the same class)
    let keep_indices = nms_per_class(&candidates, config.iou_threshold);

    // Build output array with kept detections
    let num_kept = keep_indices.len().min(config.max_detections);
    let mut result = Array2::zeros((num_kept, 6));

    for (out_idx, &keep_idx) in keep_indices.iter().take(num_kept).enumerate() {
        let (bbox, score, class) = &candidates[keep_idx];
        result[[out_idx, 0]] = bbox[0];
        result[[out_idx, 1]] = bbox[1];
        result[[out_idx, 2]] = bbox[2];
        result[[out_idx, 3]] = bbox[3];
        result[[out_idx, 4]] = *score;
        result[[out_idx, 5]] = *class as f32;
    }

    result
}

/// Post-process segmentation model output (placeholder).
#[allow(clippy::too_many_arguments)]
fn postprocess_segment(
    _output: &[f32],
    _output_shape: &[usize],
    preprocess: &PreprocessResult,
    _config: &InferenceConfig,
    names: &HashMap<usize, String>,
    orig_img: Array3<u8>,
    path: String,
    speed: Speed,
    inference_shape: (u32, u32),
) -> Results {
    // TODO: Implement segmentation post-processing
    // Segmentation models output both detection boxes and mask coefficients
    let mut results = Results::new(orig_img, path, names.clone(), speed, inference_shape);

    // Placeholder - return empty masks
    results.masks = Some(Masks::new(
        Array3::zeros((0, 160, 160)),
        preprocess.orig_shape,
    ));

    results
}

/// Post-process pose estimation model output (placeholder).
#[allow(clippy::too_many_arguments)]
fn postprocess_pose(
    _output: &[f32],
    _output_shape: &[usize],
    preprocess: &PreprocessResult,
    _config: &InferenceConfig,
    names: &HashMap<usize, String>,
    orig_img: Array3<u8>,
    path: String,
    speed: Speed,
    inference_shape: (u32, u32),
) -> Results {
    // TODO: Implement pose estimation post-processing
    // Pose models output boxes + keypoints
    let mut results = Results::new(orig_img, path, names.clone(), speed, inference_shape);

    // Placeholder - return empty keypoints
    results.keypoints = Some(Keypoints::new(
        Array3::zeros((0, 17, 3)),
        preprocess.orig_shape,
    ));

    results
}

/// Post-process classification model output.
fn postprocess_classify(
    output: &[f32],
    _output_shape: &[usize],
    names: &HashMap<usize, String>,
    orig_img: Array3<u8>,
    path: String,
    speed: Speed,
    inference_shape: (u32, u32),
) -> Results {
    let mut results = Results::new(orig_img, path, names.clone(), speed, inference_shape);

    if output.is_empty() {
        return results;
    }

    // Classification output is class probabilities (softmax applied)
    let probs = ndarray::Array1::from_vec(output.to_vec());
    results.probs = Some(Probs::new(probs));

    results
}

/// Post-process OBB (oriented bounding box) model output (placeholder).
#[allow(clippy::too_many_arguments)]
fn postprocess_obb(
    _output: &[f32],
    _output_shape: &[usize],
    preprocess: &PreprocessResult,
    _config: &InferenceConfig,
    names: &HashMap<usize, String>,
    orig_img: Array3<u8>,
    path: String,
    speed: Speed,
    inference_shape: (u32, u32),
) -> Results {
    // TODO: Implement OBB post-processing
    // OBB models output rotated bounding boxes
    let mut results = Results::new(orig_img, path, names.clone(), speed, inference_shape);

    // Placeholder - return empty OBBs
    results.obb = Some(Obb::new(Array2::zeros((0, 7)), preprocess.orig_shape));

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_detect_shape() {
        // Standard YOLO output [1, 84, 8400]
        let (nc, np, transposed) = parse_detect_shape(&[1, 84, 8400], 80);
        assert_eq!(nc, 80);
        assert_eq!(np, 8400);
        assert!(!transposed);

        // Transposed format [1, 8400, 84]
        let (nc, np, transposed) = parse_detect_shape(&[1, 8400, 84], 80);
        assert_eq!(nc, 80);
        assert_eq!(np, 8400);
        assert!(transposed);
    }

    #[test]
    fn test_parse_detect_shape_no_metadata() {
        // When metadata is missing (expected_classes == 0), infer from shape
        // Standard YOLO output [1, 84, 8400] with no metadata
        let (nc, np, transposed) = parse_detect_shape(&[1, 84, 8400], 0);
        assert_eq!(nc, 80);  // Inferred: 84 - 4 = 80 classes
        assert_eq!(np, 8400);
        assert!(!transposed);

        // Transposed format [1, 8400, 84] with no metadata
        let (nc, np, transposed) = parse_detect_shape(&[1, 8400, 84], 0);
        assert_eq!(nc, 80);  // Inferred: 84 - 4 = 80 classes
        assert_eq!(np, 8400);
        assert!(transposed);
    }

    #[test]
    fn test_empty_output() {
        let output: Vec<f32> = vec![];
        let preprocess = PreprocessResult {
            tensor: ndarray::Array4::zeros((1, 3, 640, 640)),
            tensor_f16: None,
            orig_shape: (480, 640),
            scale: (1.0, 1.0),
            padding: (0.0, 0.0),
        };
        let config = InferenceConfig::default();
        let names = HashMap::new();
        let orig_img = ndarray::Array3::zeros((480, 640, 3));

        let results = postprocess_detect(
            &output,
            &[1, 84, 0],
            &preprocess,
            &config,
            &names,
            orig_img,
            String::new(),
            Speed::default(),
            (640, 640),
        );

        assert!(results.is_empty());
    }
}
