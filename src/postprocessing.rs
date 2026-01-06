// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Post-processing for YOLO model outputs.
//!
//! This module handles task-specific post-processing of raw model outputs,
//! including NMS, coordinate transformation, and result construction.

use std::collections::HashMap;

use fast_image_resize::images::Image;
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use ndarray::{Array2, Array3, ArrayView2, Zip, s};

use crate::inference::InferenceConfig;
use crate::preprocessing::{PreprocessResult, clip_coords, scale_coords};
use crate::results::{Boxes, Keypoints, Masks, Obb, Probs, Results, Speed};
use crate::task::Task;
use crate::utils::{nms_per_class, nms_rotated_per_class};

/// Post-process raw model output based on task type.
///
/// # Arguments
///
/// * `outputs` - Vector of raw model outputs (data, shape).
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
#[allow(
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::implicit_hasher
)]
pub fn postprocess(
    outputs: Vec<(Vec<f32>, Vec<usize>)>,
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
        Task::Detect => {
            let (output, shape) = &outputs[0];
            postprocess_detect(
                output,
                shape,
                preprocess,
                config,
                names,
                orig_img,
                path,
                speed,
                inference_shape,
            )
        }
        Task::Segment => postprocess_segment(
            outputs,
            preprocess,
            config,
            names,
            orig_img,
            path,
            speed,
            inference_shape,
        ),
        Task::Pose => {
            let (output, shape) = &outputs[0];
            postprocess_pose(
                output,
                shape,
                preprocess,
                config,
                names,
                orig_img,
                path,
                speed,
                inference_shape,
            )
        }
        Task::Classify => {
            let (output, shape) = &outputs[0];
            postprocess_classify(output, shape, names, orig_img, path, speed, inference_shape)
        }
        Task::Obb => {
            let (output, shape) = &outputs[0];
            postprocess_obb(
                output,
                shape,
                preprocess,
                config,
                names,
                orig_img,
                path,
                speed,
                inference_shape,
            )
        }
    }
}

/// Post-process detection model output.
///
/// Converts raw YOLO model output into a list of bounding boxes with class scores.
///
/// # Arguments
///
/// * `output` - Flat vector of model output values.
/// * `output_shape` - Shape of the output tensor.
/// * `preprocess` - Preprocessing metadata (scaling, padding).
/// * `config` - Inference configuration (thresholds).
/// * `names` - Class ID to name mapping.
/// * `orig_img` - Original image data.
/// * `path` - Source image path.
/// * `speed` - Timing metrics.
/// * `inference_shape` - Input shape used for inference.
///
/// # Returns
///
/// `Results` struct containing detected bounding boxes.
#[allow(
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::cast_precision_loss
)]
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
    let (num_classes, num_predictions, is_transposed) =
        parse_detect_shape(output_shape, names.len());

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
    let boxes_data = extract_detect_boxes(output_2d.view(), num_classes, preprocess, config);

    if !boxes_data.is_empty() {
        results.boxes = Some(Boxes::new(boxes_data, preprocess.orig_shape));
    }

    results
}

/// Parse detection output shape to determine format.
///
/// Derives class count from output shape when metadata is missing (`expected_classes` == 0).
/// YOLO outputs are either [1, `num_features`, `num_preds`] or [1, `num_preds`, `num_features`]
/// where `num_features` = 4 (bbox) + `num_classes`.
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
                let (num_features, num_preds, transposed) =
                    if a < b { (a, b, false) } else { (b, a, true) };
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
                let (num_features, num_preds, transposed) =
                    if a < b { (a, b, false) } else { (b, a, true) };
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
///
/// Filters predictions by confidence threshold and converts coordinates to original image space.
#[allow(clippy::cast_precision_loss, clippy::needless_pass_by_value)]
fn extract_detect_boxes(
    output: ArrayView2<f32>,
    _num_classes: usize,
    preprocess: &PreprocessResult,
    config: &InferenceConfig,
) -> Array2<f32> {
    let _num_predictions = output.nrows();
    let mut candidates = Vec::new();

    // Iterate over rows efficiently
    // output shape is (num_predictions, 4 + num_classes)
    // We can iterate over raw elements if we are careful, but using outer_iter() is safer and still fast

    // Pre-calculate scaling factors to avoid repeated struct access
    let (scale_y, scale_x) = preprocess.scale;
    let (pad_top, pad_left) = preprocess.padding;
    let orig_shape = preprocess.orig_shape;
    let (max_w, max_h) = (orig_shape.1 as f32, orig_shape.0 as f32);

    for row in output.outer_iter() {
        // Row is [cx, cy, w, h, class_scores...]

        // Efficiently find the best class score without allocating a new slice or iterator chain.
        // We skip low-confidence detections early to avoid expensive coordinate scaling and NMS operations later.
        let scores = row.slice(s![4..]);

        // Find best class manually to avoid iterator overhead
        let (best_class, best_score) =
            scores
                .iter()
                .enumerate()
                .fold((0, 0.0f32), |(best_idx, best_val), (idx, &val)| {
                    if val > best_val {
                        (idx, val)
                    } else {
                        (best_idx, best_val)
                    }
                });

        // Filter by confidence threshold early to reduce computation
        if best_score < config.confidence_threshold {
            continue;
        }

        // Extract coordinates only for candidate detections
        let cx = row[0];
        let cy = row[1];
        let w = row[2];
        let h = row[3];

        let x1 = (cx - w / 2.0 - pad_left) / scale_x;
        let y1 = (cy - h / 2.0 - pad_top) / scale_y;
        let x2 = (cx + w / 2.0 - pad_left) / scale_x;
        let y2 = (cy + h / 2.0 - pad_top) / scale_y;

        // Clip (clamp)
        let x1 = x1.clamp(0.0, max_w);
        let y1 = y1.clamp(0.0, max_h);
        let x2 = x2.clamp(0.0, max_w);
        let y2 = y2.clamp(0.0, max_h);

        candidates.push(([x1, y1, x2, y2], best_score, best_class));
    }

    if candidates.is_empty() {
        return Array2::zeros((0, 6));
    }

    // Apply per-class NMS (only suppress boxes within the same class)
    let keep_indices = nms_per_class(&candidates, config.iou_threshold);

    // Build output array with kept detections
    let num_kept = keep_indices.len().min(config.max_det);
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

/// Post-process segmentation model output.
///
/// Generates bounding boxes and segmentation masks from the model output.
///
/// # Arguments
///
/// * `outputs` - Vector of model outputs (detection features and mask prototypes).
/// * `preprocess` - Preprocessing metadata.
/// * `config` - Inference configuration.
/// * `names` - Class mapping.
/// * `orig_img` - Original image.
/// * `path` - Source path.
/// * `speed` - Timing metrics.
/// * `inference_shape` - Inference input dimensions.
///
/// # Returns
///
/// `Results` struct containing boxes and masks.
#[allow(
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::needless_pass_by_value,
    clippy::manual_let_else,
    clippy::cast_possible_truncation
)]
fn postprocess_segment(
    outputs: Vec<(Vec<f32>, Vec<usize>)>,
    preprocess: &PreprocessResult,
    config: &InferenceConfig,
    names: &HashMap<usize, String>,
    orig_img: Array3<u8>,
    path: String,
    speed: Speed,
    inference_shape: (u32, u32),
) -> Results {
    let mut results = Results::new(orig_img, path, names.clone(), speed, inference_shape);

    if outputs.len() < 2 {
        // Protos output missing - log warning for user visibility
        eprintln!(
            "WARNING ⚠️ Segmentation model missing protos output (expected 2 outputs, got {}). Returning empty masks.",
            outputs.len()
        );
        return results;
    }

    let (output0, shape0) = &outputs[0];
    let (output1, shape1) = &outputs[1];

    // output0: [1, 4 + nc + 32, 8400]
    // output1: [1, 32, 160, 160] (protos)

    // 1. Process Detections
    // Standard segmentation models use 32 mask prototypes
    let num_masks = 32;
    let expected_features = 4 + names.len() + num_masks;

    // Manual shape check
    let (num_preds, is_transposed) = if shape0.len() == 3 {
        let (a, b) = (shape0[1], shape0[2]);
        if a == expected_features {
            (b, false) // [1, features, preds]
        } else if b == expected_features {
            (a, true) // [1, preds, features]
        } else {
            // Assume format [1, 116, 8400] if ambiguous
            if a < b { (b, false) } else { (a, true) }
        }
    } else {
        (0, false)
    };

    if output0.is_empty() || num_preds == 0 {
        return results;
    }

    // Convert to 2D [preds, features]
    let output_2d = if is_transposed {
        Array2::from_shape_vec((num_preds, expected_features), output0.clone())
            .unwrap_or_else(|_| Array2::zeros((0, 0)))
    } else {
        let arr = Array2::from_shape_vec((expected_features, num_preds), output0.clone())
            .unwrap_or_else(|_| Array2::zeros((0, 0)));
        arr.t().to_owned()
    };

    // Filter and NMS
    let mut candidates = Vec::new(); // (bbox, score, class, original_index)

    for i in 0..num_preds {
        let scores = output_2d.slice(s![i, 4..4 + names.len()]);
        let (best_class, best_score) = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map_or((0, 0.0), |(idx, &score)| (idx, score));

        if best_score < config.confidence_threshold {
            continue;
        }

        // Box
        let cx = output_2d[[i, 0]];
        let cy = output_2d[[i, 1]];
        let w = output_2d[[i, 2]];
        let h = output_2d[[i, 3]];
        let x1 = cx - w / 2.0;
        let y1 = cy - h / 2.0;
        let x2 = cx + w / 2.0;
        let y2 = cy + h / 2.0;

        let scaled = scale_coords(&[x1, y1, x2, y2], preprocess.scale, preprocess.padding);
        let clipped = clip_coords(&scaled, preprocess.orig_shape);

        candidates.push((
            [clipped[0], clipped[1], clipped[2], clipped[3]],
            best_score,
            best_class,
            i, // Keep index to get coefficients
        ));
    }

    if candidates.is_empty() {
        return results;
    }

    // Prepare candidates for NMS (bbox, score, class)
    let nms_candidates: Vec<_> = candidates
        .iter()
        .map(|(bbox, score, class, _)| (*bbox, *score, *class))
        .collect();

    let keep_indices = nms_per_class(&nms_candidates, config.iou_threshold);
    let num_kept = keep_indices.len().min(config.max_det);

    // 2. Extract Box Results
    let mut boxes_data = Array2::zeros((num_kept, 6));
    let mut mask_coeffs = Array2::zeros((num_kept, num_masks));

    for (out_idx, &keep_idx) in keep_indices.iter().take(num_kept).enumerate() {
        let (bbox, score, class, orig_idx) = &candidates[keep_idx];
        boxes_data[[out_idx, 0]] = bbox[0];
        boxes_data[[out_idx, 1]] = bbox[1];
        boxes_data[[out_idx, 2]] = bbox[2];
        boxes_data[[out_idx, 3]] = bbox[3];
        boxes_data[[out_idx, 4]] = *score;
        boxes_data[[out_idx, 5]] = *class as f32;

        // Extract coefficients: [orig_idx, 4+nc..]
        let start = 4 + names.len();
        let coeffs = output_2d.slice(s![*orig_idx, start..start + num_masks]);
        for m in 0..num_masks {
            mask_coeffs[[out_idx, m]] = coeffs[m];
        }
    }

    results.boxes = Some(Boxes::new(boxes_data.clone(), preprocess.orig_shape));

    // 3. Process Masks
    // Protos: [1, 32, 160, 160] -> [32, 25600]
    // Validate protos shape before indexing to prevent panic
    if shape1.len() < 4 {
        eprintln!(
            "WARNING ⚠️ Protos output has unexpected shape (expected 4 dims, got {}). Skipping mask generation.",
            shape1.len()
        );
        return results;
    }
    let mh = shape1[2];
    let mw = shape1[3];

    // Validate expected mask dimensions match
    if shape1[1] != num_masks {
        eprintln!(
            "WARNING ⚠️ Protos output has {} mask channels, expected {}. Mask quality may be affected.",
            shape1[1], num_masks
        );
    }

    let protos = match Array2::from_shape_vec((num_masks, mh * mw), output1.clone()) {
        Ok(arr) => arr,
        Err(e) => {
            eprintln!("WARNING ⚠️ Failed to create protos array: {e}. Skipping mask generation.");
            return results;
        }
    };

    // Matrix Mul: [N, 32] x [32, 25600] -> [N, 25600]
    let masks_flat = mask_coeffs.dot(&protos);

    // Resize and crop to original image size
    let (oh, ow) = preprocess.orig_shape;
    let (th, tw) = inference_shape;
    let (pad_top, pad_left) = preprocess.padding;

    // Pre-calculate crop parameters (same for all masks)
    let scale_w = mw as f32 / tw as f32;
    let scale_h = mh as f32 / th as f32;
    let crop_x = pad_left * scale_w;
    let crop_y = pad_top * scale_h;
    let crop_w = 2.0f32.mul_add(-crop_x, mw as f32);
    let crop_h = 2.0f32.mul_add(-crop_y, mh as f32);

    // Initialize output array
    let mut masks_data = Array3::zeros((num_kept, oh as usize, ow as usize));

    // Process each mask in parallel using Rayon and ndarray::Zip.
    // Each thread handles resizing and cropping for one mask.
    //
    // Inputs:
    // - mask_out: Mutable view into output masks array
    // - mask_flat: Mask coefficients for the detection
    // - box_data: Bounding box for the detection
    Zip::from(masks_data.outer_iter_mut())
        .and(masks_flat.outer_iter())
        .and(boxes_data.outer_iter())
        .par_for_each(|mut mask_out, mask_flat, box_data| {
            // Create a local resizer for each task (Resizer is not Sync)
            let mut resizer = Resizer::new();
            let resize_alg = ResizeAlg::Convolution(FilterType::Bilinear);

            // Sigmoid into a Vec<f32>
            let f32_data: Vec<f32> = mask_flat
                .iter()
                .map(|&val| 1.0 / (1.0 + (-val).exp()))
                .collect();

            // Use bytemuck for efficient f32->bytes conversion
            let src_bytes: &[u8] = bytemuck::cast_slice(&f32_data);

            // Create source image (160x160)
            let src_image = match Image::from_vec_u8(
                mw as u32,
                mh as u32,
                src_bytes.to_vec(),
                PixelType::F32,
            ) {
                Ok(img) => img,
                Err(_) => return, // Skip if creation fails
            };

            // Create dest image (orig_w x orig_h)
            let mut dst_image = Image::new(ow, oh, PixelType::F32);

            // Configure resize with crop
            let safe_crop_x = f64::from(crop_x.max(0.0));
            let safe_crop_y = f64::from(crop_y.max(0.0));
            let safe_crop_w = f64::from(crop_w.max(1.0).min(mw as f32));
            let safe_crop_h = f64::from(crop_h.max(1.0).min(mh as f32));

            let options = ResizeOptions::new().resize_alg(resize_alg).crop(
                safe_crop_x,
                safe_crop_y,
                safe_crop_w,
                safe_crop_h,
            );

            // Handle resize errors gracefully
            if resizer
                .resize(&src_image, &mut dst_image, &options)
                .is_err()
            {
                return;
            }

            // Get resized data as f32 slice
            let dst_bytes = dst_image.buffer();
            let dst_slice: &[f32] = bytemuck::cast_slice(dst_bytes);

            // Apply bbox cropping and store directly to output array
            let x1 = box_data[0].max(0.0).min(ow as f32);
            let y1 = box_data[1].max(0.0).min(oh as f32);
            let x2 = box_data[2].max(0.0).min(ow as f32);
            let y2 = box_data[3].max(0.0).min(oh as f32);

            for y in 0..oh as usize {
                for x in 0..ow as usize {
                    let val = dst_slice[y * ow as usize + x];
                    let x_f = x as f32;
                    let y_f = y as f32;
                    // Apply bounding box mask: invalid pixels outside the box are zeroed.
                    if x_f >= x1 && x_f <= x2 && y_f >= y1 && y_f <= y2 {
                        mask_out[[y, x]] = val;
                    }
                }
            }
        });

    results.masks = Some(Masks::new(masks_data, preprocess.orig_shape));

    results
}

/// Post-process pose estimation model output.
///
/// Extracts bounding boxes and keypoints (skeleton) from the model output.
///
/// # Arguments
///
/// * `output` - Flat vector of model output.
/// * `output_shape` - Output tensor dimensions.
/// * `preprocess` - Preprocessing metadata.
/// * `config` - Inference configuration.
/// * `names` - Class name mapping.
/// * `orig_img` - Original image.
/// * `path` - Source image path.
/// * `speed` - Timing data.
/// * `inference_shape` - Inference input dimensions.
///
/// # Returns
///
/// `Results` struct containing boxes and keypoints.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::type_complexity,
    clippy::cast_precision_loss,
    clippy::doc_lazy_continuation
)]
fn postprocess_pose(
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

    // Standard COCO pose has 17 keypoints, each with (x, y, conf)
    let num_keypoints = 17;
    let kpt_dim = 3; // x, y, visibility/confidence
    let kpt_features = num_keypoints * kpt_dim; // 51

    // Pose typically has 1 class (person), so features = 4 + 1 + 51 = 56
    let num_classes = names.len().max(1);
    let expected_features = 4 + num_classes + kpt_features;

    // Parse output shape
    let (num_preds, is_transposed) = if output_shape.len() == 3 {
        let (a, b) = (output_shape[1], output_shape[2]);
        if a == expected_features || (a < b && a >= 4 + kpt_features) {
            (b, false) // [1, features, preds]
        } else {
            (a, true) // [1, preds, features]
        }
    } else if output_shape.len() == 2 {
        let (a, b) = (output_shape[0], output_shape[1]);
        if a < b { (b, false) } else { (a, true) }
    } else {
        (0, false)
    };

    if output.is_empty() || num_preds == 0 {
        return results;
    }

    // Infer actual feature count from data
    let actual_features = output.len() / num_preds;
    if actual_features < 4 + kpt_features {
        eprintln!(
            "WARNING ⚠️ Pose model has insufficient features ({actual_features}), expected at least {}",
            4 + kpt_features
        );
        return results;
    }

    // Convert to 2D [preds, features]
    let output_2d = if is_transposed {
        Array2::from_shape_vec((num_preds, actual_features), output.to_vec())
            .unwrap_or_else(|_| Array2::zeros((0, 0)))
    } else {
        let arr = Array2::from_shape_vec((actual_features, num_preds), output.to_vec())
            .unwrap_or_else(|_| Array2::zeros((0, 0)));
        arr.t().to_owned()
    };

    if output_2d.is_empty() {
        return results;
    }

    // Derive number of classes from actual features
    let derived_classes = actual_features.saturating_sub(4 + kpt_features);
    let num_classes = derived_classes.max(1);

    // Filter and NMS - store candidates with keypoints
    let mut candidates: Vec<([f32; 4], f32, usize, Vec<[f32; 3]>)> = Vec::new();

    for i in 0..num_preds {
        // Get class score(s) - for pose, typically just "person" class
        let class_scores = output_2d.slice(s![i, 4..4 + num_classes]);
        let (best_class, best_score) = class_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
            .map_or((0, 0.0), |(idx, &score)| {
                (idx, if score.is_nan() { 0.0 } else { score })
            });

        if best_score < config.confidence_threshold {
            continue;
        }

        // Extract box coordinates (xywh format)
        let cx = output_2d[[i, 0]];
        let cy = output_2d[[i, 1]];
        let w = output_2d[[i, 2]];
        let h = output_2d[[i, 3]];

        // Convert to xyxy
        let x1 = cx - w / 2.0;
        let y1 = cy - h / 2.0;
        let x2 = cx + w / 2.0;
        let y2 = cy + h / 2.0;

        // Scale box to original image space
        let scaled = scale_coords(&[x1, y1, x2, y2], preprocess.scale, preprocess.padding);
        let clipped = clip_coords(&scaled, preprocess.orig_shape);

        // Extract keypoints (after class scores)
        let kpt_start = 4 + num_classes;
        let mut keypoints = Vec::with_capacity(num_keypoints);
        for k in 0..num_keypoints {
            let kpt_offset = kpt_start + k * kpt_dim;
            let kpt_x = output_2d[[i, kpt_offset]];
            let kpt_y = output_2d[[i, kpt_offset + 1]];
            let kpt_conf = output_2d[[i, kpt_offset + 2]];

            // Scale keypoint coordinates to original image space
            let scaled_kpt = scale_coords(
                &[kpt_x, kpt_y, kpt_x, kpt_y],
                preprocess.scale,
                preprocess.padding,
            );
            let (oh, ow) = preprocess.orig_shape;
            #[allow(clippy::cast_precision_loss)]
            let scaled_x = scaled_kpt[0].max(0.0).min(ow as f32);
            #[allow(clippy::cast_precision_loss)]
            let scaled_y = scaled_kpt[1].max(0.0).min(oh as f32);

            keypoints.push([scaled_x, scaled_y, kpt_conf]);
        }

        candidates.push((
            [clipped[0], clipped[1], clipped[2], clipped[3]],
            best_score,
            best_class,
            keypoints,
        ));
    }

    if candidates.is_empty() {
        results.keypoints = Some(Keypoints::new(
            Array3::zeros((0, num_keypoints, kpt_dim)),
            preprocess.orig_shape,
        ));
        return results;
    }

    // Apply NMS
    let nms_candidates: Vec<_> = candidates
        .iter()
        .map(|(bbox, score, class, _)| (*bbox, *score, *class))
        .collect();
    let keep_indices = nms_per_class(&nms_candidates, config.iou_threshold);
    let num_kept = keep_indices.len().min(config.max_det);

    // Build output arrays
    let mut boxes_data = Array2::zeros((num_kept, 6));
    let mut keypoints_data = Array3::zeros((num_kept, num_keypoints, kpt_dim));

    for (out_idx, &keep_idx) in keep_indices.iter().take(num_kept).enumerate() {
        let (bbox, score, class, kpts) = &candidates[keep_idx];

        // Store box data
        boxes_data[[out_idx, 0]] = bbox[0];
        boxes_data[[out_idx, 1]] = bbox[1];
        boxes_data[[out_idx, 2]] = bbox[2];
        boxes_data[[out_idx, 3]] = bbox[3];
        boxes_data[[out_idx, 4]] = *score;
        #[allow(clippy::cast_precision_loss)]
        let class_f32 = *class as f32;
        boxes_data[[out_idx, 5]] = class_f32;

        // Store keypoints
        for (k, kpt) in kpts.iter().enumerate() {
            keypoints_data[[out_idx, k, 0]] = kpt[0]; // x
            keypoints_data[[out_idx, k, 1]] = kpt[1]; // y
            keypoints_data[[out_idx, k, 2]] = kpt[2]; // confidence
        }
    }

    results.boxes = Some(Boxes::new(boxes_data, preprocess.orig_shape));
    results.keypoints = Some(Keypoints::new(keypoints_data, preprocess.orig_shape));

    results
}

/// Post-process classification model output.
///
/// Computes best class predictions and probabilities.
///
/// # Arguments
///
/// * `output` - Raw model output vector.
/// * `_output_shape` - Output shape (unused).
/// * `names` - Class name mapping.
/// * `orig_img` - Original image.
/// * `path` - Source path.
/// * `speed` - Timing metrics.
/// * `inference_shape` - Inference dimensions.
///
/// # Returns
///
/// `Results` struct containing classification probabilities.
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

    // Filter out NaN values and ensure valid probabilities
    let mut probs_vec: Vec<f32> = output
        .iter()
        .map(|&v| if v.is_nan() { 0.0 } else { v })
        .collect();

    // Check if softmax is already applied (sum ≈ 1.0)
    let sum: f32 = probs_vec.iter().sum();
    if (sum - 1.0).abs() > 0.1 && sum > 0.0 {
        // Apply softmax normalization
        let max_val = probs_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = probs_vec.iter().map(|&v| (v - max_val).exp()).collect();
        let exp_sum: f32 = exp_vals.iter().sum();
        if exp_sum > 0.0 {
            probs_vec = exp_vals.iter().map(|&v| v / exp_sum).collect();
        }
    }

    let probs = ndarray::Array1::from_vec(probs_vec);
    results.probs = Some(Probs::new(probs));

    results
}

/// Post-process OBB (oriented bounding box) model output.
///
/// Extracts oriented bounding boxes with rotation angle.
///
/// # Arguments
///
/// * `output` - Model output data.
/// * `output_shape` - Output tensor shape.
/// * `preprocess` - Preprocessing metadata.
/// * `config` - Inference configuration.
/// * `names` - Class name mapping.
/// * `orig_img` - Original image.
/// * `path` - Source path.
/// * `speed` - Timing metrics.
/// * `inference_shape` - Inference dimensions.
///
/// # Returns
///
/// `Results` struct containing oriented bounding boxes.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names
)]
fn postprocess_obb(
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

    // OBB format: [xywh, class_scores..., rotation_angle]
    // features = 4 (bbox) + num_classes + 1 (angle)
    let num_classes = names.len().max(1);
    let expected_features = 4 + num_classes + 1;

    // Parse output shape
    let (num_preds, is_transposed) = if output_shape.len() == 3 {
        let (a, b) = (output_shape[1], output_shape[2]);
        if a == expected_features || (a < b && a >= 6) {
            (b, false) // [1, features, preds]
        } else {
            (a, true) // [1, preds, features]
        }
    } else if output_shape.len() == 2 {
        let (a, b) = (output_shape[0], output_shape[1]);
        if a < b { (b, false) } else { (a, true) }
    } else {
        (0, false)
    };

    if output.is_empty() || num_preds == 0 {
        return results;
    }

    // Infer actual feature count from data
    let actual_features = output.len() / num_preds;
    if actual_features < 6 {
        eprintln!(
            "WARNING ⚠️ OBB model has insufficient features ({actual_features}), expected at least 6"
        );
        return results;
    }

    // Convert to 2D [preds, features]
    let output_2d = if is_transposed {
        Array2::from_shape_vec((num_preds, actual_features), output.to_vec())
            .unwrap_or_else(|_| Array2::zeros((0, 0)))
    } else {
        let arr = Array2::from_shape_vec((actual_features, num_preds), output.to_vec())
            .unwrap_or_else(|_| Array2::zeros((0, 0)));
        arr.t().to_owned()
    };

    if output_2d.is_empty() {
        return results;
    }

    // Derive number of classes from features: features = 4 + nc + 1
    let derived_classes = actual_features.saturating_sub(5); // 4 bbox + 1 angle
    let num_classes = derived_classes.max(1);

    // Filter and NMS - store candidates with angle
    let mut candidates: Vec<([f32; 5], f32, usize)> = Vec::new(); // [cx, cy, w, h, angle], conf, class

    for i in 0..num_preds {
        // Get class scores
        let class_scores = output_2d.slice(s![i, 4..4 + num_classes]);
        let (best_class, best_score) = class_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
            .map_or((0, 0.0), |(idx, &score)| {
                (idx, if score.is_nan() { 0.0 } else { score })
            });

        if best_score < config.confidence_threshold {
            continue;
        }

        // Extract OBB: xywh + rotation
        let cx = output_2d[[i, 0]];
        let cy = output_2d[[i, 1]];
        let w = output_2d[[i, 2]];
        let h = output_2d[[i, 3]];
        let angle = output_2d[[i, 4 + num_classes]]; // Last value is rotation angle (radians)

        // Scale center coordinates to original image space
        let scaled = scale_coords(&[cx, cy, cx, cy], preprocess.scale, preprocess.padding);
        let scaled_cx = scaled[0];
        let scaled_cy = scaled[1];

        // Scale width and height (note: don't apply padding, just scale)
        let scaled_w = w / preprocess.scale.1;
        let scaled_h = h / preprocess.scale.0;

        // Clip center to image bounds
        let (oh, ow) = preprocess.orig_shape;
        #[allow(clippy::cast_precision_loss)]
        let clipped_cx = scaled_cx.max(0.0).min(ow as f32);
        #[allow(clippy::cast_precision_loss)]
        let clipped_cy = scaled_cy.max(0.0).min(oh as f32);

        candidates.push((
            [clipped_cx, clipped_cy, scaled_w, scaled_h, angle],
            best_score,
            best_class,
        ));
    }

    if candidates.is_empty() {
        results.obb = Some(Obb::new(Array2::zeros((0, 7)), preprocess.orig_shape));
        return results;
    }

    // Apply Rotated NMS for precise suppression using ProbIoU (Hellinger distance).
    // This ensures that overlapping rotated boxes are correctly filtered based on their
    // actual geometric overlap, which standard axis-aligned IoU cannot handle.
    let keep_indices = nms_rotated_per_class(&candidates, config.iou_threshold);
    let num_kept = keep_indices.len().min(config.max_det);

    // Build output array: [cx, cy, w, h, rotation, conf, cls]
    let mut obb_data = Array2::zeros((num_kept, 7));

    for (out_idx, &keep_idx) in keep_indices.iter().take(num_kept).enumerate() {
        let (xywhr, score, class) = &candidates[keep_idx];
        obb_data[[out_idx, 0]] = xywhr[0]; // cx
        obb_data[[out_idx, 1]] = xywhr[1]; // cy
        obb_data[[out_idx, 2]] = xywhr[2]; // w
        obb_data[[out_idx, 3]] = xywhr[3]; // h
        obb_data[[out_idx, 4]] = xywhr[4]; // rotation (radians)
        obb_data[[out_idx, 5]] = *score;
        #[allow(clippy::cast_precision_loss)]
        let class_f32 = *class as f32;
        obb_data[[out_idx, 6]] = class_f32;
    }

    results.obb = Some(Obb::new(obb_data, preprocess.orig_shape));

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
        assert_eq!(nc, 80); // Inferred: 84 - 4 = 80 classes
        assert_eq!(np, 8400);
        assert!(!transposed);

        // Transposed format [1, 8400, 84] with no metadata
        let (nc, np, transposed) = parse_detect_shape(&[1, 8400, 84], 0);
        assert_eq!(nc, 80); // Inferred: 84 - 4 = 80 classes
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

    #[test]
    fn test_nan_scores_handled() {
        // Test that NaN scores don't cause panic
        let mut output: Vec<f32> = vec![0.0; 84]; // One prediction
        // Set box coords
        output[0] = 100.0; // cx
        output[1] = 100.0; // cy
        output[2] = 50.0; // w
        output[3] = 50.0; // h
        // Set class scores with NaN
        output[4] = f32::NAN;
        output[5] = 0.9; // This should be selected even with NaN present

        let preprocess = PreprocessResult {
            tensor: ndarray::Array4::zeros((1, 3, 640, 640)),
            tensor_f16: None,
            orig_shape: (640, 640),
            scale: (1.0, 1.0),
            padding: (0.0, 0.0),
        };
        let config = InferenceConfig::default();
        let mut names = HashMap::new();
        names.insert(0, "class0".to_string());
        names.insert(1, "class1".to_string());
        let orig_img = ndarray::Array3::zeros((640, 640, 3));

        // This should not panic
        let results = postprocess_detect(
            &output,
            &[1, 84, 1],
            &preprocess,
            &config,
            &names,
            orig_img,
            String::new(),
            Speed::default(),
            (640, 640),
        );

        // Test passed if we got here without panicking - NaN was handled gracefully
        // Note: The detection may or may not exist depending on how NaN affects max_by
        // The key is that the code didn't crash
        let _ = results;
    }

    #[test]
    fn test_malformed_shape_fallback() {
        // Test that malformed shapes return empty results instead of panicking
        let output: Vec<f32> = vec![0.0; 100]; // Some data

        let preprocess = PreprocessResult {
            tensor: ndarray::Array4::zeros((1, 3, 640, 640)),
            tensor_f16: None,
            orig_shape: (640, 640),
            scale: (1.0, 1.0),
            padding: (0.0, 0.0),
        };
        let config = InferenceConfig::default();
        let names = HashMap::new();
        let orig_img = ndarray::Array3::zeros((640, 640, 3));

        // Empty shape should not panic
        let results = postprocess_detect(
            &output,
            &[],
            &preprocess,
            &config,
            &names,
            orig_img.clone(),
            String::new(),
            Speed::default(),
            (640, 640),
        );
        assert!(results.is_empty());

        // Single dimension shape should not panic
        let results = postprocess_detect(
            &output,
            &[100],
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

    #[test]
    fn test_postprocess_pose_logic() {
        // Mock output for pose: [1, 56, 100]
        // 56 features = 4 bbox + 1 class + 51 keypoints (17*3)
        let num_preds = 100;
        let num_features = 56;
        let mut output = vec![0.0; num_preds * num_features];

        // Fill one prediction
        let idx = 0;
        // BBox: cx, cy, w, h
        output[idx] = 100.0;
        output[idx + num_preds] = 100.0;
        output[idx + num_preds * 2] = 50.0;
        output[idx + num_preds * 3] = 50.0;
        // Class score
        output[idx + num_preds * 4] = 0.9;
        // Keypoints: 17 * 3
        for k in 0..17 {
            let offset = 5 + k * 3;
            output[idx + num_preds * offset] = 100.0; // x
            output[idx + num_preds * (offset + 1)] = 100.0; // y
            output[idx + num_preds * (offset + 2)] = 0.8; // conf
        }

        let preprocess = PreprocessResult {
            tensor: ndarray::Array4::zeros((1, 3, 640, 640)),
            tensor_f16: None,
            orig_shape: (640, 640),
            scale: (1.0, 1.0),
            padding: (0.0, 0.0),
        };
        let config = InferenceConfig::default();
        let mut names = HashMap::new();
        names.insert(0, "person".to_string());

        // Shape [1, 56, 100]
        let results = postprocess_pose(
            &output,
            &[1, num_features, num_preds],
            &preprocess,
            &config,
            &names,
            ndarray::Array3::zeros((640, 640, 3)),
            "test.jpg".to_string(),
            Speed::default(),
            (640, 640),
        );

        assert!(results.keypoints.is_some());
        let kpts = results.keypoints.unwrap();
        assert_eq!(kpts.data.shape()[0], 1); // 1 detection
        assert_eq!(kpts.data.shape()[1], 17); // 17 keypoints
        assert_eq!(kpts.data.shape()[2], 3); // x, y, conf

        #[allow(clippy::float_cmp)]
        {
            // Verify values
            assert_eq!(kpts.data[[0, 0, 0]], 100.0);
            assert_eq!(kpts.data[[0, 0, 2]], 0.8);
        }
    }

    #[test]
    fn test_postprocess_obb_logic() {
        // Mock output for OBB: [1, 6, 100]
        // 6 features = 4 bbox + 1 class + 1 angle
        let num_preds = 100;
        let num_features = 6;
        let mut output = vec![0.0; num_preds * num_features];

        // Fill one prediction
        let idx = 0;
        // BBox: cx, cy, w, h
        output[idx] = 100.0;
        output[idx + num_preds] = 100.0;
        output[idx + num_preds * 2] = 50.0;
        output[idx + num_preds * 3] = 20.0;
        // Class score
        output[idx + num_preds * 4] = 0.95;
        // Angle
        output[idx + num_preds * 5] = std::f32::consts::FRAC_PI_4; // 45 degrees

        let preprocess = PreprocessResult {
            tensor: ndarray::Array4::zeros((1, 3, 640, 640)),
            tensor_f16: None,
            orig_shape: (640, 640),
            scale: (1.0, 1.0),
            padding: (0.0, 0.0),
        };
        let config = InferenceConfig::default();
        let mut names = HashMap::new();
        names.insert(0, "object".to_string());

        // Shape [1, 6, 100]
        let results = postprocess_obb(
            &output,
            &[1, num_features, num_preds],
            &preprocess,
            &config,
            &names,
            ndarray::Array3::zeros((640, 640, 3)),
            "test.jpg".to_string(),
            Speed::default(),
            (640, 640),
        );

        assert!(results.obb.is_some());
        let obb = results.obb.unwrap();
        assert_eq!(obb.len(), 1);

        // Verify values
        let data = obb.data.row(0);
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(data[0], 100.0); // cx
            assert_eq!(data[4], std::f32::consts::FRAC_PI_4); // angle
            assert_eq!(data[5], 0.95); // conf
        }
    }
}
