// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Post-processing for YOLO model outputs.
//!
//! This module handles task-specific post-processing of raw model outputs,
//! including NMS, coordinate transformation, and result construction.

#![allow(
    unsafe_code,
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::if_not_else,
    clippy::ptr_as_ptr,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::collections::HashMap;

use wide::{CmpGt, f32x8};

use fast_image_resize::images::Image;
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use ndarray::{Array2, Array3, ArrayView1, ArrayViewMut2, Zip, s};

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
    outputs: Vec<(&[f32], Vec<usize>)>,
    metadata: &crate::metadata::ModelMetadata,
    preprocess: &PreprocessResult,
    config: &InferenceConfig,
    orig_img: Array3<u8>,
    path: String,
    speed: Speed,
    inference_shape: (u32, u32),
) -> Results {
    match metadata.task {
        Task::Detect => {
            let (output, shape) = &outputs[0];
            postprocess_detect(
                output,
                shape,
                preprocess,
                config,
                &metadata.names,
                metadata.end2end,
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
            &metadata.names,
            orig_img,
            path,
            speed,
            inference_shape,
            metadata.end2end,
        ),
        Task::Pose => {
            let (output, shape) = &outputs[0];
            postprocess_pose(
                output,
                shape,
                preprocess,
                config,
                &metadata.names,
                orig_img,
                path,
                speed,
                inference_shape,
                metadata.end2end,
            )
        }
        Task::Classify => {
            let (output, shape) = &outputs[0];
            postprocess_classify(
                output,
                shape,
                &metadata.names,
                orig_img,
                path,
                speed,
                inference_shape,
            )
        }
        Task::Obb => {
            let (output, shape) = &outputs[0];
            postprocess_obb(
                output,
                shape,
                preprocess,
                config,
                &metadata.names,
                orig_img,
                path,
                speed,
                inference_shape,
                metadata.end2end,
            )
        }
    }
}

/// Post-process detection model output.
///
/// Zero-copy implementation using stride-based indexing to avoid memory allocations.
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
    end2end: bool,
    orig_img: Array3<u8>,
    path: String,
    speed: Speed,
    inference_shape: (u32, u32),
) -> Results {
    let mut results = Results::new(orig_img, path, names.clone(), speed, inference_shape);

    // Auto-detect E2E from shape if metadata flag is missing
    // E2E output is [Batch, N, 6] (or [N, 6]) -> last dim is 6.
    // Standard output is [Batch, 4+nc, N] or [Batch, N, 4+nc].
    // If we have 80 classes, standard features is 84. If we see 6, it's E2E.
    // Ambiguity exists only if nc=2 (features=6). In that case, we rely on flag.
    let is_e2e_shape = output_shape.last().is_some_and(|&d| d == 6) && names.len() != 2;
    let use_e2e = end2end || is_e2e_shape;

    let boxes_data = if use_e2e {
        extract_e2e_boxes(output, output_shape, preprocess, config)
    } else {
        // Parse output shape - handle both [1, 84, 8400] and [1, 8400, 84] formats
        let (num_classes, num_predictions, is_transposed) =
            parse_detect_shape(output_shape, names.len());

        if output.is_empty() || num_predictions == 0 {
            return results;
        }

        // Zero-copy extraction with stride-based indexing
        extract_detect_boxes(
            output,
            num_classes,
            num_predictions,
            is_transposed,
            preprocess,
            config,
        )
    };

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

/// Ultra-fast detection extraction - single-threaded tight loop.
///
/// Key optimizations:
/// - No parallelization overhead (Rayon adds ~0.5ms for small workloads)
/// - Pre-sized allocations
/// - Minimal branching in hot loops
/// - Direct unsafe indexing
#[allow(clippy::cast_precision_loss, clippy::too_many_arguments)]
#[derive(Clone, Copy)]
struct Candidate {
    bbox: [f32; 4],
    score: f32,
    class: usize,
}

#[allow(clippy::cast_precision_loss, clippy::too_many_arguments)]
fn extract_e2e_boxes(
    output: &[f32],
    output_shape: &[usize],
    preprocess: &PreprocessResult,
    config: &InferenceConfig,
) -> Array2<f32> {
    // E2E output shape is typically [Batch, N, 6] -> [1, 300, 6]
    // where 6 features are: [x1, y1, x2, y2, conf, cls]
    // We need to parse this manually as it differs from standard [Batch, Feat, N]

    if output_shape.is_empty() || output.is_empty() {
        return Array2::zeros((0, 6));
    }

    // Determine dimensions (ignore batch dim if present)
    // Likely [Batch, NumPreds, 6] or [NumPreds, 6]
    let (num_preds, num_features) = if output_shape.len() == 3 {
        (output_shape[1], output_shape[2])
    } else if output_shape.len() == 2 {
        (output_shape[0], output_shape[1])
    } else {
        // Fallback or error
        return Array2::zeros((0, 6));
    };

    if num_features < 6 {
        // Unexpected shape
        return Array2::zeros((0, 6));
    }

    let (scale_y, scale_x) = preprocess.scale;
    let (pad_top, pad_left) = preprocess.padding;
    let (max_w, max_h) = (
        preprocess.orig_shape.1 as f32,
        preprocess.orig_shape.0 as f32,
    );

    let mut valid_predictions = Vec::new();

    for i in 0..num_preds {
        let base = i * num_features;

        let score = output[base + 4];
        if score < config.confidence_threshold {
            continue;
        }

        let class_id = output[base + 5] as usize;
        if !config.keep_class(class_id) {
            continue;
        }

        let x1_raw = output[base];
        let y1_raw = output[base + 1];
        let x2_raw = output[base + 2];
        let y2_raw = output[base + 3];

        // Scale coordinates back to original image
        let x1 = (x1_raw - pad_left) / scale_x;
        let y1 = (y1_raw - pad_top) / scale_y;
        let x2 = (x2_raw - pad_left) / scale_x;
        let y2 = (y2_raw - pad_top) / scale_y;

        // Clip to image bounds
        let x1 = x1.clamp(0.0, max_w);
        let y1 = y1.clamp(0.0, max_h);
        let x2 = x2.clamp(0.0, max_w);
        let y2 = y2.clamp(0.0, max_h);

        valid_predictions.push([x1, y1, x2, y2, score, output[base + 5]]);
    }

    // Sort by confidence descending (optional but good for consistency)
    valid_predictions.sort_by(|a, b| b[4].partial_cmp(&a[4]).unwrap_or(std::cmp::Ordering::Equal));

    // Limit to max_det
    let count = valid_predictions.len().min(config.max_det);

    let mut result = Array2::zeros((count, 6));
    for (i, pred) in valid_predictions.iter().take(count).enumerate() {
        result[[i, 0]] = pred[0];
        result[[i, 1]] = pred[1];
        result[[i, 2]] = pred[2];
        result[[i, 3]] = pred[3];
        result[[i, 4]] = pred[4];
        result[[i, 5]] = pred[5];
    }

    result
}

/// Optimized detection extraction with SIMD acceleration.
///
/// Key optimizations:
/// - SIMD-accelerated candidate extraction (f32x8)
/// - Parallel Bitmask NMS (IoU 1 vs 8)
/// - Struct-of-Arrays (SoA) layout for NMS cache locality
/// - Direct unsafe indexing for performance
#[allow(clippy::cast_precision_loss, clippy::too_many_arguments)]
fn extract_detect_boxes(
    output: &[f32],
    num_classes: usize,
    num_predictions: usize,
    is_transposed: bool,
    preprocess: &PreprocessResult,
    config: &InferenceConfig,
) -> Array2<f32> {
    let feat_count = 4 + num_classes;
    let (scale_y, scale_x) = preprocess.scale;
    let (pad_top, pad_left) = preprocess.padding;
    let orig_shape = preprocess.orig_shape;
    let (max_w, max_h) = (orig_shape.1 as f32, orig_shape.0 as f32);
    let conf_thresh = config.confidence_threshold;
    let max_det = config.max_det;
    let iou_thresh = config.iou_threshold;
    let conf_v = f32x8::splat(conf_thresh);

    let mut candidates: Vec<Candidate> = Vec::with_capacity(256);

    // Candidate Extraction
    if !is_transposed {
        // Layout [feat, pred] - Cache-friendly linear scan
        let mut max_scores = vec![conf_thresh; num_predictions];
        let mut max_classes = vec![0usize; num_predictions];

        for c in 0..num_classes {
            let offset = (4 + c) * num_predictions;
            let class_scores = &output[offset..offset + num_predictions];
            for (idx, &score) in class_scores.iter().enumerate() {
                if score > max_scores[idx] {
                    max_scores[idx] = score;
                    max_classes[idx] = c;
                }
            }
        }

        for (idx, &score) in max_scores.iter().enumerate() {
            if score > conf_thresh {
                let best_class = max_classes[idx];

                // Filter by class if specified
                if !config.keep_class(best_class) {
                    continue;
                }

                let cx = unsafe { *output.get_unchecked(idx) };
                let cy = unsafe { *output.get_unchecked(num_predictions + idx) };
                let w = unsafe { *output.get_unchecked(2 * num_predictions + idx) };
                let h = unsafe { *output.get_unchecked(3 * num_predictions + idx) };

                let x1 = (cx - w * 0.5 - pad_left) / scale_x;
                let y1 = (cy - h * 0.5 - pad_top) / scale_y;
                let x2 = (cx + w * 0.5 - pad_left) / scale_x;
                let y2 = (cy + h * 0.5 - pad_top) / scale_y;

                candidates.push(Candidate {
                    bbox: [x1, y1, x2, y2],
                    score,
                    class: best_class,
                });
            }
        }
    } else {
        // Layout [pred, feat] - Process 8 classes at once
        for idx in 0..num_predictions {
            let base = idx * feat_count;
            let row_ptr = unsafe { output.as_ptr().add(base + 4) };
            let mut best_score = conf_thresh;
            let mut best_class = 0;
            let mut found = false;

            for c_idx in (0..num_classes).step_by(8) {
                if num_classes - c_idx >= 8 {
                    let scores: f32x8 =
                        unsafe { (row_ptr.add(c_idx) as *const f32x8).read_unaligned() };
                    if scores.simd_gt(conf_v).any() {
                        for i in 0..8 {
                            let s = unsafe { *row_ptr.add(c_idx + i) };
                            if s > best_score {
                                best_score = s;
                                best_class = c_idx + i;
                                found = true;
                            }
                        }
                    }
                } else {
                    for i in c_idx..num_classes {
                        let s = unsafe { *row_ptr.add(i) };
                        if s > best_score {
                            best_score = s;
                            best_class = i;
                            found = true;
                        }
                    }
                }
            }

            if found {
                // Filter by class if specified
                if !config.keep_class(best_class) {
                    continue;
                }

                let cx = unsafe { *output.get_unchecked(base) };
                let cy = unsafe { *output.get_unchecked(base + 1) };
                let w = unsafe { *output.get_unchecked(base + 2) };
                let h = unsafe { *output.get_unchecked(base + 3) };

                let x1 = (cx - w * 0.5 - pad_left) / scale_x;
                let y1 = (cy - h * 0.5 - pad_top) / scale_y;
                let x2 = (cx + w * 0.5 - pad_left) / scale_x;
                let y2 = (cy + h * 0.5 - pad_top) / scale_y;

                candidates.push(Candidate {
                    bbox: [x1, y1, x2, y2],
                    score: best_score,
                    class: best_class,
                });
            }
        }
    }

    if candidates.is_empty() {
        return Array2::zeros((0, 6));
    }

    // Top-K Selection & Sort
    let nms_limit = (max_det * 10).min(candidates.len());
    if candidates.len() > nms_limit {
        candidates.select_nth_unstable_by(nms_limit, |a, b| b.score.partial_cmp(&a.score).unwrap());
        candidates.truncate(nms_limit);
    }
    candidates.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    // Population of SoA for NMS (small copy, very fast)
    let n = candidates.len();
    let mut x1 = Vec::with_capacity(n);
    let mut y1 = Vec::with_capacity(n);
    let mut x2 = Vec::with_capacity(n);
    let mut y2 = Vec::with_capacity(n);
    let mut areas = Vec::with_capacity(n);

    for c in &candidates {
        x1.push(c.bbox[0]);
        y1.push(c.bbox[1]);
        x2.push(c.bbox[2]);
        y2.push(c.bbox[3]);
        areas.push((c.bbox[2] - c.bbox[0]) * (c.bbox[3] - c.bbox[1]));
    }

    let mut suppressed = vec![false; n];
    let mut keep = Vec::with_capacity(max_det);
    let iou_v = f32x8::splat(iou_thresh);
    // Build output array with kept detections
    // let num_kept = keep_indices.len().min(config.max_det);
    // let mut result = Array2::zeros((num_kept, 6));

    for i in 0..n {
        if suppressed[i] {
            continue;
        }
        keep.push(i);
        if keep.len() >= max_det {
            break;
        }

        let ax1 = f32x8::splat(x1[i]);
        let ay1 = f32x8::splat(y1[i]);
        let ax2 = f32x8::splat(x2[i]);
        let ay2 = f32x8::splat(y2[i]);
        let aa = f32x8::splat(areas[i]);
        let ac = candidates[i].class;

        let mut j = i + 1;
        while j < n {
            if n - j >= 8 {
                // Inline fast class and suppression check
                let mut chunk_needs_processing = false;
                for k in 0..8 {
                    if candidates[j + k].class == ac && !suppressed[j + k] {
                        chunk_needs_processing = true;
                        break;
                    }
                }

                if chunk_needs_processing {
                    let bx1 = unsafe { (x1.as_ptr().add(j) as *const f32x8).read_unaligned() };
                    let by1 = unsafe { (y1.as_ptr().add(j) as *const f32x8).read_unaligned() };
                    let bx2 = unsafe { (x2.as_ptr().add(j) as *const f32x8).read_unaligned() };
                    let by2 = unsafe { (y2.as_ptr().add(j) as *const f32x8).read_unaligned() };
                    let ba = unsafe { (areas.as_ptr().add(j) as *const f32x8).read_unaligned() };

                    let ix1 = ax1.max(bx1);
                    let iy1 = ay1.max(by1);
                    let ix2 = ax2.min(bx2);
                    let iy2 = ay2.min(by2);

                    let iw = (ix2 - ix1).max(f32x8::ZERO);
                    let ih = (iy2 - iy1).max(f32x8::ZERO);
                    let ia = iw * ih;
                    let iou = ia / (aa + ba - ia);

                    let mask = iou.simd_gt(iou_v).to_bitmask() as u8;
                    if mask != 0 {
                        for k in 0..8 {
                            if (mask & (1 << k)) != 0 && candidates[j + k].class == ac {
                                suppressed[j + k] = true;
                            }
                        }
                    }
                }
                j += 8;
            } else {
                for k in j..n {
                    if !suppressed[k] && candidates[k].class == ac {
                        let ix1 = x1[i].max(x1[k]);
                        let iy1 = y1[i].max(y1[k]);
                        let ix2 = x2[i].min(x2[k]);
                        let iy2 = y2[i].min(y2[k]);
                        let iw = (ix2 - ix1).max(0.0);
                        let ih = (iy2 - iy1).max(0.0);
                        let ia = iw * ih;
                        let iou = ia / (areas[i] + areas[k] - ia);
                        if iou > iou_thresh {
                            suppressed[k] = true;
                        }
                    }
                }
                break;
            }
        }
    }
    // Result Construction
    let num_kept = keep.len();
    let mut result = Array2::zeros((num_kept, 6));
    for (out_idx, &idx) in keep.iter().enumerate() {
        let c = &candidates[idx];
        result[[out_idx, 0]] = c.bbox[0].clamp(0.0, max_w);
        result[[out_idx, 1]] = c.bbox[1].clamp(0.0, max_h);
        result[[out_idx, 2]] = c.bbox[2].clamp(0.0, max_w);
        result[[out_idx, 3]] = c.bbox[3].clamp(0.0, max_h);
        result[[out_idx, 4]] = c.score;
        result[[out_idx, 5]] = c.class as f32;
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
    clippy::cast_possible_truncation,
    clippy::suboptimal_flops
)]
fn postprocess_segment(
    outputs: Vec<(&[f32], Vec<usize>)>,
    preprocess: &PreprocessResult,
    config: &InferenceConfig,
    names: &HashMap<usize, String>,
    orig_img: Array3<u8>,
    path: String,
    speed: Speed,
    inference_shape: (u32, u32),
    end2end: bool,
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

    // Standard segmentation models use 32 mask prototypes
    let num_masks = 32;

    // 1. Process Detections
    let (boxes_data, mask_coeffs): (Array2<f32>, Array2<f32>) = if end2end {

        // Parse similarly to extract_e2e_boxes but keep mask coeffs
        let (num_preds, num_features) = if shape0.len() == 3 {
            (shape0[1], shape0[2])
        } else {
            (shape0[0], shape0[1])
        };

        if num_features < 6 + 32 {
            // minimal check
            // Fallback
            return results;
        }

        // Manual parsing loop
        let mut candidates = Vec::new();
        let (scale_y, scale_x) = preprocess.scale;
        let (pad_top, pad_left) = preprocess.padding;
        let (max_w, max_h) = (
            preprocess.orig_shape.1 as f32,
            preprocess.orig_shape.0 as f32,
        );

        for i in 0..num_preds {
            let base = i * num_features;

            let score = output0[base + 4];
            if score < config.confidence_threshold {
                continue;
            }

            let class_id = output0[base + 5] as usize;
            if !config.keep_class(class_id) {
                continue;
            }

            // Box
            let cx: f32 = output0[base];
            let cy: f32 = output0[base + 1];
            let w: f32 = output0[base + 2];
            let h: f32 = output0[base + 3];

            let x1: f32 = (cx - w * 0.5 - pad_left) / scale_x;
            let y1: f32 = (cy - h * 0.5 - pad_top) / scale_y;
            let x2: f32 = (cx + w * 0.5 - pad_left) / scale_x;
            let y2: f32 = (cy + h * 0.5 - pad_top) / scale_y;

            let x1: f32 = x1.clamp(0.0, max_w);
            let y1: f32 = y1.clamp(0.0, max_h);
            let x2: f32 = x2.clamp(0.0, max_w);
            let y2: f32 = y2.clamp(0.0, max_h);

            // Mask coeffs
            let mut coeffs = Vec::with_capacity(num_masks);
            for m in 0..num_masks {
                coeffs.push(output0[base + 6 + m]);
            }

            candidates.push(([x1, y1, x2, y2], score, class_id as f32, coeffs));
        }

        // Sort
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let count = candidates.len().min(config.max_det);

        let mut b_data = Array2::zeros((count, 6));
        let mut m_coeffs = Array2::zeros((count, num_masks));

        for (i, (bbox, score, class, coeffs)) in candidates.iter().take(count).enumerate() {
            b_data[[i, 0]] = bbox[0];
            b_data[[i, 1]] = bbox[1];
            b_data[[i, 2]] = bbox[2];
            b_data[[i, 3]] = bbox[3];
            b_data[[i, 4]] = *score;
            b_data[[i, 5]] = *class;
            for (m, &c) in coeffs.iter().enumerate() {
                m_coeffs[[i, m]] = c;
            }
        }
        (b_data, m_coeffs)
    } else {
        // Standard NMS path
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
            Array2::from_shape_vec((num_preds, expected_features), output0.to_vec())
                .unwrap_or_else(|_| Array2::zeros((0, 0)))
        } else {
            let arr = Array2::from_shape_vec((expected_features, num_preds), output0.to_vec())
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
                .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
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

            // Filter by class if specified
            if !config.keep_class(best_class) {
                continue;
            }

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
        (boxes_data, mask_coeffs)
    };

    if boxes_data.is_empty() {
        return results;
    }

    let num_kept = boxes_data.nrows();
    results.boxes = Some(Boxes::new(boxes_data.clone(), preprocess.orig_shape));

    // 3. Process Masks
    // ...

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

    let protos = match Array2::from_shape_vec((num_masks, mh * mw), output1.to_vec()) {
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
        .par_for_each(
            |mut mask_out: ArrayViewMut2<f32>,
             mask_flat: ArrayView1<f32>,
             box_data: ArrayView1<f32>| {
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
            },
        );

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
    end2end: bool,
) -> Results {
    let mut results = Results::new(orig_img, path, names.clone(), speed, inference_shape);

    // Standard COCO pose has 17 keypoints, each with (x, y, conf)
    let num_keypoints = 17;
    let kpt_dim = 3; // x, y, visibility/confidence
    let kpt_features = num_keypoints * kpt_dim; // 51

    // Pose typically has 1 class (person), so features = 4 + 1 + 51 = 56
    let num_classes = names.len().max(1);
    let expected_features = 4 + num_classes + kpt_features;

    // Detect E2E: Output shape [Batch, N, 6 + kpt_features]
    // 6 = 4 box + 1 conf + 1 cls
    let e2e_features = 6 + kpt_features;
    let is_e2e_shape = output_shape.last().is_some_and(|&d| d == e2e_features);
    let use_e2e = end2end || is_e2e_shape;

    let mut candidates: Vec<([f32; 4], f32, usize, Vec<[f32; 3]>)> = Vec::new();

    if use_e2e {
        // E2E Parsing: [Batch, N, Features]
        // Flatten logic similar to extract_e2e_boxes but with keypoints
        let (num_preds, num_feats) = if output_shape.len() == 3 {
            (output_shape[1], output_shape[2])
        } else {
            // Fallback
            (output.len() / e2e_features, e2e_features)
        };

        if num_feats < e2e_features {
            return results;
        }

        for i in 0..num_preds {
            let base = i * num_feats;
            let score = output[base + 4];
            if score < config.confidence_threshold {
                continue;
            }
            let class_id = output[base + 5] as usize;
            if !config.keep_class(class_id) {
                continue;
            }

            // Box
            let x1 = output[base];
            let y1 = output[base + 1];
            let x2 = output[base + 2];
            let y2 = output[base + 3];

            // Scale Box
            let scaled = scale_coords(&[x1, y1, x2, y2], preprocess.scale, preprocess.padding);
            let clipped = clip_coords(&scaled, preprocess.orig_shape);

            // Keypoints
            let mut keypoints = Vec::with_capacity(num_keypoints);
            for k in 0..num_keypoints {
                let k_base = base + 6 + k * kpt_dim;
                let kx = output[k_base];
                let ky = output[k_base + 1];
                let kc = output[k_base + 2];

                let scaled_kpt =
                    scale_coords(&[kx, ky, kx, ky], preprocess.scale, preprocess.padding);
                let (oh, ow) = preprocess.orig_shape;
                keypoints.push([
                    scaled_kpt[0].clamp(0.0, ow as f32),
                    scaled_kpt[1].clamp(0.0, oh as f32),
                    kc,
                ]);
            }

            candidates.push((
                [clipped[0], clipped[1], clipped[2], clipped[3]],
                score,
                class_id,
                keypoints,
            ));
        }
    } else {
        // Standard Auto-Backend Parsing
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

            // Filter by class if specified
            if !config.keep_class(best_class) {
                continue;
            }

            candidates.push((
                [clipped[0], clipped[1], clipped[2], clipped[3]],
                best_score,
                best_class,
                keypoints,
            ));
        }
    }

    if candidates.is_empty() {
        results.keypoints = Some(Keypoints::new(
            Array3::zeros((0, num_keypoints, kpt_dim)),
            preprocess.orig_shape,
        ));
        return results;
    }

    // Apply NMS only if NOT E2E (E2E models are already filtered/sorted, we just take top)
    // Actually E2E mode above pushes to candidates, so we can unified the result building.
    // BUT E2E doesn't need NMS.

    let keep_indices = if use_e2e {
        // Just take top max_det (already naturally ordered usually, but sorting is safe)
        let mut indices: Vec<usize> = (0..candidates.len()).collect();
        indices.sort_by(|&a, &b| {
            candidates[b]
                .1
                .partial_cmp(&candidates[a].1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices
    } else {
        let nms_candidates: Vec<_> = candidates
            .iter()
            .map(|(bbox, score, class, _)| (*bbox, *score, *class))
            .collect();
        nms_per_class(&nms_candidates, config.iou_threshold)
    };

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

    // Probs::new expects an Array1, which we can create from the slice
    let mut probs_vec = output.to_vec();

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
    end2end: bool,
) -> Results {
    let mut results = Results::new(orig_img, path, names.clone(), speed, inference_shape);

    // OBB format: [xywh, class_scores..., rotation_angle]
    // features = 4 (bbox) + num_classes + 1 (angle)
    let num_classes = names.len().max(1);
    let expected_features = 4 + num_classes + 1;

    // Detect E2E: Output shape [Batch, N, 7]
    // 7 = 4 box + 1 angle + 1 conf + 1 cls
    let is_e2e_shape = output_shape.last().is_some_and(|&d| d == 7);
    let use_e2e = end2end || is_e2e_shape;

    let mut candidates: Vec<([f32; 5], f32, usize)> = Vec::new(); // [cx, cy, w, h, angle], conf, class

    if use_e2e {
        // E2E Parsing: [Batch, N, Features]
        let (num_preds, num_feats) = if output_shape.len() == 3 {
            (output_shape[1], output_shape[2])
        } else {
            (output.len() / 7, 7)
        };

        if num_feats < 7 {
            return results;
        }

        for i in 0..num_preds {
            let base = i * num_feats;
            let score = output[base + 5];
            if score < config.confidence_threshold {
                continue;
            }
            let class_id = output[base + 6] as usize;
            if !config.keep_class(class_id) {
                continue;
            }

            let cx = output[base];
            let cy = output[base + 1];
            let w = output[base + 2];
            let h = output[base + 3];
            let angle = output[base + 4];

            // Scale
            let (scale_y, scale_x) = preprocess.scale;
            let (pad_top, pad_left) = preprocess.padding;

            let cx = (cx - pad_left) / scale_x;
            let cy = (cy - pad_top) / scale_y;
            let w = w / scale_x;
            let h = h / scale_y;

            candidates.push(([cx, cy, w, h, angle], score, class_id));
        }
    } else {
        // Standard Auto-Backend Parsing
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

            // Extract OBB params
            let cx = output_2d[[i, 0]];
            let cy = output_2d[[i, 1]];
            let w = output_2d[[i, 2]];
            let h = output_2d[[i, 3]];
            // Angle is usually the last feature (or before class? No, check Ultralytics OBB head)
            // Typically: [xywh, score, ..., angle] or [xywh, angle, score?]
            // Let's assume standard behavior: output_2d has [xywh, classes..., angle]
            // Wait, previously I assumed angle was last.
            let angle = output_2d[[i, actual_features - 1]];

            // Scale
            let (scale_y, scale_x) = preprocess.scale;
            let (pad_top, pad_left) = preprocess.padding;

            let cx = (cx - pad_left) / scale_x;
            let cy = (cy - pad_top) / scale_y;
            let w = w / scale_x;
            let h = h / scale_y;

            // Filter by class if specified
            if !config.keep_class(best_class) {
                continue;
            }

            candidates.push(([cx, cy, w, h, angle], best_score, best_class));
        }
    }

    if candidates.is_empty() {
        return results;
    }

    let keep_indices = if use_e2e {
        // Just sort
        let mut indices: Vec<usize> = (0..candidates.len()).collect();
        indices.sort_by(|&a, &b| {
            candidates[b]
                .1
                .partial_cmp(&candidates[a].1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices
    } else {
        // Apply Rotated NMS for precise suppression using ProbIoU (Hellinger distance).
        nms_rotated_per_class(&candidates, config.iou_threshold)
    };

    let num_kept = keep_indices.len().min(config.max_det);
    let mut obb_data = Array2::zeros((num_kept, 7)); // [cx, cy, w, h, angle, conf, cls]

    for (out_idx, &keep_idx) in keep_indices.iter().take(num_kept).enumerate() {
        let (obb, score, class) = &candidates[keep_idx];
        obb_data[[out_idx, 0]] = obb[0];
        obb_data[[out_idx, 1]] = obb[1];
        obb_data[[out_idx, 2]] = obb[2];
        obb_data[[out_idx, 3]] = obb[3];
        obb_data[[out_idx, 4]] = obb[4];
        obb_data[[out_idx, 5]] = *score;
        obb_data[[out_idx, 6]] = *class as f32;
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
            false,
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
            false,
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
            false,
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
            false,
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
