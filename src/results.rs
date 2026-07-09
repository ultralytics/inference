// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Results classes for YOLO inference output.
//!
//! This module provides Ultralytics-compatible result classes with an
//! API for easy migration and consistent usage patterns.

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis, s};

use crate::utils::pluralize;

/// Timing information for inference operations (in milliseconds).
#[derive(Debug, Clone, Default)]
pub struct Speed {
    /// Time spent on preprocessing.
    pub preprocess: Option<f64>,
    /// Time spent on model inference.
    pub inference: Option<f64>,
    /// Time spent on postprocessing.
    pub postprocess: Option<f64>,
}

impl Speed {
    /// Create a new Speed instance with all timings.
    ///
    /// # Arguments
    ///
    /// * `preprocess` - Time in milliseconds.
    /// * `inference` - Time in milliseconds.
    /// * `postprocess` - Time in milliseconds.
    ///
    /// # Returns
    ///
    /// * A new `Speed` instance.
    #[must_use]
    pub const fn new(preprocess: f64, inference: f64, postprocess: f64) -> Self {
        Self {
            preprocess: Some(preprocess),
            inference: Some(inference),
            postprocess: Some(postprocess),
        }
    }

    /// Get total inference time.
    ///
    /// # Returns
    ///
    /// * Sum of preprocess, inference, and postprocess times in milliseconds.
    #[must_use]
    pub fn total(&self) -> f64 {
        self.preprocess.unwrap_or(0.0)
            + self.inference.unwrap_or(0.0)
            + self.postprocess.unwrap_or(0.0)
    }
}

/// Per-pixel class label map for semantic segmentation.
#[derive(Debug, Clone)]
pub struct SemanticMask {
    /// Class index for each pixel, shape [H, W].
    pub data: Array2<u16>,
    /// Original image shape (height, width).
    pub orig_shape: (u32, u32),
}

impl SemanticMask {
    /// Sentinel class index for "ignore" pixels, e.g. classes filtered out by
    /// [`InferenceConfig::with_classes`](crate::InferenceConfig::with_classes).
    /// Renderers skip it and [`class_ids`](Self::class_ids) excludes it.
    pub const IGNORE: u16 = u16::MAX;

    /// Create a new `SemanticMask`.
    #[must_use]
    pub const fn new(data: Array2<u16>, orig_shape: (u32, u32)) -> Self {
        Self { data, orig_shape }
    }

    /// Get the original image shape (height, width).
    #[must_use]
    pub const fn orig_shape(&self) -> (u32, u32) {
        self.orig_shape
    }

    /// Count unique class IDs present in the mask.
    #[must_use]
    pub fn classes_present(&self) -> usize {
        self.class_ids().len()
    }

    /// Return sorted unique class IDs present in the mask (excluding
    /// [`IGNORE`](Self::IGNORE)).
    #[must_use]
    pub fn class_ids(&self) -> Vec<usize> {
        let mut seen = vec![false; usize::from(u16::MAX) + 1];
        for &v in &self.data {
            seen[usize::from(v)] = true;
        }
        seen[usize::from(Self::IGNORE)] = false;
        seen.iter()
            .enumerate()
            .filter_map(|(i, &present)| if present { Some(i) } else { None })
            .collect()
    }
}

/// Per-pixel depth map (in meters) for monocular depth estimation.
#[derive(Debug, Clone)]
pub struct DepthMap {
    /// Depth value in meters for each pixel, shape [H, W].
    pub data: Array2<f32>,
    /// Original image shape (height, width).
    pub orig_shape: (u32, u32),
}

impl DepthMap {
    /// Create a new `DepthMap`.
    #[must_use]
    pub const fn new(data: Array2<f32>, orig_shape: (u32, u32)) -> Self {
        Self { data, orig_shape }
    }

    /// Get the original image shape (height, width).
    #[must_use]
    pub const fn orig_shape(&self) -> (u32, u32) {
        self.orig_shape
    }

    /// Minimum depth in meters over valid (`> 0`) pixels, or `None` if there are none.
    #[must_use]
    pub fn min_depth(&self) -> Option<f32> {
        self.data
            .iter()
            .copied()
            .filter(|&v| v > 0.0)
            .reduce(f32::min)
    }

    /// Maximum depth in meters over valid (`> 0`) pixels, or `None` if there are none.
    #[must_use]
    pub fn max_depth(&self) -> Option<f32> {
        self.data
            .iter()
            .copied()
            .filter(|&v| v > 0.0)
            .reduce(f32::max)
    }
}

/// Main results container for YOLO inference.
///
/// Contains the original image, detection results (boxes, masks, keypoints, etc.), timing information, and metadata.
#[derive(Debug, Clone)]
pub struct Results {
    /// Original image as HWC array (height, width, channels).
    pub orig_img: Array3<u8>,
    /// Original image shape (height, width).
    pub orig_shape: (u32, u32),
    /// Inference tensor shape (height, width) after letterboxing.
    pub inference_shape: (u32, u32),
    /// Detection bounding boxes (if applicable).
    pub boxes: Option<Boxes>,
    /// Segmentation masks (if applicable).
    pub masks: Option<Masks>,
    /// Pose keypoints (if applicable).
    pub keypoints: Option<Keypoints>,
    /// Classification probabilities (if applicable).
    pub probs: Option<Probs>,
    /// Oriented bounding boxes (if applicable).
    pub obb: Option<Obb>,
    /// Semantic segmentation class map (if applicable).
    pub semantic_mask: Option<SemanticMask>,
    /// Per-pixel depth map in meters (if applicable).
    pub depth: Option<DepthMap>,
    /// Inference timing information.
    pub speed: Speed,
    /// Class ID to name mapping.
    pub names: Arc<HashMap<usize, String>>,
    /// Path to the source image/video.
    pub path: String,
}

fn format_class_counts(
    cls: &ArrayView1<'_, f32>,
    count: usize,
    names: &HashMap<usize, String>,
) -> String {
    if count == 0 {
        return String::new();
    }
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for i in 0..count {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let class_id = cls[i] as usize;
        *counts.entry(class_id).or_insert(0) += 1;
    }
    let mut sorted: Vec<(usize, usize)> = counts.into_iter().collect();
    sorted.sort_by_key(|(id, _)| *id);
    sorted
        .iter()
        .map(|(id, n)| {
            let name = names.get(id).map_or("object", String::as_str);
            let label = if *n > 1 {
                pluralize(name)
            } else {
                name.to_string()
            };
            format!("{n} {label}")
        })
        .collect::<Vec<_>>()
        .join(", ")
}

impl Results {
    /// Create a new Results instance.
    ///
    /// # Arguments
    ///
    /// * `orig_img` - Original image as HWC array.
    /// * `path` - Path to the source image/video.
    /// * `names` - Map of class IDs to class names.
    /// * `speed` - Timing information.
    /// * `inference_shape` - Shape of the inference tensor (height, width).
    ///
    /// # Returns
    ///
    /// * A new `Results` instance.
    #[must_use]
    pub fn new(
        orig_img: Array3<u8>,
        path: String,
        names: Arc<HashMap<usize, String>>,
        speed: Speed,
        inference_shape: (u32, u32),
    ) -> Self {
        let shape = orig_img.shape();
        #[allow(clippy::cast_possible_truncation)]
        let orig_shape = (shape[0] as u32, shape[1] as u32);

        Self {
            orig_img,
            orig_shape,
            inference_shape,
            boxes: None,
            masks: None,
            keypoints: None,
            probs: None,
            obb: None,
            semantic_mask: None,
            depth: None,
            speed,
            names,
            path,
        }
    }

    /// Get the number of detections.
    ///
    /// # Returns
    ///
    /// * The count of detected objects, keypoints, or instance masks.
    ///   Semantic segmentation masks are per-pixel maps, not detections, so they return 0.
    #[must_use]
    pub fn len(&self) -> usize {
        if let Some(ref boxes) = self.boxes {
            return boxes.len();
        }
        if let Some(ref masks) = self.masks {
            return masks.len();
        }
        if let Some(ref keypoints) = self.keypoints {
            return keypoints.len();
        }
        if let Some(ref probs) = self.probs {
            return usize::from(!probs.data.is_empty());
        }
        if let Some(ref obb) = self.obb {
            return obb.len();
        }
        0
    }

    /// Check if there are no detections.
    ///
    /// # Returns
    ///
    /// * `true` if no object-like results were detected.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the original image shape (height, width).
    ///
    /// # Returns
    ///
    /// * Tuple of (height, width).
    #[must_use]
    pub const fn orig_shape(&self) -> (u32, u32) {
        self.orig_shape
    }

    /// Get the inference tensor shape (height, width) after letterboxing.
    ///
    /// # Returns
    ///
    /// * Tuple of (height, width).
    #[must_use]
    pub const fn inference_shape(&self) -> (u32, u32) {
        self.inference_shape
    }

    /// One-line detection summary suitable for per-image verbose output.
    ///
    /// Formats as "4 persons, 1 bus" (no trailing punctuation).
    /// Returns "(no detections)" when nothing was found.
    #[must_use]
    pub fn detection_summary(&self) -> String {
        if let Some(ref sm) = self.semantic_mask {
            let ids = sm.class_ids();
            if ids.is_empty() {
                return "(no detections)".to_string();
            }
            let shown: Vec<&str> = ids
                .iter()
                .map(|id| self.names.get(id).map_or("unknown", String::as_str))
                .collect();
            return shown.join(", ");
        }

        if let Some(ref depth) = self.depth {
            return match (depth.min_depth(), depth.max_depth()) {
                (Some(lo), Some(hi)) => format!("depth {lo:.2}-{hi:.2}m"),
                _ => "depth (no valid pixels)".to_string(),
            };
        }

        #[allow(clippy::option_if_let_else)]
        let summary = if let Some(ref boxes) = self.boxes {
            format_class_counts(&boxes.cls(), boxes.len(), &self.names)
        } else if let Some(ref obb) = self.obb {
            format_class_counts(&obb.cls(), obb.len(), &self.names)
        } else if let Some(ref probs) = self.probs {
            probs
                .top5()
                .iter()
                .map(|&i| {
                    let name = self.names.get(&i).map_or("unknown", String::as_str);
                    format!("{name} {:.2}", probs.data[[i]])
                })
                .collect::<Vec<_>>()
                .join(", ")
        } else {
            String::new()
        };

        if summary.is_empty() {
            "(no detections)".to_string()
        } else {
            summary
        }
    }

    /// Convert results to a list of dictionaries (summary format).
    ///
    /// # Arguments
    ///
    /// * `normalize` - Whether to normalize coordinates to [0, 1] range.
    ///
    /// # Returns
    ///
    /// * A vector of hashmaps representing the detections.
    #[must_use]
    pub fn summary(&self, normalize: bool) -> Vec<HashMap<String, SummaryValue>> {
        let mut results = Vec::new();

        if let Some(ref probs) = self.probs {
            let class_id = probs.top1();
            let mut entry = HashMap::new();
            entry.insert(
                "name".to_string(),
                SummaryValue::String(
                    self.names
                        .get(&class_id)
                        .cloned()
                        .unwrap_or_else(|| class_id.to_string()),
                ),
            );
            entry.insert("class".to_string(), SummaryValue::Int(class_id));
            entry.insert(
                "confidence".to_string(),
                SummaryValue::Float(probs.top1conf()),
            );
            results.push(entry);
            return results;
        }

        if let Some(ref boxes) = self.boxes {
            let (h, w) = if normalize {
                #[allow(clippy::cast_precision_loss)]
                (self.orig_shape.0 as f32, self.orig_shape.1 as f32)
            } else {
                (1.0, 1.0)
            };

            let xyxy = boxes.xyxy();
            let conf = boxes.conf();
            let cls = boxes.cls();

            for i in 0..boxes.len() {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let class_id = cls[i] as usize;
                let mut entry = HashMap::new();
                entry.insert(
                    "name".to_string(),
                    SummaryValue::String(
                        self.names
                            .get(&class_id)
                            .cloned()
                            .unwrap_or_else(|| class_id.to_string()),
                    ),
                );
                entry.insert("class".to_string(), SummaryValue::Int(class_id));
                entry.insert("confidence".to_string(), SummaryValue::Float(conf[i]));

                let mut box_coords = HashMap::new();
                box_coords.insert("x1".to_string(), SummaryValue::Float(xyxy[[i, 0]] / w));
                box_coords.insert("y1".to_string(), SummaryValue::Float(xyxy[[i, 1]] / h));
                box_coords.insert("x2".to_string(), SummaryValue::Float(xyxy[[i, 2]] / w));
                box_coords.insert("y2".to_string(), SummaryValue::Float(xyxy[[i, 3]] / h));
                entry.insert("box".to_string(), SummaryValue::Box(box_coords));

                results.push(entry);
            }
        }

        results
    }

    /// Save the annotated result to a file.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to save the image to.
    ///
    /// # Errors
    ///
    /// Returns an error if the image cannot be saved or if the format is unsupported.
    #[cfg(feature = "annotate")]
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> crate::error::Result<()> {
        let img = crate::utils::array_to_image(&self.orig_img)?;
        let annotated = crate::annotate::annotate_image(&img, self, None);
        annotated
            .save(path)
            .map_err(|e| crate::error::InferenceError::ImageError(e.to_string()))
    }
}

/// Values that can appear in a summary dictionary.
#[derive(Debug, Clone)]
pub enum SummaryValue {
    /// String value.
    String(String),
    /// Integer value.
    Int(usize),
    /// Float value.
    Float(f32),
    /// Box coordinates.
    Box(HashMap<String, Self>),
}

/// Detection bounding boxes.
///
/// Stores bounding boxes in xyxy format along with confidence scores and class IDs.
#[derive(Debug, Clone)]
pub struct Boxes {
    /// Raw data array with shape (N, 6) containing [x1, y1, x2, y2, conf, cls].
    /// Or shape (N, 7) if tracking: [x1, y1, x2, y2, `track_id`, conf, cls].
    pub data: Array2<f32>,
    /// Original image shape (height, width) for normalization.
    pub orig_shape: (u32, u32),
    /// Whether tracking IDs are present.
    is_track: bool,
}

/// Normalize box rows in-place by image size: columns 0 and 2 are divided by the
/// width and columns 1 and 3 by the height. Shared by the xyxy and xywh paths,
/// both of which store x-like values in columns 0/2 and y-like values in 1/3.
#[allow(clippy::cast_precision_loss)]
fn normalize_box_rows(boxes: &mut Array2<f32>, orig_shape: (u32, u32)) {
    let (h, w) = (orig_shape.0 as f32, orig_shape.1 as f32);
    for mut row in boxes.rows_mut() {
        row[0] /= w;
        row[1] /= h;
        row[2] /= w;
        row[3] /= h;
    }
}

impl Boxes {
    /// Create a new Boxes instance.
    ///
    /// # Arguments
    ///
    /// * `data` - Array with shape (N, 6) or (N, 7) containing box data.
    /// * `orig_shape` - Original image shape (height, width).
    ///
    /// # Returns
    ///
    /// * A new `Boxes` instance.
    #[must_use]
    pub fn new(data: Array2<f32>, orig_shape: (u32, u32)) -> Self {
        let is_track = data.shape()[1] == 7;
        Self {
            data,
            orig_shape,
            is_track,
        }
    }

    /// Get the number of boxes.
    ///
    /// # Returns
    ///
    /// * The count of bounding boxes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.nrows()
    }

    /// Check if there are no boxes.
    ///
    /// # Returns
    ///
    /// * `true` if the boxes array is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get boxes in xyxy format [x1, y1, x2, y2].
    ///
    /// # Returns
    ///
    /// * A view of the box coordinates.
    #[must_use]
    pub fn xyxy(&self) -> ArrayView2<'_, f32> {
        self.data.slice(s![.., 0..4])
    }

    /// Get confidence scores.
    ///
    /// # Returns
    ///
    /// * A view of confidence scores (0.0 to 1.0).
    #[must_use]
    pub fn conf(&self) -> ArrayView1<'_, f32> {
        self.data.slice(s![.., -2])
    }

    /// Get class IDs.
    ///
    /// # Returns
    ///
    /// * A view of class IDs.
    #[must_use]
    pub fn cls(&self) -> ArrayView1<'_, f32> {
        self.data.slice(s![.., -1])
    }

    /// Get tracking IDs (if available).
    ///
    /// # Returns
    ///
    /// * `Some` view of track IDs if this is a tracking result, otherwise `None`.
    #[must_use]
    pub fn id(&self) -> Option<ArrayView1<'_, f32>> {
        if self.is_track {
            Some(self.data.slice(s![.., -3]))
        } else {
            None
        }
    }

    /// Get boxes in xywh format [`x_center`, `y_center`, width, height].
    ///
    /// # Returns
    ///
    /// * An owned array of boxes in xywh format.
    #[must_use]
    pub fn xywh(&self) -> Array2<f32> {
        let xyxy = self.xyxy();
        let n = xyxy.nrows();
        let mut xywh = Array2::zeros((n, 4));

        for i in 0..n {
            let x1 = xyxy[[i, 0]];
            let y1 = xyxy[[i, 1]];
            let x2 = xyxy[[i, 2]];
            let y2 = xyxy[[i, 3]];

            xywh[[i, 0]] = f32::midpoint(x1, x2); // x_center
            xywh[[i, 1]] = f32::midpoint(y1, y2); // y_center
            xywh[[i, 2]] = x2 - x1; // width
            xywh[[i, 3]] = y2 - y1; // height
        }

        xywh
    }

    /// Get boxes in xyxy format normalized by image size.
    ///
    /// # Returns
    ///
    /// * An owned array of normalized boxes [0.0-1.0].
    #[must_use]
    pub fn xyxyn(&self) -> Array2<f32> {
        let mut xyxyn = self.xyxy().to_owned();
        normalize_box_rows(&mut xyxyn, self.orig_shape);
        xyxyn
    }

    /// Get boxes in xywh format normalized by image size.
    ///
    /// # Returns
    ///
    /// * An owned array of normalized boxes [0.0-1.0].
    #[must_use]
    pub fn xywhn(&self) -> Array2<f32> {
        let mut xywhn = self.xywh();
        normalize_box_rows(&mut xywhn, self.orig_shape);
        xywhn
    }

    /// Check if tracking IDs are available.
    ///
    /// # Returns
    ///
    /// * `true` if the boxes contain tracking information.
    #[must_use]
    pub const fn is_track(&self) -> bool {
        self.is_track
    }
}

/// Segmentation masks.
///
/// Placeholder for future segmentation support.
#[derive(Debug, Clone)]
pub struct Masks {
    /// Raw mask data with shape (N, H, W).
    pub data: Array3<f32>,
    /// Original image shape (height, width).
    pub orig_shape: (u32, u32),
}

impl Masks {
    /// Create a new Masks instance.
    ///
    /// # Arguments
    ///
    /// * `data` - Raw mask data with shape (N, H, W).
    /// * `orig_shape` - Original image shape (height, width).
    ///
    /// # Returns
    ///
    /// * A new `Masks` instance.
    #[must_use]
    pub const fn new(data: Array3<f32>, orig_shape: (u32, u32)) -> Self {
        Self { data, orig_shape }
    }

    /// Get the number of masks.
    ///
    /// # Returns
    ///
    /// * The count of masks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.shape()[0]
    }

    /// Check if there are no masks.
    ///
    /// # Returns
    ///
    /// * `true` if the masks array is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    // Note: this type exposes the raw mask tensor only. Polygon xy/xyn
    // contour properties are not derived yet.
}

/// Pose keypoints.
///
/// Placeholder for future pose estimation support.
#[derive(Debug, Clone)]
pub struct Keypoints {
    /// Raw keypoint data with shape (N, K, 2) or (N, K, 3) if confidence included.
    pub data: Array3<f32>,
    /// Original image shape (height, width).
    pub orig_shape: (u32, u32),
    /// Whether confidence values are included.
    has_visible: bool,
}

impl Keypoints {
    /// Create a new Keypoints instance.
    ///
    /// # Arguments
    ///
    /// * `data` - Raw keypoint data.
    /// * `orig_shape` - Original image shape.
    ///
    /// # Returns
    ///
    /// * A new `Keypoints` instance.
    #[must_use]
    pub fn new(data: Array3<f32>, orig_shape: (u32, u32)) -> Self {
        let has_visible = data.shape()[2] == 3;
        Self {
            data,
            orig_shape,
            has_visible,
        }
    }

    /// Get the number of detected objects with keypoints.
    ///
    /// # Returns
    ///
    /// * The count of poses.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.shape()[0]
    }

    /// Check if there are no keypoints.
    ///
    /// # Returns
    ///
    /// * `true` if no keypoints were detected.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get xy coordinates.
    ///
    /// # Returns
    ///
    /// * An owned array of keypoint coordinates.
    #[must_use]
    pub fn xy(&self) -> Array3<f32> {
        self.data.slice(s![.., .., 0..2]).to_owned()
    }

    /// Get normalized xy coordinates.
    ///
    /// # Returns
    ///
    /// * An owned array of normalized keypoint coordinates.
    #[must_use]
    pub fn xyn(&self) -> Array3<f32> {
        let mut xyn = self.xy();
        #[allow(clippy::cast_precision_loss)]
        let (h, w) = (self.orig_shape.0 as f32, self.orig_shape.1 as f32);

        // Axis(2) has two lanes: index 0 holds x (normalize by width), index 1
        // holds y (normalize by height).
        for (axis_idx, mut lane) in xyn.axis_iter_mut(Axis(2)).enumerate() {
            let divisor = if axis_idx == 0 { w } else { h };
            lane.mapv_inplace(|v| v / divisor);
        }

        xyn
    }

    /// Get confidence values (if available).
    ///
    /// # Returns
    ///
    /// * `Some` array of confidences if available, otherwise `None`.
    #[must_use]
    pub fn conf(&self) -> Option<Array2<f32>> {
        if self.has_visible {
            Some(self.data.slice(s![.., .., 2]).to_owned())
        } else {
            None
        }
    }
}

/// Classification probabilities.
///
/// Stores class probabilities with convenience methods for top predictions.
#[derive(Debug, Clone)]
pub struct Probs {
    /// Raw probability data with shape (`num_classes`,).
    pub data: Array1<f32>,
}

impl Probs {
    /// Create a new Probs instance.
    ///
    /// # Arguments
    ///
    /// * `data` - Raw probability array.
    ///
    /// # Returns
    ///
    /// * A new `Probs` instance.
    #[must_use]
    pub const fn new(data: Array1<f32>) -> Self {
        Self { data }
    }

    /// Get the index of the top-1 class.
    ///
    /// # Returns
    ///
    /// * The class ID with the highest probability.
    ///
    /// # Panics
    ///
    /// Panics if valid comparison cannot be made (e.g. NaN) in `max_by`.
    #[must_use]
    pub fn top1(&self) -> usize {
        self.data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map_or(0, |(i, _)| i)
    }
    /// Get the indices of the top-5 classes.
    ///
    /// # Returns
    ///
    /// * A vector of the top 5 class IDs sorted by probability.
    #[must_use]
    pub fn top5(&self) -> Vec<usize> {
        self.top_k(5)
    }

    /// Get the indices of the top-k classes.
    ///
    /// # Arguments
    ///
    /// * `k` - The number of classes to return.
    ///
    /// # Returns
    ///
    /// * A vector of the top k class IDs sorted by probability.
    #[must_use]
    pub fn top_k(&self, k: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.data.len()).collect();
        indices.sort_by(|&a, &b| {
            self.data[b]
                .partial_cmp(&self.data[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices.truncate(k);
        indices
    }

    /// Get the confidence of the top-1 class.
    ///
    /// # Returns
    ///
    /// * The probability of the top class.
    #[must_use]
    pub fn top1conf(&self) -> f32 {
        self.data[self.top1()]
    }

    /// Get the confidences of the top-5 classes.
    ///
    /// # Returns
    ///
    /// * A vector of the top 5 probabilities.
    #[must_use]
    pub fn top5conf(&self) -> Vec<f32> {
        self.top5().iter().map(|&i| self.data[i]).collect()
    }
}

/// Oriented bounding boxes.
///
/// Placeholder for future OBB support.
#[derive(Debug, Clone)]
pub struct Obb {
    /// Raw OBB data with shape (N, 7) containing [x, y, w, h, rotation, conf, cls].
    /// Or shape (N, 8) if tracking.
    pub data: Array2<f32>,
    /// Original image shape (height, width).
    pub orig_shape: (u32, u32),
    /// Whether tracking IDs are present.
    is_track: bool,
}

impl Obb {
    /// Create a new Obb instance.
    ///
    /// # Arguments
    ///
    /// * `data` - Raw OBB data.
    /// * `orig_shape` - Original image shape.
    ///
    /// # Returns
    ///
    /// * A new `Obb` instance.
    #[must_use]
    pub fn new(data: Array2<f32>, orig_shape: (u32, u32)) -> Self {
        let is_track = data.shape()[1] == 8;
        Self {
            data,
            orig_shape,
            is_track,
        }
    }

    /// Get the number of OBBs.
    ///
    /// # Returns
    ///
    /// * The count of oriented bounding boxes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.nrows()
    }

    /// Check if there are no OBBs.
    ///
    /// # Returns
    ///
    /// * `true` if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get boxes in xywhr format [`x_center`, `y_center`, width, height, rotation].
    ///
    /// # Returns
    ///
    /// * A view of the box parameters.
    #[must_use]
    pub fn xywhr(&self) -> ArrayView2<'_, f32> {
        self.data.slice(s![.., 0..5])
    }

    /// Get confidence scores.
    ///
    /// # Returns
    ///
    /// * A view of confidence scores.
    #[must_use]
    pub fn conf(&self) -> ArrayView1<'_, f32> {
        self.data.slice(s![.., -2])
    }

    /// Get class IDs.
    ///
    /// # Returns
    ///
    /// * A view of class IDs.
    #[must_use]
    pub fn cls(&self) -> ArrayView1<'_, f32> {
        self.data.slice(s![.., -1])
    }

    /// Get tracking IDs (if available).
    ///
    /// # Returns
    ///
    /// * `Some` view of track IDs if available, otherwise `None`.
    #[must_use]
    pub fn id(&self) -> Option<ArrayView1<'_, f32>> {
        if self.is_track {
            Some(self.data.slice(s![.., -3]))
        } else {
            None
        }
    }

    /// Get corner points for each OBB as (N, 4, 2) array.
    /// Returns the 4 corner points of each rotated bounding box.
    ///
    /// # Returns
    ///
    /// * An owned array of shape (N, 4, 2) containing corner coordinates.
    #[must_use]
    pub fn xyxyxyxy(&self) -> Array3<f32> {
        let n = self.len();
        let mut corners = Array3::zeros((n, 4, 2));

        for i in 0..n {
            let cx = self.data[[i, 0]];
            let cy = self.data[[i, 1]];
            let w = self.data[[i, 2]];
            let h = self.data[[i, 3]];
            let angle = self.data[[i, 4]];

            // Calculate corner offsets from center
            let cos_a = angle.cos();
            let sin_a = angle.sin();

            // Half dimensions
            let hw = w / 2.0;
            let hh = h / 2.0;

            // Corner offsets relative to center (before rotation)
            let corners_rel = [
                (-hw, -hh), // top-left
                (hw, -hh),  // top-right
                (hw, hh),   // bottom-right
                (-hw, hh),  // bottom-left
            ];

            // Apply rotation and translate to absolute coordinates
            for (j, (dx, dy)) in corners_rel.iter().enumerate() {
                let rotated_x = dx * cos_a - dy * sin_a;
                let rotated_y = dx * sin_a + dy * cos_a;
                corners[[i, j, 0]] = cx + rotated_x;
                corners[[i, j, 1]] = cy + rotated_y;
            }
        }

        corners
    }

    /// Get axis-aligned bounding box containing each OBB.
    /// Returns array of shape (N, 4) with [x1, y1, x2, y2] for each OBB.
    ///
    /// # Returns
    ///
    /// * An owned array of axis-aligned bounding boxes.
    #[must_use]
    pub fn xyxy(&self) -> Array2<f32> {
        let corners = self.xyxyxyxy();
        let n = self.len();
        let mut xyxy = Array2::zeros((n, 4));

        for i in 0..n {
            let mut min_x = f32::INFINITY;
            let mut min_y = f32::INFINITY;
            let mut max_x = f32::NEG_INFINITY;
            let mut max_y = f32::NEG_INFINITY;

            for j in 0..4 {
                let x = corners[[i, j, 0]];
                let y = corners[[i, j, 1]];
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }

            // Clip to image bounds
            #[allow(clippy::cast_precision_loss)]
            let (h, w) = (self.orig_shape.0 as f32, self.orig_shape.1 as f32);
            xyxy[[i, 0]] = min_x.max(0.0).min(w);
            xyxy[[i, 1]] = min_y.max(0.0).min(h);
            xyxy[[i, 2]] = max_x.max(0.0).min(w);
            xyxy[[i, 3]] = max_y.max(0.0).min(h);
        }

        xyxy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_boxes_xyxy() {
        let data = array![[10.0, 20.0, 100.0, 200.0, 0.95, 0.0]];
        let boxes = Boxes::new(data, (480, 640));

        assert_eq!(boxes.len(), 1);
        assert!((boxes.conf()[0] - 0.95).abs() < 1e-6);
        assert!((boxes.cls()[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_boxes_xywh() {
        let data = array![[0.0, 0.0, 100.0, 100.0, 0.9, 1.0]];
        let boxes = Boxes::new(data, (640, 640));
        let xywh = boxes.xywh();

        assert!((xywh[[0, 0]] - 50.0).abs() < 1e-6); // x_center
        assert!((xywh[[0, 1]] - 50.0).abs() < 1e-6); // y_center
        assert!((xywh[[0, 2]] - 100.0).abs() < 1e-6); // width
        assert!((xywh[[0, 3]] - 100.0).abs() < 1e-6); // height
    }

    #[test]
    fn test_boxes_normalized() {
        let data = array![[0.0, 0.0, 320.0, 240.0, 0.9, 0.0]];
        let boxes = Boxes::new(data, (480, 640));
        let xyxyn = boxes.xyxyn();

        assert!((xyxyn[[0, 0]] - 0.0).abs() < 1e-6);
        assert!((xyxyn[[0, 1]] - 0.0).abs() < 1e-6);
        assert!((xyxyn[[0, 2]] - 0.5).abs() < 1e-6); // 320/640
        assert!((xyxyn[[0, 3]] - 0.5).abs() < 1e-6); // 240/480
    }

    #[test]
    fn test_probs() {
        let data = array![0.1, 0.3, 0.6];
        let probs = Probs::new(data);

        assert_eq!(probs.top1(), 2);
        assert_eq!(probs.top5(), vec![2, 1, 0]);
        assert!((probs.top1conf() - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_speed() {
        let speed = Speed::new(10.0, 20.0, 5.0);
        assert!((speed.total() - 35.0).abs() < 1e-6);
    }
    #[test]
    fn test_semantic_mask_has_no_detection_len() {
        let names = Arc::new(HashMap::from([(0, "background".to_string())]));
        let speed = Speed::default();
        let orig_img = Array3::zeros((2, 2, 3));
        let mut results = Results::new(orig_img, "test.jpg".to_string(), names, speed, (2, 2));
        results.semantic_mask = Some(SemanticMask::new(array![[0u16, 1], [1, 2]], (2, 2)));

        assert_eq!(results.len(), 0);
        assert!(results.is_empty());
        assert_eq!(results.semantic_mask.as_ref().unwrap().classes_present(), 3);
    }

    #[test]
    fn test_depth_map_summary_and_len() {
        let names = Arc::new(HashMap::new());
        let speed = Speed::default();
        let orig_img = Array3::zeros((2, 2, 3));
        let mut results = Results::new(orig_img, "test.jpg".to_string(), names, speed, (2, 2));
        // Includes an invalid (0.0) pixel that must be excluded from min/max.
        let depth = DepthMap::new(array![[0.0f32, 1.5], [3.0, 4.0]], (2, 2));
        assert!((depth.min_depth().unwrap() - 1.5).abs() < 1e-6);
        assert!((depth.max_depth().unwrap() - 4.0).abs() < 1e-6);
        results.depth = Some(depth);

        assert_eq!(results.len(), 0);
        assert!(results.is_empty());
        assert_eq!(results.detection_summary(), "depth 1.50-4.00m");
    }

    #[test]
    fn test_depth_map_no_valid_pixels() {
        let depth = DepthMap::new(array![[0.0f32, 0.0], [0.0, 0.0]], (2, 2));
        assert!(depth.min_depth().is_none());
        assert!(depth.max_depth().is_none());
    }

    #[test]
    fn test_boxes_tracking_columns() {
        // 7 columns => tracking: [x1, y1, x2, y2, track_id, conf, cls]
        let data = array![[0.0, 0.0, 10.0, 10.0, 42.0, 0.9, 3.0]];
        let boxes = Boxes::new(data, (100, 100));
        assert!(boxes.is_track());
        assert!((boxes.conf()[0] - 0.9).abs() < 1e-6);
        assert!((boxes.cls()[0] - 3.0).abs() < 1e-6);
        assert!((boxes.id().unwrap()[0] - 42.0).abs() < 1e-6);

        // 6 columns => no tracking id.
        let plain = Boxes::new(array![[0.0, 0.0, 10.0, 10.0, 0.9, 3.0]], (100, 100));
        assert!(!plain.is_track());
        assert!(plain.id().is_none());
    }

    #[test]
    fn test_boxes_xywhn() {
        let boxes = Boxes::new(array![[0.0, 0.0, 320.0, 240.0, 0.9, 0.0]], (480, 640));
        let xywhn = boxes.xywhn();
        // center (160,120) normalized -> (0.25, 0.25); size (320,240) -> (0.5, 0.5)
        assert!((xywhn[[0, 0]] - 0.25).abs() < 1e-6);
        assert!((xywhn[[0, 1]] - 0.25).abs() < 1e-6);
        assert!((xywhn[[0, 2]] - 0.5).abs() < 1e-6);
        assert!((xywhn[[0, 3]] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_masks_accessors() {
        let masks = Masks::new(Array3::zeros((3, 8, 8)), (16, 16));
        assert_eq!(masks.len(), 3);
        assert!(!masks.is_empty());
        assert_eq!(masks.orig_shape, (16, 16));
        let empty = Masks::new(Array3::zeros((0, 8, 8)), (16, 16));
        assert!(empty.is_empty());
    }

    #[test]
    fn test_keypoints_with_and_without_conf() {
        // (N, K, 3) => has visibility/confidence channel.
        let mut data = Array3::zeros((1, 2, 3));
        data[[0, 0, 0]] = 320.0;
        data[[0, 0, 1]] = 240.0;
        data[[0, 0, 2]] = 0.8;
        let kpts = Keypoints::new(data, (480, 640));
        assert_eq!(kpts.len(), 1);
        assert!(!kpts.is_empty());
        let xy = kpts.xy();
        assert_eq!(xy.shape(), [1, 2, 2]);
        assert!((xy[[0, 0, 0]] - 320.0).abs() < 1e-6);
        assert!((xy[[0, 0, 1]] - 240.0).abs() < 1e-6);
        // xyn normalizes x by width and y by height independently.
        let xyn = kpts.xyn();
        assert_eq!(xyn.shape(), [1, 2, 2]);
        assert!((xyn[[0, 0, 0]] - 0.5).abs() < 1e-6); // 320 / 640
        assert!((xyn[[0, 0, 1]] - 0.5).abs() < 1e-6); // 240 / 480
        assert!(kpts.conf().is_some());

        // (N, K, 2) => no confidence.
        let no_conf = Keypoints::new(Array3::zeros((1, 2, 2)), (480, 640));
        assert!(no_conf.conf().is_none());
    }

    #[test]
    fn test_probs_top_k_and_conf() {
        let probs = Probs::new(array![0.1, 0.7, 0.15, 0.05]);
        assert_eq!(probs.top1(), 1);
        assert_eq!(probs.top_k(2), vec![1, 2]);
        assert_eq!(probs.top5(), vec![1, 2, 0, 3]); // only 4 classes
        assert!((probs.top1conf() - 0.7).abs() < 1e-6);
        let c5 = probs.top5conf();
        assert!((c5[0] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_obb_accessors_and_geometry() {
        // [x, y, w, h, rotation, conf, cls] with zero rotation.
        let data = array![[50.0, 50.0, 20.0, 10.0, 0.0, 0.9, 1.0]];
        let obb = Obb::new(data, (100, 100));
        assert_eq!(obb.len(), 1);
        assert!(!obb.is_empty());
        assert!(!obb.is_empty());
        assert!((obb.conf()[0] - 0.9).abs() < 1e-6);
        assert!((obb.cls()[0] - 1.0).abs() < 1e-6);
        assert!(obb.id().is_none());
        assert_eq!(obb.xywhr().shape(), [1, 5]);

        // With zero rotation the axis-aligned bbox is the centered w×h rectangle.
        let xyxy = obb.xyxy();
        assert!((xyxy[[0, 0]] - 40.0).abs() < 1e-4); // 50 - 10
        assert!((xyxy[[0, 1]] - 45.0).abs() < 1e-4); // 50 - 5
        assert!((xyxy[[0, 2]] - 60.0).abs() < 1e-4); // 50 + 10
        assert!((xyxy[[0, 3]] - 55.0).abs() < 1e-4); // 50 + 5
        assert_eq!(obb.xyxyxyxy().shape(), [1, 4, 2]);

        // 8 columns => tracking id present.
        let tracked = Obb::new(
            array![[50.0, 50.0, 20.0, 10.0, 0.0, 7.0, 0.9, 1.0]],
            (100, 100),
        );
        assert!(tracked.id().is_some());
    }

    #[test]
    fn test_results_len_dispatch() {
        let names = Arc::new(HashMap::new());
        let make = || {
            Results::new(
                Array3::zeros((4, 4, 3)),
                String::new(),
                Arc::clone(&names),
                Speed::default(),
                (4, 4),
            )
        };

        let mut r = make();
        r.boxes = Some(Boxes::new(array![[0.0, 0.0, 1.0, 1.0, 0.9, 0.0]], (4, 4)));
        assert_eq!(r.len(), 1);
        assert_eq!(r.orig_shape(), (4, 4));
        assert_eq!(r.inference_shape(), (4, 4));

        let mut r = make();
        r.keypoints = Some(Keypoints::new(Array3::zeros((2, 3, 3)), (4, 4)));
        assert_eq!(r.len(), 2);

        let mut r = make();
        r.probs = Some(Probs::new(array![0.2, 0.8]));
        assert_eq!(r.len(), 1);

        let empty = make();
        assert!(empty.is_empty());
    }

    #[test]
    fn test_detection_summary_variants() {
        let names = Arc::new(HashMap::from([
            (0, "person".to_string()),
            (5, "bus".to_string()),
        ]));
        let mut r = Results::new(
            Array3::zeros((4, 4, 3)),
            String::new(),
            Arc::clone(&names),
            Speed::default(),
            (4, 4),
        );

        // No detections.
        assert_eq!(r.detection_summary(), "(no detections)");

        // Two persons + one bus -> pluralized, class-id sorted.
        r.boxes = Some(Boxes::new(
            array![
                [0.0, 0.0, 1.0, 1.0, 0.9, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.8, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.7, 5.0]
            ],
            (4, 4),
        ));
        assert_eq!(r.detection_summary(), "2 persons, 1 bus");
    }

    #[test]
    fn test_summary_boxes_and_probs() {
        let names = Arc::new(HashMap::from([(0, "person".to_string())]));
        let mut r = Results::new(
            Array3::zeros((10, 10, 3)),
            String::new(),
            Arc::clone(&names),
            Speed::default(),
            (10, 10),
        );
        r.boxes = Some(Boxes::new(array![[0.0, 0.0, 5.0, 5.0, 0.9, 0.0]], (10, 10)));

        let raw = r.summary(false);
        assert_eq!(raw.len(), 1);
        assert!(matches!(raw[0].get("name"), Some(SummaryValue::String(s)) if s == "person"));
        assert!(matches!(raw[0].get("box"), Some(SummaryValue::Box(_))));

        // Probs summary returns a single top-1 entry.
        let mut rp = Results::new(
            Array3::zeros((4, 4, 3)),
            String::new(),
            names,
            Speed::default(),
            (4, 4),
        );
        rp.probs = Some(Probs::new(array![0.1, 0.9]));
        let ps = rp.summary(true);
        assert_eq!(ps.len(), 1);
        assert!(matches!(ps[0].get("class"), Some(SummaryValue::Int(1))));
    }

    #[cfg(feature = "annotate")]
    #[test]
    fn test_results_save_writes_file() {
        let names = Arc::new(HashMap::from([(0, "person".to_string())]));
        let mut r = Results::new(
            Array3::zeros((16, 16, 3)),
            "x.jpg".to_string(),
            names,
            Speed::default(),
            (16, 16),
        );
        r.boxes = Some(Boxes::new(
            array![[1.0, 1.0, 10.0, 10.0, 0.9, 0.0]],
            (16, 16),
        ));

        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("annotated.png");
        r.save(&out).unwrap();
        assert!(out.exists());
    }
}
