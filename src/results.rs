// Ultralytics YOLO 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Results classes for YOLO inference output.
//!
//! This module provides Ultralytics-compatible result classes that match
//! the Python API for easy migration and consistent usage patterns.

use std::collections::HashMap;

use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};

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
    #[must_use]
    pub const fn new(preprocess: f64, inference: f64, postprocess: f64) -> Self {
        Self {
            preprocess: Some(preprocess),
            inference: Some(inference),
            postprocess: Some(postprocess),
        }
    }

    /// Get total inference time.
    #[must_use]
    pub fn total(&self) -> f64 {
        self.preprocess.unwrap_or(0.0)
            + self.inference.unwrap_or(0.0)
            + self.postprocess.unwrap_or(0.0)
    }
}

/// Main results container for YOLO inference.
///
/// This class matches the Ultralytics Python `Results` class API.
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
    /// Inference timing information.
    pub speed: Speed,
    /// Class ID to name mapping.
    pub names: HashMap<usize, String>,
    /// Path to the source image/video.
    pub path: String,
}

impl Results {
    /// Create a new Results instance.
    #[must_use]
    pub fn new(
        orig_img: Array3<u8>,
        path: String,
        names: HashMap<usize, String>,
        speed: Speed,
        inference_shape: (u32, u32),
    ) -> Self {
        let shape = orig_img.shape();
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
            speed,
            names,
            path,
        }
    }

    /// Get the number of detections.
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
            return if probs.data.is_empty() { 0 } else { 1 };
        }
        if let Some(ref obb) = self.obb {
            return obb.len();
        }
        0
    }

    /// Check if there are no detections.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the original image shape (height, width).
    #[must_use]
    pub const fn orig_shape(&self) -> (u32, u32) {
        self.orig_shape
    }

    /// Generate a verbose log string describing the results.
    #[must_use]
    pub fn verbose(&self) -> String {
        if self.is_empty() {
            if self.probs.is_some() {
                return String::new();
            }
            return "(no detections), ".to_string();
        }

        if let Some(ref probs) = self.probs {
            let top5: Vec<String> = probs
                .top5()
                .iter()
                .map(|&i| {
                    let name = self.names.get(&i).map_or_else(
                        || i.to_string(),
                        std::clone::Clone::clone,
                    );
                    format!("{} {:.2}", name, probs.data[i])
                })
                .collect();
            return format!("{}, ", top5.join(", "));
        }

        if let Some(ref boxes) = self.boxes {
            let cls = boxes.cls();
            let mut counts: HashMap<usize, usize> = HashMap::new();
            for &c in cls.iter() {
                let c = c as usize;
                *counts.entry(c).or_insert(0) += 1;
            }

            let mut parts = Vec::new();
            for (class_id, count) in &counts {
                let name = self.names.get(class_id).map_or_else(
                    || class_id.to_string(),
                    std::clone::Clone::clone,
                );
                let suffix = if *count > 1 { "s" } else { "" };
                parts.push(format!("{count} {name}{suffix}"));
            }
            return format!("{}, ", parts.join(", "));
        }

        String::new()
    }

    /// Convert results to a list of dictionaries (summary format).
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
                (self.orig_shape.0 as f32, self.orig_shape.1 as f32)
            } else {
                (1.0, 1.0)
            };

            let xyxy = boxes.xyxy();
            let conf = boxes.conf();
            let cls = boxes.cls();

            for i in 0..boxes.len() {
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
    Box(HashMap<String, SummaryValue>),
}

/// Detection bounding boxes.
///
/// Stores bounding boxes in xyxy format along with confidence scores and class IDs.
/// Matches the Ultralytics Python `Boxes` class API.
#[derive(Debug, Clone)]
pub struct Boxes {
    /// Raw data array with shape (N, 6) containing [x1, y1, x2, y2, conf, cls].
    /// Or shape (N, 7) if tracking: [x1, y1, x2, y2, track_id, conf, cls].
    pub data: Array2<f32>,
    /// Original image shape (height, width) for normalization.
    pub orig_shape: (u32, u32),
    /// Whether tracking IDs are present.
    is_track: bool,
}

impl Boxes {
    /// Create a new Boxes instance.
    ///
    /// # Arguments
    ///
    /// * `data` - Array with shape (N, 6) or (N, 7) containing box data.
    /// * `orig_shape` - Original image shape (height, width).
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
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.nrows()
    }

    /// Check if there are no boxes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get boxes in xyxy format [x1, y1, x2, y2].
    #[must_use]
    pub fn xyxy(&self) -> ArrayView2<'_, f32> {
        self.data.slice(s![.., 0..4])
    }

    /// Get confidence scores.
    #[must_use]
    pub fn conf(&self) -> ArrayView1<'_, f32> {
        self.data.slice(s![.., -2])
    }

    /// Get class IDs.
    #[must_use]
    pub fn cls(&self) -> ArrayView1<'_, f32> {
        self.data.slice(s![.., -1])
    }

    /// Get tracking IDs (if available).
    #[must_use]
    pub fn id(&self) -> Option<ArrayView1<'_, f32>> {
        if self.is_track {
            Some(self.data.slice(s![.., -3]))
        } else {
            None
        }
    }

    /// Get boxes in xywh format [x_center, y_center, width, height].
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

            xywh[[i, 0]] = (x1 + x2) / 2.0; // x_center
            xywh[[i, 1]] = (y1 + y2) / 2.0; // y_center
            xywh[[i, 2]] = x2 - x1; // width
            xywh[[i, 3]] = y2 - y1; // height
        }

        xywh
    }

    /// Get boxes in xyxy format normalized by image size.
    #[must_use]
    pub fn xyxyn(&self) -> Array2<f32> {
        let mut xyxyn = self.xyxy().to_owned();
        let (h, w) = (self.orig_shape.0 as f32, self.orig_shape.1 as f32);

        for mut row in xyxyn.rows_mut() {
            row[0] /= w;
            row[1] /= h;
            row[2] /= w;
            row[3] /= h;
        }

        xyxyn
    }

    /// Get boxes in xywh format normalized by image size.
    #[must_use]
    pub fn xywhn(&self) -> Array2<f32> {
        let mut xywhn = self.xywh();
        let (h, w) = (self.orig_shape.0 as f32, self.orig_shape.1 as f32);

        for mut row in xywhn.rows_mut() {
            row[0] /= w;
            row[1] /= h;
            row[2] /= w;
            row[3] /= h;
        }

        xywhn
    }

    /// Check if tracking IDs are available.
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
    #[must_use]
    pub const fn new(data: Array3<f32>, orig_shape: (u32, u32)) -> Self {
        Self { data, orig_shape }
    }

    /// Get the number of masks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.shape()[0]
    }

    /// Check if there are no masks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    // TODO: Implement xy and xyn properties for segment coordinates
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
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.shape()[0]
    }

    /// Check if there are no keypoints.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get xy coordinates.
    #[must_use]
    pub fn xy(&self) -> Array3<f32> {
        self.data.slice(s![.., .., 0..2]).to_owned()
    }

    /// Get normalized xy coordinates.
    #[must_use]
    pub fn xyn(&self) -> Array3<f32> {
        let mut xyn = self.xy();
        let (h, w) = (self.orig_shape.0 as f32, self.orig_shape.1 as f32);

        for mut point in xyn.axis_iter_mut(Axis(2)) {
            if point.shape()[0] > 0 {
                point.mapv_inplace(|v| v / w);
            }
            if point.shape()[0] > 1 {
                point.mapv_inplace(|v| v / h);
            }
        }

        xyn
    }

    /// Get confidence values (if available).
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
    /// Raw probability data with shape (num_classes,).
    pub data: Array1<f32>,
}

impl Probs {
    /// Create a new Probs instance.
    #[must_use]
    pub const fn new(data: Array1<f32>) -> Self {
        Self { data }
    }

    /// Get the index of the top-1 class.
    #[must_use]
    pub fn top1(&self) -> usize {
        self.data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Get the indices of the top-5 classes.
    #[must_use]
    pub fn top5(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.data.len()).collect();
        indices.sort_by(|&a, &b| {
            self.data[b]
                .partial_cmp(&self.data[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices.truncate(5);
        indices
    }

    /// Get the confidence of the top-1 class.
    #[must_use]
    pub fn top1conf(&self) -> f32 {
        self.data[self.top1()]
    }

    /// Get the confidences of the top-5 classes.
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
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.nrows()
    }

    /// Check if there are no OBBs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get boxes in xywhr format [x_center, y_center, width, height, rotation].
    #[must_use]
    pub fn xywhr(&self) -> ArrayView2<'_, f32> {
        self.data.slice(s![.., 0..5])
    }

    /// Get confidence scores.
    #[must_use]
    pub fn conf(&self) -> ArrayView1<'_, f32> {
        self.data.slice(s![.., -2])
    }

    /// Get class IDs.
    #[must_use]
    pub fn cls(&self) -> ArrayView1<'_, f32> {
        self.data.slice(s![.., -1])
    }

    /// Get tracking IDs (if available).
    #[must_use]
    pub fn id(&self) -> Option<ArrayView1<'_, f32>> {
        if self.is_track {
            Some(self.data.slice(s![.., -3]))
        } else {
            None
        }
    }

    // TODO: Implement xyxyxyxy and xyxy conversion methods
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
}
