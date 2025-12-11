// Ultralytics YOLO 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Inference configuration and common types.
//!
//! This module defines configuration options for YOLO inference
//! and provides legacy types for backward compatibility.

/// Configuration for YOLO inference.
///
/// This struct allows customizing various inference parameters such as
/// confidence thresholds, NMS settings, and hardware options.
///
/// # Example
///
/// ```rust
/// use inference::InferenceConfig;
///
/// let config = InferenceConfig {
///     confidence_threshold: 0.5,
///     iou_threshold: 0.45,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Confidence threshold for detections (0.0 to 1.0).
    pub confidence_threshold: f32,
    /// IoU threshold for NMS (0.0 to 1.0).
    pub iou_threshold: f32,
    /// Maximum number of detections to return.
    pub max_detections: usize,
    /// Input image size (height, width). If None, uses model metadata.
    pub imgsz: Option<(usize, usize)>,
    /// Number of CPU threads for inference.
    pub num_threads: usize,
    /// Whether to use FP16 (half precision) inference.
    pub half: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.25,
            iou_threshold: 0.45,
            max_detections: 300,
            imgsz: None,
            num_threads: 4,
            half: false,
        }
    }
}

impl InferenceConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the confidence threshold.
    #[must_use]
    pub const fn with_confidence(mut self, threshold: f32) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    /// Set the IoU threshold for NMS.
    #[must_use]
    pub const fn with_iou(mut self, threshold: f32) -> Self {
        self.iou_threshold = threshold;
        self
    }

    /// Set the maximum number of detections.
    #[must_use]
    pub const fn with_max_detections(mut self, max: usize) -> Self {
        self.max_detections = max;
        self
    }

    /// Set the input image size.
    #[must_use]
    pub const fn with_imgsz(mut self, height: usize, width: usize) -> Self {
        self.imgsz = Some((height, width));
        self
    }

    /// Set the number of threads.
    #[must_use]
    pub const fn with_threads(mut self, threads: usize) -> Self {
        self.num_threads = threads;
        self
    }

    /// Enable or disable FP16 inference.
    #[must_use]
    pub const fn with_half(mut self, half: bool) -> Self {
        self.half = half;
        self
    }
}

/// Legacy detection result type for backward compatibility.
///
/// Use [`crate::results::Results`] for new code.
#[derive(Debug, Clone)]
#[deprecated(since = "0.1.0", note = "Use Results and Boxes from results module instead")]
pub struct DetectionResult {
    /// Bounding box coordinates [x1, y1, x2, y2].
    pub bbox: [f32; 4],
    /// Confidence score.
    pub confidence: f32,
    /// Class ID.
    pub class_id: usize,
    /// Class name (if available).
    pub class_name: Option<String>,
}

#[allow(deprecated)]
impl DetectionResult {
    /// Create a new detection result.
    #[must_use]
    pub const fn new(bbox: [f32; 4], confidence: f32, class_id: usize) -> Self {
        Self {
            bbox,
            confidence,
            class_id,
            class_name: None,
        }
    }

    /// Get the center point of the bounding box.
    #[must_use]
    pub fn center(&self) -> (f32, f32) {
        let x = (self.bbox[0] + self.bbox[2]) / 2.0;
        let y = (self.bbox[1] + self.bbox[3]) / 2.0;
        (x, y)
    }

    /// Get the width and height of the bounding box.
    #[must_use]
    pub fn size(&self) -> (f32, f32) {
        let w = self.bbox[2] - self.bbox[0];
        let h = self.bbox[3] - self.bbox[1];
        (w, h)
    }

    /// Get the area of the bounding box.
    #[must_use]
    pub fn area(&self) -> f32 {
        let (w, h) = self.size();
        w * h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = InferenceConfig::default();
        assert!((config.confidence_threshold - 0.25).abs() < f32::EPSILON);
        assert!((config.iou_threshold - 0.45).abs() < f32::EPSILON);
        assert_eq!(config.max_detections, 300);
    }

    #[test]
    fn test_config_builder() {
        let config = InferenceConfig::new()
            .with_confidence(0.5)
            .with_iou(0.6)
            .with_max_detections(100)
            .with_imgsz(640, 640)
            .with_threads(8);

        assert!((config.confidence_threshold - 0.5).abs() < f32::EPSILON);
        assert!((config.iou_threshold - 0.6).abs() < f32::EPSILON);
        assert_eq!(config.max_detections, 100);
        assert_eq!(config.imgsz, Some((640, 640)));
        assert_eq!(config.num_threads, 8);
    }

    #[test]
    #[allow(deprecated)]
    fn test_legacy_detection_result() {
        let det = DetectionResult::new([10.0, 20.0, 50.0, 80.0], 0.95, 0);
        assert_eq!(det.center(), (30.0, 50.0));
        assert_eq!(det.size(), (40.0, 60.0));
        assert!((det.area() - 2400.0).abs() < f32::EPSILON);
    }
}
