// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Inference configuration and common types.
//!
//! This module defines the [`InferenceConfig`] struct, which controls various parameters
//! for YOLO model inference, such as confidence thresholds, Non-Maximum Suppression (NMS),
//! input image sizing, and hardware execution options.

/// Configuration for YOLO inference.
///
/// This struct is used to customize the behavior of the inference engine.
/// It uses a builder pattern for convenient construction.
///
/// # Examples
///
/// Basic configuration:
/// ```rust
/// use ultralytics_inference::InferenceConfig;
///
/// let config = InferenceConfig::new()
///     .with_confidence(0.5)
///     .with_iou(0.45)
///     .with_max_det(300)
///     .with_imgsz(640, 640);
/// ```
///
/// With specific hardware device:
/// ```rust
/// use ultralytics_inference::{InferenceConfig, Device};
///
/// let config = InferenceConfig::new()
///     .with_confidence(0.5)
///     .with_device(Device::Cuda(0));
/// ```
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct InferenceConfig {
    /// Confidence threshold for detections (0.0 to 1.0).
    /// Detections with confidence scores lower than this value will be discarded.
    pub confidence_threshold: f32,
    /// Intersection over Union (`IoU`) threshold for Non-Maximum Suppression (NMS) (0.0 to 1.0).
    /// Used to merge overlapping boxes. Lower values filter more duplicates.
    pub iou_threshold: f32,
    /// Maximum number of detections to return per image.
    /// The top-k detections sorted by confidence will be returned.
    pub max_det: usize,
    /// Explicit input image size (height, width).
    /// If `None`, the model's metadata will be used to determine input size.
    pub imgsz: Option<(usize, usize)>,
    /// Batch size for inference when using [`BatchProcessor`](crate::batch::BatchProcessor).
    /// If `None`, defaults to 1 (single-image inference).
    pub batch: Option<usize>,
    /// Number of intra-op threads for ONNX Runtime.
    /// Setting this to `0` allows ONNX Runtime to choose the optimal number.
    pub num_threads: usize,
    /// Whether to use FP16 (half-precision) inference.
    /// This can improve performance on compatible hardware (e.g., GPUs) but may
    /// result in slight precision loss.
    pub half: bool,
    /// Hardware device to use for inference.
    /// If `None`, the best available device will be automatically selected.
    pub device: Option<crate::Device>,
    /// Whether to save annotated results.
    /// Defaults to `true`.
    pub save: bool,
    /// Whether to save individual frames instead of a video file when input is video.
    /// Defaults to `false` (save as video).
    pub save_frames: bool,
    /// Whether to use minimal padding (rectangular inference).
    /// Defaults to `true` to match Ultralytics Python.
    pub rect: bool,
    /// Class IDs to filter predictions. If `None`, all classes are returned.
    /// Useful for focusing on specific objects in multi-class detection tasks.
    pub classes: Option<Vec<usize>>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: Self::DEFAULT_CONF,
            iou_threshold: Self::DEFAULT_IOU,
            max_det: Self::DEFAULT_MAX_DET,
            imgsz: None,
            batch: None,
            num_threads: 0, // 0 = let ONNX Runtime decide (typically uses all cores efficiently)
            half: Self::DEFAULT_HALF,
            device: None,
            save: Self::DEFAULT_SAVE,
            save_frames: Self::DEFAULT_SAVE_FRAMES,
            rect: Self::DEFAULT_RECT,
            classes: None,
        }
    }
}

impl InferenceConfig {
    /// Default confidence threshold (0.0 to 1.0).
    pub const DEFAULT_CONF: f32 = 0.25;
    /// Default `IoU` threshold for NMS (0.0 to 1.0).
    pub const DEFAULT_IOU: f32 = 0.7;
    /// Default maximum number of detections per image.
    pub const DEFAULT_MAX_DET: usize = 300;
    /// Default for FP16 half-precision inference.
    pub const DEFAULT_HALF: bool = false;
    /// Default for saving annotated results.
    pub const DEFAULT_SAVE: bool = true;
    /// Default for saving individual frames (vs video).
    pub const DEFAULT_SAVE_FRAMES: bool = false;
    /// Default for rectangular (minimal padding) inference.
    pub const DEFAULT_RECT: bool = true;

    /// Create a new configuration with default values.
    ///
    /// # Returns
    ///
    /// * A new `InferenceConfig` instance with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the batch size.
    ///
    /// # Arguments
    ///
    /// * `batch` - The batch size.
    ///
    /// # Returns
    ///
    /// * The modified `InferenceConfig`.
    #[must_use]
    pub const fn with_batch(mut self, batch: usize) -> Self {
        self.batch = Some(batch);
        self
    }

    /// Set the confidence threshold.
    ///
    /// Detections with a confidence score below this threshold will be filtered out.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The minimum confidence score (0.0 to 1.0).
    ///
    /// # Returns
    ///
    /// * The modified `InferenceConfig`.
    #[must_use]
    pub const fn with_confidence(mut self, threshold: f32) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    /// Set the `IoU` threshold for Non-Maximum Suppression (NMS).
    ///
    /// NMS suppresses overlapping bounding boxes. This threshold determines how much overlap
    /// is allowed before boxes are considered duplicates.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The `IoU` threshold (0.0 to 1.0).
    ///
    /// # Returns
    ///
    /// * The modified `InferenceConfig`.
    #[must_use]
    pub const fn with_iou(mut self, threshold: f32) -> Self {
        self.iou_threshold = threshold;
        self
    }

    /// Set the maximum number of detections to return.
    ///
    /// Only the top `max` detections (sorted by confidence) will be kept after NMS.
    ///
    /// # Arguments
    ///
    /// * `max` - The maximum number of detections.
    ///
    /// # Returns
    ///
    /// * The modified `InferenceConfig`.
    #[must_use]
    pub const fn with_max_det(mut self, max: usize) -> Self {
        self.max_det = max;
        self
    }

    /// Set the input image size.
    ///
    /// This explicitly sets the size to resize images to before inference.
    /// If not set, the model's internal metadata size will be used.
    ///
    /// # Arguments
    ///
    /// * `height` - The target image height.
    /// * `width` - The target image width.
    ///
    /// # Returns
    ///
    /// * The modified `InferenceConfig`.
    #[must_use]
    pub const fn with_imgsz(mut self, height: usize, width: usize) -> Self {
        self.imgsz = Some((height, width));
        self
    }

    /// Set the number of threads for inference.
    ///
    /// # Arguments
    ///
    /// * `threads` - The number of intra-op threads. Set to `0` for auto-configuration.
    ///
    /// # Returns
    ///
    /// * The modified `InferenceConfig`.
    #[must_use]
    pub const fn with_threads(mut self, threads: usize) -> Self {
        self.num_threads = threads;
        self
    }

    /// Enable or disable FP16 (half-precision) inference.
    ///
    /// Using FP16 can significantly speed up inference on GPUs and some CPUS,
    /// at the cost of potential minor precision loss.
    ///
    /// # Arguments
    ///
    /// * `half` - `true` to enable FP16, `false` for FP32.
    ///
    /// # Returns
    ///
    /// * The modified `InferenceConfig`.
    #[must_use]
    pub const fn with_half(mut self, half: bool) -> Self {
        self.half = half;
        self
    }

    /// Set the hardware device for inference.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to use (e.g., CPU, CUDA, MPS).
    ///
    /// # Example
    ///
    /// ```rust
    /// use ultralytics_inference::{InferenceConfig, Device};
    ///
    /// let config = InferenceConfig::new()
    ///     .with_device(Device::Mps); // Use Apple Metal Performance Shaders
    /// ```
    ///
    /// # Returns
    ///
    /// * The modified `InferenceConfig`.
    #[must_use]
    pub const fn with_device(mut self, device: crate::Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Set whether to save annotated results.
    ///
    /// # Arguments
    ///
    /// * `save` - `true` to save results, `false` to skip saving.
    ///
    /// # Returns
    ///
    /// * The modified `InferenceConfig`.
    #[must_use]
    pub const fn with_save(mut self, save: bool) -> Self {
        self.save = save;
        self
    }

    /// Set whether to save individual frames for video inputs.
    ///
    /// # Arguments
    ///
    /// * `save_frames` - `true` to save frames, `false` to save as video.
    ///
    /// # Returns
    ///
    /// * The modified `InferenceConfig`.
    #[must_use]
    pub const fn with_save_frames(mut self, save_frames: bool) -> Self {
        self.save_frames = save_frames;
        self
    }

    /// Set whether to use minimal padding (rectangular inference).
    ///
    /// # Arguments
    ///
    /// * `rect` - `true` to enable, `false` to disable.
    ///
    /// # Returns
    ///
    /// * The modified `InferenceConfig`.
    #[must_use]
    pub const fn with_rect(mut self, rect: bool) -> Self {
        self.rect = rect;
        self
    }

    /// Set the class IDs to filter predictions.
    ///
    /// Only detections belonging to the specified classes will be returned.
    ///
    /// # Arguments
    ///
    /// * `classes` - A vector of class IDs to keep.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ultralytics_inference::InferenceConfig;
    ///
    /// // Only detect persons (class 0) and cars (class 2)
    /// let config = InferenceConfig::new()
    ///     .with_classes(vec![0, 2]);
    /// ```
    ///
    /// # Returns
    ///
    /// * The modified `InferenceConfig`.
    #[must_use]
    pub fn with_classes(mut self, classes: Vec<usize>) -> Self {
        self.classes = Some(classes);
        self
    }
    /// Check if a class should be included in the results.
    ///
    /// # Arguments
    ///
    /// * `class_id` - The class index to check.
    ///
    /// # Returns
    ///
    /// * `true` if the class should be kept.
    /// * `false` if the class should be filtered out.
    #[must_use]
    pub fn keep_class(&self, class_id: usize) -> bool {
        self.classes.as_ref().is_none_or(|c| c.contains(&class_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = InferenceConfig::default();
        assert!((config.confidence_threshold - InferenceConfig::DEFAULT_CONF).abs() < f32::EPSILON);
        assert!((config.iou_threshold - InferenceConfig::DEFAULT_IOU).abs() < f32::EPSILON);
        assert_eq!(config.max_det, 300);
    }

    #[test]
    fn test_config_builder() {
        let config = InferenceConfig::new()
            .with_confidence(0.5)
            .with_iou(0.6)
            .with_max_det(300)
            .with_imgsz(640, 640)
            .with_threads(8);

        assert!((config.confidence_threshold - 0.5).abs() < f32::EPSILON);
        assert!((config.iou_threshold - 0.6).abs() < f32::EPSILON);
        assert_eq!(config.max_det, 300);
        assert_eq!(config.imgsz, Some((640, 640)));
        assert_eq!(config.num_threads, 8);
    }

    #[test]
    fn test_keep_class() {
        let config = InferenceConfig::default();
        // Default: no filtering -> keep all
        assert!(config.keep_class(0));
        assert!(config.keep_class(100));

        let config_filtered = InferenceConfig::new().with_classes(vec![1, 3]);
        // Class 1 is in list -> keep
        assert!(config_filtered.keep_class(1));
        // Class 3 is in list -> keep
        assert!(config_filtered.keep_class(3));
        // Class 0 is NOT in list -> filter out (keep = false)
        assert!(!config_filtered.keep_class(0));
        // Class 2 is NOT in list -> filter out (keep = false)
        assert!(!config_filtered.keep_class(2));
    }
}
