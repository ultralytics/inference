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
///     .with_max_detections(100)
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
pub struct InferenceConfig {
    /// Confidence threshold for detections (0.0 to 1.0).
    /// Detections with confidence scores lower than this value will be discarded.
    pub confidence_threshold: f32,
    /// Intersection over Union (`IoU`) threshold for Non-Maximum Suppression (NMS) (0.0 to 1.0).
    /// Used to merge overlapping boxes. Lower values filter more duplicates.
    pub iou_threshold: f32,
    /// Maximum number of detections to return per image.
    /// The top-k detections sorted by confidence will be returned.
    pub max_detections: usize,
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
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.25,
            iou_threshold: 0.45,
            max_detections: 300,
            imgsz: None,
            batch: None,
            num_threads: 0, // 0 = let ONNX Runtime decide (typically uses all cores efficiently)
            half: false,
            device: None,
            save: true,
            save_frames: false,
            rect: true,
        }
    }
}

impl InferenceConfig {
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
    pub const fn with_max_detections(mut self, max: usize) -> Self {
        self.max_detections = max;
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
}
