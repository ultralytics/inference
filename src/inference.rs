// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Inference configuration and common types.
//!
//! This module defines the [`InferenceConfig`] struct, which controls various parameters
//! for YOLO model inference, such as confidence thresholds, Non-Maximum Suppression (NMS),
//! input image sizing, and hardware execution options.

use std::fmt;
use std::str::FromStr;

pub(crate) fn handle_deprecated_precision(
    quantize: Option<Quantization>,
    half: Option<bool>,
) -> Option<Quantization> {
    if quantize.is_some() {
        return quantize;
    }
    half.map_or(quantize, |enabled| {
        crate::warn!(
            "'half' is deprecated and will be removed in the future. Use 'quantize' instead."
        );
        enabled.then_some(Quantization::Fp16)
    })
}

/// Inference precision requested through the `quantize` argument.
///
/// Values and aliases match the Ultralytics Python package: `8`/`int8`/`w8a8`,
/// `16`/`fp16`/`w16a16`, `32`/`fp32`/`w32a32`, `w8a16`, and `w8a32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Quantization {
    /// INT8 weights and activations.
    Int8,
    /// FP16 weights and activations.
    Fp16,
    /// FP32 weights and activations.
    Fp32,
    /// INT8 weights and 16-bit activations.
    W8a16,
    /// INT8 weights and FP32 activations.
    W8a32,
}

impl Quantization {
    /// Return the canonical Ultralytics `quantize` value.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Int8 => "8",
            Self::Fp16 => "16",
            Self::Fp32 => "32",
            Self::W8a16 => "w8a16",
            Self::W8a32 => "w8a32",
        }
    }
}

impl fmt::Display for Quantization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Quantization {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "8" | "int8" | "w8a8" => Ok(Self::Int8),
            "16" | "fp16" | "w16a16" => Ok(Self::Fp16),
            "32" | "fp32" | "w32a32" => Ok(Self::Fp32),
            "w8a16" => Ok(Self::W8a16),
            "w8a32" => Ok(Self::W8a32),
            _ => Err(format!(
                "'quantize={value}' is invalid. Valid 'quantize' values are 8, 16, 32, \
                 'int8', 'fp16', 'fp32', 'w8a8', 'w16a16', 'w32a32', 'w8a16', or 'w8a32'. \
                 See https://docs.ultralytics.com/modes/export#quantization-options"
            )),
        }
    }
}

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
    /// Setting this to `0` lets [`YOLOModel::load`](crate::YOLOModel::load) resolve it to
    /// [`std::thread::available_parallelism`] when it builds the session, falling back to
    /// `4` if that cannot be determined.
    pub num_threads: usize,
    /// Requested inference precision. `None` uses the model's native precision.
    pub quantize: Option<Quantization>,
    /// Legacy FP16 inference flag. Use [`Self::quantize`] instead.
    #[doc(hidden)]
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
    /// Whether to use minimal padding (rectangular inference). Defaults to `true`.
    pub rect: bool,
    /// Class IDs to filter predictions. If `None`, all classes are returned.
    /// Useful for focusing on specific objects in multi-class detection tasks.
    pub classes: Option<Vec<usize>>,
    /// Use the `CUDA` preprocess fast path when available.
    ///
    /// Defaults to `true`. The flag is only consulted when the crate was
    /// compiled with the `cuda-preprocess` feature **and** the selected
    /// device is `CUDA` or `TensorRT` (or one of those EPs is registered by
    /// default). In every other configuration the value is ignored and the
    /// standard CPU preprocess path runs.
    pub cuda_preprocess: bool,
    /// Upper bound, in bytes, on the CUDA execution provider's memory arena.
    ///
    /// `None` (default) lets the arena grow as far as the device allows. Only
    /// consulted with the `cuda` feature and a CUDA device: a `TensorRT` device
    /// ignores it, as does a limit of `0`. The cap covers the arena alone, so
    /// peak device memory stays well above it.
    pub cuda_memory_limit: Option<usize>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: Self::DEFAULT_CONF,
            iou_threshold: Self::DEFAULT_IOU,
            max_det: Self::DEFAULT_MAX_DET,
            imgsz: None,
            batch: None,
            num_threads: 0, // 0 = resolve to `available_parallelism()` when the session is built
            quantize: Self::DEFAULT_QUANTIZE,
            half: Self::DEFAULT_HALF,
            device: None,
            save: Self::DEFAULT_SAVE,
            save_frames: Self::DEFAULT_SAVE_FRAMES,
            rect: Self::DEFAULT_RECT,
            classes: None,
            cuda_preprocess: Self::DEFAULT_CUDA_PREPROCESS,
            cuda_memory_limit: None,
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
    /// Default inference precision. `None` uses the model's native precision.
    pub const DEFAULT_QUANTIZE: Option<Quantization> = None;
    /// Legacy default retained for source compatibility.
    #[doc(hidden)]
    pub const DEFAULT_HALF: bool = false;
    /// Default for saving annotated results.
    pub const DEFAULT_SAVE: bool = true;
    /// Default for saving individual frames (vs video).
    pub const DEFAULT_SAVE_FRAMES: bool = false;
    /// Default for rectangular (minimal padding) inference.
    pub const DEFAULT_RECT: bool = true;
    /// Default input image size for standard YOLO models (height, width).
    pub const DEFAULT_IMGSZ: (usize, usize) = (640, 640);
    /// Default input image size for OBB models (height, width).
    pub const DEFAULT_OBB_IMGSZ: (usize, usize) = (1024, 1024);
    /// Default for the CUDA preprocess fast path: on whenever the crate is
    /// built with the `cuda-preprocess` feature and the device permits it.
    pub const DEFAULT_CUDA_PREPROCESS: bool = true;

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
    /// * `threads` - The number of intra-op threads. Set to `0` to use every available core.
    ///
    /// # Returns
    ///
    /// * The modified `InferenceConfig`.
    #[must_use]
    pub const fn with_threads(mut self, threads: usize) -> Self {
        self.num_threads = threads;
        self
    }

    /// Set the requested inference precision.
    ///
    /// The accepted schemes match the Python package's `quantize` argument.
    /// On CPU this selects the requested precision where the execution provider
    /// supports it; an FP16 ONNX model always runs at FP32 weights there, because
    /// ONNX Runtime widens the graph while building the session.
    ///
    /// # Returns
    ///
    /// * The modified `InferenceConfig`.
    #[must_use]
    pub const fn with_quantize(mut self, quantize: Quantization) -> Self {
        self.quantize = Some(quantize);
        self
    }

    /// Set FP16 inference using the legacy precision argument.
    #[doc(hidden)]
    #[must_use]
    pub const fn with_half(mut self, half: bool) -> Self {
        self.half = half;
        self
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn normalize_precision(&mut self) {
        self.quantize = handle_deprecated_precision(self.quantize, self.half.then_some(true));
        self.half = false;
    }

    /// Enable or disable the CUDA preprocess fast path.
    ///
    /// When `true` (default), and the crate was built with the
    /// `cuda-preprocess` feature, and the selected device is CUDA or
    /// `TensorRT`, [`YOLOModel::predict_image`](crate::YOLOModel::predict_image)
    /// dispatches to a fused CUDA kernel for letterbox + normalize +
    /// HWC→CHW and feeds the result to ORT as a zero-copy device tensor.
    /// Set to `false` to force the standard CPU preprocess path even when
    /// the feature is available.
    ///
    /// In any configuration where the fast path can't run (feature off,
    /// non-CUDA device, or runtime fallback), the value is silently ignored.
    #[must_use]
    pub const fn with_cuda_preprocess(mut self, enabled: bool) -> Self {
        self.cuda_preprocess = enabled;
        self
    }

    /// Set the hardware device for inference.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to use (e.g. `Device::Cpu`, `Device::Cuda(0)`,
    ///   `Device::CoreMl`, `Device::IntelGpu`).
    ///
    /// # Example
    ///
    /// ```rust
    /// use ultralytics_inference::{Device, InferenceConfig};
    ///
    /// let config = InferenceConfig::new()
    ///     .with_device(Device::CoreMl); // CoreML on Apple Silicon
    ///
    /// // OpenVINO on Intel hardware (intel:cpu, intel:gpu, intel:npu)
    /// let intel = InferenceConfig::new()
    ///     .with_device(Device::IntelGpu);
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

    /// Cap the CUDA execution provider's memory arena at `limit` bytes.
    ///
    /// Use this to stop ONNX Runtime from reserving most of the GPU so the
    /// device can be shared with other processes. Has no effect unless the
    /// crate is built with the `cuda` feature and a CUDA device is selected.
    /// A limit the graph cannot fit in fails the load with an ONNX Runtime arena
    /// error, so leave the model room to run.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ultralytics_inference::InferenceConfig;
    ///
    /// // Limit the CUDA arena to 2 GiB.
    /// let config = InferenceConfig::new().with_cuda_memory_limit(2 * 1024 * 1024 * 1024);
    /// ```
    ///
    /// # Returns
    ///
    /// * The modified `InferenceConfig`.
    #[must_use]
    pub const fn with_cuda_memory_limit(mut self, limit: usize) -> Self {
        self.cuda_memory_limit = Some(limit);
        self
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
        assert!(config.keep_class(0));
        assert!(config.keep_class(100));

        let config_filtered = InferenceConfig::new().with_classes(vec![1, 3]);
        assert!(config_filtered.keep_class(1));
        assert!(config_filtered.keep_class(3));
        assert!(!config_filtered.keep_class(0));
        assert!(!config_filtered.keep_class(2));
    }

    #[test]
    fn test_remaining_builders() {
        let config = InferenceConfig::new()
            .with_batch(4)
            .with_quantize(Quantization::Fp16)
            .with_cuda_preprocess(false)
            .with_device(crate::Device::Cpu)
            .with_save(false)
            .with_save_frames(true)
            .with_rect(false)
            .with_cuda_memory_limit(2 * 1024 * 1024 * 1024);

        assert_eq!(config.batch, Some(4));
        assert_eq!(config.quantize, Some(Quantization::Fp16));
        assert!(!config.cuda_preprocess);
        assert_eq!(config.device, Some(crate::Device::Cpu));
        assert!(!config.save);
        assert!(config.save_frames);
        assert!(!config.rect);
        assert_eq!(config.cuda_memory_limit, Some(2 * 1024 * 1024 * 1024));
    }

    #[test]
    fn test_default_constants() {
        // Defaults applied by `default()` match the public constants.
        let c = InferenceConfig::default();
        assert_eq!(c.max_det, InferenceConfig::DEFAULT_MAX_DET);
        assert_eq!(c.save, InferenceConfig::DEFAULT_SAVE);
        assert_eq!(c.rect, InferenceConfig::DEFAULT_RECT);
        assert!(c.batch.is_none());
        assert!(c.device.is_none());
        assert!(c.classes.is_none());
        assert_eq!(c.quantize, InferenceConfig::DEFAULT_QUANTIZE);
    }

    #[test]
    fn test_quantization_aliases() {
        for (value, expected) in [
            ("8", Quantization::Int8),
            ("int8", Quantization::Int8),
            ("w8a8", Quantization::Int8),
            ("16", Quantization::Fp16),
            ("fp16", Quantization::Fp16),
            ("w16a16", Quantization::Fp16),
            ("32", Quantization::Fp32),
            ("fp32", Quantization::Fp32),
            ("w32a32", Quantization::Fp32),
            ("w8a16", Quantization::W8a16),
            ("W8A32", Quantization::W8a32),
        ] {
            assert_eq!(value.parse::<Quantization>().unwrap(), expected);
        }
        assert_eq!(Quantization::Int8.to_string(), "8");
        assert!("4".parse::<Quantization>().is_err());
    }

    #[test]
    fn test_deprecated_half_mapping() {
        let mut config = InferenceConfig::new().with_half(true);
        config.normalize_precision();
        assert_eq!(config.quantize, Some(Quantization::Fp16));
        assert!(!config.half);

        let mut config = InferenceConfig::new()
            .with_half(true)
            .with_quantize(Quantization::Fp32);
        config.normalize_precision();
        assert_eq!(config.quantize, Some(Quantization::Fp32));
        assert!(!config.half);
    }
}
