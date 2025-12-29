// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! YOLO model loading and inference.
//!
//! This module provides the main `YOLOModel` struct for loading ONNX models
//! and running inference.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use half::f16;
use image::DynamicImage;
use ndarray::Array3;
use ort::session::Session;
use ort::tensor::TensorElementType;
use ort::value::TensorRef;
use ort::value::ValueType;

use crate::download::try_download_model;
use crate::error::{InferenceError, Result};
use crate::inference::InferenceConfig;
use crate::metadata::ModelMetadata;
use crate::postprocessing::postprocess;
use crate::preprocessing::{
    image_to_array, preprocess_image_center_crop, preprocess_image_with_precision,
};
use crate::results::{Results, Speed};
use crate::task::Task;

/// YOLO model for inference.
///
/// This struct wraps an ONNX Runtime session and provides methods for
/// running inference on images, videos, and other sources.
///
/// # Example
///
/// ```no_run
/// use ultralytics_inference::YOLOModel;
///
/// let mut model = YOLOModel::load("yolo11n.onnx").unwrap();
/// let results = model.predict("image.jpg").unwrap();
/// println!("Found {} detections", results.len());
/// ```
pub struct YOLOModel {
    /// ONNX Runtime session.
    session: Session,
    /// Model metadata (task, classes, etc.).
    metadata: ModelMetadata,
    /// Input tensor name.
    input_name: String,
    /// Output tensor names.
    output_names: Vec<String>,
    /// Inference configuration.
    config: InferenceConfig,
    /// Whether model has been warmed up.
    warmed_up: bool,
    /// Whether model expects FP16 input.
    fp16_input: bool,
    /// Execution provider used for inference
    execution_provider: String,
}

#[allow(
    clippy::too_many_lines,
    clippy::needless_pass_by_value,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::cast_possible_truncation
)]
impl YOLOModel {
    /// Load a YOLO model from an ONNX file.
    ///
    /// The model metadata (class names, task type, input size) is automatically
    /// extracted from the ONNX model's custom metadata properties.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the ONNX model file.
    ///
    /// # Errors
    ///
    /// Returns an error if the model file doesn't exist or can't be loaded.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ultralytics_inference::YOLOModel;
    ///
    /// let model = YOLOModel::load("yolo11n.onnx")?;
    /// # Ok::<(), ultralytics_inference::InferenceError>(())
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::load_with_config(path, InferenceConfig::default())
    }

    /// Load a YOLO model with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the ONNX model file.
    /// * `config` - Custom inference configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the model file doesn't exist or can't be loaded.
    pub fn load_with_config<P: AsRef<Path>>(path: P, config: InferenceConfig) -> Result<Self> {
        let path = path.as_ref();

        // Check if file exists, attempt auto-download if not
        let path = if path.exists() {
            std::borrow::Cow::Borrowed(path)
        } else {
            // Try to download the model if it's a known downloadable model
            std::borrow::Cow::Owned(try_download_model(path)?)
        };
        let path = path.as_ref();

        // Determine optimal thread count based on available parallelism
        let num_threads = if config.num_threads > 0 {
            config.num_threads
        } else {
            // Use all available cores for intra-op parallelism (single inference)
            std::thread::available_parallelism()
                .map(std::num::NonZero::get)
                .unwrap_or(4)
        };

        // Create ONNX Runtime session with optimizations
        let mut session_builder = Session::builder().map_err(|e| {
            InferenceError::ModelLoadError(format!("Failed to create session builder: {e}"))
        })?;

        // Register execution providers based on features and device config
        #[allow(unused_mut)]
        let mut eps: Vec<ort::execution_providers::ExecutionProviderDispatch> = Vec::new();
        let mut provider_name = "CPUExecutionProvider";

        if let Some(device) = &config.device {
            // User requested specific device
            match device {
                crate::Device::Cpu => {
                    // No specific provider needed, will fall back to CPU
                    provider_name = "CPUExecutionProvider";
                }
                #[cfg(feature = "cuda")]
                crate::Device::Cuda(i) => {
                    eps.push(
                        ort::execution_providers::CUDAExecutionProvider::default()
                            .with_device_id(*i as i32)
                            .build(),
                    );
                    provider_name = "CUDAExecutionProvider";
                }
                #[cfg(feature = "coreml")]
                crate::Device::CoreMl | crate::Device::Mps => {
                    // Map both CoreML and MPS to CoreMLExecutionProvider
                    eps.push(ort::execution_providers::CoreMLExecutionProvider::default().build());
                    provider_name = "CoreMLExecutionProvider";
                }
                #[cfg(feature = "tensorrt")]
                crate::Device::TensorRt(i) => {
                    eps.push(
                        ort::execution_providers::TensorRTExecutionProvider::default()
                            .with_device_id(*i as i32)
                            .build(),
                    );
                    provider_name = "TensorRTExecutionProvider";
                }
                #[cfg(feature = "rocm")]
                crate::Device::Rocm(i) => {
                    eps.push(
                        ort::execution_providers::ROCmExecutionProvider::default()
                            .with_device_id(*i as i32)
                            .build(),
                    );
                    provider_name = "ROCmExecutionProvider";
                }
                #[cfg(feature = "directml")]
                crate::Device::DirectMl(i) => {
                    eps.push(
                        ort::execution_providers::DirectMLExecutionProvider::default()
                            .with_device_id(*i as i32)
                            .build(),
                    );
                    provider_name = "DirectMLExecutionProvider";
                }
                #[cfg(feature = "openvino")]
                crate::Device::OpenVino => {
                    eps.push(
                        ort::execution_providers::OpenVINOExecutionProvider::default().build(),
                    );
                    provider_name = "OpenVINOExecutionProvider";
                }
                #[cfg(feature = "xnnpack")]
                crate::Device::Xnnpack => {
                    eps.push(ort::execution_providers::XNNPACKExecutionProvider::default().build());
                    provider_name = "XNNPACKExecutionProvider";
                }
                // Handle cases where feature is disabled but enum variant exists
                #[allow(unreachable_patterns)]
                _ => {
                    eprintln!(
                        "\x1b[33mWARNING ⚠️ Device '{device}' requested but feature not enabled or supported. Falling back to available providers.\x1b[0m"
                    );
                }
            }
        } else {
            // Default: Register all available providers in preference order
            provider_name = "CPUExecutionProvider";

            #[cfg(feature = "tensorrt")]
            {
                eps.push(ort::execution_providers::TensorRTExecutionProvider::default().build());
                provider_name = "TensorRTExecutionProvider";
            }

            #[cfg(feature = "cuda")]
            {
                eps.push(ort::execution_providers::CUDAExecutionProvider::default().build());
                if provider_name == "CPUExecutionProvider" {
                    provider_name = "CUDAExecutionProvider";
                }
            }

            #[cfg(feature = "coreml")]
            {
                eps.push(ort::execution_providers::CoreMLExecutionProvider::default().build());
                if provider_name == "CPUExecutionProvider" {
                    provider_name = "CoreMLExecutionProvider";
                }
            }

            #[cfg(feature = "rocm")]
            {
                eps.push(ort::execution_providers::ROCmExecutionProvider::default().build());
                if provider_name == "CPUExecutionProvider" {
                    provider_name = "ROCmExecutionProvider";
                }
            }

            #[cfg(feature = "directml")]
            {
                eps.push(ort::execution_providers::DirectMLExecutionProvider::default().build());
                if provider_name == "CPUExecutionProvider" {
                    provider_name = "DirectMLExecutionProvider";
                }
            }

            #[cfg(feature = "openvino")]
            {
                eps.push(ort::execution_providers::OpenVINOExecutionProvider::default().build());
                if provider_name == "CPUExecutionProvider" {
                    provider_name = "OpenVINOExecutionProvider";
                }
            }

            #[cfg(feature = "xnnpack")]
            {
                eps.push(ort::execution_providers::XNNPACKExecutionProvider::default().build());
                if provider_name == "CPUExecutionProvider" {
                    provider_name = "XNNPACKExecutionProvider";
                }
            }
        }

        if !eps.is_empty() {
            session_builder = session_builder.with_execution_providers(eps).map_err(|e| {
                InferenceError::ModelLoadError(format!("Failed to set execution providers: {e}"))
            })?;
        }

        let session = session_builder
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| {
                InferenceError::ModelLoadError(format!("Failed to set optimization level: {e}"))
            })?
            .with_intra_threads(num_threads)
            .map_err(|e| {
                InferenceError::ModelLoadError(format!("Failed to set intra-op thread count: {e}"))
            })?
            .with_memory_pattern(true)
            .map_err(|e| {
                InferenceError::ModelLoadError(format!("Failed to enable memory pattern: {e}"))
            })?
            .commit_from_file(path)
            .map_err(|e| InferenceError::ModelLoadError(format!("Failed to load model: {e}")))?;

        // Extract metadata from model
        let metadata = Self::extract_metadata(&session)?;

        // Get input/output names and detect input type
        let input_info = session.inputs().first();
        let input_name = input_info.map_or_else(|| "images".to_string(), |i| i.name().to_string());

        // Check if model input tensor expects FP16 (rare - most models use FP32 input even with half weights)
        let model_input_fp16 = input_info.is_some_and(|i| {
            matches!(
                i.dtype(),
                ValueType::Tensor {
                    ty: TensorElementType::Float16,
                    ..
                }
            )
        });

        // Check for dynamic input dimensions
        // Dimensions are typically [-1, 3, -1, -1] for dynamic batch/height/width
        // ort 2.0 returns Option<i64> (None means dynamic/unknown)
        let is_dynamic = input_info.is_some_and(|i| {
            if let ValueType::Tensor { shape, .. } = i.dtype() {
                shape.iter().any(|d| *d == -1 || *d == 0)
            } else {
                false
            }
        });

        // Use FP16 input if model tensor type requires it
        let fp16_input = model_input_fp16;

        let output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        // Resolve image size
        // Priority:
        // 1. User config
        // 2. Model metadata
        // 3. Dynamic default (1024 for OBB, 640 for others)
        // 4. Static input shape
        // 5. Hard default (640)
        let resolved_imgsz = if let Some(sz) = config.imgsz {
            sz
        } else if let Some(sz) = metadata.imgsz {
            sz
        } else if is_dynamic {
            // Dynamic input without metadata -> apply robust defaults
            match metadata.task {
                Task::Obb => (1024, 1024),
                _ => (640, 640),
            }
        } else {
            // Static input without metadata -> try to read from tensor shape
            // Typically [1, 3, H, W]
            input_info
                .and_then(|i| {
                    if let ValueType::Tensor { shape, .. } = i.dtype() {
                        if shape.len() == 4 && shape[2] > 0 && shape[3] > 0 {
                            #[allow(clippy::cast_sign_loss)]
                            Some((shape[2] as usize, shape[3] as usize))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .unwrap_or((640, 640))
        };

        // Update config with resolved values
        let config = InferenceConfig {
            imgsz: Some(resolved_imgsz),
            half: config.half || metadata.half, // Use half if user requested OR model was exported with half
            ..config
        };

        // Ensure metadata reflects the resolved size
        let mut metadata = metadata;
        metadata.imgsz = Some(resolved_imgsz);

        let mut model = Self {
            session,
            metadata,
            input_name,
            output_names,
            config,
            warmed_up: false,
            fp16_input,
            execution_provider: provider_name.to_string(),
        };

        // Warmup inference to trigger JIT compilation and memory allocation
        model.warmup()?;

        Ok(model)
    }

    /// Maximum allowed image dimension to prevent OOM during warmup.
    const MAX_IMGSZ: usize = 8192;

    /// Warm up the model by running inference with a dummy input.
    ///
    /// This pre-allocates memory and optimizes the execution graph for faster
    /// subsequent inferences. Warmup is automatically called on first predict.
    pub fn warmup(&mut self) -> Result<()> {
        if self.warmed_up {
            return Ok(());
        }

        let target_size = self
            .config
            .imgsz
            .or(self.metadata.imgsz)
            .unwrap_or((640, 640));

        // Sanity check to prevent huge allocations from invalid imgsz
        if target_size.0 > Self::MAX_IMGSZ || target_size.1 > Self::MAX_IMGSZ {
            return Err(InferenceError::ConfigError(format!(
                "Image size {}x{} exceeds maximum allowed {}x{}",
                target_size.0,
                target_size.1,
                Self::MAX_IMGSZ,
                Self::MAX_IMGSZ
            )));
        }

        if self.fp16_input {
            // Use FP16 dummy input if model expects FP16
            let dummy_input = ndarray::Array4::<f16>::zeros((1, 3, target_size.0, target_size.1));
            self.run_inference_f16(&dummy_input)?;
        } else {
            let dummy_input = ndarray::Array4::<f32>::zeros((1, 3, target_size.0, target_size.1));
            self.run_inference(&dummy_input)?;
        }
        self.warmed_up = true;
        Ok(())
    }

    /// Extract metadata from the ONNX model session.
    fn extract_metadata(session: &Session) -> Result<ModelMetadata> {
        // Get metadata from the model
        let model_metadata = session.metadata().map_err(|e| {
            InferenceError::ModelLoadError(format!("Failed to get model metadata: {e}"))
        })?;

        // Ultralytics stores metadata under individual keys
        // Try to get each key separately and build a YAML string
        let mut metadata_map: HashMap<String, String> = HashMap::new();

        // List of all Ultralytics metadata keys
        let keys = [
            "description",
            "author",
            "date",
            "version",
            "license",
            "docs",
            "stride",
            "task",
            "batch",
            "imgsz",
            "names",
            "half",
            "channels",
            "args",
        ];

        for key in &keys {
            if let Some(value) = model_metadata.custom(key) {
                metadata_map.insert((*key).to_string(), value);
            }
        }

        // If we found individual keys, build a YAML string from them
        if !metadata_map.is_empty() {
            let mut yaml_parts = Vec::new();
            for (key, value) in &metadata_map {
                yaml_parts.push(format!("{key}: {value}"));
            }
            let combined_yaml = yaml_parts.join("\n");
            let mut combined_map = HashMap::new();
            combined_map.insert(String::new(), combined_yaml);
            return ModelMetadata::from_onnx_metadata(&combined_map);
        }

        // Also try getting metadata from a single combined key
        for key in &["", "metadata", "model_metadata"] {
            if let Some(value) = model_metadata.custom(key) {
                metadata_map.insert((*key).to_string(), value);
            }
        }

        if metadata_map.is_empty() {
            // Return defaults
            return Ok(ModelMetadata::default());
        }

        ModelMetadata::from_onnx_metadata(&metadata_map)
    }

    /// Returns the execution provider used for inference.
    pub fn execution_provider(&self) -> &str {
        &self.execution_provider
    }

    /// Run inference on an image file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the image file.
    ///
    /// # Returns
    ///
    /// Vector of Results (one per image in batch).
    ///
    /// # Errors
    ///
    /// Returns an error if the image can't be loaded or inference fails.
    pub fn predict<P: AsRef<Path>>(&mut self, path: P) -> Result<Vec<Results>> {
        let path = path.as_ref();

        // Warmup first to fail fast before loading/decoding the image
        self.warmup()?;

        // Load image using standard image crate
        let img = image::open(path).map_err(|e| {
            InferenceError::ImageError(format!("Failed to load image {}: {e}", path.display()))
        })?;

        self.predict_image(&img, path.to_string_lossy().to_string())
    }

    /// Run inference on a `DynamicImage`.
    ///
    /// # Arguments
    ///
    /// * `image` - The image to run inference on.
    /// * `path` - Optional path/identifier for the image.
    ///
    /// # Returns
    ///
    /// Vector of Results.
    pub fn predict_image(&mut self, image: &DynamicImage, path: String) -> Result<Vec<Results>> {
        // Get target size from config or metadata
        let target_size = self
            .config
            .imgsz
            .or(self.metadata.imgsz)
            .unwrap_or((640, 640));

        // Preprocess - generate FP16 tensor if model expects FP16 input
        let start_preprocess = Instant::now();
        let preprocess_result = if self.metadata.task == Task::Classify {
            preprocess_image_center_crop(image, target_size, self.fp16_input)
        } else {
            preprocess_image_with_precision(
                image,
                target_size,
                self.metadata.stride,
                self.fp16_input,
            )
        };
        let preprocess_time = start_preprocess.elapsed().as_secs_f64() * 1000.0;

        // Convert original image to array for results
        let orig_img = image_to_array(image);

        // Run inference with appropriate precision
        let start_inference = Instant::now();
        let outputs = if self.fp16_input {
            // Use FP16 tensor directly (no round-trip
            let tensor_f16 = preprocess_result
                .tensor_f16
                .as_ref()
                .expect("FP16 tensor should be available");
            self.run_inference_f16(tensor_f16)?
        } else {
            let input_tensor = &preprocess_result.tensor;

            self.run_inference(input_tensor)?
        };
        let inference_time = start_inference.elapsed().as_secs_f64() * 1000.0;

        // Post-process
        let _start_postprocess = Instant::now();

        // Post-process
        let start_postprocess = Instant::now();

        let speed = Speed::new(preprocess_time, inference_time, 0.0);

        // Extract inference tensor shape (height, width) from preprocessed tensor
        let tensor_shape = preprocess_result.tensor.shape();
        let inference_shape = (tensor_shape[2] as u32, tensor_shape[3] as u32);

        let result = postprocess(
            outputs,
            self.metadata.task,
            &preprocess_result,
            &self.config,
            &self.metadata.names,
            orig_img,
            path,
            speed,
            inference_shape,
        );

        let postprocess_time = start_postprocess.elapsed().as_secs_f64() * 1000.0;

        // Update speed with postprocess time
        let mut final_result = result;
        final_result.speed.postprocess = Some(postprocess_time);

        Ok(vec![final_result])
    }

    /// Run inference on a raw array.
    ///
    /// # Arguments
    ///
    /// * `image` - HWC u8 array.
    /// * `path` - Optional identifier.
    ///
    /// # Returns
    ///
    /// Vector of Results.
    pub fn predict_array(&mut self, image: &Array3<u8>, path: String) -> Result<Vec<Results>> {
        // Convert array to DynamicImage
        let shape = image.shape();
        let (height, width) = (shape[0] as u32, shape[1] as u32);

        let mut rgb_data = Vec::with_capacity((height * width * 3) as usize);
        for y in 0..height as usize {
            for x in 0..width as usize {
                rgb_data.push(image[[y, x, 0]]);
                rgb_data.push(image[[y, x, 1]]);
                rgb_data.push(image[[y, x, 2]]);
            }
        }

        let img_buffer = image::RgbImage::from_raw(width, height, rgb_data).ok_or_else(|| {
            InferenceError::ImageError("Failed to create image buffer".to_string())
        })?;

        let dynamic_img = DynamicImage::ImageRgb8(img_buffer);
        self.predict_image(&dynamic_img, path)
    }

    /// Run the ONNX model inference with FP32 input.
    fn run_inference(
        &mut self,
        input: &ndarray::Array4<f32>,
    ) -> Result<Vec<(Vec<f32>, Vec<usize>)>> {
        // Ensure input is contiguous in memory (CowArray)
        let input_contiguous = input.as_standard_layout();

        // Create input tensor reference from ndarray view
        let input_tensor = TensorRef::from_array_view(input_contiguous.view()).map_err(|e| {
            InferenceError::InferenceError(format!("Failed to create input tensor: {e}"))
        })?;

        // Run session - inputs! macro returns a Vec, not a Result
        let inputs = ort::inputs![&self.input_name => input_tensor];

        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| InferenceError::InferenceError(format!("Inference failed: {e}")))?;

        let mut results = Vec::new();

        for output_name in &self.output_names {
            let output = outputs.get(output_name.as_str()).ok_or_else(|| {
                InferenceError::InferenceError(format!("Output '{output_name}' not found"))
            })?;

            // Get output as f32 tensor - use try_extract_tensor which returns (shape, data)
            let (shape, data) = output.try_extract_tensor::<f32>().map_err(|e| {
                InferenceError::InferenceError(format!("Failed to extract output: {e}"))
            })?;

            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            let shape_vec: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            let data_vec: Vec<f32> = data.to_vec();
            results.push((data_vec, shape_vec));
        }

        Ok(results)
    }

    /// Run the ONNX model inference with FP16 input.
    fn run_inference_f16(
        &mut self,
        input: &ndarray::Array4<f16>,
    ) -> Result<Vec<(Vec<f32>, Vec<usize>)>> {
        // Ensure input is contiguous in memory (CowArray)
        let input_contiguous = input.as_standard_layout();

        // Create input tensor reference from ndarray view
        let input_tensor = TensorRef::from_array_view(&input_contiguous).map_err(|e| {
            InferenceError::InferenceError(format!("Failed to create FP16 input tensor: {e}"))
        })?;

        // Run session
        let inputs = ort::inputs![&self.input_name => input_tensor];

        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| InferenceError::InferenceError(format!("FP16 inference failed: {e}")))?;

        let mut results = Vec::new();

        for output_name in &self.output_names {
            let output = outputs.get(output_name.as_str()).ok_or_else(|| {
                InferenceError::InferenceError(format!("Output '{output_name}' not found"))
            })?;

            // Try to extract as f32 first (model may have FP32 output even with FP16 input)
            // If that fails, extract as f16 and convert
            let (shape_vec, data_vec) = if let Ok((shape, data)) =
                output.try_extract_tensor::<f32>()
            {
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                let shape_vec: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
                let data_vec: Vec<f32> = data.to_vec();
                (shape_vec, data_vec)
            } else {
                // Extract as f16 and convert to f32 for postprocessing
                let (shape, data) = output.try_extract_tensor::<f16>().map_err(|e| {
                    InferenceError::InferenceError(format!("Failed to extract FP16 output: {e}"))
                })?;

                #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                let shape_vec: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
                let data_vec: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
                (shape_vec, data_vec)
            };
            results.push((data_vec, shape_vec));
        }

        Ok(results)
    }

    /// Get the model's task type.
    #[must_use]
    pub const fn task(&self) -> Task {
        self.metadata.task
    }

    /// Get the model's class names.
    #[must_use]
    pub const fn names(&self) -> &HashMap<usize, String> {
        &self.metadata.names
    }

    /// Get the number of classes.
    #[must_use]
    pub fn num_classes(&self) -> usize {
        self.metadata.num_classes()
    }

    /// Get the model's input size.
    #[must_use]
    pub fn imgsz(&self) -> (usize, usize) {
        self.metadata.imgsz.unwrap_or((640, 640))
    }

    /// Get the model's stride.
    #[must_use]
    pub const fn stride(&self) -> u32 {
        self.metadata.stride
    }

    /// Check if model is using FP16 (half precision) inference.
    #[must_use]
    pub const fn is_half(&self) -> bool {
        self.fp16_input
    }

    /// Get the model metadata.
    ///
    /// Example:
    ///
    /// ```no_run
    /// use ultralytics_inference::YOLOModel;
    /// let mut model = YOLOModel::load("yolo11n.onnx")?;
    /// println!("Model name: {}", model.metadata().model_name());
    /// # Ok::<(), ultralytics_inference::InferenceError>(())
    /// ```
    #[must_use]
    pub const fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    /// Get the model path.
    #[must_use]
    pub const fn model_path(&self) -> &'static str {
        // Note: ONNX Runtime doesn't expose the original path
        // This is a placeholder - in practice, users should track the path themselves
        ""
    }
}

#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for YOLOModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("YOLOModel")
            .field("task", &self.metadata.task)
            .field("num_classes", &self.metadata.num_classes())
            .field("imgsz", &self.metadata.imgsz)
            .field("stride", &self.metadata.stride)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_not_found() {
        let result = YOLOModel::load("nonexistent.onnx");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            InferenceError::ModelLoadError(_)
        ));
    }
}
