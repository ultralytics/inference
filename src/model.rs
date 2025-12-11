// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! YOLO model loading and inference.
//!
//! This module provides the main `YOLOModel` struct for loading ONNX models
//! and running inference.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use image::DynamicImage;
use ndarray::Array3;
use ort::session::Session;
#[cfg(feature = "coreml")]
use ort::execution_providers::CoreMLExecutionProvider;
use ort::value::TensorRef;

use crate::error::{InferenceError, Result};
use crate::inference::InferenceConfig;
use crate::metadata::ModelMetadata;
use crate::postprocessing::postprocess;
use crate::preprocessing::{image_to_array, preprocess_image};
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
/// use inference::YOLOModel;
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
}

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
    /// use inference::YOLOModel;
    ///
    /// let model = YOLOModel::load("yolo11n.onnx")?;
    /// # Ok::<(), inference::InferenceError>(())
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

        // Check if file exists
        if !path.exists() {
            return Err(InferenceError::ModelLoadError(format!(
                "Model file not found: {}",
                path.display()
            )));
        }

        // Create ONNX Runtime session with optimizations
        #[allow(unused_mut)]
        let mut builder = Session::builder()
            .map_err(|e| InferenceError::ModelLoadError(format!("Failed to create session builder: {e}")))?;

        // Configure execution providers (hardware acceleration)
        #[cfg(feature = "coreml")]
        {
            builder = builder
                .with_execution_providers([
                    CoreMLExecutionProvider::default()
                        .with_subgraphs(true) // Enable on subgraphs for better coverage
                        .build()
                ])
                .map_err(|e| InferenceError::ModelLoadError(format!("Failed to register CoreML EP: {e}")))?;
        }

        // Apply session optimizations
        let session = builder
            // Graph optimization - Level3 enables all optimizations including extended graph optimizations
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| InferenceError::ModelLoadError(format!("Failed to set optimization level: {e}")))?
            // Intra-op threads: parallelization within individual operators (e.g., matrix multiply)
            .with_intra_threads(config.num_threads)
            .map_err(|e| InferenceError::ModelLoadError(format!("Failed to set intra-thread count: {e}")))?
            .commit_from_file(path)
            .map_err(|e| InferenceError::ModelLoadError(format!("Failed to load model: {e}")))?;

        // Extract metadata from model
        let metadata = Self::extract_metadata(&session)?;

        // Get input/output names
        let input_name = session
            .inputs
            .first()
            .map(|i| i.name.clone())
            .unwrap_or_else(|| "images".to_string());

        let output_names: Vec<String> = session.outputs.iter().map(|o| o.name.clone()).collect();

        // Update config with model metadata if not overridden
        let config = InferenceConfig {
            imgsz: config.imgsz.or(Some(metadata.imgsz)),
            ..config
        };

        Ok(Self {
            session,
            metadata,
            input_name,
            output_names,
            config,
            warmed_up: false,
        })
    }

    /// Warm up the model by running inference with a dummy input.
    ///
    /// This pre-allocates memory and optimizes the execution graph for faster
    /// subsequent inferences. Warmup is automatically called on first predict.
    pub fn warmup(&mut self) -> Result<()> {
        if self.warmed_up {
            return Ok(());
        }

        let target_size = self.config.imgsz.unwrap_or(self.metadata.imgsz);

        // Create dummy input tensor (zeros)
        let dummy_input = ndarray::Array4::<f32>::zeros((1, 3, target_size.0, target_size.1));

        // Run warmup inference (discard results)
        let _ = self.run_inference(&dummy_input)?;

        self.warmed_up = true;
        Ok(())
    }

    /// Extract metadata from the ONNX model session.
    fn extract_metadata(session: &Session) -> Result<ModelMetadata> {
        // Get metadata from the model
        let model_metadata = session
            .metadata()
            .map_err(|e| InferenceError::ModelLoadError(format!("Failed to get model metadata: {e}")))?;

        // Ultralytics stores metadata under individual keys
        // Try to get each key separately and build a YAML string
        let mut metadata_map: HashMap<String, String> = HashMap::new();

        // List of all Ultralytics metadata keys
        let keys = [
            "description", "author", "date", "version", "license", "docs",
            "stride", "task", "batch", "imgsz", "names", "half", "channels",
        ];

        for key in &keys {
            if let Ok(Some(value)) = model_metadata.custom(key) {
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
            if let Ok(Some(value)) = model_metadata.custom(key) {
                metadata_map.insert((*key).to_string(), value);
            }
        }

        if metadata_map.is_empty() {
            // Return defaults
            return Ok(ModelMetadata::default());
        }

        ModelMetadata::from_onnx_metadata(&metadata_map)
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

        // Load image
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
        // Warmup on first inference (pre-allocates memory, optimizes graph)
        if !self.warmed_up {
            self.warmup()?;
        }

        // Get target size from config or metadata
        let target_size = self.config.imgsz.unwrap_or(self.metadata.imgsz);

        // Preprocess
        let start_preprocess = Instant::now();
        let preprocess_result = preprocess_image(image, target_size, self.metadata.stride);
        let preprocess_time = start_preprocess.elapsed().as_secs_f64() * 1000.0;

        // Convert original image to array for results
        let orig_img = image_to_array(image);

        // Run inference
        let start_inference = Instant::now();
        let outputs = self.run_inference(&preprocess_result.tensor)?;
        let inference_time = start_inference.elapsed().as_secs_f64() * 1000.0;

        // Post-process
        let start_postprocess = Instant::now();

        // Get output data
        let (output_data, output_shape) = outputs;

        let speed = Speed::new(preprocess_time, inference_time, 0.0);

        // Extract inference tensor shape (height, width) from preprocessed tensor
        let tensor_shape = preprocess_result.tensor.shape();
        let inference_shape = (tensor_shape[2] as u32, tensor_shape[3] as u32);

        let result = postprocess(
            &output_data,
            &output_shape,
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

        let img_buffer = image::RgbImage::from_raw(width, height, rgb_data)
            .ok_or_else(|| InferenceError::ImageError("Failed to create image buffer".to_string()))?;

        let dynamic_img = DynamicImage::ImageRgb8(img_buffer);
        self.predict_image(&dynamic_img, path)
    }

    /// Run the ONNX model inference.
    fn run_inference(
        &mut self,
        input: &ndarray::Array4<f32>,
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        // Ensure input is contiguous in memory (CowArray)
        let input_contiguous = input.as_standard_layout();

        // Create input tensor reference from ndarray view (pass reference to CowArray)
        let input_tensor = TensorRef::from_array_view(&input_contiguous)
            .map_err(|e| InferenceError::InferenceError(format!("Failed to create input tensor: {e}")))?;

        // Run session - inputs! macro returns a Vec, not a Result
        let inputs = ort::inputs![&self.input_name => input_tensor];

        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| InferenceError::InferenceError(format!("Inference failed: {e}")))?;

        // Extract output
        let output_name = &self.output_names[0];
        let output = outputs
            .get(output_name.as_str())
            .ok_or_else(|| InferenceError::InferenceError(format!("Output '{}' not found", output_name)))?;

        // Get output as f32 tensor - use try_extract_tensor which returns (shape, data)
        let (shape, data) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| InferenceError::InferenceError(format!("Failed to extract output: {e}")))?;

        let shape_vec: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let data_vec: Vec<f32> = data.to_vec();

        Ok((data_vec, shape_vec))
    }

    /// Get the model's task type.
    #[must_use]
    pub const fn task(&self) -> Task {
        self.metadata.task
    }

    /// Get the model's class names.
    #[must_use]
    pub fn names(&self) -> &HashMap<usize, String> {
        &self.metadata.names
    }

    /// Get the number of classes.
    #[must_use]
    pub fn num_classes(&self) -> usize {
        self.metadata.num_classes()
    }

    /// Get the model's input size.
    #[must_use]
    pub const fn imgsz(&self) -> (usize, usize) {
        self.metadata.imgsz
    }

    /// Get the model's stride.
    #[must_use]
    pub const fn stride(&self) -> u32 {
        self.metadata.stride
    }

    /// Get the model metadata.
    #[must_use]
    pub const fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    /// Get the model path.
    #[must_use]
    pub fn model_path(&self) -> &str {
        // Note: ONNX Runtime doesn't expose the original path
        // This is a placeholder - in practice, users should track the path themselves
        ""
    }
}

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
        assert!(matches!(result.unwrap_err(), InferenceError::ModelLoadError(_)));
    }
}
