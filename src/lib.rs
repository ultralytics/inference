// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! # Ultralytics YOLO Inference Library
//!
//! High-performance YOLO model inference library written in Rust.
//! This library provides a safe and efficient interface for running YOLO models
//! on images and video streams.
//!
//! ## Features
//!
//! - Fast inference using ONNX Runtime
//! - Support for all YOLO versions (YOLOv5, YOLOv8, YOLO11, etc.)
//! - Support for all YOLO tasks (detection, segmentation, pose, classification, OBB)
//! - Ultralytics-compatible Results API
//! - Thread-safe model loading and inference
//! - Multiple input sources (images, video, webcam, streams)
//!
//! ## Quick Start
//!
//! ```no_run
//! use inference::{YOLOModel, InferenceConfig};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Load model - metadata (classes, task, imgsz) is read automatically
//!     let mut model = YOLOModel::load("yolo11n.onnx")?;
//!
//!     // Run inference
//!     let results = model.predict("image.jpg")?;
//!
//!     // Process results
//!     for result in &results {
//!         if let Some(ref boxes) = result.boxes {
//!             println!("Found {} detections", boxes.len());
//!             for i in 0..boxes.len() {
//!                 let cls = boxes.cls()[i] as usize;
//!                 let conf = boxes.conf()[i];
//!                 let name = result.names.get(&cls).map(|s| s.as_str()).unwrap_or("unknown");
//!                 println!("  {} {:.2}", name, conf);
//!             }
//!         }
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Custom Configuration
//!
//! ```rust
//! use inference::InferenceConfig;
//!
//! let config = InferenceConfig::new()
//!     .with_confidence(0.5)
//!     .with_iou(0.45)
//!     .with_max_detections(100);
//! ```

// Modules
#[cfg(feature = "annotate")]
pub mod annotate;
pub mod download;
pub mod error;
pub mod inference;
pub mod metadata;
pub mod model;
pub mod postprocessing;
pub mod preprocessing;
pub mod results;
pub mod source;
pub mod task;
pub mod utils;
pub mod color;

// Re-export main types for convenience
pub use error::{InferenceError, Result};
pub use inference::InferenceConfig;
pub use model::YOLOModel;
pub use results::{Boxes, Keypoints, Masks, Obb, Probs, Results, Speed};
pub use source::{Source, SourceIterator, SourceMeta};
pub use task::Task;

// Re-export metadata for advanced use
pub use metadata::ModelMetadata;

// Re-export preprocessing utilities
pub use preprocessing::{
    preprocess_image, preprocess_image_with_precision, PreprocessResult, TensorData,
};

/// Library version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Library name.
pub const NAME: &str = env!("CARGO_PKG_NAME");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_name() {
        assert_eq!(NAME, "inference");
    }
}
