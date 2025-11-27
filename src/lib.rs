// © 2014-2025 Ultralytics Inc. 🚀 All rights reserved. CONFIDENTIAL: Unauthorized use or distribution prohibited.

//! # Ultralytics YOLO Inference Library
//!
//! High-performance YOLO model inference library written in Rust.
//! This library provides a safe and efficient interface for running YOLO models
//! on images and video streams.
//!
//! ## Features
//!
//! - Fast inference using ONNX Runtime
//! - Support for all YOLO versions (YOLOv5, YOLOv8, YOLOv11, etc.)
//! - Thread-safe model loading and inference
//! - Python bindings via PyO3 (future)
//!
//! ## Example
//!
//! ```no_run
//! use inference::YoloModel;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let model = YoloModel::load("yolov8n.onnx")?;
//!     let results = model.predict("image.jpg")?;
//!     println!("Detected {} objects", results.len());
//!     Ok(())
//! }
//! ```

pub mod error;
pub mod model;
pub mod inference;
pub mod utils;

// Re-export main types
pub use error::{InferenceError, Result};
pub use model::YoloModel;
pub use inference::{DetectionResult, InferenceConfig};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
