// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#![allow(clippy::multiple_crate_versions)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! # Ultralytics YOLO Inference Library
//!
//! [![crates.io](https://img.shields.io/crates/v/ultralytics-inference.svg)](https://crates.io/crates/ultralytics-inference)
//! [![docs.rs](https://docs.rs/ultralytics-inference/badge.svg)](https://docs.rs/ultralytics-inference)
//! [![License](https://img.shields.io/crates/l/ultralytics-inference.svg)](https://github.com/ultralytics/inference/blob/main/LICENSE)
//!
//! High-performance YOLO model inference library written in Rust, providing a safe
//! and efficient interface for running [Ultralytics](https://ultralytics.com) YOLO
//! models on images, videos, and streams.
//!
//! ## Features
//!
//! - **High Performance** - Pure Rust with zero-cost abstractions and SIMD-optimized preprocessing
//! - **ONNX Runtime** - Leverages ONNX Runtime for cross-platform hardware acceleration
//! - **All YOLO Versions** - Supports `YOLOv5`, `YOLOv8`, YOLO11, and future versions
//! - **All Tasks** - Detection, segmentation, pose estimation, classification, and OBB
//! - **Ultralytics API** - Results API matches the Python package for easy migration
//! - **Multiple Backends** - CPU, CUDA, `TensorRT`, `CoreML`, `OpenVINO`, and more
//! - **Multiple Sources** - Images, directories, glob patterns, video, webcam, streams
//!
//! ## Installation
//!
//! Add to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! ultralytics-inference = "0.0.5"
//! ```
//!
//! Or install the CLI tool:
//!
//! ```bash
//! cargo install ultralytics-inference
//! ```
//!
//! ## Quick Start (Library)
//!
//! ```no_run
//! use ultralytics_inference::{YOLOModel, InferenceConfig};
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
//! ## CLI Usage
//!
//! The `ultralytics-inference` CLI provides a command-line interface for running YOLO inference:
//!
//! ```bash
//! # Install the CLI
//! cargo install ultralytics-inference
//!
//! # Run with defaults (auto-downloads model and sample images)
//! ultralytics-inference predict
//!
//! # Run on a specific image
//! ultralytics-inference predict --model yolo11n.onnx --source image.jpg
//!
//! # Run on a directory of images
//! ultralytics-inference predict --model yolo11n.onnx --source images/
//!
//! # With custom thresholds
//! ultralytics-inference predict -m yolo11n.onnx -s image.jpg --conf 0.5 --iou 0.45
//!
//! # With visualization window
//! ultralytics-inference predict --model yolo11n.onnx --source video.mp4 --show
//!
//! # Save annotated results
//! ultralytics-inference predict --model yolo11n.onnx --source image.jpg --save
//!
//! # Show help
//! ultralytics-inference help
//! ```
//!
//! **CLI Options:**
//!
//! | Option | Short | Description | Default |
//! |--------|-------|-------------|---------|
//! | `--model` | `-m` | Path to ONNX model | `yolo11n.onnx` |
//! | `--source` | `-s` | Input source | Sample images |
//! | `--conf` | | Confidence threshold | `0.25` |
//! | `--iou` | | `IoU` threshold for NMS | `0.45` |
//! | `--imgsz` | | Inference image size | `640` |
//! | `--half` | | Use FP16 inference | `false` |
//! | `--save` | | Save annotated images | `false` |
//! | `--show` | | Display results window | `false` |
//!
//! ## Task-Specific Examples
//!
//! The library supports all YOLO tasks. Export models from Python:
//!
//! ```bash
//! # Detection (default)
//! yolo export model=yolo11n.pt format=onnx
//!
//! # Segmentation
//! yolo export model=yolo11n-seg.pt format=onnx
//!
//! # Pose Estimation
//! yolo export model=yolo11n-pose.pt format=onnx
//!
//! # Classification
//! yolo export model=yolo11n-cls.pt format=onnx
//!
//! # Oriented Bounding Boxes
//! yolo export model=yolo11n-obb.pt format=onnx
//! ```
//!
//! The task is auto-detected from ONNX metadata:
//!
//! ```no_run
//! use ultralytics_inference::YOLOModel;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Segmentation model - returns masks
//! let mut model = YOLOModel::load("yolo11n-seg.onnx")?;
//! let results = model.predict("image.jpg")?;
//! if let Some(ref masks) = results[0].masks {
//!     println!("Found {} instance masks", masks.len());
//! }
//!
//! // Pose model - returns keypoints
//! let mut model = YOLOModel::load("yolo11n-pose.onnx")?;
//! let results = model.predict("image.jpg")?;
//! if let Some(ref keypoints) = results[0].keypoints {
//!     println!("Found {} poses", keypoints.len());
//! }
//!
//! // Classification model - returns probabilities
//! let mut model = YOLOModel::load("yolo11n-cls.onnx")?;
//! let results = model.predict("image.jpg")?;
//! if let Some(ref probs) = results[0].probs {
//!     println!("Top-1: class {} ({:.1}%)", probs.top1(), probs.top1conf() * 100.0);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Docker Deployment
//!
//! Deploy with Docker for consistent cross-platform inference:
//!
//! ```dockerfile
//! # Dockerfile
//! FROM rust:1.85-slim AS builder
//! WORKDIR /app
//! COPY . .
//! RUN cargo build --release
//!
//! FROM debian:bookworm-slim
//! RUN apt-get update && apt-get install -y libssl3 ca-certificates && rm -rf /var/lib/apt/lists/*
//! COPY --from=builder /app/target/release/ultralytics-inference /usr/local/bin/
//! ENTRYPOINT ["ultralytics-inference"]
//! CMD ["predict", "--help"]
//! ```
//!
//! ```bash
//! # Build and run
//! docker build -t ultralytics-inference .
//! docker run -v $(pwd)/models:/models -v $(pwd)/images:/images \
//!     ultralytics-inference predict --model /models/yolo11n.onnx --source /images/
//! ```
//!
//! ## Shell/Bash Integration
//!
//! Use in shell scripts for batch processing and automation:
//!
//! ```bash
//! #!/bin/bash
//! # Process all images in a directory
//! ultralytics-inference predict \
//!     --model yolo11n.onnx \
//!     --source ./images/*.jpg \
//!     --conf 0.5 \
//!     --save
//!
//! # Process with different models
//! for model in yolo11n yolo11s yolo11m; do
//!     ultralytics-inference predict \
//!         --model "${model}.onnx" \
//!         --source image.jpg
//! done
//! ```
//!
//! ## Python Integration
//!
//! Call the CLI from Python for high-performance inference:
//!
//! ```python
//! import subprocess
//! import json
//!
//! def run_inference(model: str, source: str, conf: float = 0.25) -> str:
//!     """Run YOLO inference using the Rust CLI."""
//!     result = subprocess.run(
//!         ["ultralytics-inference", "predict",
//!          "--model", model,
//!          "--source", source,
//!          "--conf", str(conf)],
//!         capture_output=True,
//!         text=True,
//!         check=True
//!     )
//!     return result.stdout
//!
//! # Single image
//! output = run_inference("yolo11n.onnx", "image.jpg")
//! print(output)
//!
//! # Batch processing
//! from pathlib import Path
//! for img in Path("images").glob("*.jpg"):
//!     run_inference("yolo11n.onnx", str(img), conf=0.5)
//! ```
//!
//! For native Python bindings with the full Ultralytics API:
//! ```bash
//! pip install ultralytics
//! ```
//!
//! ## Node.js / JavaScript
//!
//! ```javascript
//! const { execSync, spawn } = require('child_process');
//!
//! // Synchronous execution
//! const output = execSync(
//!     'ultralytics-inference predict --model yolo11n.onnx --source image.jpg'
//! );
//! console.log(output.toString());
//!
//! // Async with streaming output
//! const proc = spawn('ultralytics-inference', [
//!     'predict', '--model', 'yolo11n.onnx', '--source', 'video.mp4'
//! ]);
//! proc.stdout.on('data', (data) => console.log(data.toString()));
//! proc.on('close', (code) => console.log(`Exited with code ${code}`));
//! ```
//!
//! ## Go
//!
//! ```go
//! package main
//!
//! import (
//!     "fmt"
//!     "os/exec"
//! )
//!
//! func runInference(model, source string) (string, error) {
//!     cmd := exec.Command("ultralytics-inference", "predict",
//!         "--model", model, "--source", source)
//!     output, err := cmd.Output()
//!     return string(output), err
//! }
//!
//! func main() {
//!     output, err := runInference("yolo11n.onnx", "image.jpg")
//!     if err != nil {
//!         panic(err)
//!     }
//!     fmt.Println(output)
//! }
//! ```
//!
//! ## Ruby
//!
//! ```ruby
//! # Run inference
//! output = `ultralytics-inference predict --model yolo11n.onnx --source image.jpg`
//! puts output
//!
//! # With error handling
//! require 'open3'
//! stdout, stderr, status = Open3.capture3(
//!   'ultralytics-inference', 'predict',
//!   '--model', 'yolo11n.onnx',
//!   '--source', 'image.jpg'
//! )
//! puts stdout if status.success?
//! ```
//!
//! ## Java / Kotlin
//!
//! ```java
//! import java.io.*;
//!
//! public class YOLOInference {
//!     public static String predict(String model, String source) throws Exception {
//!         ProcessBuilder pb = new ProcessBuilder(
//!             "ultralytics-inference", "predict",
//!             "--model", model, "--source", source
//!         );
//!         pb.redirectErrorStream(true);
//!         Process p = pb.start();
//!         BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
//!         StringBuilder output = new StringBuilder();
//!         String line;
//!         while ((line = reader.readLine()) != null) {
//!             output.append(line).append("\n");
//!         }
//!         p.waitFor();
//!         return output.toString();
//!     }
//! }
//! ```
//!
//! ## C/C++ Integration
//!
//! For embedded systems or performance-critical applications, call the CLI or link directly:
//!
//! ```c
//! #include <stdlib.h>
//! #include <stdio.h>
//!
//! int main() {
//!     // Simple CLI invocation
//!     int result = system("ultralytics-inference predict "
//!                         "--model yolo11n.onnx --source image.jpg");
//!     return result;
//! }
//! ```
//!
//! For native FFI bindings (C-compatible shared library), contact Ultralytics for enterprise licensing.
//!
//! ## Custom Configuration
//!
//! Use the builder pattern to customize inference settings:
//!
//! ```rust
//! use ultralytics_inference::InferenceConfig;
//!
//! let config = InferenceConfig::new()
//!     .with_confidence(0.5)    // Confidence threshold
//!     .with_iou(0.45)          // NMS IoU threshold
//!     .with_max_detections(100) // Max detections per image
//!     .with_imgsz(640, 640);   // Input image size
//! ```
//!
//! ## Hardware Acceleration
//!
//! Enable hardware acceleration with Cargo features:
//!
//! ```bash
//! # NVIDIA CUDA
//! cargo build --release --features cuda
//!
//! # NVIDIA TensorRT
//! cargo build --release --features tensorrt
//!
//! # Apple CoreML
//! cargo build --release --features coreml
//!
//! # Intel OpenVINO
//! cargo build --release --features openvino
//! ```
//!
//! ## Results API
//!
//! The [`Results`] struct provides access to inference outputs:
//!
//! - [`Boxes`] - Bounding boxes with `xyxy()`, `xywh()`, `conf()`, `cls()` methods
//! - [`Masks`] - Segmentation masks for each detection
//! - [`Keypoints`] - Pose keypoints with `xy()`, `conf()` methods
//! - [`Probs`] - Classification probabilities with `top1()`, `top5()` methods
//! - [`Obb`] - Oriented bounding boxes with rotation
//!
//! ## Module Overview
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`model`] | Core [`YOLOModel`] for loading models and running inference |
//! | [`results`] | Output types ([`Results`], [`Boxes`], [`Masks`], etc.) |
//! | [`inference`] | [`InferenceConfig`] for customizing inference settings |
//! | [`source`] | Input source handling ([`Source`], [`SourceIterator`]) |
//! | [`task`] | YOLO task types ([`Task`]: Detect, Segment, Pose, etc.) |
//! | [`error`] | Error types ([`InferenceError`], [`Result`]) |
//! | [`preprocessing`] | Image preprocessing utilities |
//! | [`postprocessing`] | Detection post-processing (NMS, decode) |
//! | [`metadata`] | ONNX model metadata parsing |
//!
//! ## Feature Flags
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `annotate` | Image annotation support (default) |
//! | `visualize` | Real-time window display (default) |
//! | `video` | Video file and stream support |
//! | `cuda` | NVIDIA CUDA acceleration |
//! | `tensorrt` | NVIDIA `TensorRT` optimization |
//! | `coreml` | Apple `CoreML` (macOS/iOS) |
//! | `openvino` | Intel `OpenVINO` |
//!
//! ## License
//!
//! This project is dual-licensed under [AGPL-3.0](https://github.com/ultralytics/inference/blob/main/LICENSE)
//! for open-source use or [Ultralytics Enterprise License](https://ultralytics.com/license)
//! for commercial applications.

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
pub mod visualizer;

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
    PreprocessResult, TensorData, preprocess_image, preprocess_image_with_precision,
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
        // Version should be semver format like "0.0.5"
        assert!(VERSION.contains('.'));
    }

    #[test]
    fn test_name() {
        assert_eq!(NAME, "ultralytics-inference");
    }
}
