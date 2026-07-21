// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#![allow(clippy::multiple_crate_versions)]
#![deny(dead_code)]
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! # Ultralytics YOLO Inference Library
//!
//! [![crates.io](https://img.shields.io/crates/v/ultralytics-inference.svg)](https://crates.io/crates/ultralytics-inference)
//! [![docs.rs](https://docs.rs/ultralytics-inference/badge.svg)](https://docs.rs/ultralytics-inference)
//! [![Downloads](https://img.shields.io/crates/d/ultralytics-inference?logo=rust&logoColor=white&label=downloads&color=CE422B)](https://crates.io/crates/ultralytics-inference)
//! [![License](https://img.shields.io/crates/l/ultralytics-inference.svg)](https://github.com/ultralytics/inference/blob/main/LICENSE)
//! [![MSRV](https://img.shields.io/crates/msrv/ultralytics-inference?logo=rust&logoColor=white&color=CE422B)](https://crates.io/crates/ultralytics-inference)
//! [![dependency status](https://deps.rs/repo/github/ultralytics/inference/status.svg)](https://deps.rs/repo/github/ultralytics/inference)
//!
//!
//! High-performance YOLO model inference library written in Rust, providing a safe
//! and efficient interface for running [Ultralytics](https://ultralytics.com) YOLO
//! models on images, videos, and streams.
//!
//! ## Features
//!
//! - **High Performance** - Pure Rust with zero-cost abstractions and SIMD-optimized preprocessing
//! - **ONNX Runtime** - Leverages ONNX Runtime for cross-platform hardware acceleration
//! - **Supported YOLO Versions** - `YOLO26`, `YOLO11`, and `YOLOv8` (including YOLO26 end-to-end NMS-free exports)
//! - **All Tasks** - Detection, segmentation, pose estimation, classification, OBB, semantic segmentation, and depth estimation (last two YOLO26 only)
//! - **Ultralytics API** - Results API for easy migration
//! - **Multiple Backends** - CPU, CUDA, `TensorRT`, `CoreML`, `OpenVINO`, and more
//! - **Multiple Sources** - Images, directories, glob patterns, video, webcam, streams
//!
//! ## Installation
//!
//! Add to your `Cargo.toml`:
//!
#![doc = concat!("```toml\n[dependencies]\nultralytics-inference = \"", env!("CARGO_PKG_VERSION"), "\"\n```")]
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
//!     let mut model = YOLOModel::load("yolo26n.onnx")?;
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
//! For runnable programs you can copy and adapt, see the
//! [examples](https://github.com/ultralytics/inference/tree/main/examples) directory.
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
//! # Select task: auto-downloads the matching nano model
//! ultralytics-inference predict --task segment
//! ultralytics-inference predict --task pose
//! ultralytics-inference predict --task obb
//! ultralytics-inference predict --task classify
//!
//! # Run on a specific image
//! ultralytics-inference predict --model yolo26n.onnx --source image.jpg
//!
//! # Run on a directory of images
//! ultralytics-inference predict --model yolo26n.onnx --source images/
//!
//! # With custom thresholds
//! ultralytics-inference predict -m yolo26n.onnx -s image.jpg --conf 0.5 --iou 0.7
//!
//! # Filter by class IDs
//! ultralytics-inference predict --source image.jpg --classes "0,1,2"
//!
//! # With visualization window
//! ultralytics-inference predict --model yolo26n.onnx --source video.mp4 --show
//!
//! # Save annotated results
//! ultralytics-inference predict --model yolo26n.onnx --source image.jpg --save
//!
//! # Save individual frames for video input
//! ultralytics-inference predict --source video.mp4 --save-frames
//!
//! # Show help
//! ultralytics-inference help
//!
//! # Show version
//! ultralytics-inference version
//! ```
//!
//! **CLI Options:**
//!
//! | Option | Short | Description | Default |
//! |--------|-------|-------------|---------|
//! | `--model` | `-m` | Path to ONNX model file; auto-downloaded if a known YOLO26/YOLO11/YOLOv8 name | `yolo26n.onnx` |
//! | `--task` | | Task type (`detect`, `segment`, `pose`, `obb`, `classify`, `semantic`\*, `depth`\*); selects nano model when `--model` is omitted | `detect` |
//! | `--source` | `-s` | Input source (image, directory, glob, video, webcam index, or URL) | Task-dependent sample assets |
//! | `--conf` | | Confidence threshold | `0.25` |
//! | `--iou` | | `IoU` threshold for NMS | `0.7` |
//! | `--max-det` | | Maximum number of detections | `300` |
//! | `--imgsz` | | Inference image size | Model metadata |
//! | `--rect` | | Enable rectangular inference (minimal padding) | `true` |
//! | `--batch` | | Batch size for inference | `1` |
//! | `--half` | | Use FP16 half-precision inference | `false` |
//! | `--save` | | Save annotated results to runs/\<task\>/predict | `true` |
//! | `--save-frames` | | Save individual frames for video input | `false` |
//! | `--save-json` | | Save semantic segmentation class-map PNGs for external evaluation | `false` |
//! | `--show` | | Display results in a window | `false` |
//! | `--device` | | Device (cpu, cuda:0, coreml, directml:0, intel:cpu, intel:gpu, intel:npu, tensorrt:0, xnnpack) | `cpu` |
//! | `--verbose` | | Show verbose output | `true` |
//! | `--classes` | | Filter by class IDs, e.g. `0` or `"0,1,2"` or `"[0, 1, 2]"` | all classes |
//!
//! \* `semantic` (semantic segmentation) and `depth` (depth estimation) are YOLO26-only.
//!
//! ## Task-Specific Examples
//!
//! The library supports all YOLO tasks. Export models to ONNX:
//!
//! ```bash
//! # Detection (default)
//! yolo export model=yolo26n.pt format=onnx
//!
//! # Segmentation
//! yolo export model=yolo26n-seg.pt format=onnx
//!
//! # Pose Estimation
//! yolo export model=yolo26n-pose.pt format=onnx
//!
//! # Classification
//! yolo export model=yolo26n-cls.pt format=onnx
//!
//! # Oriented Bounding Boxes
//! yolo export model=yolo26n-obb.pt format=onnx
//!
//! # Semantic Segmentation (YOLO26 only)
//! yolo export model=yolo26n-sem.pt format=onnx
//!
//! # Depth Estimation (YOLO26 only)
//! yolo export model=yolo26n-depth.pt format=onnx
//! ```
//!
//! Add `quantize=16` for an FP16 (half-precision) ONNX, or `quantize=8` for INT8
//! (which also needs a calibration dataset via `data=`). Requires Ultralytics
//! >= 8.4; `quantize` replaces the deprecated `half`/`int8` flags.
//!
//! The task is auto-detected from ONNX metadata:
//!
//! ```no_run
//! use ultralytics_inference::YOLOModel;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Detection model - returns bounding boxes
//! let mut model = YOLOModel::load("yolo26n.onnx")?;
//! let results = model.predict("image.jpg")?;
//! if let Some(ref boxes) = results[0].boxes {
//!     println!("Found {} detections", boxes.len());
//!     for i in 0..boxes.len() {
//!         let cls = boxes.cls()[i] as usize;
//!         let conf = boxes.conf()[i];
//!         let name = results[0].names.get(&cls).map(|s| s.as_str()).unwrap_or("unknown");
//!         println!("  {} {:.2}", name, conf);
//!     }
//! }
//!
//! // Segmentation model - returns instance masks
//! let mut model = YOLOModel::load("yolo26n-seg.onnx")?;
//! let results = model.predict("image.jpg")?;
//! if let Some(ref masks) = results[0].masks {
//!     println!("Found {} instance masks", masks.len());
//! }
//!
//! // Pose model - returns keypoints
//! let mut model = YOLOModel::load("yolo26n-pose.onnx")?;
//! let results = model.predict("image.jpg")?;
//! if let Some(ref keypoints) = results[0].keypoints {
//!     println!("Found {} poses", keypoints.len());
//! }
//!
//! // OBB model - returns oriented bounding boxes
//! let mut model = YOLOModel::load("yolo26n-obb.onnx")?;
//! let results = model.predict("image.jpg")?;
//! if let Some(ref obb) = results[0].obb {
//!     println!("Found {} oriented boxes", obb.len());
//!     for i in 0..obb.len() {
//!         let conf = obb.conf()[i];
//!         let cls = obb.cls()[i] as usize;
//!         let name = results[0].names.get(&cls).map(|s| s.as_str()).unwrap_or("unknown");
//!         println!("  {} {:.2}", name, conf);
//!     }
//! }
//!
//! // Classification model - returns probabilities
//! let mut model = YOLOModel::load("yolo26n-cls.onnx")?;
//! let results = model.predict("image.jpg")?;
//! if let Some(ref probs) = results[0].probs {
//!     println!("Top-1: class {} ({:.1}%)", probs.top1(), probs.top1conf() * 100.0);
//! }
//!
//! // Semantic segmentation model (YOLO26 only) - returns a per-pixel class map
//! let mut model = YOLOModel::load("yolo26n-sem.onnx")?;
//! let results = model.predict("image.jpg")?;
//! if let Some(ref sem) = results[0].semantic_mask {
//!     println!("Semantic mask shape: {:?}", sem.data.shape());
//! }
//!
//! // Depth model (YOLO26 only) - returns a per-pixel depth map in meters
//! let mut model = YOLOModel::load("yolo26n-depth.onnx")?;
//! let results = model.predict("image.jpg")?;
//! if let Some(ref depth) = results[0].depth {
//!     println!("Depth map shape: {:?}", depth.data.shape());
//!     if let (Some(lo), Some(hi)) = (depth.min_depth(), depth.max_depth()) {
//!         println!("Depth range: {:.2}-{:.2} m", lo, hi);
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! Annotated depth is the colorized map blended over the image at `alpha = 0.6` with the
//! `jet` colormap and `disparity` normalization, matching Python's `Annotator.depth_map`.
//! [`annotate_image_with`](annotate::annotate_image_with) selects a different
//! [`Colormap`](visualizer::color::Colormap) or [`DepthViz`](visualizer::color::DepthViz);
//! the CLI always renders the default.
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
//!     .with_max_det(300) // Max detections per image
//!     .with_imgsz(640, 640);   // Input image size
//! ```
//!
//! ## Hardware Acceleration
//!
//! See the [CUDA / `TensorRT` acceleration guide](crate::cuda_guide) for setup,
//! requirements, and the zero-copy GPU preprocess fast path.
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
//! # NVIDIA GPU preprocess + zero-copy device input
//! # (requires CUDA toolkit; see docs/CUDA.md)
//! cargo build --release --features cuda-preprocess
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
//! - [`Boxes`] - Bounding boxes with `xyxy()`, `xywh()`, `xyxyn()`, `xywhn()`, `conf()`, `cls()` methods
//! - [`Masks`] - Segmentation masks with `data`, `orig_shape` fields
//! - [`Keypoints`] - Pose keypoints with `xy()`, `xyn()`, `conf()` methods
//! - [`Probs`] - Classification probabilities with `top1()`, `top5()`, `top1conf()`, `top5conf()` methods
//! - [`Obb`] - Oriented bounding boxes with `xyxyxyxy()`, `xywhr()`, `conf()`, `cls()` methods
//!
//! ## Module Overview
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`model`] | Core [`YOLOModel`] for loading models and running inference |
//! | [`results`] | Output types ([`Results`], [`Boxes`], [`Masks`], etc.) |
//! | [`inference`] | [`InferenceConfig`] for customizing inference settings |
//! | [`source`] | Input source handling ([`Source`], [`SourceIterator`]) |
//! | [`task`] | YOLO task types ([`Task`]: Detect, Segment, Pose, Classify, Obb, Semantic, Depth) |
//! | [`mod@error`] | Error types ([`InferenceError`], [`Result`]) |
//! | [`preprocessing`] | Image preprocessing utilities |
//! | [`postprocessing`] | Post-processing for all tasks (NMS/decode for detection; argmax for semantic segmentation; letterbox-crop resize for depth) |
//! | [`metadata`] | ONNX model metadata parsing |
//!
//! ## Feature Flags
//!
//! Default features (enabled unless `--no-default-features` is passed): `annotate`, `visualize`.
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `annotate` | Image annotation for `--save` (default) |
//! | `visualize` | Real-time window display for `--show` (default) |
//! | `video` | Video file decoding/encoding (requires `FFmpeg`) |
//! | `cuda` | NVIDIA CUDA acceleration |
//! | `tensorrt` | NVIDIA `TensorRT` optimization |
//! | `coreml` | Apple `CoreML` (macOS/iOS) |
//! | `openvino` | Intel `OpenVINO` |
//! | `onednn` | Intel oneDNN |
//! | `rocm` | AMD `ROCm` |
//! | `migraphx` | AMD `MIGraphX` |
//! | `directml` | `DirectML` (Windows) |
//! | `nnapi` | Android Neural Networks API |
//! | `qnn` | Qualcomm Neural Networks |
//! | `xnnpack` | XNNPACK (cross-platform) |
//! | `acl` | ARM Compute Library |
//! | `armnn` | ARM NN |
//! | `tvm` | Apache TVM |
//! | `rknpu` | Rockchip NPU |
//! | `cann` | Huawei CANN |
//! | `webgpu` | WebGPU |
//! | `azure` | Azure |
//! | `nvidia` | Convenience: `cuda` + `tensorrt` |
//! | `amd` | Convenience: `rocm` + `migraphx` |
//! | `intel` | Convenience: `openvino` + `onednn` |
//! | `mobile` | Convenience: `nnapi` + `coreml` + `qnn` |
//! | `all` | Convenience: `annotate` + `visualize` + `video` |
//!
//! ## License
//!
//! This project is dual-licensed under [AGPL-3.0](https://github.com/ultralytics/inference/blob/main/LICENSE)
//! for open-source use or [Ultralytics Enterprise License](https://ultralytics.com/license)
//! for commercial applications.

/// CUDA / `TensorRT` acceleration guide rendered from `docs/CUDA.md`.
#[allow(clippy::doc_markdown)]
#[doc = include_str!("../docs/CUDA.md")]
pub mod cuda_guide {}

// Modules
#[cfg(feature = "annotate")]
pub mod annotate;
pub mod device;
pub mod error;
pub mod inference;

pub mod logging;
pub mod metadata;
pub mod postprocessing;

// Palettes + pose skeleton. Pure data (the native window viewer inside is gated
// behind the `visualize` feature), so it compiles on wasm and the browser crate
// reuses the exact same colors/skeleton as the native annotator.
pub mod visualizer;

// Internal rayon/sequential abstraction. On native targets it re-exports the
// rayon prelude; on `wasm32` (no OS threads) it provides sequential shims so the
// shared preprocessing/postprocessing code compiles unchanged. See `src/parallel.rs`.
mod parallel;

// Backend and host-only modules. These depend on the native ONNX Runtime (`ort`),
// OS threads, filesystem, sockets, or windowing, none of which are available on
// `wasm32-unknown-unknown`. The browser build (the `ultralytics-inference-web`
// crate) reuses only the wasm-safe modules above plus its own `ort-web` session.
#[cfg(not(target_arch = "wasm32"))]
pub mod batch;
#[cfg(not(target_arch = "wasm32"))]
pub mod cli;
#[cfg(not(target_arch = "wasm32"))]
pub mod download;
#[cfg(not(target_arch = "wasm32"))]
pub mod io;
#[cfg(not(target_arch = "wasm32"))]
pub mod model;
#[cfg(not(target_arch = "wasm32"))]
pub mod source;

// CUDA-side preprocess + zero-copy device input - internal fast path used by
// `YOLOModel` when the `cuda-preprocess` feature is enabled and the device is
// CUDA/TensorRT. Gated by `InferenceConfig::with_cuda_preprocess(false)` to
// opt back into CPU preprocess.
#[cfg(feature = "cuda-preprocess")]
mod cuda_inference;
pub mod preprocessing;
pub mod results;
pub mod task;
pub mod utils;

// Re-export main types for convenience
pub use device::Device;
pub use error::{InferenceError, Result};
pub use inference::InferenceConfig;
#[cfg(not(target_arch = "wasm32"))]
pub use model::YOLOModel;
pub use results::{Boxes, DepthMap, Keypoints, Masks, Obb, Probs, Results, SemanticMask, Speed};
#[cfg(not(target_arch = "wasm32"))]
pub use source::{Source, SourceIterator, SourceMeta};
pub use task::Task;

// Re-export metadata for advanced use
pub use metadata::ModelMetadata;

// Re-export preprocessing utilities
pub use preprocessing::{PreprocessResult, preprocess_image, preprocess_image_with_precision};

/// Library version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Library name.
pub const NAME: &str = env!("CARGO_PKG_NAME");

/// Application display name.
pub const DISPLAY_NAME: &str = "Ultralytics Inference";

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
