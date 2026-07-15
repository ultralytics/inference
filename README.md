<!-- Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license -->

<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# 🦀 Ultralytics YOLO Rust Inference

[English](README.md) | [简体中文](README.zh-CN.md)

High-performance YOLO inference library written in Rust. This library provides a fast, safe, and efficient interface for running YOLO models using ONNX Runtime, with an API designed to match the [Ultralytics Python package](https://github.com/ultralytics/ultralytics).

[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://www.reddit.com/r/Ultralytics/)
[![codecov](https://codecov.io/github/ultralytics/inference/branch/main/graph/badge.svg)](https://app.codecov.io/github/ultralytics/inference)
[![CI](https://github.com/ultralytics/inference/actions/workflows/ci.yml/badge.svg)](https://github.com/ultralytics/inference/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2606.03748-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.03748)

[![Crates.io](https://img.shields.io/crates/v/ultralytics-inference?logo=rust&logoColor=white&label=crates.io&color=CE422B)](https://crates.io/crates/ultralytics-inference)
[![docs.rs](https://img.shields.io/docsrs/ultralytics-inference?logo=docs.rs&logoColor=white&label=docs.rs&color=CE422B)](https://docs.rs/ultralytics-inference)
[![Downloads](https://img.shields.io/crates/d/ultralytics-inference?logo=rust&logoColor=white&label=downloads&color=CE422B)](https://crates.io/crates/ultralytics-inference)
[![MSRV](https://img.shields.io/crates/msrv/ultralytics-inference?logo=rust&logoColor=white&color=CE422B)](https://crates.io/crates/ultralytics-inference)
[![License](https://img.shields.io/crates/l/ultralytics-inference?label=license&color=blue)](https://github.com/ultralytics/inference/blob/main/LICENSE)
[![dependency status](https://deps.rs/repo/github/ultralytics/inference/status.svg)](https://deps.rs/repo/github/ultralytics/inference)

## ✨ Features

- 🚀 **High Performance** - Pure Rust implementation with zero-cost abstractions
- 🎯 **Ultralytics API Compatible** - `Results`, `Boxes`, `Masks`, `Keypoints`, `Probs`, `SemanticMask`, and `DepthMap` types matching the Python API shape
- 🔧 **Multiple Backends** - CPU, XNNPACK, CUDA, TensorRT, CoreML, OpenVINO, and more via ONNX Runtime
- 📦 **Dual Use** - Library for Rust projects + standalone CLI application
- 🏷️ **Auto Metadata** - Automatically reads class names, task type, and input size from ONNX models
- ⬇️ **Auto Download** - Downloads supported YOLO26, YOLO11, and YOLOv8 ONNX models (sizes: n/s/m/l/x) when not found locally
- 🖼️ **Multiple Sources** - Images, directories, glob patterns, video files, webcams, and streams
- 🪶 **Lean Runtime** - No PyTorch, TensorFlow, or Python runtime required

## ✨ Models

<a href="https://docs.ultralytics.com/tasks" target="_blank">
    <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/docs/ultralytics-yolov8-tasks-banner.avif" alt="Ultralytics YOLO supported tasks">
</a>
<br>
<br>

This crate runs [YOLOv8](https://docs.ultralytics.com/models/yolov8), [YOLO11](https://docs.ultralytics.com/models/yolo11), and [YOLO26](https://docs.ultralytics.com/models/yolo26) ONNX models. They are pretrained on [COCO](https://docs.ultralytics.com/datasets/detect/coco) for [Detection](https://docs.ultralytics.com/tasks/detect), [Segmentation](https://docs.ultralytics.com/tasks/segment), and [Pose Estimation](https://docs.ultralytics.com/tasks/pose); on [DOTA](https://docs.ultralytics.com/datasets/obb/dota-v2) for [OBB](https://docs.ultralytics.com/tasks/obb); on [Cityscapes](https://docs.ultralytics.com/datasets/semantic/cityscapes) for [Semantic Segmentation](https://docs.ultralytics.com/tasks/semantic); on [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet) for [Classification](https://docs.ultralytics.com/tasks/classify); and for monocular Depth Estimation<!-- TODO: re-add the tasks/depth docs link once the page is published --> (YOLO26 only). All [models](https://docs.ultralytics.com/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

## 🚀 Quick Start

### Prerequisites

- [Rust 1.89+](https://rustup.rs/) (install via rustup)
- A YOLO ONNX model (export from Ultralytics: `yolo export model=yolo26n.pt format=onnx`)

### System dependencies

Building from source (this includes `cargo install`) compiles native crates, so you need a C compiler. On Linux you additionally need `pkg-config` and the OpenSSL development headers, which the HTTPS model/asset downloader links against. macOS and Windows use their system TLS backends, so a C toolchain is all that is required.

```bash
# Debian/Ubuntu
sudo apt install build-essential pkg-config libssl-dev

# Fedora/RHEL
sudo dnf install gcc gcc-c++ pkgconf-pkg-config openssl-devel

# Arch
sudo pacman -S base-devel openssl pkgconf

# macOS (Xcode Command Line Tools provide the clang compiler)
xcode-select --install

# Windows: install the "Desktop development with C++" workload from
# Visual Studio Build Tools (https://visualstudio.microsoft.com/downloads/)
```

### Installation

```bash
# Install CLI globally from crates.io
cargo install ultralytics-inference

# Install CLI globally with custom features
# Minimal build (no default features)
cargo install ultralytics-inference --no-default-features

# Enable video support
cargo install ultralytics-inference --features video

# Enable multiple accelerators
cargo install ultralytics-inference --features "cuda,tensorrt"
```

### Development install

```bash
# Install CLI directly from the git repository
cargo install --git https://github.com/ultralytics/inference.git ultralytics-inference

# Or clone, build, and install from source
git clone https://github.com/ultralytics/inference.git
cd inference
cargo build --release

# Install from local checkout
cargo install --path . --locked
```

`cargo install` places binaries in Cargo's default bin directory:

- macOS/Linux: `~/.cargo/bin`
- Windows: `%USERPROFILE%\\.cargo\\bin`

Ensure this directory is in your `PATH`, then run from anywhere:

```bash
ultralytics-inference help
```

### Export a YOLO Model to ONNX

```bash
# Using Ultralytics CLI (FP32, default)
yolo export model=yolo26n.pt format=onnx

# FP16 (half precision) - ~50% smaller model
yolo export model=yolo26n.pt format=onnx quantize=16
```

```python
# Or with Python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.export(format="onnx")  # FP32 (default)
model.export(format="onnx", quantize=16)  # FP16 (half precision)
```

> **Precision / quantization:** Ultralytics ≥8.4 uses a single `quantize`
> argument instead of the deprecated `half=True` / `int8=True` flags. For ONNX
> the supported values are `32`/`fp32` (FP32, the default), `16`/`fp16` (FP16),
> and `8`/`int8` (INT8 - requires a calibration dataset via `data=`). The old
> `half=True` (→ `quantize=16`) and `int8=True` (→ `quantize=8`) still work but
> emit a deprecation warning. See the
> [export docs](https://docs.ultralytics.com/modes/export) and the
> [ONNX integration guide](https://docs.ultralytics.com/integrations/onnx).

### Run Inference

```bash
# With defaults (auto-downloads yolo26n.onnx and sample images)
ultralytics-inference predict

# Select task: auto-downloads the nano model for that task
ultralytics-inference predict --task segment  # downloads yolo26n-seg.onnx
ultralytics-inference predict --task pose     # downloads yolo26n-pose.onnx
ultralytics-inference predict --task obb      # downloads yolo26n-obb.onnx
ultralytics-inference predict --task classify # downloads yolo26n-cls.onnx
ultralytics-inference predict --task semantic # downloads yolo26n-sem.onnx (YOLO26 only)
ultralytics-inference predict --task depth    # downloads yolo26n-depth.onnx (YOLO26 only)

# With explicit model (task is read from model metadata)
ultralytics-inference predict --model yolo26n.onnx --source image.jpg

# Auto-download any supported size (n/s/m/l/x) across YOLO26, YOLO11, and YOLOv8
ultralytics-inference predict --model yolo26l.onnx --source image.jpg
ultralytics-inference predict --model yolo11x-seg.onnx --source image.jpg
ultralytics-inference predict --model yolov8n.onnx --source image.jpg

# On a directory of images
ultralytics-inference predict --model yolo26n.onnx --source assets/

# With custom thresholds
ultralytics-inference predict -m yolo26n.onnx -s image.jpg --conf 0.5 --iou 0.45

# Filter by class IDs
ultralytics-inference predict --model yolo26n.onnx --source image.jpg --classes 0
ultralytics-inference predict --model yolo26n.onnx --source image.jpg --classes "0,1,2"

# With visualization and custom image size
ultralytics-inference predict --model yolo26n.onnx --source video.mp4 --show --imgsz 1280

# Save individual frames for video input
ultralytics-inference predict --model yolo26n.onnx --source video.mp4 --save-frames

# Rectangular inference
ultralytics-inference predict --model yolo26n.onnx --source image.jpg --rect

# Semantic segmentation: write per-image PNG class maps to runs/semantic/predictN/results/
ultralytics-inference predict --task semantic --source cityscapes/ --save-json

# Depth estimation: save a colorized side-by-side (image | depth) to runs/depth/predictN/
ultralytics-inference predict --task depth --source image.jpg
```

### Example Output

```bash
ultralytics-inference predict
```

```text
WARNING ⚠️ 'model' argument is missing. Using default '--model=yolo26n.onnx'.
WARNING ⚠️ 'source' argument is missing. Using default images: https://ultralytics.com/images/bus.jpg, https://ultralytics.com/images/zidane.jpg
Ultralytics Inference 0.0.29 🚀 Rust ONNX FP32 CPU
Using ONNX Runtime CPUExecutionProvider
YOLO26n summary: 80 classes, imgsz=(640, 640)

image 1/2 /home/ultralytics/inference/bus.jpg: 640x480 4 persons, 1 bus, 36.4ms
image 2/2 /home/ultralytics/inference/zidane.jpg: 384x640 2 persons, 1 tie, 28.6ms
Speed: 1.5ms preprocess, 32.5ms inference, 0.5ms postprocess per image at shape (1, 3, 384, 640)
Results saved to runs/detect/predict1
💡 Learn more at https://docs.ultralytics.com/modes/predict
```

**With `--task` (auto-downloads the matching nano model):**

```bash
ultralytics-inference predict --task segment
```

```text
WARNING ⚠️ 'model' argument is missing. Using default '--model=yolo26n-seg.onnx'.
WARNING ⚠️ 'source' argument is missing. Using default images: https://ultralytics.com/images/bus.jpg, https://ultralytics.com/images/zidane.jpg
Ultralytics Inference 0.0.29 🚀 Rust ONNX FP32 CPU
Using ONNX Runtime CPUExecutionProvider
YOLO26n-seg summary: 80 classes, imgsz=(640, 640)

image 1/2 /home/ultralytics/inference/bus.jpg: 640x480 4 persons, 1 bus, 48.2ms
image 2/2 /home/ultralytics/inference/zidane.jpg: 384x640 2 persons, 1 tie, 38.1ms
Speed: 1.6ms preprocess, 44.3ms inference, 1.2ms postprocess per image at shape (1, 3, 384, 640)
Results saved to runs/segment/predict1
💡 Learn more at https://docs.ultralytics.com/modes/predict
```

## 📚 Usage

### As a CLI Tool

```bash
# Show help
ultralytics-inference help

# Show version
ultralytics-inference version

# Run inference
ultralytics-inference predict --model <model.onnx> --source <source>
```

`--help` and `--version` are also supported as standard flag aliases.

**CLI Options:**

| Option          | Short | Description                                                                                                                                                                    | Default                               |
| --------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------- |
| `--model`       | `-m`  | Path to ONNX model file; auto-downloaded if a known YOLOv8/YOLO11/YOLO26 name                                                                                                  | `yolo26n.onnx`                        |
| `--task`        |       | Task type (`detect`, `segment`, `pose`, `obb`, `classify`, `semantic`\*, `depth`\*); selects nano model when `--model` is omitted                                              | `detect`                              |
| `--source`      | `-s`  | Input source (image, directory, glob, video, webcam index, or URL)                                                                                                             | Task-dependent Ultralytics URL assets |
| `--conf`        |       | Confidence threshold                                                                                                                                                           | `0.25`                                |
| `--iou`         |       | IoU threshold for NMS                                                                                                                                                          | `0.7`                                 |
| `--max-det`     |       | Maximum number of detections                                                                                                                                                   | `300`                                 |
| `--imgsz`       |       | Inference image size                                                                                                                                                           | `Model metadata`                      |
| `--rect`        |       | Enable rectangular inference (minimal padding)                                                                                                                                 | `true`                                |
| `--batch`       |       | Batch size for inference                                                                                                                                                       | `1`                                   |
| `--half`        |       | Use FP16 half-precision inference                                                                                                                                              | `false`                               |
| `--save`        |       | Save annotated results to runs/\<task\>/predict                                                                                                                                | `true`                                |
| `--save-frames` |       | Save individual frames for video input (instead of video file)                                                                                                                 | `false`                               |
| `--save-json`   |       | Save semantic segmentation class-map PNGs for external evaluation                                                                                                              | `false`                               |
| `--colormap`    |       | Depth colormap: `jet`, `inferno`, `spectral`, or `gray` (depth task only)                                                                                                      | `jet`                                 |
| `--show`        |       | Display results in a window                                                                                                                                                    | `false`                               |
| `--device`      |       | Device string, e.g. cpu, cuda:0, coreml, directml:0, openvino, tensorrt:0, rocm:0, xnnpack; additional providers selectable when their feature is enabled (see Features table) | `cpu`                                 |
| `--verbose`     |       | Show verbose output                                                                                                                                                            | `true`                                |
| `--classes`     |       | Filter by class IDs, e.g. `0` or `"0,1,2"` or `"[0, 1, 2]"`                                                                                                                    | all classes                           |

**Task and Model Resolution:**

| Invocation                                        | Model used             | Notes                                                               |
| ------------------------------------------------- | ---------------------- | ------------------------------------------------------------------- |
| `predict`                                         | `yolo26n.onnx`         | Default detect model, auto-downloaded                               |
| `predict --task segment`                          | `yolo26n-seg.onnx`     | Nano seg model, auto-downloaded                                     |
| `predict --task pose`                             | `yolo26n-pose.onnx`    | Nano pose model, auto-downloaded                                    |
| `predict --task obb`                              | `yolo26n-obb.onnx`     | Nano OBB model, auto-downloaded                                     |
| `predict --task classify`                         | `yolo26n-cls.onnx`     | Nano classify model, auto-downloaded                                |
| `predict --task semantic`                         | `yolo26n-sem.onnx`\*   | Nano semantic segmentation model, auto-downloaded (YOLO26 only)     |
| `predict --task depth`                            | `yolo26n-depth.onnx`\* | Nano depth estimation model, auto-downloaded (YOLO26 only)          |
| `predict --model yolo26l-seg.onnx`                | `yolo26l-seg.onnx`     | Task read from model metadata                                       |
| `predict --task segment --model yolo26l-seg.onnx` | `yolo26l-seg.onnx`     | `--task` matches metadata, proceeds normally                        |
| `predict --task segment --model yolo26n.onnx`     | error                  | `--task` conflicts with model metadata (`detect`), exits with error |

\* `semantic` (semantic segmentation) and `depth` (depth estimation) are YOLO26-only.

**Auto-downloadable models:**

YOLOv8, YOLO11, and YOLO26 ONNX models in sizes **n / s / m / l / x** are supported for auto-download across the standard task variants. YOLO26 also includes `-sem` for semantic segmentation and `-depth` for depth estimation:

| Family | Variants                                                                                              |
| ------ | ----------------------------------------------------------------------------------------------------- |
| YOLO26 | `yolo26{n,s,m,l,x}.onnx`, `yolo26{n,s,m,l,x}-seg.onnx`, `-pose`, `-obb`, `-cls`, `-sem`\*, `-depth`\* |
| YOLO11 | `yolo11{n,s,m,l,x}.onnx`, `yolo11{n,s,m,l,x}-seg.onnx`, `-pose`, `-obb`, `-cls`                       |
| YOLOv8 | `yolov8{n,s,m,l,x}.onnx`, `yolov8{n,s,m,l,x}-seg.onnx`, `-pose`, `-obb`, `-cls`                       |

\* `-sem` (semantic segmentation) and `-depth` (depth estimation) are YOLO26-only.

**Source Options:**

| Source Type | Example Input                   | Description                       |
| ----------- | ------------------------------- | --------------------------------- |
| Image       | `image.jpg`                     | Single image file                 |
| Directory   | `images/`                       | Directory of images               |
| Glob        | `images/*.jpg`                  | Glob pattern for images           |
| Video       | `video.mp4`                     | Video file                        |
| Webcam      | `0`,`1`                         | Webcam index (0 = default webcam) |
| URL         | `https://example.com/image.jpg` | Remote image URL                  |

### As a Rust Library

Add to your `Cargo.toml` (choose one):

```toml
# Stable release from crates.io
[dependencies]
ultralytics-inference = "0.0.29"
```

```toml
# Development version (latest unreleased code from GitHub)
[dependencies]
ultralytics-inference = { git = "https://github.com/ultralytics/inference.git" }
```

**Basic Usage:**

```rust
use ultralytics_inference::{YOLOModel, InferenceConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load model - metadata (classes, task, imgsz) is read automatically
    let mut model = YOLOModel::load("yolo26n.onnx")?;

    // Run inference
    let results = model.predict("image.jpg")?;

    // Process results
    for result in &results {
        if let Some(ref boxes) = result.boxes {
            println!("Found {} detections", boxes.len());
            for i in 0..boxes.len() {
                let cls = boxes.cls()[i] as usize;
                let conf = boxes.conf()[i];
                let name = result.names.get(&cls).map(|s| s.as_str()).unwrap_or("unknown");
                println!("  {} {:.2}", name, conf);
            }
        }
    }

    Ok(())
}
```

**With Custom Configuration:**

```rust
use ultralytics_inference::{YOLOModel, InferenceConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = InferenceConfig::new()
        .with_confidence(0.5)
        .with_iou(0.45)
        .with_max_det(300);

    let mut model = YOLOModel::load_with_config("yolo26n.onnx", config)?;
    let results = model.predict("image.jpg")?;

    Ok(())
}
```

**Accessing Detection Data:**

```rust
if let Some(ref boxes) = result.boxes {
    // Bounding boxes in different formats
    let xyxy = boxes.xyxy();      // [x1, y1, x2, y2]
    let xywh = boxes.xywh();      // [x_center, y_center, width, height]
    let xyxyn = boxes.xyxyn();    // Normalized [0-1]
    let xywhn = boxes.xywhn();    // Normalized [0-1]

    // Confidence scores and class IDs
    let conf = boxes.conf();      // Confidence scores
    let cls = boxes.cls();        // Class IDs
}
```

**Selecting a Device:**

```rust
use ultralytics_inference::{Device, InferenceConfig, YOLOModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Select a device (e.g., CUDA, CoreML, CPU)
    let device = Device::Cuda(0);

    // Configure the model to use this device
    let config = InferenceConfig::new().with_device(device);

    let mut model = YOLOModel::load_with_config("yolo26n.onnx", config)?;
    let results = model.predict("image.jpg")?;

    Ok(())
}
```

## 🗂️ Project Structure

```text
inference/
├── src/
│   ├── lib.rs              # Library entry point and public exports
│   ├── main.rs             # CLI application
│   ├── model.rs            # YOLOModel - ONNX session and inference
│   ├── results.rs          # Results, Boxes, Masks, Keypoints, Probs, Obb, SemanticMask, DepthMap
│   ├── preprocessing.rs    # Image preprocessing (letterbox, normalize, SIMD)
│   ├── postprocessing.rs   # Post-processing for all tasks (NMS/decode for detection, argmax for semantic, resize for depth)
│   ├── metadata.rs         # ONNX model metadata parsing
│   ├── source.rs           # Input source handling (images, video, webcam)
│   ├── task.rs             # Task enum (Detect, Segment, Pose, Classify, Obb, Semantic, Depth)
│   ├── inference.rs        # InferenceConfig
│   ├── batch.rs            # Batch processing pipeline
│   ├── device.rs           # Device enum (CPU, CUDA, CoreML, etc.)
│   ├── cuda_inference.rs   # Fused CUDA preprocess kernel (cuda-preprocess feature)
│   ├── parallel.rs         # Rayon parallelism shims (sequential on wasm)
│   ├── download.rs         # Model and asset downloading
│   ├── annotate.rs         # Image annotation (bounding boxes, instance masks, keypoints, semantic overlay, depth colormap)
│   ├── io.rs               # Result saving (images, videos)
│   ├── logging.rs          # Logging macros
│   ├── error.rs            # Error types
│   ├── utils.rs            # Utility functions (NMS, IoU)
│   ├── cli/                # CLI module
│   │   ├── mod.rs          # CLI module exports
│   │   ├── args.rs         # CLI argument parsing
│   │   └── predict.rs      # Predict command implementation
│   └── visualizer/         # Real-time visualization (minifb)
├── tests/
│   └── integration_test.rs # Integration tests
├── assets/                 # Test images
│   ├── boats.jpg
│   ├── bus.jpg
│   └── zidane.jpg
├── Cargo.toml              # Rust dependencies and features
├── LICENSE                 # AGPL-3.0 License
├── README.md               # English README
└── README.zh-CN.md         # Simplified Chinese README
```

## ⚡ Hardware Acceleration

Enable hardware acceleration by adding features to your build:

```bash
# NVIDIA GPU (CUDA)
cargo build --release --features cuda

# NVIDIA TensorRT
cargo build --release --features tensorrt

# NVIDIA GPU preprocessing + zero-copy TensorRT input (fastest; needs CUDA toolkit)
cargo build --release --features cuda-preprocess

# Apple CoreML (macOS/iOS)
cargo build --release --features coreml

# Intel OpenVINO
cargo build --release --features openvino

# Multiple features
cargo build --release --features "cuda,tensorrt"
```

> NVIDIA setup, requirements, and the GPU preprocessing fast path are documented in [`docs/CUDA.md`](docs/CUDA.md).

**Available Features:**

Default features (enabled unless `--no-default-features` is passed): `annotate`, `visualize`.

| Feature           | Description                                                                                           |
| ----------------- | ----------------------------------------------------------------------------------------------------- |
| `annotate`        | Image annotation for `--save` (default)                                                               |
| `visualize`       | Real-time window display for `--show` (default)                                                       |
| `video`           | Video file decoding/encoding (requires FFmpeg)                                                        |
| `cuda`            | NVIDIA CUDA support                                                                                   |
| `tensorrt`        | NVIDIA TensorRT optimization                                                                          |
| `cuda-preprocess` | GPU preprocessing + zero-copy TensorRT input (needs CUDA toolkit; see [`docs/CUDA.md`](docs/CUDA.md)) |
| `coreml`          | Apple CoreML (macOS/iOS)                                                                              |
| `openvino`        | Intel OpenVINO                                                                                        |
| `onednn`          | Intel oneDNN                                                                                          |
| `rocm`            | AMD ROCm                                                                                              |
| `migraphx`        | AMD MIGraphX                                                                                          |
| `directml`        | DirectML (Windows)                                                                                    |
| `nnapi`           | Android Neural Networks API                                                                           |
| `qnn`             | Qualcomm Neural Networks                                                                              |
| `xnnpack`         | XNNPACK (cross-platform)                                                                              |
| `acl`             | ARM Compute Library                                                                                   |
| `armnn`           | ARM NN                                                                                                |
| `tvm`             | Apache TVM                                                                                            |
| `rknpu`           | Rockchip NPU                                                                                          |
| `cann`            | Huawei CANN                                                                                           |
| `webgpu`          | WebGPU                                                                                                |
| `azure`           | Azure                                                                                                 |
| `nvidia`          | Convenience: CUDA + TensorRT                                                                          |
| `amd`             | Convenience: ROCm + MIGraphX                                                                          |
| `intel`           | Convenience: OpenVINO + oneDNN                                                                        |
| `mobile`          | Convenience: NNAPI + CoreML + QNN                                                                     |
| `all`             | Convenience: annotate + visualize + video                                                             |

## 🌐 Browser / WebGPU (WASM)

[![npm version](https://img.shields.io/npm/v/@ultralytics/yolo?logo=npm&logoColor=white&label=npm&color=CB3837)](https://www.npmjs.com/package/@ultralytics/yolo)
[![npm downloads](https://img.shields.io/npm/dm/@ultralytics/yolo?logo=npm&logoColor=white&label=downloads&color=CB3837)](https://www.npmjs.com/package/@ultralytics/yolo)

The same engine runs in the browser on **WebGPU**, compiled to WebAssembly. The
shared Rust preprocessing and postprocessing run in wasm, so results match the
native path, while the forward pass runs on a pluggable backend: the official
ONNX Runtime Web build (bridged through [`ort-web`](https://ort.pyke.io/backends/web))
for `.onnx` models, or [**LiteRT.js**](https://developers.google.com/edge/litert/web)
for `.tflite` models.

It ships as the [`@ultralytics/yolo`](web/README.md) npm package:

```ts
import { YOLO } from "@ultralytics/yolo";

const model = await YOLO.load("yolo26n.onnx");
const results = await model.predict("bus.jpg");
console.log(results.boxes); // [{ x1, y1, x2, y2, conf, cls, name, color }, ...]
```

Pass `{ device: "webgpu" | "cpu" }` to pick the accelerator (`"auto"` is the
default), and read `model.device` to see what actually ran.

The backend is picked automatically from the model format (its extension when
available, otherwise the model bytes), so switching is just a matter of the model
you load. LiteRT.js (Google's LiteRT for Web) is optional and
often **~2× faster than ONNX Runtime Web on WebGPU** — point `YOLO.load` at an
Ultralytics `.tflite` export and `npm install @litertjs/core` alongside the package.
See the [LiteRT.js section](web/README.md#-litertjs-backend) for details.

The browser bindings live in [`crates/web`](crates/web) (the
`ultralytics-inference-web` cdylib); the JS/TS wrapper and build instructions are
in [`web/`](web/README.md). A WebGPU-capable browser and a secure context
(`https`/`localhost`) are required.

## 📦 Dependencies

One of the key benefits of this library is a Rust/ONNX Runtime stack with no PyTorch, TensorFlow, or Python runtime required.

### Core Dependencies (always included)

| Crate               | Purpose                         |
| ------------------- | ------------------------------- |
| `ort`               | ONNX Runtime bindings           |
| `ndarray`           | N-dimensional arrays            |
| `image`             | Image loading/decoding          |
| `jpeg-decoder`      | JPEG decoding                   |
| `fast_image_resize` | SIMD-optimized resizing         |
| `half`              | FP16 support                    |
| `lru`               | LRU cache for preprocessing LUT |
| `wide`              | SIMD for fast preprocessing     |

### Optional Dependencies (for the `annotate` feature)

| Crate       | Purpose                        |
| ----------- | ------------------------------ |
| `imageproc` | Drawing boxes and shapes       |
| `ab_glyph`  | Text rendering (embedded font) |

### Optional Dependencies (for Video & Visualization)

| Crate      | Purpose                            |
| ---------- | ---------------------------------- |
| `minifb`   | Window creation and buffer display |
| `video-rs` | Video decoding/encoding (ffmpeg)   |

### Video Support (FFmpeg)

Video features require FFmpeg (7 or 8) installed on your system:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
apt-get install -y ffmpeg libavutil-dev libavformat-dev libavfilter-dev libavdevice-dev libclang-dev

# Build with video support
cargo build --release --features video
```

To build without annotation and visualization support (smaller binary):

```bash
cargo build --release --no-default-features
```

## 🧪 Testing

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_boxes_creation
```

## 📊 Performance

Benchmarks on Apple M4 MacBook Pro (CPU, ONNX Runtime):

### YOLO26n Detection Model (640x640)

| Precision | Model Size | Preprocess | Inference | Postprocess | Total |
| --------- | ---------- | ---------- | --------- | ----------- | ----- |
| FP32      | 10.2 MB    | ~9ms       | ~21ms     | <1ms        | ~31ms |
| FP16      | 5.2 MB     | ~9ms       | ~24ms     | <1ms        | ~34ms |

**Key findings:**

- **FP16 models are ~50% smaller** (5.2 MB vs 10.2 MB)
- **FP32 is slightly faster on CPU** (~21ms vs ~24ms) due to CPU's native FP32 support
- FP16 requires upcasting to FP32 for computation on most CPUs, adding overhead
- Use **FP32 for CPU** inference, **FP16 for GPU** (where it provides speedup)

### Threading Optimization

ONNX Runtime threading is set to auto (`num_threads: 0`) which lets ORT choose optimal thread count:

- Manual threading (4 threads): ~40ms inference
- Auto threading (0 = ORT decides): ~21ms inference

## 🔮 Roadmap

### Completed

- [x] Detection, Segmentation, Pose, Classification, OBB, Semantic Segmentation, and Depth Estimation inference
- [x] ONNX model metadata parsing (auto-detect classes, task, imgsz)
- [x] Hardware acceleration support (CUDA, TensorRT, CoreML, OpenVINO, XNNPACK)
- [x] Ultralytics-compatible Results API (`Boxes`, `Masks`, `Keypoints`, `Probs`, `Obb`, `SemanticMask`, `DepthMap`)
- [x] Multiple input sources (images, directories, globs, URLs)
- [x] Video file support and webcam/RTSP streaming
- [x] Image annotation and visualization
- [x] FP16 half-precision inference
- [x] Batch inference support
- [x] Rectangular inference support and optimization
- [x] Class filtering support
- [x] Auto-download all YOLO26, YOLO11, and YOLOv8 ONNX models (all sizes n/s/m/l/x, all tasks)
- [x] `--task` CLI flag: selects and auto-downloads the matching nano model when `--model` is omitted; errors on task/model metadata conflict
- [x] WebAssembly (WASM) browser inference on WebGPU (npm [`@ultralytics/yolo`](https://www.npmjs.com/package/@ultralytics/yolo))

### In Progress

- [ ] Python bindings (PyO3)

## 💡 Contributing

Ultralytics thrives on community collaboration, and we deeply value your contributions! Whether it's reporting bugs,
suggesting features, or submitting code changes, your involvement is crucial.

- **Report Issues**: Found a bug? [Open an issue](https://github.com/ultralytics/inference/issues)
- **Feature Requests**: Have an idea? [Share it](https://github.com/ultralytics/inference/issues)
- **Pull Requests**: Read our [Contributing Guide](https://docs.ultralytics.com/help/contributing) first
- **Feedback**: Take our [Survey](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey)

A heartfelt thank you 🙏 goes out to all our contributors! Your efforts help make Ultralytics tools better for everyone.

[![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/ultralytics/graphs/contributors)

## 📜 License

Ultralytics offers two licensing options to suit different needs:

- **AGPL-3.0 License**: This [OSI-approved](https://opensource.org/license/agpl-3.0) open-source license is perfect for students, researchers, and enthusiasts. It encourages open collaboration and knowledge sharing. See the [LICENSE](https://github.com/ultralytics/inference/blob/main/LICENSE) file for full details.
- **Ultralytics Enterprise License**: Designed for commercial use, this license allows for the seamless integration of Ultralytics software and AI models into commercial products and services, bypassing the open-source requirements of AGPL-3.0. If your use case involves commercial deployment, please contact us via [Ultralytics Licensing](https://www.ultralytics.com/license).

## 📮 Contact

- **GitHub Issues**: [Bug reports and feature requests](https://github.com/ultralytics/inference/issues)
- **Discord**: [Join our community](https://discord.com/invite/ultralytics)
- **Documentation**: [docs.ultralytics.com](https://docs.ultralytics.com)

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://x.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
