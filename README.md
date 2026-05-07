<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# 🦀 Ultralytics YOLO Rust Inference

High-performance YOLO inference library written in Rust. This library provides a fast, safe, and efficient interface for running YOLO models using ONNX Runtime, with an API designed to match the [Ultralytics Python package](https://github.com/ultralytics/ultralytics).

[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://reddit.com/r/ultralytics)
[![codecov](https://codecov.io/gh/ultralytics/inference/graph/badge.svg?token=AVE5n6yvnf)](https://codecov.io/gh/ultralytics/inference)
[![CI](https://github.com/ultralytics/inference/actions/workflows/ci.yml/badge.svg)](https://github.com/ultralytics/inference/actions/workflows/ci.yml)
[![MSRV](https://img.shields.io/crates/msrv/ultralytics-inference?logo=rust&color=CE422B)](https://crates.io/crates/ultralytics-inference)

[![Crates.io](https://img.shields.io/crates/v/ultralytics-inference.svg)](https://crates.io/crates/ultralytics-inference)
[![docs.rs](https://img.shields.io/docsrs/ultralytics-inference)](https://docs.rs/ultralytics-inference)
![Crates.io Total Downloads](https://img.shields.io/crates/d/ultralytics-inference)

## ✨ Features

- 🚀 **High Performance** - Pure Rust implementation with zero-cost abstractions
- 🎯 **Ultralytics API Compatible** - `Results`, `Boxes`, `Masks`, `Keypoints`, `Probs` classes matching Python
- 🔧 **Multiple Backends** - CPU, CUDA, TensorRT, CoreML, OpenVINO, and more via ONNX Runtime
- 📦 **Dual Use** - Library for Rust projects + standalone CLI application
- 🏷️ **Auto Metadata** - Automatically reads class names, task type, and input size from ONNX models
- ⬇️ **Auto Download** - Automatically downloads YOLO11 and YOLO26 ONNX models (all sizes: n/s/m/l/x) when not found locally
- 🖼️ **Multiple Sources** - Images, directories, glob patterns, video files, webcams, and streams
- 🪶 **Lean Runtime** - No PyTorch, TensorFlow, or Python runtime required

## 🚀 Quick Start

### Prerequisites

- [Rust 1.88+](https://rustup.rs/) (install via rustup)
- A YOLO ONNX model (export from Ultralytics: `yolo export model=yolo26n.pt format=onnx`)

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
# Using Ultralytics CLI
yolo export model=yolo26n.pt format=onnx

# Or with Python
from ultralytics import YOLO
model = YOLO("yolo26n.pt")
model.export(format="onnx")
```

### Run Inference

```bash
# With defaults (auto-downloads yolo26n.onnx and sample images)
ultralytics-inference predict

# Select task — auto-downloads the nano model for that task
ultralytics-inference predict --task segment  # downloads yolo26n-seg.onnx
ultralytics-inference predict --task pose     # downloads yolo26n-pose.onnx
ultralytics-inference predict --task obb      # downloads yolo26n-obb.onnx
ultralytics-inference predict --task classify # downloads yolo26n-cls.onnx

# With explicit model (task is read from model metadata)
ultralytics-inference predict --model yolo26n.onnx --source image.jpg

# Auto-download any supported size (n/s/m/l/x)
ultralytics-inference predict --model yolo26l.onnx --source image.jpg
ultralytics-inference predict --model yolo11x-seg.onnx --source image.jpg

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
```

### Example Output

```
# ultralytics-inference predict

WARNING ⚠️ 'model' argument is missing. Using default '--model=yolo26n.onnx'.
WARNING ⚠️ 'source' argument is missing. Using default images: https://ultralytics.com/images/bus.jpg, https://ultralytics.com/images/zidane.jpg
Ultralytics 0.0.12 🚀 Rust ONNX FP32 CPU
Using ONNX Runtime CPUExecutionProvider
YOLO26n summary: 80 classes, imgsz=(640, 640)

image 1/2 /home/ultralytics/inference/bus.jpg: 640x480 640x480 4 persons, 1 bus, 36.4ms
image 2/2 /home/ultralytics/inference/zidane.jpg: 384x640 2 persons, 1 tie, 28.6ms
Speed: 1.5ms preprocess, 32.5ms inference, 0.5ms postprocess per image at shape (1, 3, 384, 640)
Results saved to runs/detect/predict1
💡 Learn more at https://docs.ultralytics.com/modes/predict
```

**With `--task` (auto-downloads the matching nano model):**

```bash
# ultralytics-inference predict --task segment

WARNING ⚠️ 'model' argument is missing. Using default '--model=yolo26n-seg.onnx'.
WARNING ⚠️ 'source' argument is missing. Using default images: https://ultralytics.com/images/bus.jpg, https://ultralytics.com/images/zidane.jpg
Ultralytics 0.0.12 🚀 Rust ONNX FP32 CPU
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

| Option          | Short | Description                                                                                              | Default                                 |
| --------------- | ----- | -------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| `--model`       | `-m`  | Path to ONNX model file; auto-downloaded if a known YOLO11/YOLO26 name                                   | `yolo26n.onnx`                          |
| `--task`        |       | Task type (`detect`, `segment`, `pose`, `obb`, `classify`); selects nano model when `--model` is omitted | `detect`                                |
| `--source`      | `-s`  | Input source (image, directory, glob, video, webcam index, or URL)                                       | `Task dependent Ultralytics URL assets` |
| `--conf`        |       | Confidence threshold                                                                                     | `0.25`                                  |
| `--iou`         |       | IoU threshold for NMS                                                                                    | `0.7`                                   |
| `--max-det`     |       | Maximum number of detections                                                                             | `300`                                   |
| `--imgsz`       |       | Inference image size                                                                                     | `Model metadata`                        |
| `--rect`        |       | Enable rectangular inference (minimal padding)                                                           | `true`                                  |
| `--batch`       |       | Batch size for inference                                                                                 | `1`                                     |
| `--half`        |       | Use FP16 half-precision inference                                                                        | `false`                                 |
| `--save`        |       | Save annotated results to runs/\<task\>/predict                                                          | `true`                                  |
| `--save-frames` |       | Save individual frames for video                                                                         | `false`                                 |
| `--show`        |       | Display results in a window                                                                              | `false`                                 |
| `--device`      |       | Device (cpu, cuda:0, mps, coreml, directml:0, openvino, tensorrt:0, xnnpack)                             | `cpu`                                   |
| `--verbose`     |       | Show verbose output                                                                                      | `true`                                  |
| `--classes`     |       | Filter by class IDs, e.g. `0` or `"0,1,2"` or `"[0, 1, 2]"`                                              | all classes                             |

**Task and Model Resolution:**

| Invocation                                        | Model used          | Notes                                                               |
| ------------------------------------------------- | ------------------- | ------------------------------------------------------------------- |
| `predict`                                         | `yolo26n.onnx`      | Default detect model, auto-downloaded                               |
| `predict --task segment`                          | `yolo26n-seg.onnx`  | Nano seg model, auto-downloaded                                     |
| `predict --task pose`                             | `yolo26n-pose.onnx` | Nano pose model, auto-downloaded                                    |
| `predict --task obb`                              | `yolo26n-obb.onnx`  | Nano OBB model, auto-downloaded                                     |
| `predict --task classify`                         | `yolo26n-cls.onnx`  | Nano classify model, auto-downloaded                                |
| `predict --model yolo26l-seg.onnx`                | `yolo26l-seg.onnx`  | Task read from model metadata                                       |
| `predict --task segment --model yolo26l-seg.onnx` | `yolo26l-seg.onnx`  | `--task` matches metadata, proceeds normally                        |
| `predict --task segment --model yolo26n.onnx`     | error               | `--task` conflicts with model metadata (`detect`), exits with error |

**Auto-downloadable models:**

All YOLO11 and YOLO26 ONNX models in sizes **n / s / m / l / x** across all five task variants are supported for auto-download:

| Family | Variants                                                                        |
| ------ | ------------------------------------------------------------------------------- |
| YOLO26 | `yolo26{n,s,m,l,x}.onnx`, `yolo26{n,s,m,l,x}-seg.onnx`, `-pose`, `-obb`, `-cls` |
| YOLO11 | `yolo11{n,s,m,l,x}.onnx`, `yolo11{n,s,m,l,x}-seg.onnx`, `-pose`, `-obb`, `-cls` |

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
ultralytics-inference = "0.0.12"
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
    // Select a device (e.g., CUDA, MPS, CPU)
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
│   ├── results.rs          # Results, Boxes, Masks, Keypoints, Probs, Obb
│   ├── preprocessing.rs    # Image preprocessing (letterbox, normalize, SIMD)
│   ├── postprocessing.rs   # Detection post-processing (NMS, decode, SIMD)
│   ├── metadata.rs         # ONNX model metadata parsing
│   ├── source.rs           # Input source handling (images, video, webcam)
│   ├── task.rs             # Task enum (Detect, Segment, Pose, Classify, Obb)
│   ├── inference.rs        # InferenceConfig
│   ├── batch.rs            # Batch processing pipeline
│   ├── device.rs           # Device enum (CPU, CUDA, MPS, CoreML, etc.)
│   ├── download.rs         # Model and asset downloading
│   ├── annotate.rs         # Image annotation (bounding boxes, masks, keypoints)
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
│   ├── bus.jpg
│   └── zidane.jpg
├── Cargo.toml              # Rust dependencies and features
├── LICENSE                 # AGPL-3.0 License
└── README.md               # This file
```

## ⚡ Hardware Acceleration

Enable hardware acceleration by adding features to your build:

```bash
# NVIDIA GPU (CUDA)
cargo build --release --features cuda

# NVIDIA TensorRT
cargo build --release --features tensorrt

# Apple CoreML (macOS/iOS)
cargo build --release --features coreml

# Intel OpenVINO
cargo build --release --features openvino

# Multiple features
cargo build --release --features "cuda,tensorrt"
```

**Available Features:**

| Feature    | Description                       |
| ---------- | --------------------------------- |
| `cuda`     | NVIDIA CUDA support               |
| `tensorrt` | NVIDIA TensorRT optimization      |
| `coreml`   | Apple CoreML (macOS/iOS)          |
| `openvino` | Intel OpenVINO                    |
| `onednn`   | Intel oneDNN                      |
| `rocm`     | AMD ROCm                          |
| `directml` | DirectML (Windows)                |
| `nnapi`    | Android Neural Networks API       |
| `xnnpack`  | XNNPACK (cross-platform)          |
| `nvidia`   | Convenience: CUDA + TensorRT      |
| `intel`    | Convenience: OpenVINO + oneDNN    |
| `mobile`   | Convenience: NNAPI + CoreML + QNN |

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

### Optional Dependencies (for `--save` feature)

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

To build without annotation support (smaller binary):

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

- [x] Detection, Segmentation, Pose, Classification, OBB inference
- [x] ONNX model metadata parsing (auto-detect classes, task, imgsz)
- [x] Hardware acceleration support (CUDA, TensorRT, CoreML, OpenVINO, XNNPACK)
- [x] Ultralytics-compatible Results API (`Boxes`, `Masks`, `Keypoints`, `Probs`, `Obb`)
- [x] Multiple input sources (images, directories, globs, URLs)
- [x] Video file support and webcam/RTSP streaming
- [x] Image annotation and visualization
- [x] FP16 half-precision inference
- [x] Batch inference support
- [x] Rectangular inference support and optimization
- [x] Class filtering support
- [x] Auto-download all YOLO11 and YOLO26 ONNX models (all sizes n/s/m/l/x, all tasks)
- [x] `--task` CLI flag: selects and auto-downloads the matching nano model when `--model` is omitted; errors on task/model metadata conflict

### In Progress

- [ ] Python bindings (PyO3)
- [ ] WebAssembly (WASM) support for browser inference

## 💡 Contributing

Ultralytics thrives on community collaboration! We deeply value your contributions.

- **Report Issues**: Found a bug? [Open an issue](https://github.com/ultralytics/inference/issues)
- **Feature Requests**: Have an idea? [Share it](https://github.com/ultralytics/inference/issues)
- **Pull Requests**: Read our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) first
- **Feedback**: Take our [Survey](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey)

## 📄 License

Ultralytics offers two licensing options:

- **AGPL-3.0 License**: Open-source license for students, researchers, and enthusiasts. See [LICENSE](LICENSE).
- **Enterprise License**: For commercial applications. Contact [Ultralytics Licensing](https://www.ultralytics.com/license).

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
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
