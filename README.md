<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# 🦀 Ultralytics YOLO Rust Inference

High-performance YOLO inference library written in Rust. This library provides a fast, safe, and efficient interface for running YOLO models using ONNX Runtime, with an API designed to match the [Ultralytics Python package](https://github.com/ultralytics/ultralytics).

[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://reddit.com/r/ultralytics)

## ✨ Features

- 🚀 **High Performance** - Pure Rust implementation with zero-cost abstractions
- 🎯 **Ultralytics API Compatible** - `Results`, `Boxes`, `Masks`, `Keypoints`, `Probs` classes matching Python
- 🔧 **Multiple Backends** - CPU, CUDA, TensorRT, CoreML, OpenVINO, and more via ONNX Runtime
- 📦 **Dual Use** - Library for Rust projects + standalone CLI application
- 🏷️ **Auto Metadata** - Automatically reads class names, task type, and input size from ONNX models
- 🖼️ **Multiple Sources** - Images, directories, glob patterns, video files, webcams, and streams
- 🪶 **Minimal Dependencies** - No PyTorch, no heavy ML frameworks - just 5 core crates

## 🚀 Quick Start

### Prerequisites

- [Rust 1.85+](https://rustup.rs/) (install via rustup, edition 2024 required)
- A YOLO ONNX model (export from Ultralytics: `yolo export model=yolo11n.pt format=onnx`)

### Installation

```bash
# Clone the repository
git clone https://github.com/ultralytics/inference.git
cd inference

# Build release version
cargo build --release
```

### Export a YOLO Model to ONNX

```bash
# Using Ultralytics CLI
yolo export model=yolo11n.pt format=onnx

# Or with Python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.export(format="onnx")
```

### Run Inference

```bash
# With defaults (auto-downloads model and sample images)
cargo run --release -- predict

# With explicit arguments
cargo run --release -- predict --model yolo11n.onnx --source image.jpg

# On a directory of images
cargo run --release -- predict --model yolo11n.onnx --source assets/

# With custom thresholds
cargo run --release -- predict -m yolo11n.onnx -s image.jpg --conf 0.5 --iou 0.45

# With visualization and custom image size
cargo run --release -- predict --model yolo11n.onnx --source video.mp4 --show --imgsz 1280
```

### Example Output

```
WARNING ⚠️ 'source' argument is missing. Using default images: https://ultralytics.com/images/bus.jpg, https://ultralytics.com/images/zidane.jpg
Ultralytics 0.0.4 🚀 Rust ONNX CPU
YOLO11 summary: 80 classes, imgsz=(640, 640)

image 1/2 bus.jpg: 810x1080 4 persons, 1 bus, 27.3ms
image 2/2 zidane.jpg: 1280x720 2 persons, 1 tie, 24.9ms
Speed: 9.4ms preprocess, 26.1ms inference, 0.8ms postprocess per image at shape (1, 3, 720, 1280)
💡 Learn more at https://docs.ultralytics.com/modes/predict
```

## 📚 Usage

### As a CLI Tool

```bash
# Show help
cargo run --release -- help

# Show version
cargo run --release -- version

# Run inference
cargo run --release -- predict --model <model.onnx> --source <source>
```

**CLI Options:**

| Option      | Short | Description                                       | Default                                 |
| ----------- | ----- | ------------------------------------------------- | --------------------------------------- |
| `--model`   | `-m`  | Path to ONNX model file                           | `yolo11n.onnx`                          |
| `--source`  | `-s`  | Input source (image, video, webcam index, or URL) | `Task dependent Ultralytics URL assets` |
| `--device`  |       | Device to use (cpu, cuda:0, mps, coreml, etc.)    | `cpu`                                   |
| `--conf`    |       | Confidence threshold                              | `0.25`                                  |
| `--iou`     |       | IoU threshold for NMS                             | `0.45`                                  |
| `--imgsz`   |       | Inference image size                              | `Model metadata`                        |
| `--half`    |       | Use FP16 half-precision inference                 | `false`                                 |
| `--save`    |       | Save annotated images to runs/<task>/predict      | `false`                                 |
| `--show`    |       | Display results in a window                       | `false`                                 |
| `--verbose` |       | Show verbose output                               | `true`                                  |

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

Add to your `Cargo.toml`:

```toml
[dependencies]
ultralytics-inference = { git = "https://github.com/ultralytics/inference.git" }
```

**Basic Usage:**

```rust
use ultralytics_inference::{YOLOModel, InferenceConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load model - metadata (classes, task, imgsz) is read automatically
    let mut model = YOLOModel::load("yolo11n.onnx")?;

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
        .with_max_detections(100);

    let mut model = YOLOModel::load_with_config("yolo11n.onnx", config)?;
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

    let mut model = YOLOModel::load_with_config("yolo11n.onnx", config)?;
    let results = model.predict("image.jpg")?;

    Ok(())
}
```

## 🗂️ Project Structure

```
inference/
├── src/
│   ├── lib.rs              # Library entry point and public exports
│   ├── main.rs             # CLI application
│   ├── model.rs            # YOLOModel - ONNX session and inference
│   ├── results.rs          # Results, Boxes, Masks, Keypoints, Probs, Obb
│   ├── preprocessing.rs    # Image preprocessing (letterbox, normalize)
│   ├── postprocessing.rs   # Detection post-processing (NMS, decode)
│   ├── metadata.rs         # ONNX model metadata parsing
│   ├── source.rs           # Input source handling
│   ├── task.rs             # Task enum (Detect, Segment, Pose, etc.)
│   ├── inference.rs        # InferenceConfig
│   ├── download.rs         # Model and asset downloading
│   ├── visualizer/         # Visualization tools (Viewer)
│   ├── error.rs            # Error types
│   └── utils.rs            # Utility functions (NMS, IoU)
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

One of the key benefits of this library is **minimal dependencies** - no PyTorch, TensorFlow, or heavy ML frameworks required.

### Core Dependencies (always included)

| Crate               | Purpose                 |
| ------------------- | ----------------------- |
| `ort`               | ONNX Runtime bindings   |
| `ndarray`           | N-dimensional arrays    |
| `image`             | Image loading/decoding  |
| `fast_image_resize` | SIMD-optimized resizing |
| `half`              | FP16 support            |

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

### YOLO11n Detection Model (640x640)

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
- [x] Ultralytics-compatible Results API (`Boxes`, `Masks`, `Keypoints`, `Probs`, `Obb`)
- [x] Multiple input sources (images, directories, globs, URLs)
- [x] Video file support and webcam/RTSP streaming
- [x] Image annotation and visualization
- [x] FP16 half-precision inference

### In Progress

- [ ] Python bindings (PyO3)
- [ ] Batch inference optimization
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
  <a href="https://youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
