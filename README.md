<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# 🛠 Ultralytics Inference

Welcome to Ultralytics Inference! This repository provides a high-performance Rust library and CLI application for running YOLO models. Built with Rust for maximum performance and safety, it can be used as both a standalone application and a reusable library for Rust and Python projects.

**Key Features:**

- 🚀 **Modular Architecture:** Choose only the acceleration backends you need (CUDA, TensorRT, OpenVINO, CoreML, etc.)
- ⚡ **High Performance:** Built in Rust for zero-cost abstractions and memory safety
- 🎯 **Multiple Deployment Options:** Use as a library or standalone CLI
- 🔧 **Flexible:** Support for CPU, GPU, and specialized hardware accelerators

📖 **[Feature Flags Guide](FEATURES.md)** - Learn how to enable specific hardware acceleration features.

Explore our [Ultralytics Solutions](https://www.ultralytics.com/solutions) to see how we apply AI in real-world applications.

[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://reddit.com/r/ultralytics)

## 🗂️ Repository Structure

This project is organized following Rust best practices with a clear separation between library and application code:

- `src/lib.rs`: Core library code that can be reused in other Rust projects or exposed to Python via PyO3
- `src/main.rs`: CLI application entry point for running YOLO inference from the command line
- `tests/`: Integration tests for the library
- `docs/`: Project documentation
- `Cargo.toml`: Rust project configuration, dependencies, and metadata
- `.gitignore`: Configured to exclude Rust build artifacts (`target/`), editor files, etc.
- `LICENSE`: Open-source license (AGPL-3.0)

```plaintext
inference/
│
├── src/
│   ├── lib.rs              # Library entry point
│   ├── main.rs             # CLI application
│   ├── model/              # YOLO model implementations
│   ├── inference/          # Inference engine
│   └── utils/              # Utility functions
│
├── tests/                  # Integration tests
│   └── integration_test.rs
│
├── docs/                   # Documentation
│   └── README.md
│
├── Cargo.toml              # Rust project configuration
├── .gitignore              # Git ignore rules
├── LICENSE                 # Project license
└── README.md               # This file
```

### 📦 Source Code Directory (`src/`)

The `src/` directory contains the core Rust code for the YOLO inference engine. It's organized into:

- **`lib.rs`**: The library entry point, exposing public APIs for use in other Rust projects or Python bindings
- **`main.rs`**: The CLI application for running inference from the command line
- Additional modules for model loading, inference execution, and utilities

### 🧪 Testing Directory (`tests/`)

Integration tests ensure the reliability of the inference engine across different models and inputs. Unit tests are typically co-located with source files in Rust.

### 📚 Documentation Directory (`docs/`)

Comprehensive documentation for using the library and CLI, including examples and API references.

## ✨ Getting Started

### Prerequisites

- Rust 1.70 or later (install via [rustup](https://rustup.rs/))

### Installation

#### As a Library

Add to your `Cargo.toml` with the features you need:

```toml
[dependencies]
# Basic installation (CPU only)
inference = { git = "https://github.com/ultralytics/inference.git" }

# With NVIDIA GPU support
inference = { git = "https://github.com/ultralytics/inference.git", features = ["cuda"] }

# With multiple accelerators
inference = { git = "https://github.com/ultralytics/inference.git", features = ["cuda", "tensorrt"] }

# Convenience feature groups
inference = { git = "https://github.com/ultralytics/inference.git", features = ["nvidia"] }  # CUDA + TensorRT
inference = { git = "https://github.com/ultralytics/inference.git", features = ["intel"] }   # OpenVINO + oneDNN
inference = { git = "https://github.com/ultralytics/inference.git", features = ["mobile"] }  # NNAPI + CoreML + QNN

# All features (not recommended for most users)
inference = { git = "https://github.com/ultralytics/inference.git", features = ["all"] }
```

#### Available Features

**GPU Acceleration:**

- `cuda` - NVIDIA CUDA support
- `tensorrt` - NVIDIA TensorRT optimization
- `rocm` - AMD ROCm support
- `nvidia` - Convenience feature (CUDA + TensorRT)
- `amd` - Convenience feature (ROCm + MIGraphX)

**CPU Acceleration:**

- `openvino` - Intel OpenVINO
- `onednn` - Intel oneDNN
- `xnnpack` - XNNPACK (cross-platform)
- `intel` - Convenience feature (OpenVINO + oneDNN)

**Mobile/Embedded:**

- `nnapi` - Android Neural Networks API
- `coreml` - Apple CoreML (iOS/macOS)
- `qnn` - Qualcomm AI Engine
- `mobile` - Convenience feature (NNAPI + CoreML + QNN)

**Platform-Specific:**

- `directml` - DirectML (Windows)
- `webgpu` - WebGPU support
- `azure` - Azure acceleration

**Advanced:**

- `acl` - ARM Compute Library
- `armnn` - ARM NN
- `tvm` - Apache TVM
- `migraphx` - AMD MIGraphX
- `rknpu` - Rockchip NPU
- `vitis` - Xilinx Vitis AI
- `can` - Huawei CAN

#### As a CLI Tool

```bash
# Basic installation (CPU only)
cargo install --git https://github.com/ultralytics/inference.git

# With specific features
cargo install --git https://github.com/ultralytics/inference.git --features cuda
cargo install --git https://github.com/ultralytics/inference.git --features nvidia
```

### Development Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/ultralytics/inference.git
   cd inference
   ```

2. Build the project:

   ```bash
   cargo build --release
   ```

3. Run tests:

   ```bash
   cargo test
   ```

4. Run the CLI:

   ```bash
   cargo run -- --help
   ```

## 🚀 Usage

### As a Rust Library

```rust
use inference::YoloModel;

fn main() {
    let model = YoloModel::load("path/to/model.onnx").unwrap();
    let results = model.predict("path/to/image.jpg").unwrap();
    println!("Detections: {:?}", results);
}
```

### As a CLI

```bash
# Run inference on an image
inference predict --model yolov8n.onnx --source image.jpg

# Run inference on a video
inference predict --model yolov8n.onnx --source video.mp4

# Run inference on a webcam
inference predict --model yolov8n.onnx --source 0
```

## 🔧 Building for Production

For optimized release builds:

```bash
cargo build --release
```

The compiled binary will be available in `target/release/inference`.

## 💡 Contribute

Ultralytics thrives on community collaboration, and we deeply value your contributions! Whether it's reporting bugs, suggesting features, or submitting code changes, your involvement is crucial.

- **Reporting Issues**: Encounter a bug? Please report it on [GitHub Issues](https://github.com/ultralytics/inference/issues).
- **Feature Requests**: Have an idea for improvement? Share it via [GitHub Issues](https://github.com/ultralytics/inference/issues).
- **Pull Requests**: Want to contribute code? Please read our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) first, then submit a Pull Request.
- **Feedback**: Share your thoughts and experiences by participating in our official [Survey](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey).

A heartfelt thank you 🙏 goes out to all our contributors! Your efforts help make Ultralytics tools better for everyone.

[![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/ultralytics/graphs/contributors)

## 📄 License

Ultralytics offers two licensing options to accommodate diverse needs:

- **AGPL-3.0 License**: Ideal for students, researchers, and enthusiasts passionate about open collaboration and knowledge sharing. This [OSI-approved](https://opensource.org/license/agpl-v3) open-source license promotes transparency and community involvement. See the [LICENSE](LICENSE) file for details.
- **Enterprise License**: Designed for commercial applications, this license permits the seamless integration of Ultralytics software and AI models into commercial products and services, bypassing the copyleft requirements of AGPL-3.0. For commercial use cases, please inquire about an [Ultralytics Enterprise License](https://www.ultralytics.com/license).

## 📮 Contact

For bug reports or feature suggestions related to this project, please use [GitHub Issues](https://github.com/ultralytics/inference/issues). For general questions, discussions, and community support, join our [Discord](https://discord.com/invite/ultralytics) server!

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
