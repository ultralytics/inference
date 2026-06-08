<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics 标志"></a>

# 🦀 Ultralytics YOLO Rust Inference

[English](README.md) | [简体中文](README.zh-CN.md)

用 Rust 编写的高性能 YOLO 推理库。本库基于 ONNX Runtime，为运行 YOLO 模型提供快速、安全、高效的接口，API 设计与 [Ultralytics Python 包](https://github.com/ultralytics/ultralytics)保持一致。

[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://reddit.com/r/ultralytics)
[![codecov](https://codecov.io/gh/ultralytics/inference/graph/badge.svg?token=AVE5n6yvnf)](https://codecov.io/gh/ultralytics/inference)
[![CI](https://github.com/ultralytics/inference/actions/workflows/ci.yml/badge.svg)](https://github.com/ultralytics/inference/actions/workflows/ci.yml)

[![Crates.io](https://img.shields.io/crates/v/ultralytics-inference?logo=rust&logoColor=white&label=crates.io&color=CE422B)](https://crates.io/crates/ultralytics-inference)
[![docs.rs](https://img.shields.io/docsrs/ultralytics-inference?logo=docs.rs&logoColor=white&label=docs.rs)](https://docs.rs/ultralytics-inference)
[![Downloads](https://img.shields.io/crates/d/ultralytics-inference?logo=rust&logoColor=white&label=downloads&color=CE422B)](https://crates.io/crates/ultralytics-inference)
[![MSRV](https://img.shields.io/crates/msrv/ultralytics-inference?logo=rust&logoColor=white&color=CE422B)](https://crates.io/crates/ultralytics-inference)
[![dependency status](https://deps.rs/repo/github/ultralytics/inference/status.svg)](https://deps.rs/repo/github/ultralytics/inference)

## ✨ 功能

- 🚀 **高性能**：纯 Rust 实现，使用零成本抽象。
- 🎯 **兼容 Ultralytics API**：`Results`、`Boxes`、`Masks`、`Keypoints`、`Probs` 和 `SemanticMask` 类型与 Python API 形态保持一致。
- 🔧 **多后端支持**：通过 ONNX Runtime 支持 CPU、XNNPACK、CUDA、TensorRT、CoreML、OpenVINO 等后端。
- 📦 **双用途**：既可作为 Rust 项目的库，也可作为独立 CLI 应用。
- 🏷️ **自动读取元数据**：自动读取 ONNX 模型中的类别名称、任务类型和输入尺寸。
- ⬇️ **自动下载**：本地不存在时，自动下载支持的 YOLO26、YOLO11 和 YOLOv8 ONNX 模型（尺寸：n/s/m/l/x）。
- 🖼️ **多输入源**：支持图片、目录、glob 模式、视频文件、摄像头和流。
- 🪶 **轻量运行时**：不需要 PyTorch、TensorFlow 或 Python 运行时。

## 🚀 快速开始

### 前置条件

- [Rust 1.89+](https://rustup.rs/)（通过 rustup 安装）。
- YOLO ONNX 模型（从 Ultralytics 导出：`yolo export model=yolo26n.pt format=onnx`）。

### 安装

```bash
# 从 crates.io 全局安装 CLI
cargo install ultralytics-inference

# 使用自定义 features 全局安装 CLI
# 最小构建（禁用默认 features）
cargo install ultralytics-inference --no-default-features

# 启用视频支持
cargo install ultralytics-inference --features video

# 启用多个加速后端
cargo install ultralytics-inference --features "cuda,tensorrt"
```

### 开发安装

```bash
# 直接从 git 仓库安装 CLI
cargo install --git https://github.com/ultralytics/inference.git ultralytics-inference

# 或克隆源码、构建并安装
git clone https://github.com/ultralytics/inference.git
cd inference
cargo build --release

# 从本地 checkout 安装
cargo install --path . --locked
```

`cargo install` 会把二进制文件安装到 Cargo 默认 bin 目录：

- macOS/Linux：`~/.cargo/bin`
- Windows：`%USERPROFILE%\\.cargo\\bin`

确认该目录已加入 `PATH` 后，可在任意位置运行：

```bash
ultralytics-inference help
```

### 将 YOLO 模型导出为 ONNX

```bash
# 使用 Ultralytics CLI
yolo export model=yolo26n.pt format=onnx
```

```python
# 或使用 Python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.export(format="onnx")
```

### 运行推理

```bash
# 使用默认参数（自动下载 yolo26n.onnx 和示例图片）
ultralytics-inference predict

# 选择任务：自动下载对应任务的 nano 模型
ultralytics-inference predict --task segment  # 下载 yolo26n-seg.onnx
ultralytics-inference predict --task pose     # 下载 yolo26n-pose.onnx
ultralytics-inference predict --task obb      # 下载 yolo26n-obb.onnx
ultralytics-inference predict --task classify # 下载 yolo26n-cls.onnx
ultralytics-inference predict --task semantic # 下载 yolo26n-sem.onnx（仅 YOLO26）

# 使用显式模型（任务从模型元数据读取）
ultralytics-inference predict --model yolo26n.onnx --source image.jpg

# 自动下载任意支持尺寸（n/s/m/l/x）的 YOLO26、YOLO11 和 YOLOv8 模型
ultralytics-inference predict --model yolo26l.onnx --source image.jpg
ultralytics-inference predict --model yolo11x-seg.onnx --source image.jpg
ultralytics-inference predict --model yolov8n.onnx --source image.jpg

# 对图片目录运行推理
ultralytics-inference predict --model yolo26n.onnx --source assets/

# 自定义阈值
ultralytics-inference predict -m yolo26n.onnx -s image.jpg --conf 0.5 --iou 0.45

# 按类别 ID 过滤
ultralytics-inference predict --model yolo26n.onnx --source image.jpg --classes 0
ultralytics-inference predict --model yolo26n.onnx --source image.jpg --classes "0,1,2"

# 可视化并设置自定义图片尺寸
ultralytics-inference predict --model yolo26n.onnx --source video.mp4 --show --imgsz 1280

# 为视频输入保存单帧
ultralytics-inference predict --model yolo26n.onnx --source video.mp4 --save-frames

# 矩形推理
ultralytics-inference predict --model yolo26n.onnx --source image.jpg --rect

# 语义分割：将每张图片的 PNG 类别图写入 runs/semantic/predictN/results/
ultralytics-inference predict --task semantic --source cityscapes/ --save-json
```

### 示例输出

```bash
ultralytics-inference predict
```

```text
WARNING ⚠️ 'model' argument is missing. Using default '--model=yolo26n.onnx'.
WARNING ⚠️ 'source' argument is missing. Using default images: https://ultralytics.com/images/bus.jpg, https://ultralytics.com/images/zidane.jpg
Ultralytics Inference 0.0.18 🚀 Rust ONNX FP32 CPU
Using ONNX Runtime CPUExecutionProvider
YOLO26n summary: 80 classes, imgsz=(640, 640)

image 1/2 /home/ultralytics/inference/bus.jpg: 640x480 4 persons, 1 bus, 36.4ms
image 2/2 /home/ultralytics/inference/zidane.jpg: 384x640 2 persons, 1 tie, 28.6ms
Speed: 1.5ms preprocess, 32.5ms inference, 0.5ms postprocess per image at shape (1, 3, 384, 640)
Results saved to runs/detect/predict1
💡 Learn more at https://docs.ultralytics.com/modes/predict
```

**使用 `--task`（自动下载匹配的 nano 模型）：**

```bash
ultralytics-inference predict --task segment
```

```text
WARNING ⚠️ 'model' argument is missing. Using default '--model=yolo26n-seg.onnx'.
WARNING ⚠️ 'source' argument is missing. Using default images: https://ultralytics.com/images/bus.jpg, https://ultralytics.com/images/zidane.jpg
Ultralytics Inference 0.0.18 🚀 Rust ONNX FP32 CPU
Using ONNX Runtime CPUExecutionProvider
YOLO26n-seg summary: 80 classes, imgsz=(640, 640)

image 1/2 /home/ultralytics/inference/bus.jpg: 640x480 4 persons, 1 bus, 48.2ms
image 2/2 /home/ultralytics/inference/zidane.jpg: 384x640 2 persons, 1 tie, 38.1ms
Speed: 1.6ms preprocess, 44.3ms inference, 1.2ms postprocess per image at shape (1, 3, 384, 640)
Results saved to runs/segment/predict1
💡 Learn more at https://docs.ultralytics.com/modes/predict
```

## 📚 使用方式

### 作为 CLI 工具

```bash
# 显示帮助
ultralytics-inference help

# 显示版本
ultralytics-inference version

# 运行推理
ultralytics-inference predict --model <model.onnx> --source <source>
```

`--help` 和 `--version` 也可作为标准别名使用。

**CLI 选项：**

| 选项            | 简写 | 说明                                                                                                                     | 默认值                            |
| --------------- | ---- | ------------------------------------------------------------------------------------------------------------------------ | --------------------------------- |
| `--model`       | `-m` | ONNX 模型文件路径；若为已知 YOLOv8/YOLO11/YOLO26 名称则自动下载                                                          | `yolo26n.onnx`                    |
| `--task`        |      | 任务类型（`detect`、`segment`、`pose`、`obb`、`classify`、`semantic`\*）；省略 `--model` 时选择 nano 模型                | `detect`                          |
| `--source`      | `-s` | 输入源（图片、目录、glob、视频、摄像头索引或 URL）                                                                       | 与任务相关的 Ultralytics URL 资源 |
| `--conf`        |      | 置信度阈值                                                                                                               | `0.25`                            |
| `--iou`         |      | NMS IoU 阈值                                                                                                             | `0.7`                             |
| `--max-det`     |      | 最大检测数量                                                                                                             | `300`                             |
| `--imgsz`       |      | 推理图片尺寸                                                                                                             | 模型元数据                        |
| `--rect`        |      | 启用矩形推理（最小填充）                                                                                                 | `true`                            |
| `--batch`       |      | 推理 batch size                                                                                                          | `1`                               |
| `--half`        |      | 使用 FP16 半精度推理                                                                                                     | `false`                           |
| `--save`        |      | 将标注结果保存到 runs/\<task\>/predict                                                                                   | `true`                            |
| `--save-frames` |      | 为视频输入保存单帧（而不是视频文件）                                                                                     | `false`                           |
| `--save-json`   |      | 保存语义分割类别图 PNG，便于外部评估                                                                                     | `false`                           |
| `--show`        |      | 在窗口中显示结果                                                                                                         | `false`                           |
| `--device`      |      | 设备字符串，例如 cpu、cuda:0、coreml、directml:0、openvino、tensorrt:0、rocm:0、xnnpack；启用 feature 后可选择更多提供方 | `cpu`                             |
| `--verbose`     |      | 显示详细输出                                                                                                             | `true`                            |
| `--classes`     |      | 按类别 ID 过滤，例如 `0`、`"0,1,2"` 或 `"[0, 1, 2]"`                                                                     | 所有类别                          |

**任务和模型解析：**

| 调用方式                                          | 使用模型             | 说明                                                    |
| ------------------------------------------------- | -------------------- | ------------------------------------------------------- |
| `predict`                                         | `yolo26n.onnx`       | 默认检测模型，自动下载                                  |
| `predict --task segment`                          | `yolo26n-seg.onnx`   | nano 分割模型，自动下载                                 |
| `predict --task pose`                             | `yolo26n-pose.onnx`  | nano 姿态模型，自动下载                                 |
| `predict --task obb`                              | `yolo26n-obb.onnx`   | nano OBB 模型，自动下载                                 |
| `predict --task classify`                         | `yolo26n-cls.onnx`   | nano 分类模型，自动下载                                 |
| `predict --task semantic`                         | `yolo26n-sem.onnx`\* | nano 语义分割模型，自动下载（仅 YOLO26）                |
| `predict --model yolo26l-seg.onnx`                | `yolo26l-seg.onnx`   | 从模型元数据读取任务                                    |
| `predict --task segment --model yolo26l-seg.onnx` | `yolo26l-seg.onnx`   | `--task` 与元数据一致，正常执行                         |
| `predict --task segment --model yolo26n.onnx`     | error                | `--task` 与模型元数据（`detect`）冲突，程序以错误退出。 |

\* `semantic`（语义分割）仅支持 YOLO26。

**可自动下载的模型：**

YOLOv8、YOLO11 和 YOLO26 ONNX 模型支持 **n / s / m / l / x** 尺寸，并覆盖标准任务变体。YOLO26 还包含用于语义分割的 `-sem`：

| 系列   | 变体                                                                                      |
| ------ | ----------------------------------------------------------------------------------------- |
| YOLO26 | `yolo26{n,s,m,l,x}.onnx`、`yolo26{n,s,m,l,x}-seg.onnx`、`-pose`、`-obb`、`-cls`、`-sem`\* |
| YOLO11 | `yolo11{n,s,m,l,x}.onnx`、`yolo11{n,s,m,l,x}-seg.onnx`、`-pose`、`-obb`、`-cls`           |
| YOLOv8 | `yolov8{n,s,m,l,x}.onnx`、`yolov8{n,s,m,l,x}-seg.onnx`、`-pose`、`-obb`、`-cls`           |

\* `-sem`（语义分割）仅支持 YOLO26。

**输入源选项：**

| 输入源类型 | 示例输入                        | 说明                   |
| ---------- | ------------------------------- | ---------------------- |
| 图片       | `image.jpg`                     | 单张图片文件           |
| 目录       | `images/`                       | 图片目录               |
| Glob       | `images/*.jpg`                  | 图片 glob 模式         |
| 视频       | `video.mp4`                     | 视频文件               |
| 摄像头     | `0`,`1`                         | 摄像头索引（0 为默认） |
| URL        | `https://example.com/image.jpg` | 远程图片 URL           |

### 作为 Rust 库

添加到 `Cargo.toml`（二选一）：

```toml
# crates.io 稳定版本
[dependencies]
ultralytics-inference = "0.0.18"
```

```toml
# 开发版本（GitHub 上最新未发布代码）
[dependencies]
ultralytics-inference = { git = "https://github.com/ultralytics/inference.git" }
```

**基础用法：**

```rust
use ultralytics_inference::{YOLOModel, InferenceConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 加载模型 - 自动读取元数据（类别、任务、imgsz）
    let mut model = YOLOModel::load("yolo26n.onnx")?;

    // 运行推理
    let results = model.predict("image.jpg")?;

    // 处理结果
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

**自定义配置：**

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

**访问检测数据：**

```rust
if let Some(ref boxes) = result.boxes {
    // 不同格式的边界框
    let xyxy = boxes.xyxy();      // [x1, y1, x2, y2]
    let xywh = boxes.xywh();      // [x_center, y_center, width, height]
    let xyxyn = boxes.xyxyn();    // 归一化 [0-1]
    let xywhn = boxes.xywhn();    // 归一化 [0-1]

    // 置信度和类别 ID
    let conf = boxes.conf();      // 置信度
    let cls = boxes.cls();        // 类别 ID
}
```

**选择设备：**

```rust
use ultralytics_inference::{Device, InferenceConfig, YOLOModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 选择设备（例如 CUDA、CoreML、CPU）
    let device = Device::Cuda(0);

    // 配置模型使用该设备
    let config = InferenceConfig::new().with_device(device);

    let mut model = YOLOModel::load_with_config("yolo26n.onnx", config)?;
    let results = model.predict("image.jpg")?;

    Ok(())
}
```

## 🗂️ 项目结构

```text
inference/
├── src/
│   ├── lib.rs              # 库入口和公开导出
│   ├── main.rs             # CLI 应用
│   ├── model.rs            # YOLOModel - ONNX session 和推理
│   ├── results.rs          # Results, Boxes, Masks, Keypoints, Probs, Obb, SemanticMask
│   ├── preprocessing.rs    # 图片预处理（letterbox、normalize、SIMD）
│   ├── postprocessing.rs   # 所有任务的后处理（检测 NMS/decode、语义分割 argmax）
│   ├── metadata.rs         # ONNX 模型元数据解析
│   ├── source.rs           # 输入源处理（图片、视频、摄像头）
│   ├── task.rs             # Task 枚举（Detect, Segment, Pose, Classify, Obb, Semantic）
│   ├── inference.rs        # InferenceConfig
│   ├── batch.rs            # Batch 处理流程
│   ├── device.rs           # Device 枚举（CPU, CUDA, CoreML 等）
│   ├── download.rs         # 模型和资源下载
│   ├── annotate.rs         # 图片标注（边界框、实例 mask、关键点、语义叠加）
│   ├── io.rs               # 结果保存（图片、视频）
│   ├── logging.rs          # 日志宏
│   ├── error.rs            # 错误类型
│   ├── utils.rs            # 工具函数（NMS、IoU）
│   ├── cli/                # CLI 模块
│   │   ├── mod.rs          # CLI 模块导出
│   │   ├── args.rs         # CLI 参数解析
│   │   └── predict.rs      # predict 命令实现
│   └── visualizer/         # 实时可视化（minifb）
├── tests/
│   └── integration_test.rs # 集成测试
├── assets/                 # 测试图片
│   ├── boats.jpg
│   ├── bus.jpg
│   └── zidane.jpg
├── Cargo.toml              # Rust 依赖和 features
├── LICENSE                 # AGPL-3.0 License
├── README.md               # 英文 README
└── README.zh-CN.md         # 简体中文 README
```

## ⚡ 硬件加速

通过添加 features 启用硬件加速：

```bash
# NVIDIA GPU（CUDA）
cargo build --release --features cuda

# NVIDIA TensorRT
cargo build --release --features tensorrt

# NVIDIA GPU 预处理 + TensorRT 零拷贝输入（最快；需要 CUDA toolkit）
cargo build --release --features cuda-preprocess

# Apple CoreML（macOS/iOS）
cargo build --release --features coreml

# Intel OpenVINO
cargo build --release --features openvino

# 多个 features
cargo build --release --features "cuda,tensorrt"
```

> NVIDIA 安装、要求和 GPU 预处理快速路径见 [`docs/CUDA.md`](docs/CUDA.md)。

**可用 Features：**

默认 features（除非传入 `--no-default-features`）：`annotate`、`visualize`。

| Feature           | 说明                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------- |
| `annotate`        | 为 `--save` 提供图片标注（默认）                                                         |
| `visualize`       | 为 `--show` 提供实时窗口显示（默认）                                                     |
| `video`           | 视频文件解码/编码（需要 FFmpeg）                                                         |
| `cuda`            | NVIDIA CUDA 支持                                                                         |
| `tensorrt`        | NVIDIA TensorRT 优化                                                                     |
| `cuda-preprocess` | GPU 预处理 + TensorRT 零拷贝输入（需要 CUDA toolkit；见 [`docs/CUDA.md`](docs/CUDA.md)） |
| `coreml`          | Apple CoreML（macOS/iOS）                                                                |
| `openvino`        | Intel OpenVINO                                                                           |
| `onednn`          | Intel oneDNN                                                                             |
| `rocm`            | AMD ROCm                                                                                 |
| `migraphx`        | AMD MIGraphX                                                                             |
| `directml`        | DirectML（Windows）                                                                      |
| `nnapi`           | Android Neural Networks API                                                              |
| `qnn`             | Qualcomm Neural Networks                                                                 |
| `xnnpack`         | XNNPACK（跨平台）                                                                        |
| `acl`             | ARM Compute Library                                                                      |
| `armnn`           | ARM NN                                                                                   |
| `tvm`             | Apache TVM                                                                               |
| `rknpu`           | Rockchip NPU                                                                             |
| `cann`            | Huawei CANN                                                                              |
| `webgpu`          | WebGPU                                                                                   |
| `azure`           | Azure                                                                                    |
| `nvidia`          | 便捷组合：CUDA + TensorRT                                                                |
| `amd`             | 便捷组合：ROCm + MIGraphX                                                                |
| `intel`           | 便捷组合：OpenVINO + oneDNN                                                              |
| `mobile`          | 便捷组合：NNAPI + CoreML + QNN                                                           |
| `all`             | 便捷组合：annotate + visualize + video                                                   |

## 📦 依赖

本库的关键优势之一是使用 Rust/ONNX Runtime 技术栈，不需要 PyTorch、TensorFlow 或 Python 运行时。

### 核心依赖（始终包含）

| Crate               | 用途                    |
| ------------------- | ----------------------- |
| `ort`               | ONNX Runtime 绑定       |
| `ndarray`           | N 维数组                |
| `image`             | 图片加载/解码           |
| `jpeg-decoder`      | JPEG 解码               |
| `fast_image_resize` | SIMD 优化 resize        |
| `half`              | FP16 支持               |
| `lru`               | 预处理 LUT 的 LRU cache |
| `wide`              | 快速预处理使用的 SIMD   |

### 可选依赖（用于 `annotate` feature）

| Crate       | 用途         |
| ----------- | ------------ |
| `imageproc` | 绘制框和形状 |
| `ab_glyph`  | 文本渲染字体 |

### 可选依赖（用于视频和可视化）

| Crate      | 用途                    |
| ---------- | ----------------------- |
| `minifb`   | 窗口创建和缓冲区显示    |
| `video-rs` | 视频解码/编码（ffmpeg） |

### 视频支持（FFmpeg）

视频 features 需要系统安装 FFmpeg（7 或 8）：

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
apt-get install -y ffmpeg libavutil-dev libavformat-dev libavfilter-dev libavdevice-dev libclang-dev

# 使用视频支持构建
cargo build --release --features video
```

如需构建不含标注和可视化支持的更小二进制：

```bash
cargo build --release --no-default-features
```

## 🧪 测试

```bash
# 运行全部测试
cargo test

# 运行并显示输出
cargo test -- --nocapture

# 运行指定测试
cargo test test_boxes_creation
```

## 📊 性能

Apple M4 MacBook Pro（CPU，ONNX Runtime）上的基准：

### YOLO26n 检测模型（640x640）

| 精度 | 模型大小 | 预处理 | 推理  | 后处理 | 总耗时 |
| ---- | -------- | ------ | ----- | ------ | ------ |
| FP32 | 10.2 MB  | ~9ms   | ~21ms | <1ms   | ~31ms  |
| FP16 | 5.2 MB   | ~9ms   | ~24ms | <1ms   | ~34ms  |

**关键结论：**

- **FP16 模型体积约小 50%**（5.2 MB vs 10.2 MB）。
- **FP32 在 CPU 上略快**（~21ms vs ~24ms），因为 CPU 对 FP32 有原生支持。
- FP16 在大多数 CPU 上需要上转换为 FP32 计算，会增加开销。
- CPU 推理建议使用 **FP32**，GPU 推理建议使用 **FP16**（通常能带来速度提升）。

### 线程优化

ONNX Runtime 线程数设置为 auto（`num_threads: 0`），由 ORT 自动选择最佳线程数：

- 手动线程（4 threads）：~40ms 推理。
- 自动线程（0 = ORT 决定）：~21ms 推理。

## 🔮 路线图

### 已完成

- [x] 检测、分割、姿态、分类、OBB 和语义分割推理。
- [x] ONNX 模型元数据解析（自动检测类别、任务和 imgsz）。
- [x] 硬件加速支持（CUDA、TensorRT、CoreML、OpenVINO、XNNPACK）。
- [x] 兼容 Ultralytics 的 Results API（`Boxes`、`Masks`、`Keypoints`、`Probs`、`Obb`、`SemanticMask`）。
- [x] 多输入源（图片、目录、glob、URL）。
- [x] 视频文件支持和摄像头/RTSP 流。
- [x] 图片标注和可视化。
- [x] FP16 半精度推理。
- [x] Batch 推理支持。
- [x] 矩形推理支持和优化。
- [x] 类别过滤支持。
- [x] 自动下载所有 YOLO26、YOLO11 和 YOLOv8 ONNX 模型（所有 n/s/m/l/x 尺寸、所有任务）。
- [x] `--task` CLI 参数：省略 `--model` 时选择并自动下载匹配的 nano 模型；任务和模型元数据冲突时退出并报错。

### 进行中

- [ ] Python 绑定（PyO3）。
- [ ] 浏览器推理的 WebAssembly（WASM）支持。

## 💡 贡献

Ultralytics 依靠社区协作持续发展，我们重视每一份贡献。无论是报告 bug、提出功能建议，还是提交代码改动，都欢迎参与。

- **报告问题**：[打开 issue](https://github.com/ultralytics/inference/issues)。
- **功能请求**：[提交想法](https://github.com/ultralytics/inference/issues)。
- **Pull Request**：请先阅读[贡献指南](https://docs.ultralytics.com/help/contributing/)。
- **反馈**：填写 [Ultralytics 调查问卷](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey)。

感谢所有贡献者！你们的努力让 Ultralytics 工具持续变得更好。

[![Ultralytics 开源贡献者](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/ultralytics/graphs/contributors)

## 📜 许可证

Ultralytics 提供两种许可方式：

- **AGPL-3.0 许可证**：经 [OSI 批准](https://opensource.org/license/agpl-3.0)的开源许可证，适合学生、研究者和爱好者，鼓励开放协作和知识共享。完整详情请参阅 [LICENSE](https://github.com/ultralytics/inference/blob/main/LICENSE) 文件。
- **Ultralytics 企业许可证**：面向商业使用，允许将 Ultralytics 软件和 AI 模型集成到商业产品与服务中，而无需遵循 AGPL-3.0 的开源要求。如需商业部署，请通过 [Ultralytics Licensing](https://www.ultralytics.com/license) 联系我们。

## 📮 联系方式

- **GitHub Issues**：[bug 报告和功能请求](https://github.com/ultralytics/inference/issues)。
- **Discord**：[加入社区](https://discord.com/invite/ultralytics)。
- **文档**：[docs.ultralytics.com](https://docs.ultralytics.com)。

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
