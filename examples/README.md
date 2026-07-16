<a href="https://www.ultralytics.com/" target="_blank"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Ultralytics YOLO Rust Examples

<img alt="Rust" src="https://img.shields.io/badge/Rust-1.89%2B-CE422B.svg?logo=rust&logoColor=white"> <a href="https://docs.rs/ultralytics-inference" target="_blank"><img alt="docs.rs" src="https://img.shields.io/docsrs/ultralytics-inference?logo=docs.rs&logoColor=white&label=docs.rs&color=CE422B"></a> <a href="https://docs.ultralytics.com/" target="_blank"><img alt="Ultralytics Docs" src="https://img.shields.io/badge/Ultralytics-Docs-042AFF.svg?logo=ultralytics&logoColor=white"></a>

Runnable examples for the [`ultralytics-inference`](https://crates.io/crates/ultralytics-inference) library. Each example is a single file you run with `cargo run --example`. They show how to load and run [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/), [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/), and [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) models from Rust.

> [!NOTE]
> Model metadata (classes, task, image size) is read from the ONNX file, so Ultralytics YOLOv8, Ultralytics YOLO11, and Ultralytics YOLO26 models all work without extra configuration. Each example takes an optional image path and downloads a sample image when none is given.

## 📂 Examples

| Example                 | Feature    | Description                                     | Run                                                |
| ----------------------- | ---------- | ----------------------------------------------- | -------------------------------------------------- |
| [basic](basic.rs)       | none       | Load a model, run inference, print detections   | `cargo run --example basic`                        |
| [annotate](annotate.rs) | `annotate` | Draw boxes and labels, save the annotated image | `cargo run --example annotate --features annotate` |

## ✅ How to Run

Run any example from the repository root. The model and a sample image are downloaded on first use, so a network connection is needed once.

```bash
# Print detections for the sample image
cargo run --example basic

# Use your own image
cargo run --example basic -- path/to/image.jpg

# Annotate and save to runs/predict/annotated_*.jpg
cargo run --example annotate --features annotate
```

To run your own model, export one with the Ultralytics Python package and pass its path:

```bash
pip install ultralytics
yolo export model=yolo26n.pt format=onnx

cargo run --example basic -- image.jpg
```

## 🤝 Contributing

Contributions are welcome. If you find an issue with an example or want to add one, open an issue or pull request on the [Ultralytics inference repository](https://github.com/ultralytics/inference).
