<a href="https://www.ultralytics.com/" target="_blank"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Ultralytics YOLO Rust Examples

<img alt="Rust" src="https://img.shields.io/badge/Rust-1.89%2B-CE422B.svg?logo=rust&logoColor=white"> <a href="https://crates.io/crates/ultralytics-inference" target="_blank"><img alt="crates.io" src="https://img.shields.io/crates/v/ultralytics-inference?logo=rust&logoColor=white&label=crates.io&color=CE422B"></a> <a href="https://docs.rs/ultralytics-inference" target="_blank"><img alt="docs.rs" src="https://img.shields.io/docsrs/ultralytics-inference?logo=docs.rs&logoColor=white&label=docs.rs&color=CE422B"></a> <a href="https://docs.ultralytics.com/" target="_blank"><img alt="Ultralytics Docs" src="https://img.shields.io/badge/Ultralytics-Docs-042AFF.svg?logo=ultralytics&logoColor=white"></a>

Runnable examples for the [`ultralytics-inference`](https://crates.io/crates/ultralytics-inference) library. Each example is a single file you run with `cargo run --example`. They show how to load and run [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/), [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/), and [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) models from Rust.

> [!NOTE]
> Model metadata (classes, task, image size) is read from the ONNX file, so Ultralytics YOLOv8, Ultralytics YOLO11, and Ultralytics YOLO26 models all work without extra configuration. Each example takes an optional image path and downloads a sample image when none is given.

## 📂 Examples

| Example                 | Feature    | Description                                                          | Run                                                |
| ----------------------- | ---------- | -------------------------------------------------------------------- | -------------------------------------------------- |
| [basic](basic.rs)       | none       | Load a model, run inference, print detections                        | `cargo run --example basic`                        |
| [config](config.rs)     | none       | Set confidence, IoU, image size, and device                          | `cargo run --example config`                       |
| [tasks](tasks.rs)       | none       | Summary for every task, plus raw arrays for segment, semantic, depth | `cargo run --example tasks`                        |
| [annotate](annotate.rs) | `annotate` | Draw boxes and labels, save the annotated image                      | `cargo run --example annotate --features annotate` |

## ✅ How to Run

Run any example from the repository root. The model and a sample image are downloaded on first use, so a network connection is needed once.

```bash
# Print detections for the sample image
cargo run --example basic

# Use your own image
cargo run --example basic -- path/to/image.jpg

# Set thresholds, image size, and device
cargo run --example config

# Annotate and save to runs/predict/annotated_*.jpg
cargo run --example annotate --features annotate
```

The `tasks` example defaults to detection and works with the other tasks when you pass their model. The matching sample image is downloaded automatically. The segment, semantic, and depth branches also print the raw output array, which `ndarray` truncates with `...`:

```bash
cargo run --example tasks                       # detect (default)
cargo run --example tasks -- yolo26n-seg.onnx   # segment
cargo run --example tasks -- yolo26n-pose.onnx  # pose
cargo run --example tasks -- yolo26n-obb.onnx   # obb
cargo run --example tasks -- yolo26n-cls.onnx   # classify
cargo run --example tasks -- yolo26n-sem.onnx   # semantic (YOLO26)
cargo run --example tasks -- yolo26n-depth.onnx # depth (YOLO26)
```

The `basic` example prints one block per image:

```text
Found 5 detections
  bus 0.93 [5.9 228.8 807.3 748.9]
  person 0.92 [46.7 398.6 237.0 901.8]
  person 0.90 [223.0 405.3 345.2 863.1]
  person 0.86 [668.6 391.2 810.0 879.7]
  person 0.51 [0.0 553.4 65.4 874.1]
```

The examples load `yolo26n.onnx`, downloading it on first use. To provide the model yourself, export it with the Ultralytics Python package. The command below writes `yolo26n.onnx` into the current directory, which the example then loads instead of downloading. The positional argument is still the input image:

```bash
pip install ultralytics
yolo export model=yolo26n.pt format=onnx

cargo run --example basic -- image.jpg
```

To run a different model, change the model path in the example.

## 🤝 Contributing

Contributions are welcome. See the [Ultralytics Contributing Guide](https://docs.ultralytics.com/help/contributing/) for details. If you find an issue with an example or want to add one, open an issue or pull request on the [Ultralytics inference repository](https://github.com/ultralytics/inference).
