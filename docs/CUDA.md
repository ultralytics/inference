# CUDA Acceleration

`ultralytics-inference` supports NVIDIA GPUs through three opt-in cargo features.

| Feature           | Path                                                | When to use                                                                                      |
| ----------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `cuda`            | ORT CUDA EP                                         | NVIDIA GPU, fast to set up, CUDA-only deps                                                       |
| `tensorrt`        | ORT TensorRT EP (FP16, engine cache, opt-level 5)   | NVIDIA GPU with TensorRT installed; 2–3× faster than `cuda`                                      |
| `cuda-preprocess` | GPU-side preprocess + zero-copy device input to TRT | maximum throughput; `YOLOModel::predict_image` transparently uses a fused CUDA preprocess kernel |

`cuda-preprocess` implies `cuda` + `tensorrt`. When it's compiled in, no API change is required - `YOLOModel::predict_image` automatically routes through the GPU preprocess path on CUDA/TensorRT devices. Opt out per-model with [`InferenceConfig::with_cuda_preprocess(false)`](crate::InferenceConfig::with_cuda_preprocess).

## Requirements

| Component              | Tested      | How to verify                                                                                                            |
| ---------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------ |
| NVIDIA driver          | 580+        | `nvidia-smi`                                                                                                             |
| CUDA toolkit           | 11.4 – 13.3 | `nvcc --version` _(toolkit only required for `cuda-preprocess`; `cuda` and `tensorrt` ship their EP libs through `ort`)_ |
| TensorRT               | 10.x        | `ldconfig -p \| grep libnvinfer` (only for `tensorrt` / `cuda-preprocess`)                                               |
| GPU compute capability | sm_70+      | `nvidia-smi --query-gpu=compute_cap --format=csv`                                                                        |

`cuda-preprocess` only needs `libcudart.so` and `libnvrtc.so` at **runtime**. Kernel code is compiled in-process via NVRTC, so `nvcc` is not invoked at runtime.

## Enable in your project

Add the feature you need in your `Cargo.toml`:

```toml
[dependencies]
ultralytics-inference = { version = "0.0.29", features = ["tensorrt"] }
# or, for the fastest path:
ultralytics-inference = { version = "0.0.29", features = ["cuda-preprocess"] }
```

Then `cargo build --release` - no extra flags needed.

For the CLI / examples in this repo directly:

```bash
cargo build --release --features tensorrt        # TensorRT EP
cargo build --release --features cuda-preprocess # GPU preprocess (fastest)
```

### Selecting the CUDA toolkit version (`cuda-preprocess` only)

`cuda-preprocess` depends on [`cudarc`](https://crates.io/crates/cudarc), which must be matched to your installed CUDA toolkit. By default it auto-detects via `nvcc --version`. If `nvcc` is not on `PATH`, set one of:

```bash
# Option 1: put nvcc on PATH
export PATH=/usr/local/cuda/bin:$PATH

# Option 2: tell cudarc directly (CUDA 13.2 -> 13020, CUDA 12.6 -> 12060)
export CUDARC_CUDA_VERSION=13020
```

Supported toolkits: 11.4, 11.5, 11.6, 11.7, 11.8, 12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.8, 12.9, 13.0, 13.1, 13.2, 13.3. (Full list and feature names: [cudarc Cargo.toml](https://github.com/chelsea0x3b/cudarc/blob/main/Cargo.toml).)

If you need to pin a specific version at compile time instead, override the cudarc dep in your project's `Cargo.toml`:

```toml
[dependencies]
ultralytics-inference = { version = "0.0.29", features = ["cuda-preprocess"] }
# Replace the default feature with a pinned one (e.g. CUDA 12.8):
cudarc = { version = "0.19", default-features = false, features = ["driver", "nvrtc", "dynamic-loading", "cuda-12080"] }
```

## Use

### `tensorrt` feature

```rust,no_run
use ultralytics_inference::{Device, InferenceConfig, YOLOModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = InferenceConfig::new()
        .with_device(Device::TensorRt(0))
        .with_half(true);
    let mut model = YOLOModel::load_with_config("yolo26n.onnx", cfg)?;
    let results = model.predict("image.jpg")?;
    println!("{} detections", results.len());
    Ok(())
}
```

The first run builds and caches a TensorRT engine at `<model_dir>/.trt_cache/<model_stem>_fp16/` (one-time cost, ~1–3 minutes for medium models). Subsequent runs are instant.

### ⏳ First-run engine build (warm-up) time

The TensorRT EP compiles a hardware-specific engine the **first time a given
model + input shape + precision is loaded**. This happens _during model load_
(inside `YOLOModel::load*`), and it can take from tens of seconds to several
minutes, it is **not** a hang.

| Model input                            | Approx. first-build time |
| -------------------------------------- | ------------------------ |
| 640×640 (detect/seg/pose)              | ~30 s – 1 min            |
| 1024×1024 (OBB) / 1024×2048 (semantic) | ~2 – 5 min               |

What to expect and how to avoid surprises:

- **It's cached.** Builds are written to `<model_dir>/.trt_cache/<stem>_{fp16,fp32}/`
  (engine **and** timing cache). Later loads of the same model reuse them and
  start in seconds. **Keep `.trt_cache/` between runs** (don't delete it / add it
  to `.gitignore`, not to clean builds) to avoid paying the cost again.
- **Cache is keyed to the build context.** A new engine is built whenever the
  model file, GPU/driver/TensorRT version, precision (`--half`), or **input
  shape** changes. **Dynamic-shape models rebuild per new input size** - feed a
  consistent size (the fast path uses the model's resolved `imgsz`) to keep it
  to a single cached engine.
- **Warm up before timing.** The first `predict*` call also triggers an
  inference-time warm-up. Always discard the first few iterations when
  benchmarking (the examples do this).
- **Pre-build in deployment.** Run one inference at startup (or ship a populated
  `.trt_cache/`) so the first user request isn't stuck behind a multi-minute
  build.

> **Note on `.engine` files:** this crate runs models through ONNX Runtime's
> TensorRT EP, which consumes **ONNX** and compiles/caches the engine internally.
> You cannot load a standalone `.engine` file directly (that needs the native
> TensorRT runtime); the `.trt_cache/` engine is an internal ORT artifact.

### `cuda-preprocess` feature

No separate type or API. With the feature compiled in and a CUDA/TensorRT
device, [`YOLOModel::predict_image`] automatically runs the fused GPU
preprocess kernel (bilinear letterbox + `/255` normalize + HWC→CHW) and hands
the result to ORT as a zero-copy device tensor:

```rust,no_run
use ultralytics_inference::{Device, InferenceConfig, YOLOModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = InferenceConfig::new()
        .with_device(Device::TensorRt(0))
        .with_half(true);
    let mut model = YOLOModel::load_with_config("yolo26n.onnx", cfg)?;

    // predict() decodes the frame and calls predict_image(), which
    // transparently uses the GPU preprocess fast path:
    let results = model.predict("image.jpg")?;
    println!("{} detections", results.len());
    Ok(())
}
```

To force the standard CPU preprocess path (e.g. for an A/B comparison) without
recompiling, set the flag to `false`:

```rust,no_run
use ultralytics_inference::{Device, InferenceConfig, YOLOModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = InferenceConfig::new()
        .with_device(Device::TensorRt(0))
        .with_half(true)
        .with_cuda_preprocess(false); // opt out; CPU letterbox + host→device copy
    let _model = YOLOModel::load_with_config("yolo26n.onnx", cfg)?;
    Ok(())
}
```

The fast path is selected at load time when **all** of these hold; otherwise
the CPU path runs and the flag is silently ignored:

- the crate was built with the `cuda-preprocess` feature,
- `cuda_preprocess` is `true` (the default),
- the device is `Cuda(_)`, `TensorRt(_)`, or unset (auto-detect),
- the task is not `Classify` (which uses center-crop, not letterbox),
- the model takes an FP32 input tensor (FP16-input models keep the CPU path).

It is wired into `predict_image` specifically (single-frame). `predict_batch`
and the multi-image batch path always use CPU preprocess.

## CLI

The CLI selects the GPU EP via `--device`:

```bash
ultralytics-inference predict --model yolo26n.onnx --source image.jpg \
  --device tensorrt:0 --half
```

This uses the TensorRT EP (FP16 + engine cache). The `cuda-preprocess` kernel
fast path is **not** used by the CLI - the CLI runs through the batch
processor, which uses CPU preprocess. The GPU preprocess path is reached only
through `YOLOModel::predict_image` in library code.

[`YOLOModel::predict_image`]: crate::YOLOModel::predict_image

## Troubleshooting

| Symptom                                                                    | Fix                                                                                      |
| -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `cudarc-* build script failed: \`nvcc --version\` failed`                  | Set `PATH` to include the toolkit's `bin/`, or set `CUDARC_CUDA_VERSION` (see above).    |
| `libcudart.so.13: cannot open shared object file`                          | Toolkit not installed or not on `ld.so` path. Verify `ldconfig -p \| grep libcudart.so`. |
| `libnvinfer.so.10: cannot open shared object file`                         | TensorRT not installed. Required for `tensorrt` and `cuda-preprocess` features.          |
| TRT engine build is slow on first run                                      | Expected - engines are cached under `.trt_cache/`. Subsequent runs reuse them.           |
| Build hits `Must specify one of the following features: [cuda-13020, ...]` | Your environment has neither `nvcc` on `PATH` nor `CUDARC_CUDA_VERSION` set. Pick one.   |
