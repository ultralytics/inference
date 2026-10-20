# OpenVINO Models

`ultralytics-inference` runs native OpenVINO IR exports (`.xml` + `.bin`) on Intel CPUs, GPUs and NPUs through the opt-in `openvino-native` feature. Models load with `YOLOModel::load` and `--model`, the same way as ONNX.

## Enable

```toml
# Cargo.toml
[dependencies]
ultralytics-inference = { version = "0.0.54", features = ["openvino-native"] }
```

```bash
# build the CLI
cargo build --release --features openvino-native
```

Building needs no OpenVINO install. The runtime libraries are loaded when the first model opens.

## Requirements

- OpenVINO runtime (tested with 2026.4), installed with the official guide for [Linux](https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-linux.html), [Windows](https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-windows.html) or [macOS](https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-macos.html)
- For `intel:gpu` or `intel:npu`, an Intel GPU or NPU with its driver ([GPU setup][gpu], [NPU setup][npu]). macOS has the CPU device only.
- Rust 1.92+

`python -c "import openvino as ov; print(ov.Core().available_devices)"` lists the devices OpenVINO can see.

## Make the runtime findable

The loader looks for the OpenVINO C library (`libopenvino_c`) in this order:

1. `OPENVINO_INSTALL_DIR`, the OpenVINO install folder
2. `INTEL_OPENVINO_DIR`, set by the archive's `setupvars` script
3. The library path: `LD_LIBRARY_PATH` on Linux, `PATH` on Windows, `DYLD_LIBRARY_PATH` on macOS
4. Known install locations, so a system APT or YUM package needs no setup

For a pip or archive install, set one of them:

**Linux**

```bash
# pip
export LD_LIBRARY_PATH=$(python -c "import openvino, os; print(os.path.dirname(openvino.__file__))")/libs

# archive
source "/path/to/extracted-folder/setupvars.sh"
```

**Windows**

```bat
:: pip
for /f "delims=" %i in ('python -c "import openvino, os; print(os.path.dirname(openvino.__file__))"') do set "PATH=%i\libs;%PATH%"

:: archive
call "C:\path\to\extracted-folder\setupvars.bat"
```

**macOS**

```bash
# pip
export DYLD_LIBRARY_PATH=$(python -c "import openvino, os; print(os.path.dirname(openvino.__file__))")/libs

# archive
source "/path/to/extracted-folder/setupvars.sh"
```

## Export and run

```bash
yolo export model=yolo26n.pt format=openvino # add quantize=16, quantize=8, dynamic=True or batch=4
ultralytics-inference predict --model yolo26n_openvino_model --source bus.jpg --device intel:gpu
```

```rust,no_run
use ultralytics_inference::{Device, InferenceConfig, YOLOModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = InferenceConfig::new()
        .with_device(Device::IntelNpu) // IntelCpu, IntelGpu or IntelNpu
        .with_threads(8); // CPU inference threads
    let mut model = YOLOModel::load_with_config("yolo26n_openvino_model", config)?;
    let results = model.predict("assets/bus.jpg")?;
    println!("{} results", results.len());
    Ok(())
}
```

The run prints `Using OpenVINO Runtime` when the native path is active.

## Good to know

- The export folder needs its `metadata.yaml` beside the `.xml` and `.bin`.
- Dynamic exports are pinned to a static shape, using `--imgsz` and `--batch`.
- The NPU runs at batch 1, and larger batches are split into single images.
- The export fixes the precision (FP32, FP16 or INT8), so `--quantize` has no effect and prints a warning.
- Only `cpu` and the Intel devices (`intel:cpu`, `intel:gpu`, `intel:npu`) are accepted, other devices give an error.
- Depth models skip the graph's final upsample and are resampled on the host. NPU depth values still differ from the CPU, an NPU precision limit also seen in Python.

[gpu]: https://docs.openvino.ai/2026/get-started/install-openvino/configurations/configurations-intel-gpu.html
[npu]: https://docs.openvino.ai/2026/get-started/install-openvino/configurations/configurations-intel-npu.html
