# NVIDIA DGX Spark

Setup notes for running `ultralytics-inference` on a DGX Spark (GB10 Grace Blackwell, aarch64).

| Component    | Value on DGX Spark                                                     |
| ------------ | ---------------------------------------------------------------------- |
| GPU          | NVIDIA GB10, compute capability 12.1 (`sm_121`)                        |
| Driver       | 580.173.02                                                             |
| CUDA toolkit | 13.0                                                                   |
| CPU          | 20 cores, 10 x Cortex-X925 at 3.9 GHz plus 10 x Cortex-A725 at 2.8 GHz |
| Memory       | 121 GB unified                                                         |
| Target       | `aarch64-unknown-linux-gnu`                                            |

## The `ort` prebuilt binaries have no aarch64 GPU build

This is the first thing to know, because the failure looks like a local misconfiguration and is not.
`ort` downloads prebuilt ONNX Runtime binaries, and for `aarch64-unknown-linux-gnu` only a CPU
distribution is published. Building with `--features cuda`, `tensorrt`, `cuda-preprocess` or
`xnnpack` fails at link time:

```text
= note: some `extern` functions couldn't be found

  !!! The ort-sys crate did not download prebuilt binaries because there are no builds
  available that satisfy the requested feature set 'cuda13'.

  Builds with these feature sets are available for the current target (separated by ;):
      '(no features)' (*)
```

`undefined reference to OrtGetApiBase` in the same error is a symptom, not the cause. To confirm,
check the build script output rather than guessing at CUDA or driver problems:

```bash
grep -r "link_error_bad_dist_features" target/*/build/ort-sys-*/output
```

A plain `cargo build` with no GPU feature works normally and gives you the CPU execution provider.

## Linking a GPU ONNX Runtime

To get the CUDA and TensorRT execution providers you need an ONNX Runtime built for aarch64 with
those providers, and you point `ort` at it. PyPI publishes no aarch64 `onnxruntime-gpu` wheels, but
Ultralytics hosts them in
[ultralytics/assets](https://github.com/ultralytics/assets/releases/tag/v0.0.0). The wheel is a zip
archive, so the shared libraries can be extracted without installing Python packages:

```bash
mkdir -p ~/ortgpu && cd ~/ortgpu
curl -fLO https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.24.0-cp312-cp312-linux_aarch64.whl
unzip -j -o onnxruntime_gpu-1.24.0-cp312-cp312-linux_aarch64.whl 'onnxruntime/capi/libonnxruntime*.so*'
ln -sf libonnxruntime.so.1.24.0 libonnxruntime.so
ln -sf libonnxruntime.so.1.24.0 libonnxruntime.so.1 # the SONAME
```

That leaves `libonnxruntime.so.1.24.0` plus the `cuda`, `tensorrt` and `shared` provider libraries
in one directory.

Both symlinks matter. `ort` links against `-lonnxruntime`, which needs `libonnxruntime.so`, and the
library's SONAME is `libonnxruntime.so.1`, which is what the loader asks for at runtime.

Nothing about ONNX Runtime is compiled here. It is the same prebuilt binary the Python
`onnxruntime-gpu` package ships; only this crate is built from source, as any Rust crate is. The
difference from Python is packaging, not building: `pip install onnxruntime-gpu` is a supported
one-step install, while `ort` has no aarch64 GPU distribution to download, so the libraries have to
be extracted and pointed at by hand.

Build against it:

```bash
export ORT_LIB_PATH=$HOME/ortgpu
export ORT_PREFER_DYNAMIC_LINK=1
cargo build --release --features cuda-preprocess
```

`ORT_PREFER_DYNAMIC_LINK=1` is required. Without it `ort-sys` attempts a static link and fails with
`ort-sys could not link to the ONNX Runtime build in ...`.

An ONNX Runtime older than the one `ort` targets still works. `ort` requests API version 17 as a
minimum, not an exact match, so a 1.24 runtime satisfies a build that targets 1.28.

At runtime the CUDA and TensorRT provider libraries need cuDNN 9 and TensorRT 10 on the library
path. If you do not have them installed system-wide, the `nvidia-cudnn-cu13` and `tensorrt-cu13`
Python packages carry the same shared libraries and can be unpacked the same way as above:

```bash
export LD_LIBRARY_PATH=$HOME/ortgpu:/path/to/cudnn/lib:/path/to/tensorrt_libs:/usr/local/cuda/lib64
ultralytics-inference predict --model yolo26n.onnx --source image.jpg --device tensorrt:0 --quantize 16
```

If the CUDA provider fails to load, check for the missing dependency directly:

```bash
ldd ~/ortgpu/libonnxruntime_providers_cuda.so | grep "not found"
```

## Building ONNX Runtime from source

The prebuilt wheel above covers the CUDA and TensorRT providers, which is what this crate's GPU
features need. Building ONNX Runtime yourself is only worth it when you need something the wheel
does not contain, for example:

- an execution provider it was not built with, such as `xnnpack` or `acl`,
- a CUDA version other than the one it targets (13.0 here),
- a newer ONNX Runtime than the published wheels.

Build a shared library with the providers you want, following the
[ONNX Runtime build instructions](https://onnxruntime.ai/docs/build/eps.html):

```bash
./build.sh --config Release --build_shared_lib --parallel \
  --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr/local/cuda \
  --use_tensorrt --tensorrt_home /path/to/TensorRT
```

Expect a long compile. Then link it exactly as above, pointing `ORT_LIB_PATH` at the directory
holding the resulting `libonnxruntime.so`:

```bash
export ORT_LIB_PATH=/path/to/onnxruntime/build/Linux/Release
export ORT_PREFER_DYNAMIC_LINK=1
cargo build --release --features cuda-preprocess
```

`ORT_PREFER_DYNAMIC_LINK=1` matters here for the same reason as before. Without it `ort-sys` takes
its static path, which expects a set of static archives (`libonnxruntime_common.a` and the
per-provider archives) rather than the single `libonnxruntime.so` that `--build_shared_lib`
produces. See [ort linking](https://ort.pyke.io/setup/linking) for the full set of options.

## Pin CPU inference to the performance cores

The Grace CPU is heterogeneous: 10 cores at 3.9 GHz and 10 at 2.8 GHz. ONNX Runtime splits work
evenly across all 20 intra-op threads, so every parallel region finishes at the pace of the slowest
thread. Restricting the process to the fast cores is faster than using all of them:

| Configuration                       | FPS  |
| ----------------------------------- | ---- |
| default, all 20 cores               | 32.9 |
| pinned to the 10 cores at 2.8 GHz   | 24.6 |
| pinned to 6 of the cores at 3.9 GHz | 37.8 |
| pinned to the 10 cores at 3.9 GHz   | 41.6 |

Find the fast cores and pin to them:

```bash
for c in $(seq 0 19); do
  echo "$c $(cat /sys/devices/system/cpu/cpu$c/cpufreq/cpuinfo_max_freq)"
done | sort -k2 -rn | head -10

taskset -c 5,6,7,8,9,15,16,17,18,19 ultralytics-inference predict --source image.jpg --device cpu
```

Do not also set `num_threads`. The default of `0` asks the standard library for the available
parallelism, which already respects the CPU affinity mask and picks 10. Forcing 20 threads onto 10
pinned cores oversubscribes them and gives back most of the gain.

## Benchmarks

yolo26n at 640x640 over coco128 (128 images), mean and sample standard deviation across independent
runs, measured with the TensorRT execution provider and the `cuda-preprocess` feature.

| Configuration                           | preprocess ms | inference ms  | end to end ms | FPS            |
| --------------------------------------- | ------------- | ------------- | ------------- | -------------- |
| CPU FP32                                | 0.53 +- 0.03  | 29.83 +- 0.10 | 30.40 +- 0.09 | 32.9 +- 0.1    |
| CUDA FP32                               | 0.79 +- 0.06  | 4.35 +- 0.09  | 5.19 +- 0.15  | 192.8 +- 5.7   |
| CUDA FP32 with GPU preprocess           | 0.06 +- 0.00  | 4.46 +- 0.22  | 4.53 +- 0.22  | 221.4 +- 10.2  |
| TensorRT FP32                           | 0.70 +- 0.06  | 2.15 +- 0.01  | 2.91 +- 0.07  | 344.4 +- 8.7   |
| TensorRT FP16                           | 0.56 +- 0.02  | 1.13 +- 0.00  | 1.73 +- 0.03  | 577.5 +- 8.1   |
| TensorRT FP16 with GPU preprocess       | 0.06 +- 0.00  | 1.07 +- 0.01  | 1.13 +- 0.00  | 884.1 +- 3.4   |
| TensorRT FP16, GPU preprocess, batch 8  | 0.19 +- 0.01  | 0.65 +- 0.01  | 0.85 +- 0.02  | 1182.6 +- 21.4 |
| TensorRT FP16, GPU preprocess, batch 16 | 0.14 +- 0.01  | 0.68 +- 0.00  | 0.83 +- 0.01  | 1213.9 +- 20.5 |

Batch 8 and batch 16 are within about one and a half standard deviations of each other, so either is
a reasonable choice.

Batching raises throughput but not single-frame latency: at batch 16 a frame waits for the other 15,
so a webcam or video pipeline that wants the freshest frame should stay at batch 1.

The first TensorRT run builds an engine and takes tens of seconds; engines are cached under
`.trt_cache/` next to the model and reused afterwards.
