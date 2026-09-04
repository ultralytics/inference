// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! YOLO model loading and inference.
//!
//! This module provides the main `YOLOModel` struct for loading ONNX models
//! and running inference.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use half::f16;
use image::{DynamicImage, GenericImageView};
use ndarray::Array3;
// `model` is native-only, so rayon is used directly rather than via `crate::parallel`
// (which exists to give the wasm-shared pipeline sequential shims).
use ort::session::Session;
use ort::value::TensorElementType;
use ort::value::TensorRef;
use ort::value::ValueType;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;

use crate::download::{DEFAULT_IMAGES, DEFAULT_OBB_IMAGE, download_image, try_download_model};
use crate::error::{InferenceError, Result};
use crate::inference::{InferenceConfig, Quantization};
use crate::metadata::ModelMetadata;
use crate::postprocessing::postprocess;
use crate::preprocessing::{
    calculate_rect_size, image_to_array, preprocess_image_center_crop,
    preprocess_image_with_precision,
};
use crate::results::{Results, Speed};
use crate::task::Task;
use crate::{verbose, warn};

/// Finalize a CUDA/`TensorRT` EP builder, binding it to the `cuda-preprocess`
/// compute stream when one is supplied.
///
/// With a stream, `error_on_failure()` is set because we hand ORT a device
/// pointer: a silent CPU fallback would panic at `bind_input` (#251). The `None`
/// arm keeps ORT's default multi-EP fallback.
///
/// SAFETY: `with_compute_stream` requires the `CudaStreamHandle` to outlive
/// every `Session` bound to it; `load_with_config` guarantees this.
#[cfg(any(feature = "cuda", feature = "tensorrt"))]
macro_rules! bind_compute_stream {
    ($ep:expr, $compute_stream:expr) => {
        match $compute_stream {
            Some(s) => unsafe { $ep.with_compute_stream(s).build().error_on_failure() },
            None => $ep.build(),
        }
    };
}

/// YOLO model for inference.
///
/// This struct wraps an ONNX Runtime session and provides methods for
/// running inference on images, videos, and other sources.
///
/// # Example
///
/// ```no_run
/// use ultralytics_inference::YOLOModel;
///
/// let mut model = YOLOModel::load("yolo26n.onnx").unwrap();
/// let results = model.predict("image.jpg").unwrap();
/// println!("Found {} detections", results.len());
/// ```
pub struct YOLOModel {
    /// ONNX Runtime session.
    session: Session,
    /// Model metadata (task, classes, etc.).
    metadata: ModelMetadata,
    /// Input tensor name.
    input_name: String,
    /// Output tensor names.
    output_names: Vec<String>,
    /// Inference configuration.
    config: InferenceConfig,
    /// Whether model has been warmed up.
    warmed_up: bool,
    /// Whether model expects FP16 input.
    fp16_input: bool,
    /// Execution provider used for inference
    execution_provider: String,
    /// Whether the model accepts dynamic input shapes.
    is_dynamic: bool,
    /// Fast-path GPU preprocessor (`cuda-preprocess` feature only).
    ///
    /// Populated at load time when the user hasn't opted out and the device
    /// is CUDA/TensorRT. When `Some`, [`Self::predict_image`] routes through
    /// a fused CUDA kernel + zero-copy device input. `None` means the
    /// standard CPU preprocess path runs.
    #[cfg(feature = "cuda-preprocess")]
    cuda_preprocessor: Option<crate::cuda_inference::CudaPreprocessor>,
}

#[allow(
    clippy::too_many_lines,
    clippy::needless_pass_by_value,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::if_not_else,
    clippy::manual_is_multiple_of,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
impl YOLOModel {
    /// Load a YOLO model from an ONNX file.
    ///
    /// The model metadata (class names, task type, input size) is automatically
    /// extracted from the ONNX model's custom metadata properties.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the ONNX model file.
    ///
    /// # Errors
    ///
    /// Returns an error if the model file doesn't exist or can't be loaded.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ultralytics_inference::YOLOModel;
    ///
    /// let model = YOLOModel::load("yolo26n.onnx")?;
    /// # Ok::<(), ultralytics_inference::InferenceError>(())
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::load_with_config(path, InferenceConfig::default())
    }

    /// Load a YOLO model with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the ONNX model file.
    /// * `config` - Custom inference configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the model file doesn't exist or can't be loaded.
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn load_with_config<P: AsRef<Path>>(path: P, mut config: InferenceConfig) -> Result<Self> {
        config.normalize_precision();
        let path = path.as_ref();

        // Check if file exists, attempt auto-download if not
        let path = if path.exists() {
            std::borrow::Cow::Borrowed(path)
        } else {
            // Try to download the model if it's a known downloadable model
            std::borrow::Cow::Owned(try_download_model(path)?)
        };
        let path = path.as_ref();

        // Establish a shared cudarc stream up-front when the cuda-preprocess
        // fast path is eligible. The TRT/CUDA EPs below bind to this stream
        // via `with_compute_stream`, so the preprocess kernel and ORT see
        // each other's enqueued ops without an explicit synchronize.
        //
        // The handle is consumed by `CudaPreprocessor::finalize` further down
        // (after metadata is known) or dropped silently if the fast path
        // can't engage. Either way it outlives EP construction.
        #[cfg(feature = "cuda-preprocess")]
        let cuda_pre_stream: Option<crate::cuda_inference::CudaStreamHandle> = {
            let device_eligible = matches!(
                config.device,
                None | Some(crate::Device::Cuda(_) | crate::Device::TensorRt(_))
            );
            // Open the context on the device the session will run on. Defaulting to 0 here
            // would hand a GPU-0 pointer to a session executing on another GPU.
            let device_index = match config.device {
                Some(crate::Device::Cuda(i) | crate::Device::TensorRt(i)) => i,
                _ => 0,
            };
            if config.cuda_preprocess && device_eligible {
                match crate::cuda_inference::CudaStreamHandle::open(device_index) {
                    Ok(h) => Some(h),
                    Err(e) => {
                        warn!("cuda-preprocess: stream init failed ({e:?}); using CPU preprocess");
                        None
                    }
                }
            } else {
                None
            }
        };
        #[cfg(any(feature = "cuda", feature = "tensorrt"))]
        let cuda_pre_stream_ptr: Option<*mut ()> = {
            #[cfg(feature = "cuda-preprocess")]
            {
                cuda_pre_stream
                    .as_ref()
                    .map(crate::cuda_inference::CudaStreamHandle::raw_stream_ptr)
            }
            #[cfg(not(feature = "cuda-preprocess"))]
            {
                None
            }
        };

        // Determine optimal thread count based on available parallelism
        let num_threads = if config.num_threads > 0 {
            config.num_threads
        } else {
            // Use all available cores for intra-op parallelism (single inference)
            std::thread::available_parallelism().map_or(4, std::num::NonZero::get)
        };

        // Create ONNX Runtime session with optimizations
        let mut session_builder = Session::builder().map_err(|e| {
            InferenceError::ModelLoadError(format!("Failed to create session builder: {e}"))
        })?;

        // Register execution providers based on features and device config, as
        // `(provider, name)` pairs in registration order: the first one registered is the
        // one ORT will prefer, so it names the session (CPU when none are registered).
        #[allow(unused_mut)]
        let mut eps: Vec<(ort::ep::ExecutionProviderDispatch, &str)> = Vec::new();

        if let Some(device) = &config.device {
            // User requested specific device
            match device {
                crate::Device::Cpu => {}
                #[cfg(feature = "cuda")]
                crate::Device::Cuda(i) => eps.push((
                    Self::build_cuda_ep(*i as i32, cuda_pre_stream_ptr, config.cuda_memory_limit),
                    "CUDAExecutionProvider",
                )),
                #[cfg(feature = "coreml")]
                crate::Device::CoreMl => {
                    if matches!(Self::macos_version(), Some((major, _)) if major >= 11) {
                        eps.push((Self::build_coreml_ep(path), "CoreMLExecutionProvider"));
                    } else {
                        warn!("CoreML requires macOS 11+; falling back to CPU.");
                    }
                }
                #[cfg(feature = "tensorrt")]
                crate::Device::TensorRt(i) => eps.push((
                    Self::build_tensorrt_ep(path, *i as i32, config.quantize, cuda_pre_stream_ptr),
                    "TensorRTExecutionProvider",
                )),
                #[cfg(feature = "rocm")]
                crate::Device::Rocm(i) => eps.push((
                    ort::ep::ROCm::default().with_device_id(*i as i32).build(),
                    "ROCmExecutionProvider",
                )),
                #[cfg(feature = "directml")]
                crate::Device::DirectMl(i) => eps.push((
                    ort::ep::DirectML::default()
                        .with_device_id(*i as i32)
                        .build(),
                    "DirectMLExecutionProvider",
                )),
                #[cfg(feature = "openvino")]
                crate::Device::IntelCpu | crate::Device::IntelGpu | crate::Device::IntelNpu => {
                    let dt = match device {
                        crate::Device::IntelGpu => "GPU",
                        crate::Device::IntelNpu => "NPU",
                        _ => "CPU",
                    };
                    eps.push((
                        Self::build_openvino_ep(path, Some(dt)),
                        "OpenVINOExecutionProvider",
                    ));
                }
                #[cfg(feature = "xnnpack")]
                crate::Device::Xnnpack => eps.push((
                    ort::ep::XNNPACK::default().build(),
                    "XNNPACKExecutionProvider",
                )),
                // Handle cases where feature is disabled but enum variant exists
                #[allow(unreachable_patterns)]
                _ => {
                    warn!(
                        "Device '{device}' requested but feature not enabled or supported. Falling back to available providers."
                    );
                }
            }
        } else {
            // Default: Register all available providers in preference order
            #[cfg(feature = "tensorrt")]
            eps.push((
                Self::build_tensorrt_ep(path, 0, config.quantize, cuda_pre_stream_ptr),
                "TensorRTExecutionProvider",
            ));

            #[cfg(feature = "cuda")]
            eps.push((
                Self::build_cuda_ep(0, cuda_pre_stream_ptr, config.cuda_memory_limit),
                "CUDAExecutionProvider",
            ));

            #[cfg(feature = "coreml")]
            if matches!(Self::macos_version(), Some((major, _)) if major >= 11) {
                eps.push((Self::build_coreml_ep(path), "CoreMLExecutionProvider"));
            }

            #[cfg(feature = "rocm")]
            eps.push((ort::ep::ROCm::default().build(), "ROCmExecutionProvider"));

            #[cfg(feature = "directml")]
            eps.push((
                ort::ep::DirectML::default().build(),
                "DirectMLExecutionProvider",
            ));

            #[cfg(feature = "openvino")]
            eps.push((
                Self::build_openvino_ep(path, None),
                "OpenVINOExecutionProvider",
            ));

            #[cfg(feature = "xnnpack")]
            eps.push((
                ort::ep::XNNPACK::default().build(),
                "XNNPACKExecutionProvider",
            ));
        }

        let provider_name = eps
            .first()
            .map_or("CPUExecutionProvider", |&(_, name)| name);
        // XNNPACK runs its own threadpool whatever its priority, so this is a "registered
        // at all" check rather than "is primary": in the auto path it is registered last
        // and still executes the subgraphs the providers ahead of it decline.
        let uses_xnnpack = eps
            .iter()
            .any(|&(_, name)| name == "XNNPACKExecutionProvider");

        if !eps.is_empty() {
            crate::info!(
                "Registering {} execution providers (primary: {})",
                eps.len(),
                provider_name
            );
            let providers: Vec<_> = eps.into_iter().map(|(ep, _)| ep).collect();
            session_builder = session_builder
                .with_execution_providers(providers)
                .map_err(|e| {
                    // The GPU EPs are marked `error_on_failure` only on the
                    // cuda-preprocess fast path (see `build_cuda_ep`), so a dlopen
                    // failure surfaces here as a clear, actionable error instead of
                    // silently falling back to CPU and then panicking later at
                    // `bind_input` with "no data transfer registered" (#251).
                    let gpu = matches!(
                        provider_name,
                        "CUDAExecutionProvider" | "TensorRTExecutionProvider"
                    );
                    if gpu {
                        InferenceError::ModelLoadError(format!(
                            "{provider_name} failed to load, so the cuda-preprocess fast path \
                         cannot run on the GPU: {e}\n\
                         Hint: the downloaded ONNX Runtime is built against CUDA 13, so it needs \
                         a CUDA 13 runtime and cuDNN 9 (plus TensorRT 10 for TensorRt) installed \
                         and on the library path. On CUDA 12, compile ONNX Runtime for it and \
                         link that build with `ORT_LIB_PATH`. To use CPU preprocessing instead, \
                         set `InferenceConfig::with_cuda_preprocess(false)`."
                        ))
                    } else {
                        InferenceError::ModelLoadError(format!(
                            "Failed to set execution providers: {e}"
                        ))
                    }
                })?;
        }
        // CPU is the default - no warning needed when no accelerators are registered

        // `All` (ORT_ENABLE_ALL), not `Level3` (ORT_ENABLE_LAYOUT), reaches ONNX Runtime's
        // Level 4 pass, where `FuseFp16InitializerToFp32NodeTransformer` folds FP16
        // initializers into FP32 ones; without it an FP16 graph runs with a Cast around
        // every node. That pass only rewrites CPU-EP nodes, so other providers stay at
        // `Level3`, which also avoids an XNNPACK layout-fusion bug (onnxruntime#28540).
        let optimization_level = if provider_name == "CPUExecutionProvider" {
            ort::session::builder::GraphOptimizationLevel::All
        } else {
            ort::session::builder::GraphOptimizationLevel::Level3
        };

        if uses_xnnpack {
            session_builder = session_builder.with_intra_op_spinning(false).map_err(|e| {
                InferenceError::ModelLoadError(format!("Failed to disable intra-op spinning: {e}"))
            })?;
        }

        session_builder = session_builder
            .with_optimization_level(optimization_level)
            .map_err(|e| {
                InferenceError::ModelLoadError(format!("Failed to set optimization level: {e}"))
            })?
            .with_intra_threads(num_threads)
            .map_err(|e| {
                InferenceError::ModelLoadError(format!("Failed to set intra-op thread count: {e}"))
            })?
            .with_inter_threads(1)
            .map_err(|e| {
                InferenceError::ModelLoadError(format!("Failed to set inter-op thread count: {e}"))
            })?
            .with_memory_pattern(true)
            .map_err(|e| {
                InferenceError::ModelLoadError(format!("Failed to enable memory pattern: {e}"))
            })?;

        let session = session_builder
            .commit_from_file(path)
            .map_err(|e| InferenceError::ModelLoadError(format!("Failed to load model: {e}")))?;

        // Extract metadata from model
        let metadata = Self::extract_metadata(&session)?;

        // Get input/output names and detect input type
        let input_info = session.inputs().first();
        let input_name = input_info.map_or_else(|| "images".to_string(), |i| i.name().to_string());

        // Check if model input tensor expects FP16 (rare most models use FP32 input even with half weights)
        let fp16_input = input_info.is_some_and(|i| {
            matches!(
                i.dtype(),
                ValueType::Tensor {
                    ty: TensorElementType::Float16,
                    ..
                }
            )
        });

        // Check for a dynamic spatial input shape, i.e. `[N, 3, -1, -1]`.
        // ort 2.0 reports a dynamic/unknown dim as -1 or 0. Only the height and width
        // matter: a dynamic batch (`[-1, 3, 640, 640]`) still pins the spatial size, and
        // ONNX Runtime rejects any other height/width.
        let is_dynamic = input_info.is_some_and(|i| {
            matches!(i.dtype(), ValueType::Tensor { shape, .. }
                if shape.len() == 4 && (shape[2] <= 0 || shape[3] <= 0))
        });

        let output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        // The height and width the export pinned, whatever the batch dimension does.
        let fixed_imgsz = input_info.and_then(|i| match i.dtype() {
            ValueType::Tensor { shape, .. } if shape.len() == 4 && shape[2] > 0 && shape[3] > 0 =>
            {
                #[allow(clippy::cast_sign_loss)]
                Some((shape[2] as usize, shape[3] as usize))
            }
            _ => None,
        });

        let resolved_imgsz = if let Some(fixed) = fixed_imgsz {
            if let Some(requested) = config.imgsz
                && requested != fixed
            {
                warn!(
                    "imgsz={requested:?} ignored: this model has a fixed input shape of {fixed:?}."
                );
            }
            fixed
        } else if let Some(sz) = config.imgsz {
            sz
        } else if let Some(sz) = metadata.imgsz {
            sz
        } else {
            match metadata.task {
                Task::Obb => InferenceConfig::DEFAULT_OBB_IMGSZ,
                _ => InferenceConfig::DEFAULT_IMGSZ,
            }
        };

        let quantize = if provider_name == "TensorRTExecutionProvider" {
            Some(if config.quantize == Some(Quantization::Fp16) {
                Quantization::Fp16
            } else {
                Quantization::Fp32
            })
        } else if fp16_input {
            Some(Quantization::Fp16)
        } else if provider_name == "CPUExecutionProvider"
            && metadata.quantize == Some(Quantization::Fp16)
        {
            // ONNX Runtime widens FP16 weights to FP32 on the CPU EP, so an FP16
            // export runs at FP32 there whatever the model metadata claims.
            Some(Quantization::Fp32)
        } else {
            metadata.quantize
        };
        // `quantize` is a request, not a guarantee: the ONNX file carries its own precision
        // and each provider supports a different subset, so say so instead of running silently.
        let effective = quantize.unwrap_or(Quantization::Fp32);
        if let Some(requested) = config.quantize
            && requested != effective
        {
            crate::warn!(
                "'quantize={requested}' is unavailable for this model on {provider_name}; \
                 running at 'quantize={effective}'."
            );
        }
        let config = InferenceConfig {
            imgsz: Some(resolved_imgsz),
            quantize,
            ..config
        };

        // Ensure metadata reflects the resolved size
        let mut metadata = metadata;
        metadata.imgsz = Some(resolved_imgsz);

        // Finalize the GPU preprocessor now that we know the model input edge.
        // The handle MUST be consumed (not dropped) once `with_compute_stream`
        // has been wired into the EPs above - dropping the last `Arc<CudaStream>`
        // would invalidate the raw pointer held by ORT. So if the stream was
        // opened we always finalize the preprocessor; `predict_image`'s runtime gate
        // keeps it unused for Classify / fp16-input models.
        //
        // The buffer is sized to the resolved (possibly non-square) model input rounded
        // up to the stride, which is the largest target `rect` can ask for:
        // `calculate_rect_size` shrinks the short axis but rounds each axis *up* to the
        // stride, so a non-stride-aligned `imgsz` (e.g. 1000 -> 1024) exceeds the model
        // input itself.
        #[cfg(feature = "cuda-preprocess")]
        let cuda_preprocessor = if let Some(handle) = cuda_pre_stream {
            let stride = metadata.stride as usize;
            let round_up = |v: usize| v.div_ceil(stride) * stride;
            let (dst_h, dst_w) = (round_up(resolved_imgsz.0), round_up(resolved_imgsz.1));
            // One slot per image the model's input accepts, so a batched run fills
            // `[N, 3, H, W]` on the device instead of concatenating on the host.
            let slots = Self::fixed_batch(&session)
                .or(config.batch)
                .unwrap_or(1)
                .max(1);
            match crate::cuda_inference::CudaPreprocessor::finalize(handle, dst_h, dst_w, slots) {
                Ok(p) => Some(p),
                Err(e) => {
                    return Err(InferenceError::ModelLoadError(format!(
                        "cuda-preprocess: finalize failed ({e:?}) but stream was already \
                         bound to EPs; cannot safely drop the stream. Disable the \
                         `cuda-preprocess` feature or set with_cuda_preprocess(false)."
                    )));
                }
            }
        } else {
            None
        };

        let mut model = Self {
            session,
            metadata,
            input_name,
            output_names,
            config,
            warmed_up: false,
            fp16_input,
            execution_provider: provider_name.to_string(),
            is_dynamic,
            #[cfg(feature = "cuda-preprocess")]
            cuda_preprocessor,
        };

        // Warmup inference to trigger JIT compilation and memory allocation
        model.warmup()?;

        Ok(model)
    }

    /// Build the CUDA execution provider.
    ///
    /// TF32 is enabled. TF32 is a reduced-precision format available on
    /// NVIDIA Tensor cores (Ampere and newer) that accelerates FP32
    /// `MatMul` and `Conv` ops by truncating the mantissa to 10 bits before
    /// multiplying, while keeping FP32 range for accumulation. It typically
    /// gives a significant speedup on FP32 models with accuracy loss well
    /// below detection-threshold sensitivity. On GPUs without TF32
    /// hardware (pre-Ampere), the flag is a silent no-op cuDNN falls
    /// back to standard FP32.
    ///
    /// `compute_stream` (when `Some`) binds the EP to an external cudarc stream
    /// for the `cuda-preprocess` fast path; see [`bind_compute_stream`].
    ///
    /// `mem_limit` (when `Some`) caps the EP's memory arena at that many bytes
    /// so the GPU can be shared with other processes. The arena keeps ONNX
    /// Runtime's default extend strategy, which reuses its chunks better than
    /// an exact-size one and so peaks lower under the same cap.
    #[cfg(feature = "cuda")]
    #[allow(unsafe_code)]
    fn build_cuda_ep(
        device_id: i32,
        compute_stream: Option<*mut ()>,
        mem_limit: Option<usize>,
    ) -> ort::ep::ExecutionProviderDispatch {
        let mut ep = ort::ep::CUDA::default()
            .with_device_id(device_id)
            .with_tf32(true);

        if let Some(size) = mem_limit.filter(|&size| size > 0) {
            ep = ep.with_memory_limit(size);
        }

        bind_compute_stream!(ep, compute_stream)
    }

    /// Build the `TensorRT` execution provider with engine + timing caches enabled.
    ///
    /// FP16 is enabled for `quantize=16`. On Ada and
    /// newer GPUs this is ~2x faster than FP32 with negligible accuracy delta
    /// for YOLO detection. Engine and timing caches are written under
    /// `<model_dir>/.trt_cache/<model_stem>_{fp16,fp32}/` so subsequent loads
    /// skip the multi-minute TRT engine compile.
    ///
    /// `compute_stream` (when `Some`) binds the EP to an external cudarc stream
    /// for the `cuda-preprocess` fast path; see [`bind_compute_stream`].
    #[cfg(feature = "tensorrt")]
    #[allow(unsafe_code)]
    fn build_tensorrt_ep(
        model_path: &Path,
        device_id: i32,
        quantize: Option<Quantization>,
        compute_stream: Option<*mut ()>,
    ) -> ort::ep::ExecutionProviderDispatch {
        let stem = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model");
        let parent = model_path.parent().unwrap_or_else(|| Path::new("."));
        let fp16 = quantize == Some(Quantization::Fp16);
        let suffix = if fp16 { "fp16" } else { "fp32" };
        let cache_dir = parent.join(".trt_cache").join(format!("{stem}_{suffix}"));
        let mut ep = ort::ep::TensorRT::default()
            .with_device_id(device_id)
            .with_fp16(fp16)
            .with_max_workspace_size(4 * 1024 * 1024 * 1024)
            .with_builder_optimization_level(5);
        if crate::io::is_writable_dir(&cache_dir) {
            let cache_str = cache_dir.to_string_lossy().into_owned();
            ep = ep
                .with_engine_cache(true)
                .with_engine_cache_path(cache_str.clone())
                .with_timing_cache(true)
                .with_timing_cache_path(cache_str);
        }

        bind_compute_stream!(ep, compute_stream)
    }

    /// Read the running macOS `(major, minor)` version from `sw_vers`, or `None` if unavailable.
    #[cfg(feature = "coreml")]
    fn macos_version() -> Option<(u32, u32)> {
        let out = std::process::Command::new("sw_vers")
            .arg("-productVersion")
            .output()
            .ok()?;
        let s = std::str::from_utf8(&out.stdout).ok()?.trim();
        let mut p = s.split('.');
        Some((
            p.next()?.parse().ok()?,
            p.next().unwrap_or("0").parse().ok()?,
        ))
    }

    /// Build the `CoreML` execution provider, tuned for the model at `model_path`.
    #[cfg(feature = "coreml")]
    fn build_coreml_ep(model_path: &Path) -> ort::ep::ExecutionProviderDispatch {
        use ort::ep::coreml::{ModelFormat, SpecializationStrategy};

        // MLProgram with static input shapes and FastPrediction avoids the ORT CoreML EP
        // input-rename bug: the rename ("images" -> "graph_input_cast_0") only occurs when
        // CoreML inserts a dynamic FP32->FP16 cast, which static-shape specialization skips.
        let mut ep = ort::ep::CoreML::default()
            .with_model_format(ModelFormat::MLProgram)
            .with_specialization_strategy(SpecializationStrategy::FastPrediction)
            .with_static_input_shapes(true);

        if let Some(cache_base) = dirs::cache_dir() {
            let fnv1a = |bytes: &[u8]| {
                bytes.iter().fold(14_695_981_039_346_656_037u64, |h, &b| {
                    h.wrapping_mul(1_099_511_628_211) ^ u64::from(b)
                })
            };
            let canonical = model_path
                .canonicalize()
                .unwrap_or_else(|_| model_path.to_path_buf());
            let ort_build = fnv1a(ort::info().as_bytes());
            let stem = canonical
                .file_stem()
                .map_or_else(|| "model".to_owned(), |s| s.to_string_lossy().into_owned());
            let hash = fnv1a(canonical.as_os_str().as_encoded_bytes());
            let cache_dir = cache_base
                .join("ultralytics-inference")
                .join("coreml")
                .join(format!("{stem}_{hash:016x}_{ort_build:016x}_mlprogram"));
            if crate::io::is_writable_dir(&cache_dir) {
                ep = ep.with_model_cache_dir(cache_dir.to_string_lossy());
            }
        }
        ep.build()
    }

    /// Build the `OpenVINO` EP with a compiled-model cache, mirroring the `TensorRT` engine cache.
    ///
    /// The cache (`.ov_cache/`) lets repeat loads skip kernel recompilation, which is the dominant
    /// GPU model-load cost. Precision, streams, and thread count are intentionally left to
    /// `OpenVINO`'s defaults, which already pick the optimal low-latency configuration for a
    /// single-stream batch-1 workload (GPU runs FP16, one stream, auto thread scheduling). Forcing
    /// those knobs was measured to be a no-op at best and a regression on hybrid CPUs at worst.
    ///
    /// `device_type` is `None` when no device was requested, leaving the choice to `OpenVINO`.
    /// The cache is only passed once its directory is writable, since the GPU plugin fails
    /// compilation on a `CACHE_DIR` it cannot write.
    #[cfg(feature = "openvino")]
    fn build_openvino_ep(
        model_path: &Path,
        device_type: Option<&str>,
    ) -> ort::ep::ExecutionProviderDispatch {
        let stem = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model");
        let parent = model_path.parent().unwrap_or_else(|| Path::new("."));
        let suffix = device_type.map_or_else(|| "default".to_owned(), str::to_ascii_lowercase);
        let cache_dir = parent.join(".ov_cache").join(format!("{stem}_{suffix}"));
        let mut ep = ort::ep::OpenVINO::default();
        if let Some(device_type) = device_type {
            ep = ep.with_device_type(device_type);
        }
        if crate::io::is_writable_dir(&cache_dir) {
            ep = ep.with_cache_dir(cache_dir.to_string_lossy());
        }
        ep.build()
    }

    /// Distribute the elapsed wall time since `start` evenly across every result in the
    /// batch and stamp it onto `res.speed.postprocess`. Shared by both postprocess closures.
    fn apply_postprocess_time(batch: &mut [Vec<Results>], start: Instant, n_images_f: f64) {
        #[allow(clippy::cast_precision_loss)]
        let ms = start.elapsed().as_secs_f64() * 1000.0 / n_images_f;
        for img_results in batch {
            for res in img_results {
                res.speed.postprocess = Some(ms);
            }
        }
    }

    /// Concatenate per-image input tensor views along the batch axis into a 4D array.
    ///
    /// `label` names the precision in error messages (e.g. "FP32"). Shared tail for
    /// the FP32 and FP16 batch builders.
    fn concat_views<T: Copy + Send + Sync + Default>(
        arrays: &[ndarray::ArrayView4<'_, T>],
        label: &str,
    ) -> Result<ndarray::Array4<T>> {
        // The preprocessor hands back one contiguous `[1, 3, H, W]` block per image, all
        // the same shape, so the batch is just those blocks laid end to end. Copy them in
        // parallel: `ndarray::concatenate` clones element by element on one thread, which
        // costs more than the forward pass itself on a batched GPU run.
        if let Some((first, rest)) = arrays.split_first()
            && rest.iter().all(|a| a.shape() == first.shape())
            && let Some(slices) = arrays
                .iter()
                .map(|a| a.as_slice())
                .collect::<Option<Vec<_>>>()
        {
            let per_image = first.len();
            let (_, c, h, w) = first.dim();
            let mut buf = vec![T::default(); per_image * arrays.len()];
            buf.par_chunks_mut(per_image)
                .zip(slices)
                .for_each(|(dst, src)| dst.copy_from_slice(src));
            return ndarray::Array4::from_shape_vec((arrays.len(), c, h, w), buf).map_err(|e| {
                InferenceError::InferenceError(format!("Failed to build {label} batch: {e}"))
            });
        }

        let batch = ndarray::concatenate(ndarray::Axis(0), arrays).map_err(|e| {
            InferenceError::InferenceError(format!("Failed to concatenate {label} tensors: {e}"))
        })?;
        batch.into_dimensionality::<ndarray::Ix4>().map_err(|e| {
            InferenceError::InferenceError(format!(
                "Failed to convert concatenated tensor to 4D: {e}"
            ))
        })
    }

    /// Concatenate per-image FP32 input tensors along the batch axis.
    fn concat_f32_batch(
        preprocessed: &[crate::preprocessing::PreprocessResult],
    ) -> Result<ndarray::Array4<f32>> {
        let arrays: Vec<_> = preprocessed.iter().map(|r| r.tensor.view()).collect();
        Self::concat_views(&arrays, "FP32")
    }

    /// Concatenate per-image FP16 input tensors along the batch axis.
    fn concat_f16_batch(
        preprocessed: &[crate::preprocessing::PreprocessResult],
    ) -> Result<ndarray::Array4<f16>> {
        let arrays: Vec<_> = preprocessed
            .iter()
            .map(|r| {
                r.tensor_f16
                    .as_ref()
                    .expect("FP16 tensor should be available")
                    .view()
            })
            .collect();
        Self::concat_views(&arrays, "FP16")
    }

    /// Build per-image semantic-mask results from a batched `uint8` model output.
    ///
    /// Shared by the FP16 and FP32 semantic fast paths (models with ArgMax+Cast
    /// baked into the ONNX graph). Consumes `preprocessed_results` and
    /// `image_arrays`, slicing the batched output into per-image class maps.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::too_many_arguments,
        clippy::needless_pass_by_value
    )]
    fn semantic_mask_batch_results(
        outputs: &[(&[u8], Vec<usize>)],
        inference_ms_total: f64,
        n_images_f: f64,
        preprocess_time: f64,
        preprocessed_results: Vec<crate::preprocessing::PreprocessResult>,
        image_arrays: Vec<Array3<u8>>,
        paths: &[String],
        names: &Arc<HashMap<usize, String>>,
    ) -> Vec<Vec<Results>> {
        let inference_time = inference_ms_total / n_images_f;
        let start_postprocess = Instant::now();
        let mut batch_results: Vec<Vec<Results>> = Vec::with_capacity(image_arrays.len());

        for (i, (orig_img, preprocess_res)) in image_arrays
            .into_iter()
            .zip(preprocessed_results)
            .enumerate()
        {
            let path_i = paths.get(i).cloned().unwrap_or_default();
            let speed = Speed::new(preprocess_time, inference_time, 0.0);

            // Build the per-image slice from the batch output.
            let (data, shape) = &outputs[0];
            let actual_batch = if shape[0] > 0 { shape[0] } else { 1 };
            let elems_per_img = data.len() / actual_batch;
            let img_slice = &data[i * elems_per_img..(i + 1) * elems_per_img];
            // Per-image shape view (drops the batch dim). Zero-copy slice.
            let img_shape: &[usize] = &shape[1..];

            let tensor_shape = preprocess_res.tensor.shape();
            let inference_shape = (tensor_shape[2] as u32, tensor_shape[3] as u32);

            let result = crate::postprocessing::postprocess_semantic_mask(
                img_slice,
                img_shape,
                Arc::clone(names),
                orig_img,
                path_i,
                speed,
                inference_shape,
            );
            batch_results.push(vec![result]);
        }

        Self::apply_postprocess_time(&mut batch_results, start_postprocess, n_images_f);
        batch_results
    }

    /// Returns true when the ONNX has `ArgMax` + `Cast(uint8)` baked in, so the only output is
    /// a `[B, H, W] uint8` class map. Lets us skip f32 logits extraction + CPU argmax for semantic segmentation.
    fn has_semantic_mask_output(&self) -> bool {
        let outs = self.session.outputs();
        outs.len() == 1
            && matches!(
                outs[0].dtype(),
                ValueType::Tensor { ty: ort::value::TensorElementType::Uint8, shape, .. }
                    if shape.len() == 3
            )
    }

    /// Batch size the export pinned on the model input, or `None` when it is dynamic.
    /// Authoritative over a configured batch, the way the pinned height/width is.
    fn fixed_batch(session: &Session) -> Option<usize> {
        match session.inputs().first()?.dtype() {
            ValueType::Tensor { shape, .. } if shape.len() == 4 && shape[0] > 0 =>
            {
                #[allow(clippy::cast_sign_loss)]
                Some(shape[0] as usize)
            }
            _ => None,
        }
    }

    /// Maximum allowed image dimension to prevent OOM during warmup.
    const MAX_IMGSZ: usize = 8192;

    /// Warm up the model by running inference with a dummy input.
    ///
    /// This pre-allocates memory and optimizes the execution graph for faster
    /// subsequent inferences. Warmup is automatically called on first predict.
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn warmup(&mut self) -> Result<()> {
        if self.warmed_up {
            return Ok(());
        }

        let target_size = self
            .config
            .imgsz
            .or(self.metadata.imgsz)
            .unwrap_or(InferenceConfig::DEFAULT_IMGSZ);

        // Sanity check to prevent huge allocations from invalid imgsz
        if target_size.0 > Self::MAX_IMGSZ || target_size.1 > Self::MAX_IMGSZ {
            return Err(InferenceError::ConfigError(format!(
                "Image size {}x{} exceeds maximum allowed {}x{}",
                target_size.0,
                target_size.1,
                Self::MAX_IMGSZ,
                Self::MAX_IMGSZ
            )));
        }

        // Match whatever the export pinned on the batch axis, exactly as the height and
        // width are matched above: a `[8, 3, 640, 640]` model rejects a batch-1 warmup and
        // would fail the whole `load`. A dynamic batch takes the configured size.
        let batch =
            Self::fixed_batch(&self.session).unwrap_or_else(|| self.config.batch.unwrap_or(1));

        let warmup_result = Self::run_warmup(
            &mut self.session,
            &self.input_name,
            self.fp16_input,
            target_size,
            batch,
        );

        if let Err(e) = warmup_result {
            let msg = e.to_string();
            if !is_benign_coreml_warmup_error(&self.execution_provider, &msg) {
                return Err(e);
            }
        }

        self.warmed_up = true;
        Ok(())
    }

    /// Extract metadata from the ONNX model session.
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub(crate) fn extract_metadata(session: &Session) -> Result<ModelMetadata> {
        // Get metadata from the model
        let model_metadata = session.metadata().map_err(|e| {
            InferenceError::ModelLoadError(format!("Failed to get model metadata: {e}"))
        })?;

        // Ultralytics stores metadata under individual keys
        // Try to get each key separately and build a YAML string
        // List of all Ultralytics metadata keys
        let keys = [
            "description",
            "author",
            "date",
            "version",
            "license",
            "docs",
            "stride",
            "task",
            "batch",
            "imgsz",
            "names",
            "quantize",
            "half",
            "channels",
            "args",
            "end2end",
            "kpt_shape",
        ];

        let metadata_yaml: Vec<_> = keys
            .iter()
            .filter_map(|key| {
                model_metadata
                    .custom(key)
                    .filter(|value| !value.is_empty())
                    .map(|value| format!("{key}: {value}"))
            })
            .collect();
        if !metadata_yaml.is_empty() {
            return ModelMetadata::from_yaml_str(&metadata_yaml.join("\n"));
        }

        // Also try getting metadata from a single combined key
        for key in &["", "metadata", "model_metadata"] {
            if let Some(value) = model_metadata.custom(key)
                && !value.is_empty()
            {
                return ModelMetadata::from_yaml_str(&value);
            }
        }
        Ok(ModelMetadata::default())
    }

    /// Returns the execution provider used for inference.
    #[must_use]
    pub fn execution_provider(&self) -> &str {
        &self.execution_provider
    }

    /// Run inference on an image file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the image file.
    ///
    /// # Returns
    ///
    /// Vector of Results (one per image in batch).
    ///
    /// # Errors
    ///
    /// Returns an error if the image can't be loaded or inference fails.
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn predict<P: AsRef<Path>>(&mut self, path: P) -> Result<Vec<Results>> {
        let path = path.as_ref();

        // Warmup first to fail fast before loading/decoding the image
        self.warmup()?;

        // Load image using standard image crate
        let img = image::open(path).map_err(|e| {
            InferenceError::ImageError(format!("Failed to load image {}: {e}", path.display()))
        })?;

        self.predict_image(&img, path.to_string_lossy().to_string())
    }

    /// Run inference on a `DynamicImage`.
    ///
    /// # Arguments
    ///
    /// * `image` - The image to run inference on.
    /// * `path` - Optional path/identifier for the image.
    ///
    /// # Returns
    ///
    /// Vector of Results.
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn predict_image(&mut self, image: &DynamicImage, path: String) -> Result<Vec<Results>> {
        // Fast path: GPU preprocess + zero-copy device input.
        //
        // Allowlisted to the tasks whose preprocessing is a letterbox (square or
        // non-square) + f32 input. Classify uses center-crop (not letterbox), so
        // it's excluded. Semantic is included: `predict_image_cuda_pre` handles
        // both its f32-logits and baked-in ArgMax (u8) output forms. Depth is
        // included too: it is a plain letterbox + f32 input with a single f32
        // output, post-processed through the shared pipeline like every other
        // task. The only requirement is f32 input (the kernel writes f32, not f16).
        #[cfg(feature = "cuda-preprocess")]
        if self.cuda_preprocessor.is_some()
            && !self.fp16_input
            && matches!(
                self.metadata.task,
                Task::Detect
                    | Task::Segment
                    | Task::Pose
                    | Task::Obb
                    | Task::Semantic
                    | Task::Depth
            )
        {
            let results = self.predict_image_cuda_pre(image, path)?;
            Self::log_first_result(&results);
            return Ok(results);
        }

        let images = [image];
        let paths = [path];
        let mut results = self.predict_internal(&images, &paths)?;
        let results = results.pop().unwrap_or_default();
        Self::log_first_result(&results);
        Ok(results)
    }

    /// Whether rectangular inference applies: requested in the config and supported by
    /// the model. Fixed-shape models must letterbox to their own input size.
    const fn rect_enabled(&self) -> bool {
        self.config.rect && self.is_dynamic
    }

    /// Log the standard `image 1/1 ...` verbose line for the first result (no-op when
    /// empty), shared by both the CPU and cuda-preprocess arms of [`Self::predict_image`].
    fn log_first_result(results: &[Results]) {
        if let Some(result) = results.first() {
            let shape = result.inference_shape();
            verbose!(
                "image 1/1 {}: {}x{} {}, {:.1}ms",
                result.path,
                shape.0,
                shape.1,
                result.detection_summary(),
                result.speed.inference.unwrap_or(0.0)
            );
        }
    }

    /// Memory descriptors for a device-resident input and host-readable outputs.
    ///
    /// `device_id` must name the GPU the input buffer lives on. Naming device 0 for a
    /// session running on another GPU hands ORT a pointer it cannot read.
    #[cfg(feature = "cuda-preprocess")]
    fn device_memory_info(
        device_id: usize,
    ) -> Result<(
        ort::memory::MemoryInfo<'static>,
        ort::memory::MemoryInfo<'static>,
    )> {
        use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};

        let build = |device: AllocationDevice, id: usize, what: &str| {
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            MemoryInfo::new(
                device,
                id as i32,
                AllocatorType::Device,
                MemoryType::Default,
            )
            .map_err(|e| InferenceError::InferenceError(format!("{what} meminfo: {e}")))
        };
        Ok((
            build(AllocationDevice::CUDA, device_id, "cuda")?,
            build(AllocationDevice::CPU, 0, "cpu")?,
        ))
    }

    /// CUDA-preprocess fast path used by [`Self::predict_image`] when
    /// `cuda_preprocessor` is populated. Runs the fused letterbox+normalize kernel,
    /// hands the resulting device buffer to ORT via `TensorRefMut::from_raw`,
    /// then post-processes with the standard pipeline.
    #[cfg(feature = "cuda-preprocess")]
    #[allow(unsafe_code)]
    fn predict_image_cuda_pre(
        &mut self,
        image: &DynamicImage,
        path: String,
    ) -> Result<Vec<Results>> {
        use ort::value::TensorRefMut;

        // Computed before `run_binding` borrows the session: true when the
        // ONNX bakes in ArgMax+Cast(u8) so the single output is a uint8 class
        // map (semantic segmentation fast form).
        let semantic_u8 = self.has_semantic_mask_output();

        let start_preprocess = Instant::now();
        let rgb_img = image.to_rgb8();
        let (w, h) = (rgb_img.width(), rgb_img.height());
        let rgb_bytes = rgb_img.into_raw();

        // Same letterbox target the CPU path would pick, so both paths feed the model
        // identical pixels. Computed from the model input, not the (stride-rounded)
        // device buffer, and read before `cuda_preprocessor` borrows `self`.
        let imgsz = self
            .metadata
            .imgsz
            .unwrap_or(InferenceConfig::DEFAULT_IMGSZ);
        let target = if self.rect_enabled() {
            calculate_rect_size(w, h, imgsz, self.metadata.stride)
        } else {
            imgsz
        };
        let pre = self
            .cuda_preprocessor
            .as_mut()
            .expect("predict_image_cuda_pre invariant: cuda_preprocessor.is_some()");
        let geom = pre.preprocess(&rgb_bytes, h, w, false, target, 0)?;
        let (dst_h, dst_w) = target;
        let dev_ptr = pre.input_dev_ptr();
        let device_id = pre.device_id();
        #[allow(clippy::cast_precision_loss)]
        let preprocess_time = start_preprocess.elapsed().as_secs_f64() * 1000.0;

        let (cuda_mem, cpu_mem) = Self::device_memory_info(device_id)?;
        let shape: Vec<i64> = vec![1, 3, dst_h as i64, dst_w as i64];
        // SAFETY: dev_ptr is owned by `cuda_preprocessor` (stored on `self`) and remains
        // valid for the duration of this call. ORT consumes it during `run_binding`, which
        // is synchronized via the shared cuda stream.
        let in_tensor = unsafe {
            TensorRefMut::<f32>::from_raw(cuda_mem, dev_ptr as *mut core::ffi::c_void, shape.into())
                .map_err(|e| InferenceError::InferenceError(format!("from_raw: {e}")))?
        };

        let start_inference = Instant::now();
        let mut binding = self
            .session
            .create_binding()
            .map_err(|e| InferenceError::InferenceError(format!("create_binding: {e}")))?;
        binding
            .bind_input(&self.input_name, &in_tensor)
            .map_err(|e| InferenceError::InferenceError(format!("bind_input: {e}")))?;
        for n in &self.output_names {
            binding
                .bind_output_to_device(n, &cpu_mem)
                .map_err(|e| InferenceError::InferenceError(format!("bind_output: {e}")))?;
        }
        let outputs = self
            .session
            .run_binding(&binding)
            .map_err(|e| InferenceError::InferenceError(format!("run_binding: {e}")))?;
        binding
            .synchronize_outputs()
            .map_err(|e| InferenceError::InferenceError(format!("sync_outputs: {e}")))?;
        #[allow(clippy::cast_precision_loss)]
        let inference_time = start_inference.elapsed().as_secs_f64() * 1000.0;

        // HWC u8 ndarray for annotators/postprocess reuses the rgb buffer
        // (moved in), no copy.
        let orig_img = ndarray::Array3::from_shape_vec((h as usize, w as usize, 3), rgb_bytes)
            .map_err(|e| InferenceError::InferenceError(format!("Array3 from rgb: {e}")))?;

        let start_postprocess = Instant::now();
        let speed = Speed::new(preprocess_time, inference_time, 0.0);

        // Semantic fast form: the ONNX emits a single uint8 class map. Extract
        // it directly (no f32 logits, no CPU argmax) and run the dedicated
        // mask post-processor - mirrors the CPU `extract_and_invoke_u8` path.
        if semantic_u8 {
            let name = self.output_names.first().ok_or_else(|| {
                InferenceError::InferenceError("semantic model has no output".into())
            })?;
            let output = outputs.get(name.as_str()).ok_or_else(|| {
                InferenceError::InferenceError(format!("Output '{name}' not found"))
            })?;
            let (oshape, data) = output.try_extract_tensor::<u8>().map_err(|e| {
                InferenceError::InferenceError(format!("extract uint8 semantic output: {e}"))
            })?;
            let shape_vec = shape_to_usize(oshape);
            // Drop the leading batch dim (batch == 1 here).
            let img_shape: &[usize] = if shape_vec.len() > 1 {
                &shape_vec[1..]
            } else {
                &shape_vec
            };
            let mut result = crate::postprocessing::postprocess_semantic_mask(
                data,
                img_shape,
                Arc::clone(&self.metadata.names),
                orig_img,
                path,
                speed,
                (dst_h as u32, dst_w as u32),
            );
            #[allow(clippy::cast_precision_loss)]
            let postprocess_time = start_postprocess.elapsed().as_secs_f64() * 1000.0;
            result.speed.postprocess = Some(postprocess_time);
            return Ok(vec![result]);
        }

        // Minimal PreprocessResult - postprocess reads orig_shape, scale, padding.
        // tensor/tensor_f16 are unused in the GPU path (preprocess ran on device).
        let pre = crate::preprocessing::PreprocessResult {
            tensor: ndarray::Array4::<f32>::zeros((0, 0, 0, 0)),
            tensor_f16: None,
            orig_shape: (h, w),
            scale: (geom.scale, geom.scale),
            padding: (geom.pad_y as f32, geom.pad_x as f32),
        };
        // Reuse the shared zero-copy extraction helper (it handles the f16→f32
        // fallback) rather than duplicating the borrow-or-own here.
        let mut result =
            Self::extract_and_invoke(&outputs, &self.output_names, inference_time, |outs, _ms| {
                let img_outputs: Vec<(&[f32], Vec<usize>)> =
                    outs.iter().map(|(d, s)| (*d, s.clone())).collect();
                Ok(postprocess(
                    img_outputs,
                    self.metadata.task,
                    &pre,
                    &self.config,
                    Arc::clone(&self.metadata.names),
                    orig_img,
                    path,
                    speed,
                    (dst_h as u32, dst_w as u32),
                    self.metadata.end2end,
                    self.metadata.kpt_shape,
                ))
            })?;
        #[allow(clippy::cast_precision_loss)]
        let postprocess_time = start_postprocess.elapsed().as_secs_f64() * 1000.0;
        result.speed.postprocess = Some(postprocess_time);

        Ok(vec![result])
    }

    /// Batched twin of [`Self::predict_image_cuda_pre`]: letterboxes every image straight
    /// into its slot of the device-side `[N, 3, H, W]` input, so a batch costs no host
    /// tensor allocation and no concatenation - which otherwise dominate a batched GPU run.
    #[cfg(feature = "cuda-preprocess")]
    #[allow(unsafe_code, clippy::cast_precision_loss)]
    fn predict_batch_cuda_pre(
        &mut self,
        images: &[&DynamicImage],
        paths: &[String],
    ) -> Result<Vec<Vec<Results>>> {
        use ort::value::TensorRefMut;

        let n_images = images.len();
        let n_images_f = n_images as f64;
        let start_preprocess = Instant::now();

        // One letterbox target for the whole batch, chosen the same way the CPU batch path
        // chooses it: `rect` applies only when the model is dynamic and every source shares
        // a shape, since a mixed batch cannot share one padded target.
        let imgsz = self
            .metadata
            .imgsz
            .unwrap_or(InferenceConfig::DEFAULT_IMGSZ);
        let first_dims = images[0].dimensions();
        let uniform_shape = images.iter().all(|img| img.dimensions() == first_dims);
        let target = if self.rect_enabled() && uniform_shape {
            let (w, h) = first_dims;
            calculate_rect_size(w, h, imgsz, self.metadata.stride)
        } else {
            imgsz
        };

        // RGB extraction is per-image and host-side, so fan it out; the device uploads
        // below must stay ordered on the shared stream.
        let frames: Vec<(Vec<u8>, u32, u32)> = images
            .par_iter()
            .map(|image| {
                let rgb = image.to_rgb8();
                let (w, h) = (rgb.width(), rgb.height());
                (rgb.into_raw(), h, w)
            })
            .collect();

        let mut geoms = Vec::with_capacity(n_images);
        for (slot, (bytes, h, w)) in frames.iter().enumerate() {
            let pre = self
                .cuda_preprocessor
                .as_mut()
                .expect("predict_batch_cuda_pre invariant: cuda_preprocessor.is_some()");
            geoms.push(pre.preprocess(bytes, *h, *w, false, target, slot)?);
        }
        let preprocess_time = start_preprocess.elapsed().as_secs_f64() * 1000.0 / n_images_f;

        let (dst_h, dst_w) = target;
        let pre = self.cuda_preprocessor.as_ref().expect("cuda_preprocessor");
        let dev_ptr = pre.input_dev_ptr();
        let device_id = pre.device_id();

        let (cuda_mem, cpu_mem) = Self::device_memory_info(device_id)?;
        // The buffer was sized for `slots >= n_images` at load, checked before dispatch.
        let shape: Vec<i64> = vec![n_images as i64, 3, dst_h as i64, dst_w as i64];
        // SAFETY: as in `predict_image_cuda_pre`; the pointer is owned by
        // `cuda_preprocessor` and outlives this call.
        let in_tensor = unsafe {
            TensorRefMut::<f32>::from_raw(cuda_mem, dev_ptr as *mut core::ffi::c_void, shape.into())
                .map_err(|e| InferenceError::InferenceError(format!("from_raw: {e}")))?
        };

        let start_inference = Instant::now();
        let mut binding = self
            .session
            .create_binding()
            .map_err(|e| InferenceError::InferenceError(format!("create_binding: {e}")))?;
        binding
            .bind_input(&self.input_name, &in_tensor)
            .map_err(|e| InferenceError::InferenceError(format!("bind_input: {e}")))?;
        for n in &self.output_names {
            binding
                .bind_output_to_device(n, &cpu_mem)
                .map_err(|e| InferenceError::InferenceError(format!("bind_output: {e}")))?;
        }
        let outputs = self
            .session
            .run_binding(&binding)
            .map_err(|e| InferenceError::InferenceError(format!("run_binding: {e}")))?;
        binding
            .synchronize_outputs()
            .map_err(|e| InferenceError::InferenceError(format!("sync_outputs: {e}")))?;
        let inference_time = start_inference.elapsed().as_secs_f64() * 1000.0 / n_images_f;

        // Reuse each frame's RGB buffer as the annotator's HWC array; no copy.
        let mut origs = Vec::with_capacity(n_images);
        for (bytes, h, w) in frames {
            origs.push(
                ndarray::Array3::from_shape_vec((h as usize, w as usize, 3), bytes)
                    .map_err(|e| InferenceError::InferenceError(format!("Array3 from rgb: {e}")))?,
            );
        }

        let start_postprocess = Instant::now();
        let mut batch_results =
            Self::extract_and_invoke(&outputs, &self.output_names, inference_time, |outs, _ms| {
                let mut results = Vec::with_capacity(n_images);
                for (i, (orig_img, geom)) in origs.into_iter().zip(&geoms).enumerate() {
                    let (h, w) = (orig_img.shape()[0] as u32, orig_img.shape()[1] as u32);
                    // Slice image `i` out of each batched output. `postprocess` wants the
                    // full rank with a batch of one, not the batch dim stripped.
                    let img_outputs: Vec<(&[f32], Vec<usize>)> = outs
                        .iter()
                        .map(|(data, shape)| {
                            let per_image = data.len() / n_images;
                            let mut img_shape = shape.clone();
                            img_shape[0] = 1;
                            (&data[i * per_image..(i + 1) * per_image], img_shape)
                        })
                        .collect();
                    let pre = crate::preprocessing::PreprocessResult {
                        tensor: ndarray::Array4::<f32>::zeros((0, 0, 0, 0)),
                        tensor_f16: None,
                        orig_shape: (h, w),
                        scale: (geom.scale, geom.scale),
                        padding: (geom.pad_y as f32, geom.pad_x as f32),
                    };
                    results.push(vec![postprocess(
                        img_outputs,
                        self.metadata.task,
                        &pre,
                        &self.config,
                        Arc::clone(&self.metadata.names),
                        orig_img,
                        paths.get(i).cloned().unwrap_or_default(),
                        Speed::new(preprocess_time, inference_time, 0.0),
                        (dst_h as u32, dst_w as u32),
                        self.metadata.end2end,
                        self.metadata.kpt_shape,
                    )]);
                }
                Ok(results)
            })?;
        Self::apply_postprocess_time(&mut batch_results, start_postprocess, n_images_f);

        Ok(batch_results)
    }

    /// Run inference on the default Ultralytics sample images.
    ///
    /// Downloads default `bus.jpg` and `zidane.jpg` if not present, then runs inference on both.
    ///
    /// # Errors
    ///
    /// Returns an error if the download or inference fails.
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn predict_default(&mut self) -> Result<Vec<Results>> {
        let urls: &[&str] = if self.task() == crate::task::Task::Obb {
            &[DEFAULT_OBB_IMAGE]
        } else {
            DEFAULT_IMAGES
        };
        let mut all_results = Vec::new();
        for url in urls {
            let path = download_image(url)?;
            let results = self.predict(&path)?;
            if let Some(result) = results.into_iter().next() {
                all_results.push(result);
            }
        }
        Ok(all_results)
    }

    /// Run inference on a batch of `DynamicImage`s.
    ///
    /// # Arguments
    ///
    /// * `images` - A slice of images to run inference on.
    /// * `paths` - A slice of optional paths/identifiers for the images.
    ///
    /// # Returns
    ///
    /// Vector of Vectors of Results (one vector of results per image).
    ///
    /// # Errors
    ///
    /// Returns an error if inference fails.
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn predict_batch(
        &mut self,
        images: &[DynamicImage],
        paths: &[String],
    ) -> Result<Vec<Vec<Results>>> {
        // Create vector of references
        let image_refs: Vec<&DynamicImage> = images.iter().collect();
        self.predict_internal(&image_refs, paths)
    }

    /// Internal method to run inference on a batch of image references.
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn predict_internal(
        &mut self,
        images: &[&DynamicImage],
        paths: &[String],
    ) -> Result<Vec<Vec<Results>>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        // Same fast path (and same allowlist) as `predict_image`, filling one device slot
        // per image. The CLI reaches this even for a single image, because `BatchProcessor`
        // always calls `predict_batch`.
        // `predict_batch` is public and its length is not tied to `config.batch`, so a batch
        // larger than the buffer was sized for falls back to the CPU path rather than
        // indexing past the last slot.
        #[cfg(feature = "cuda-preprocess")]
        if images.len() > 1
            && self
                .cuda_preprocessor
                .as_ref()
                .is_some_and(|p| images.len() <= p.slots())
            && !self.fp16_input
            && !self.has_semantic_mask_output()
            && matches!(
                self.metadata.task,
                Task::Detect | Task::Segment | Task::Pose | Task::Obb | Task::Depth
            )
        {
            return self.predict_batch_cuda_pre(images, paths);
        }

        // Get target size from config or metadata
        let target_size = self
            .config
            .imgsz
            .or(self.metadata.imgsz)
            .unwrap_or(InferenceConfig::DEFAULT_IMGSZ);

        // Check if target_size is divisible by stride (one-time warning logic per batch call)
        // We only warn if the configured size itself is not divisible.
        // If rect adjusts it, that's expected.
        let stride = self.metadata.stride as usize;
        if target_size.0 % stride != 0 || target_size.1 % stride != 0 {
            warn!(
                "imgsz=[{:?}] must be multiple of max stride {}, updating to [{}, {}]",
                target_size,
                stride,
                (target_size.0 as f32 / stride as f32).ceil() as usize * stride,
                (target_size.1 as f32 / stride as f32).ceil() as usize * stride
            );
        }

        // Preprocess all images
        let start_preprocess = Instant::now();

        // Rect inference additionally needs a homogeneous batch: mixed sizes can't share
        // one padded shape, so they fall back to square below.
        let use_rect = self.rect_enabled();
        let uniform_shape = if images.len() > 1 {
            let first_dims = images[0].dimensions();
            images.iter().all(|img| img.dimensions() == first_dims)
        } else {
            true
        };
        let actual_rect = use_rect && uniform_shape;

        // Warn if rect was actually in play but disabled due to mixed batch. `use_rect` is
        // already false on a fixed-shape input, where rect never applied to begin with.
        if use_rect && !uniform_shape {
            warn!(
                "Batch contains images of different sizes. Rectangular inference disabled for this batch (falling back to square padding)."
            );
        }

        // Each image letterboxes independently, so the batch fans out across cores
        // instead of paying the per-image resize serially.
        let (stride, task, fp16_input) =
            (self.metadata.stride, self.metadata.task, self.fp16_input);
        let preprocessed_results: Vec<_> = images
            .par_iter()
            .map(|image| {
                // Determine target size for this image
                let current_target_size = if actual_rect {
                    let (w, h) = image.dimensions();
                    calculate_rect_size(w, h, target_size, stride)
                } else {
                    target_size
                };

                if task == Task::Classify {
                    preprocess_image_center_crop(
                        image,
                        current_target_size,
                        fp16_input.then_some(Quantization::Fp16),
                    )
                } else {
                    preprocess_image_with_precision(
                        image,
                        current_target_size,
                        stride,
                        fp16_input.then_some(Quantization::Fp16),
                    )
                }
            })
            .collect();
        #[allow(clippy::cast_precision_loss)]
        let preprocess_time =
            start_preprocess.elapsed().as_secs_f64() * 1000.0 / images.len() as f64;

        // Postprocess driver: runs INSIDE the inference closure so we can read the
        // ORT output buffers without copying. Returns the final batch_results.
        let n_images = images.len();
        #[allow(clippy::cast_precision_loss)]
        let n_images_f = n_images as f64;
        let task = self.metadata.task;
        let names = &self.metadata.names;
        let cfg = &self.config;
        let end2end = self.metadata.end2end;
        let kpt_shape = self.metadata.kpt_shape;

        // Compute orig_img arrays now. `image_to_array` copies the full frame, so run the
        // batch across cores like the preprocess fan-out above.
        let image_arrays: Vec<Array3<u8>> = images.par_iter().map(|i| image_to_array(i)).collect();
        // Move preprocessed_results into an Option so the closure can consume it.
        let preprocessed_results_opt = std::cell::RefCell::new(Some(preprocessed_results));
        let image_arrays_opt = std::cell::RefCell::new(Some(image_arrays));
        let paths_ref = paths;

        let postprocess_cb = |outputs: &[(&[f32], Vec<usize>)],
                              inference_ms_total: f64|
         -> Result<Vec<Vec<Results>>> {
            let inference_time = inference_ms_total / n_images_f;
            let start_postprocess = Instant::now();
            let preprocessed_results = preprocessed_results_opt
                .borrow_mut()
                .take()
                .expect("preprocessed_results");
            let image_arrays = image_arrays_opt.borrow_mut().take().expect("image_arrays");

            let mut batch_results: Vec<Vec<Results>> = Vec::with_capacity(n_images);
            for (i, (orig_img, preprocess_res)) in image_arrays
                .into_iter()
                .zip(preprocessed_results)
                .enumerate()
            {
                let path = paths_ref.get(i).cloned().unwrap_or_default();
                let speed = Speed::new(preprocess_time, inference_time, 0.0);

                let mut img_outputs = Vec::new();
                for (data, shape) in outputs {
                    let batch_size = shape[0];
                    let actual_batch_size = if batch_size > 0 { batch_size } else { 1 };
                    let total_elements = data.len();
                    let elements_per_img = total_elements / actual_batch_size;
                    let start = i * elements_per_img;
                    let end = start + elements_per_img;
                    let img_data = &data[start..end];
                    let mut img_shape = shape.clone();
                    img_shape[0] = 1;
                    img_outputs.push((img_data, img_shape));
                }

                let tensor_shape = preprocess_res.tensor.shape();
                let inference_shape = (tensor_shape[2] as u32, tensor_shape[3] as u32);

                let result = postprocess(
                    img_outputs,
                    task,
                    &preprocess_res,
                    cfg,
                    Arc::clone(names),
                    orig_img,
                    path,
                    speed,
                    inference_shape,
                    end2end,
                    kpt_shape,
                );

                batch_results.push(vec![result]);
            }

            Self::apply_postprocess_time(&mut batch_results, start_postprocess, n_images_f);
            Ok(batch_results)
        };

        // Resolve the output dtype path before the session is mutably borrowed below.
        let semantic_mask_output = self.has_semantic_mask_output();

        // One forward pass, whatever the input precision: the batch tensor only lives
        // long enough to be uploaded, while the outputs outlive it.
        // A batch of one is the common case (single image, video, webcam) and there is
        // nothing to join: `concatenate` would allocate and copy the whole NCHW tensor to
        // reproduce the one already sitting in `preprocessed_results`, so feed that instead.
        let (outputs, inference_ms_total) = {
            let pre_borrow = preprocessed_results_opt.borrow();
            let pre = pre_borrow.as_deref().expect("preprocessed_results");
            if self.fp16_input {
                match pre {
                    [single] => {
                        let tensor = single.tensor_f16.as_ref().expect("fp16 tensor");
                        Self::run_f16_input(&mut self.session, &self.input_name, tensor)?
                    }
                    batch => {
                        let batch_tensor = Self::concat_f16_batch(batch)?;
                        Self::run_f16_input(&mut self.session, &self.input_name, &batch_tensor)?
                    }
                }
            } else {
                match pre {
                    [single] => {
                        Self::run_f32_input(&mut self.session, &self.input_name, &single.tensor)?
                    }
                    batch => {
                        let batch_tensor = Self::concat_f32_batch(batch)?;
                        Self::run_f32_input(&mut self.session, &self.input_name, &batch_tensor)?
                    }
                }
            }
        };

        if semantic_mask_output {
            // Fast path: the exported ONNX graph already contains ArgMax+Cast(uint8) nodes,
            // so ONNX Runtime returns a uint8 class map directly (no f32 logits, no CPU argmax).
            debug_assert_eq!(self.metadata.task, crate::task::Task::Semantic);
            let names = &self.metadata.names;
            let preprocessed_results = preprocessed_results_opt.borrow_mut().take().unwrap();
            let image_arrays = image_arrays_opt.borrow_mut().take().unwrap();
            let mut batch_results: Vec<Vec<Results>> = Vec::new();
            Self::extract_and_invoke_u8(
                &outputs,
                &self.output_names,
                inference_ms_total,
                |outputs, inference_ms_total| {
                    batch_results = Self::semantic_mask_batch_results(
                        outputs,
                        inference_ms_total,
                        n_images_f,
                        preprocess_time,
                        preprocessed_results,
                        image_arrays,
                        paths_ref,
                        names,
                    );
                    Ok(())
                },
            )?;
            return Ok(batch_results);
        }

        Self::extract_and_invoke(
            &outputs,
            &self.output_names,
            inference_ms_total,
            postprocess_cb,
        )
    }

    /// Run the ORT session, returning outputs and measured inference time in ms.
    ///
    /// Run a single forward pass for warmup, discarding outputs.
    ///
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn run_warmup(
        session: &mut Session,
        input_name: &str,
        fp16_input: bool,
        size: (usize, usize),
        batch: usize,
    ) -> Result<()> {
        let (h, w) = size;
        if fp16_input {
            let dummy = ndarray::Array4::<f16>::zeros((batch, 3, h, w));
            let cont = dummy.as_standard_layout();
            let tensor = TensorRef::from_array_view(&cont).map_err(|e| {
                InferenceError::InferenceError(format!("Failed to create FP16 input tensor: {e}"))
            })?;
            Self::run_timed(session, ort::inputs![input_name => tensor])?;
        } else {
            let dummy = ndarray::Array4::<f32>::zeros((batch, 3, h, w));
            let cont = dummy.as_standard_layout();
            let tensor = TensorRef::from_array_view(cont.view()).map_err(|e| {
                InferenceError::InferenceError(format!("Failed to create input tensor: {e}"))
            })?;
            Self::run_timed(session, ort::inputs![input_name => tensor])?;
        }
        Ok(())
    }

    /// Associated fn (not method) so callers can split-borrow other fields of `YOLOModel`.
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn run_timed<'s>(
        session: &'s mut Session,
        inputs: Vec<(
            std::borrow::Cow<'_, str>,
            ort::session::SessionInputValue<'_>,
        )>,
    ) -> Result<(ort::session::SessionOutputs<'s>, f64)> {
        let t = Instant::now();
        let outputs = session
            .run(inputs)
            .map_err(|e| InferenceError::InferenceError(format!("Inference failed: {e}")))?;
        Ok((outputs, t.elapsed().as_secs_f64() * 1000.0))
    }

    /// Feed an FP32 input tensor and run timed inference, returning the ORT outputs.
    ///
    /// Associated fn (not method) so callers can split-borrow other fields of `YOLOModel`.
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn run_f32_input<'s>(
        session: &'s mut Session,
        input_name: &str,
        input: &ndarray::Array4<f32>,
    ) -> Result<(ort::session::SessionOutputs<'s>, f64)> {
        let input_contiguous = input.as_standard_layout();
        let input_tensor = TensorRef::from_array_view(input_contiguous.view()).map_err(|e| {
            InferenceError::InferenceError(format!("Failed to create input tensor: {e}"))
        })?;
        Self::run_timed(session, ort::inputs![input_name => input_tensor])
    }

    /// Feed an FP16 input tensor and run timed inference, returning the ORT outputs.
    fn run_f16_input<'s>(
        session: &'s mut Session,
        input_name: &str,
        input: &ndarray::Array4<f16>,
    ) -> Result<(ort::session::SessionOutputs<'s>, f64)> {
        let input_contiguous = input.as_standard_layout();
        let input_tensor = TensorRef::from_array_view(&input_contiguous).map_err(|e| {
            InferenceError::InferenceError(format!("Failed to create FP16 input tensor: {e}"))
        })?;
        Self::run_timed(session, ort::inputs![input_name => input_tensor])
    }

    /// Build zero-copy slice views over ORT output tensors and call `cb`.
    ///
    /// `cb` receives `&[(&[f32], shape)]` borrowing directly into ORT-owned device-to-host
    /// buffers (no extra Vec allocation), plus the measured `session.run()` time in ms.
    /// This avoids a ~40 ms memcpy for large semantic segmentation outputs.
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn extract_and_invoke<R>(
        outputs: &ort::session::SessionOutputs<'_>,
        output_names: &[String],
        inference_ms: f64,
        cb: impl FnOnce(&[(&[f32], Vec<usize>)], f64) -> Result<R>,
    ) -> Result<R> {
        // Holds each output as either a zero-copy borrow into the ORT buffer or, for f16
        // outputs needing dtype conversion, an owned Vec<f32>. Vec reallocation moves the
        // enum value but not the heap allocation behind `Vec<f32>`, so `.as_slice()` taken
        // in the second pass is stable.
        enum OutBuf<'a> {
            Borrow(&'a [f32]),
            Owned(Vec<f32>),
        }
        let mut bufs: Vec<(OutBuf<'_>, Vec<usize>)> = Vec::with_capacity(output_names.len());
        for output_name in output_names {
            let output = outputs.get(output_name.as_str()).ok_or_else(|| {
                InferenceError::InferenceError(format!("Output '{output_name}' not found"))
            })?;
            let (buf, shape) = if let Ok((shape, data)) = output.try_extract_tensor::<f32>() {
                let shape_vec = shape_to_usize(shape);
                (OutBuf::Borrow(data), shape_vec)
            } else {
                let (shape, data) = output.try_extract_tensor::<f16>().map_err(|e| {
                    InferenceError::InferenceError(format!("Failed to extract output: {e}"))
                })?;
                let shape_vec = shape_to_usize(shape);
                let converted: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
                (OutBuf::Owned(converted), shape_vec)
            };
            bufs.push((buf, shape));
        }
        let views: Vec<(&[f32], Vec<usize>)> = bufs
            .iter()
            .map(|(b, s)| {
                let slice: &[f32] = match b {
                    OutBuf::Borrow(d) => d,
                    OutBuf::Owned(v) => v.as_slice(),
                };
                (slice, s.clone())
            })
            .collect();

        cb(&views, inference_ms)
    }

    /// Build zero-copy `&[u8]` slice views over ORT output tensors and call `cb`
    /// (e.g. a semantic segmentation model that has ArgMax+Cast(uint8) baked in).
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn extract_and_invoke_u8<R>(
        outputs: &ort::session::SessionOutputs<'_>,
        output_names: &[String],
        inference_ms: f64,
        cb: impl FnOnce(&[(&[u8], Vec<usize>)], f64) -> Result<R>,
    ) -> Result<R> {
        // All outputs are direct borrows from ORT - no fallback path, no unsafe needed.
        let views: Vec<(&[u8], Vec<usize>)> = output_names
            .iter()
            .map(|name| {
                let output = outputs.get(name.as_str()).ok_or_else(|| {
                    InferenceError::InferenceError(format!("Output '{name}' not found"))
                })?;
                let (shape, data) = output.try_extract_tensor::<u8>().map_err(|e| {
                    InferenceError::InferenceError(format!("Failed to extract uint8 output: {e}"))
                })?;
                let shape_vec = shape_to_usize(shape);
                Ok((data, shape_vec))
            })
            .collect::<Result<_>>()?;

        cb(&views, inference_ms)
    }

    /// Get the model's task type as detected from ONNX metadata.
    ///
    /// The task is parsed automatically when the model is loaded. Use
    /// [`set_task`](Self::set_task) to override it at runtime if needed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ultralytics_inference::{Task, YOLOModel};
    /// let model = YOLOModel::load("yolo26n-seg.onnx")?;
    /// assert_eq!(model.task(), Task::Segment);
    /// # Ok::<(), ultralytics_inference::InferenceError>(())
    /// ```
    #[must_use]
    pub const fn task(&self) -> Task {
        self.metadata.task
    }

    /// Get the model's class names.
    #[must_use]
    pub fn names(&self) -> &HashMap<usize, String> {
        self.metadata.names.as_ref()
    }

    /// Get the number of classes.
    #[must_use]
    pub fn num_classes(&self) -> usize {
        self.metadata.num_classes()
    }

    /// Get the model's input size.
    #[must_use]
    pub fn imgsz(&self) -> (usize, usize) {
        self.config
            .imgsz
            .or(self.metadata.imgsz)
            .unwrap_or(match self.metadata.task {
                Task::Obb => InferenceConfig::DEFAULT_OBB_IMGSZ,
                _ => InferenceConfig::DEFAULT_IMGSZ,
            })
    }

    /// Get the model's stride.
    #[must_use]
    pub const fn stride(&self) -> u32 {
        self.metadata.stride
    }

    /// Get the resolved model precision.
    #[must_use]
    pub const fn quantize(&self) -> Option<Quantization> {
        self.config.quantize
    }

    /// Get the model metadata.
    ///
    /// Example:
    ///
    /// ```no_run
    /// use ultralytics_inference::YOLOModel;
    /// let mut model = YOLOModel::load("yolo26n.onnx")?;
    /// println!("Model name: {}", model.metadata().model_name());
    /// # Ok::<(), ultralytics_inference::InferenceError>(())
    /// ```
    #[must_use]
    pub const fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    /// Override the task type read from model metadata.
    ///
    /// Use this when you want to force a specific post-processing path without
    /// reloading the model. The model architecture must actually match the target
    /// task; overriding to a mismatched task will produce empty or incorrect
    /// results.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ultralytics_inference::{Task, YOLOModel};
    /// let mut model = YOLOModel::load("yolo26n-seg.onnx")?;
    /// // confirm the task before overriding
    /// assert_eq!(model.task(), Task::Segment);
    /// model.set_task(Task::Segment); // no-op here, but illustrates the API
    /// # Ok::<(), ultralytics_inference::InferenceError>(())
    /// ```
    pub const fn set_task(&mut self, task: crate::task::Task) {
        self.metadata.task = task;
    }
}

#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for YOLOModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("YOLOModel")
            .field("task", &self.metadata.task)
            .field("num_classes", &self.metadata.num_classes())
            .field("imgsz", &self.metadata.imgsz)
            .field("stride", &self.metadata.stride)
            .finish()
    }
}

/// Returns true if `err` is the benign `GatherElements` out-of-range error that `CoreML` produces
/// on all-zeros dummy input (issue #148). All other errors, including `graph_input_cast_0`,
/// must propagate so callers see real failures.
fn is_benign_coreml_warmup_error(provider: &str, msg: &str) -> bool {
    provider == "CoreMLExecutionProvider"
        && msg.contains("GatherElements")
        && msg.contains("Out of range")
}

/// Convert an ONNX Runtime tensor shape (`i64` dims) to `usize` for indexing.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
fn shape_to_usize(shape: &[i64]) -> Vec<usize> {
    shape.iter().map(|&d| d as usize).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_model_not_found() {
        let result = YOLOModel::load("nonexistent.onnx");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            InferenceError::ModelLoadError(_)
        ));
    }

    #[test]
    fn test_model_load_invalid_file() {
        // Create a temporary file with garbage data
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "garbage").unwrap();
        let path = file.path();

        let result = YOLOModel::load(path);
        // ONNX Runtime should fail to load this
        assert!(result.is_err());
    }

    // `#[serial]` with the other session-building test: providers compile into one shared
    // cache, and two sessions building it at once collide there.
    #[test]
    #[serial]
    fn test_model_accessors_with_dummy() {
        // The accessors need a real instance (YOLOModel wraps an ORT session and can't be
        // mocked), so this only asserts when yolo26n.onnx is present or downloadable.
        if let Ok(model) = YOLOModel::load("yolo26n.onnx") {
            assert_eq!(model.task(), Task::Detect);
            assert!(model.num_classes() > 0);
            assert_eq!(model.stride(), 32);
            assert_eq!(model.imgsz(), (640, 640)); // Default for yolo26n
            assert!(!model.names().is_empty());
            // Test Debug impl
            let debug_str = format!("{model:?}");
            assert!(debug_str.contains("YOLOModel"));
            assert!(debug_str.contains("task"));
            assert!(debug_str.contains("num_classes"));
        }
    }

    // Only the `CoreML` GatherElements warmup error may be swallowed; everything else must
    // reach the caller.
    #[test]
    fn test_warmup_error_suppression() {
        const GATHER: &str = "GatherElements op: Out of range value in index tensor";
        let cases = [
            // Issue #148, PR #149: out-of-range on an all-zeros dummy input is benign during
            // `CoreML` warmup (the DFL head gathers invalid indices for zero activations).
            ("CoreMLExecutionProvider", GATHER, true),
            // The same error on CPU/CUDA is a real bug and must not be hidden.
            ("CPUExecutionProvider", GATHER, false),
            ("CUDAExecutionProvider", GATHER, false),
            // graph_input_cast_0 is a real `CoreML` misconfiguration (MLProgram adds a cast node
            // that renames the ONNX input). The `NeuralNetwork` format prevents it, but it must
            // never be silently swallowed if it somehow reappears.
            (
                "CoreMLExecutionProvider",
                "Feature graph_input_cast_0 is required but not specified",
                false,
            ),
            // Any unrecognized `CoreML` error must propagate.
            (
                "CoreMLExecutionProvider",
                "Some unexpected `CoreML` error",
                false,
            ),
        ];
        for (provider, msg, benign) in cases {
            assert_eq!(
                is_benign_coreml_warmup_error(provider, msg),
                benign,
                "{provider}: {msg}"
            );
        }
    }

    #[cfg(all(feature = "coreml", target_os = "macos"))]
    #[test]
    fn test_macos_version_parses_on_macos() {
        let version = YOLOModel::macos_version();
        assert!(
            version.is_some(),
            "macos_version() must return Some on macOS"
        );
        let (major, _minor) = version.unwrap();
        assert!(major >= 10, "macOS major version should be >= 10");
    }

    #[test]
    fn test_apply_postprocess_time_stamps_all_results() {
        let names = Arc::new(HashMap::new());
        let result = || {
            vec![Results::new(
                Array3::zeros((4, 4, 3)),
                String::new(),
                Arc::clone(&names),
                Speed::new(0.0, 0.0, 0.0),
                (4, 4),
            )]
        };
        let mut batch: Vec<Vec<Results>> = vec![result(), result()];
        YOLOModel::apply_postprocess_time(&mut batch, Instant::now(), 2.0);
        for img in &batch {
            for r in img {
                assert!(r.speed.postprocess.is_some());
            }
        }
    }

    #[test]
    fn test_concat_f32_and_f16_batch() {
        let img = image::DynamicImage::new_rgb8(64, 48);
        // FP16 preprocessing keeps both tensors, so one pair feeds both concat paths.
        let pair = || {
            [
                crate::preprocessing::preprocess_image_with_precision(
                    &img,
                    (64, 64),
                    32,
                    Some(Quantization::Fp16),
                ),
                crate::preprocessing::preprocess_image_with_precision(
                    &img,
                    (64, 64),
                    32,
                    Some(Quantization::Fp16),
                ),
            ]
        };
        // two images stacked on the batch axis
        assert_eq!(YOLOModel::concat_f32_batch(&pair()).unwrap().dim().0, 2);
        assert_eq!(YOLOModel::concat_f16_batch(&pair()).unwrap().dim().0, 2);
    }

    #[test]
    fn test_concat_views_shape_mismatch_errors() {
        let a = ndarray::Array4::<f32>::zeros((1, 3, 4, 4));
        let b = ndarray::Array4::<f32>::zeros((1, 3, 5, 5));
        let views = [a.view(), b.view()];
        assert!(YOLOModel::concat_views(&views, "FP32").is_err());
    }

    #[test]
    fn test_semantic_mask_batch_results_builds_per_image() {
        // Baked-in ArgMax semantic output: one 2x2 uint8 class map for one image.
        let names = Arc::new(HashMap::from([
            (0usize, "bg".to_string()),
            (1, "road".to_string()),
        ]));
        let data: Vec<u8> = vec![0, 1, 1, 0];
        let outputs: Vec<(&[u8], Vec<usize>)> = vec![(data.as_slice(), vec![1, 2, 2])];

        let img = image::DynamicImage::new_rgb8(2, 2);
        let preprocessed = vec![crate::preprocessing::preprocess_image(&img, (32, 32), 32)];
        let image_arrays = vec![Array3::<u8>::zeros((2, 2, 3))];
        let paths = vec!["frame.jpg".to_string()];

        let batch = YOLOModel::semantic_mask_batch_results(
            &outputs,
            10.0,
            1.0,
            1.0,
            preprocessed,
            image_arrays,
            &paths,
            &names,
        );
        assert_eq!(batch.len(), 1);
        let results = &batch[0][0];
        let sm = results
            .semantic_mask
            .as_ref()
            .expect("semantic mask present");
        assert_eq!(sm.data.shape(), [2, 2]);
        assert!(results.speed.postprocess.is_some());
    }
}
