// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Browser/WebAssembly bindings for `ultralytics-inference`.
//!
//! Reuses the core crate's preprocessing/postprocessing so browser results match
//! the native path, and offers two ways to run a model:
//!
//! - [`YoloModel`] runs an ONNX model end to end in wasm on WebGPU via ONNX
//!   Runtime Web (bridged by [`ort-web`](https://ort.pyke.io/backends/web)):
//!   preprocess, inference, and postprocess all happen in Rust.
//! - [`YoloPipeline`] runs only the Rust preprocess/postprocess for a `.tflite`
//!   model whose inference happens in JavaScript (LiteRT.js), which has no Rust
//!   binding to drive from wasm.
//!
//! The published npm package wraps both behind a small `YOLO` class that picks
//! the path from the model file. The whole crate is gated to `wasm32`; elsewhere
//! it compiles to an empty library.
#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use js_sys::Float32Array;
use ndarray::Array3;
use ort::ep::WebGPU;
use ort::session::{RunOptions, Session};
use ort::value::{Tensor, TensorElementType, ValueType};
use ort_web::sync_outputs;
use serde::Serialize;
use wasm_bindgen::prelude::*;

use ultralytics_inference::metadata::ModelMetadata;
use ultralytics_inference::postprocessing::{postprocess, postprocess_semantic_mask};
use ultralytics_inference::preprocessing::{
    PreprocessResult, preprocess_image_center_crop, preprocess_image_with_precision,
};
use ultralytics_inference::results::Speed;
use ultralytics_inference::visualizer::color::Color;
use ultralytics_inference::visualizer::skeleton::{
    KPT_COLOR_INDICES, LIMB_COLOR_INDICES, SKELETON,
};
use ultralytics_inference::{InferenceConfig, Task};

use payload::JsResults;

mod onnx_meta;
mod payload;
mod tflite_meta;

/// Default inference image size used when a model does not record `imgsz` in its
/// metadata. Mirrors the native crate's fallback.
const DEFAULT_IMGSZ: usize = 640;

/// The device (accelerator) a model load asks for, mirroring the native
/// [`Device`](ultralytics_inference::Device) concept for the browser.
///
/// `WebGpu` needs the accelerated ONNX Runtime build (jsep); `Cpu` is the
/// portable wasm fallback that runs everywhere.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Device {
    WebGpu,
    Cpu,
}

impl Device {
    /// Parse the JS device string; anything unrecognized is the CPU fallback.
    fn parse(s: &str) -> Self {
        if s == "webgpu" {
            Self::WebGpu
        } else {
            Self::Cpu
        }
    }

    /// Whether this device needs the accelerated (jsep) runtime build versus the
    /// portable CPU/wasm build.
    fn accelerated(self) -> bool {
        self == Self::WebGpu
    }

    /// Lowercase label reported via [`YoloModel::device`], matching the native
    /// `Device` display style (`"cpu"`, `"coreml"`, ...).
    fn label(self) -> &'static str {
        match self {
            Self::WebGpu => "webgpu",
            Self::Cpu => "cpu",
        }
    }
}

/// High-resolution timestamp in milliseconds (`performance.now()`), for the
/// per-stage `speed` breakdown. Falls back to 0 if unavailable.
fn now_ms() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map_or(0.0, |p| p.now())
}

thread_local! {
    /// The backend the page's ort API was installed with: `(webgpu_runtime,
    /// self-hosted base URL)`. `ort::set_api` installs the ONNX Runtime
    /// process-globally and only once, so every model on a page shares it. We
    /// record the choice so a later load that asks for a different one is
    /// rejected rather than silently reusing the first. Wasm is single-threaded,
    /// so a thread-local is a sufficient guard.
    static BACKEND: RefCell<Option<(bool, Option<String>)>> = const { RefCell::new(None) };
}

/// Install the ort-web backend for this page, or verify a later load matches the
/// one already installed.
///
/// Loads either the WebGPU build (`webgpu = true`) or the portable CPU/wasm build
/// (`webgpu = false`, the universal fallback for browsers without WebGPU). The
/// build is fetched from `cdn.pyke.io` unless `ort_base_url` is given to
/// self-host it. The ONNX Runtime is process-global and installed once: the first
/// load wins, and a later load requesting a different backend or runtime source is
/// rejected (reload the page to switch) instead of silently ignored.
///
/// # Errors
/// Returns a JS error if the backend fails to initialize, or if it was already
/// initialized with a different `webgpu`/`ort_base_url`.
async fn ensure_backend(ort_base_url: Option<String>, webgpu: bool) -> Result<(), JsError> {
    let requested = (webgpu, ort_base_url.filter(|s| !s.is_empty()));

    if let Some(active) = BACKEND.with(|b| b.borrow().clone()) {
        if active == requested {
            return Ok(());
        }
        let label = |w: bool| if w { "WebGPU" } else { "CPU (wasm)" };
        let detail = if active.0 == requested.0 {
            format!(
                "runtime source cannot change (already loaded from {})",
                active.1.as_deref().unwrap_or("the default CDN")
            )
        } else {
            format!(
                "backend {} cannot change to {}",
                label(active.0),
                label(requested.0)
            )
        };
        return Err(JsError::new(&format!(
            "the ONNX Runtime is initialized once per page and shared by every model: \
             {detail}. Load all models with the same options, or reload the page to switch."
        )));
    }

    let api = match &requested.1 {
        // Self-hosted: pick the entrypoint + binary for the chosen build. The
        // wrapper (.mjs) name defaults to the binary with `.wasm` -> `.mjs`.
        Some(base) => {
            let (script, binary) = if webgpu {
                ("ort.webgpu.min.js", "ort-wasm-simd-threaded.jsep.wasm")
            } else {
                ("ort.wasm.min.js", "ort-wasm-simd-threaded.wasm")
            };
            let dist = ort_web::Dist::new(base.clone())
                .with_script_name(script)
                .with_binary_name(binary);
            ort_web::api(dist).await
        }
        None => {
            let feature = if webgpu {
                ort_web::FEATURE_WEBGPU
            } else {
                ort_web::FEATURE_NONE
            };
            ort_web::api(feature).await
        }
    }
    .map_err(err_ctx("failed to initialize ort-web backend"))?;
    ort::set_api(api);
    BACKEND.with(|b| *b.borrow_mut() = Some(requested));
    Ok(())
}

/// Parse Ultralytics `key: value` metadata `text` into [`ModelMetadata`], shared
/// by the ONNX and `.tflite` readers. Returns `missing` as a JS error when the
/// model carries no metadata (empty/absent text).
fn metadata_from_text(text: Option<String>, missing: &str) -> Result<ModelMetadata, JsError> {
    let text = text
        .filter(|t| !t.is_empty())
        .ok_or_else(|| JsError::new(missing))?;
    ModelMetadata::from_yaml_str(&text).map_err(err_ctx("failed to parse model metadata"))
}

/// Parse the Ultralytics metadata embedded in an ONNX model's `metadata_props`.
///
/// `ort-web` does not implement ONNX session metadata retrieval, so we read the
/// `metadata_props` (key/value pairs such as `task`, `names`, `imgsz`) straight
/// from the model protobuf and rebuild the `key: value` text the shared parser
/// consumes.
fn build_metadata(model_bytes: &[u8]) -> Result<ModelMetadata, JsError> {
    let props = onnx_meta::parse_metadata_props(model_bytes);
    let text = (!props.is_empty()).then(|| {
        props
            .iter()
            .map(|(k, v)| format!("{k}: {v}"))
            .collect::<Vec<_>>()
            .join("\n")
    });
    metadata_from_text(
        text,
        "no metadata found in ONNX model. Export it with Ultralytics \
         (`model.export(format='onnx')`) so the task, class names, and imgsz are embedded.",
    )
}

/// Parse the Ultralytics metadata embedded in a single-file `.tflite`.
///
/// LiteRT exports append a small zip holding `metadata.json` to the model bytes,
/// so, like the ONNX path's [`build_metadata`], the browser reads the task, class
/// names, and `imgsz` straight from the model file.
fn build_tflite_metadata(model_bytes: &[u8]) -> Result<ModelMetadata, JsError> {
    metadata_from_text(
        tflite_meta::metadata_text(model_bytes),
        "no Ultralytics metadata found in the .tflite model. Export it with Ultralytics \
         (`model.export(format='litert')`) so the task, class names, and imgsz are embedded.",
    )
}

/// Build the original RGB image (HWC u8, for postprocess coordinate scaling) and
/// the NCHW f32 input tensor. Classification center-crops, every other task
/// letterboxes. Shared by the ONNX and LiteRT paths.
fn preprocess_image(
    dynimg: &image::DynamicImage,
    imgsz: (usize, usize),
    stride: u32,
    task: Task,
) -> Result<(Array3<u8>, PreprocessResult), JsError> {
    let rgb = dynimg.to_rgb8();
    let (w, h) = rgb.dimensions();
    let orig_img = Array3::from_shape_vec((h as usize, w as usize, 3), rgb.into_raw())
        .map_err(err_ctx("failed to build image array"))?;
    let pre = if task == Task::Classify {
        preprocess_image_center_crop(dynimg, imgsz, false)
    } else {
        preprocess_image_with_precision(dynimg, imgsz, stride, false)
    };
    Ok((orig_img, pre))
}

/// Build the shared `InferenceConfig` from the JS thresholds and class filter.
fn make_config(conf: f32, iou: f32, classes: Option<Vec<u32>>) -> InferenceConfig {
    let mut config = InferenceConfig::new().with_confidence(conf).with_iou(iou);
    if let Some(classes) = classes {
        config = config.with_classes(classes.into_iter().map(|c| c as usize).collect());
    }
    config
}

/// A loaded ONNX model that runs end to end in wasm (preprocess, inference on
/// ONNX Runtime Web, and postprocess), for the `.onnx` path.
///
/// Created via [`YoloModel::load_bytes`]; run with [`YoloModel::predict`]. The
/// TypeScript `YOLO` class wraps this for ONNX models, and [`YoloPipeline`] for
/// `.tflite` ones.
#[wasm_bindgen]
pub struct YoloModel {
    session: Session,
    metadata: ModelMetadata,
    output_names: Vec<String>,
    /// Network input size as `(height, width)`.
    imgsz: (usize, usize),
    /// Active device label (`"webgpu"` or `"cpu"`).
    device: &'static str,
}

#[wasm_bindgen]
impl YoloModel {
    /// Load a model from raw ONNX bytes (fetched by the JS wrapper).
    ///
    /// Initializes the accelerated backend on first use, reads the embedded
    /// Ultralytics metadata from the model bytes, then commits an ONNX Runtime
    /// session on the requested execution provider.
    ///
    /// `device` is `"webgpu"` or `"cpu"` (the browser picks the GPU adapter
    /// automatically). WebGPU is registered with `error_on_failure`, so if the
    /// session cannot commit on it the load falls back to CPU and
    /// [`device`](Self::device) reports what actually ran.
    ///
    /// # Errors
    /// Returns a JS error if the backend cannot start, the bytes are not a valid
    /// model, or the model lacks Ultralytics metadata.
    pub async fn load_bytes(
        bytes: Vec<u8>,
        ort_base_url: Option<String>,
        device: String,
    ) -> Result<YoloModel, JsError> {
        let want = Device::parse(&device);
        ensure_backend(ort_base_url, want.accelerated()).await?;
        let metadata = build_metadata(&bytes)?;
        if want != Device::Cpu
            && let Ok(session) = Self::commit(&bytes, want).await
        {
            return Self::from_session(session, metadata, want);
        }
        let session = Self::commit(&bytes, Device::Cpu).await?;
        Self::from_session(session, metadata, Device::Cpu)
    }

    /// The model's task (`"detect"`, `"segment"`, `"pose"`, `"classify"`,
    /// `"obb"`, or `"semantic"`).
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn task(&self) -> String {
        self.metadata.task.as_str().to_owned()
    }

    /// The active device: `"webgpu"` or `"cpu"` (the fallback when WebGPU is
    /// unavailable). Mirrors the native `Device` display.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn device(&self) -> String {
        self.device.to_string()
    }

    /// Class id -> name map (like Ultralytics `model.names`), as a JS object.
    ///
    /// # Errors
    /// Returns a JS error only if serialization fails (not expected).
    #[wasm_bindgen(getter)]
    pub fn names(&self) -> Result<JsValue, JsError> {
        to_js(&*self.metadata.names, "names")
    }

    /// Run inference on a single encoded image (JPEG or PNG bytes).
    ///
    /// `conf` and `iou` are the confidence and NMS IoU thresholds (pass the model
    /// defaults 0.25 / 0.7 to match Ultralytics); `classes` optionally keeps only
    /// the given class ids (for semantic, other pixels become background).
    /// Returns a plain JS object whose shape mirrors the Ultralytics `Results` API.
    ///
    /// # Errors
    /// Returns a JS error if the image cannot be decoded or inference fails.
    pub async fn predict(
        &mut self,
        image: Vec<u8>,
        conf: f32,
        iou: f32,
        classes: Option<Vec<u32>>,
    ) -> Result<JsValue, JsError> {
        let dynimg = image::load_from_memory(&image).map_err(err_ctx("failed to decode image"))?;
        self.run(dynimg, conf, iou, classes).await
    }

    /// Run inference on raw `RGBA` pixels (e.g. a canvas/webcam `ImageData`).
    ///
    /// Skips image encoding/decoding entirely, so it is the fast path for live
    /// video. `rgba` is `width * height * 4` bytes, row-major.
    ///
    /// # Errors
    /// Returns a JS error if the buffer size is wrong or inference fails.
    pub async fn predict_rgba(
        &mut self,
        rgba: Vec<u8>,
        width: u32,
        height: u32,
        conf: f32,
        iou: f32,
        classes: Option<Vec<u32>>,
    ) -> Result<JsValue, JsError> {
        let expected = (width as usize) * (height as usize) * 4;
        if rgba.len() != expected {
            return Err(JsError::new(&format!(
                "rgba buffer is {} bytes, expected {expected} for {width}x{height}",
                rgba.len()
            )));
        }
        let img = image::RgbaImage::from_raw(width, height, rgba)
            .ok_or_else(|| JsError::new("failed to build image from rgba buffer"))?;
        self.run(image::DynamicImage::ImageRgba8(img), conf, iou, classes)
            .await
    }
}

impl YoloModel {
    /// Commit an ONNX Runtime session from model bytes on `device`.
    ///
    /// WebGPU is registered with `error_on_failure`, so a missing or broken
    /// adapter surfaces as an error (which [`load_bytes`](Self::load_bytes)
    /// catches to fall back to CPU) instead of a silent downgrade we would
    /// mislabel.
    async fn commit(bytes: &[u8], device: Device) -> Result<Session, JsError> {
        let mut builder = Session::builder().map_err(map_ort)?;
        if device == Device::WebGpu {
            builder = builder
                .with_execution_providers([WebGPU::default().build().error_on_failure()])
                .map_err(map_ort)?;
        }
        builder
            .commit_from_memory(bytes)
            .await
            .map_err(err_ctx("failed to load model from bytes"))
    }

    /// Finish constructing a model from a committed session and its parsed
    /// metadata by resolving the input size and output names.
    fn from_session(
        session: Session,
        metadata: ModelMetadata,
        device: Device,
    ) -> Result<Self, JsError> {
        let imgsz = metadata.imgsz.unwrap_or((DEFAULT_IMGSZ, DEFAULT_IMGSZ));
        let output_names = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        Ok(Self {
            session,
            metadata,
            output_names,
            imgsz,
            device: device.label(),
        })
    }

    /// Returns `true` when the ONNX has `ArgMax` + `Cast(uint8)` baked in, so the only
    /// output is a `[B, H, W] uint8` class map. Lets `run` skip f32 logits extraction
    /// and use the dedicated semantic-mask postprocessor (matches native model.rs).
    fn semantic_baked(&self) -> bool {
        let outputs = self.session.outputs();
        outputs.len() == 1
            && matches!(
                outputs[0].dtype(),
                ValueType::Tensor { ty: TensorElementType::Uint8, shape, .. }
                    if shape.len() == 3
            )
    }

    /// Core async predict: decode -> preprocess -> `run_async` -> sync -> postprocess.
    ///
    /// The three timed stages mirror the Ultralytics `speed` breakdown:
    /// `preprocess` = JPEG/PNG decode + letterbox/normalize; `inference` =
    /// `run_async` + cross-context output sync; `postprocess` = NMS, decoding,
    /// and coordinate scaling. (Image *download* happens in JS before this and is
    /// not counted here.)
    async fn run(
        &mut self,
        dynimg: image::DynamicImage,
        conf: f32,
        iou: f32,
        classes: Option<Vec<u32>>,
    ) -> Result<JsValue, JsError> {
        let t_pre = now_ms();

        let (orig_img, pre) = preprocess_image(
            &dynimg,
            self.imgsz,
            self.metadata.stride,
            self.metadata.task,
        )?;

        // Resolve the output dtype path before borrowing the session for inference.
        let semantic_baked = self.semantic_baked();

        // Upload the NCHW f32 tensor into the ORT (WebGPU) context and run.
        let t_inf = now_ms();
        let input = Tensor::from_array(pre.tensor.clone()).map_err(map_ort)?;
        let run_options = RunOptions::new().map_err(map_ort)?;
        let mut outputs = self
            .session
            .run_async(ort::inputs![input.view()], &run_options)
            .await
            .map_err(err_ctx("inference failed"))?;
        // Outputs live in the ONNX Runtime wasm context; copy them back to Rust.
        sync_outputs(&mut outputs)
            .await
            .map_err(err_ctx("failed to sync outputs"))?;

        let t_post = now_ms();
        let config = make_config(conf, iou, classes);
        let names: Arc<HashMap<usize, String>> = Arc::clone(&self.metadata.names);
        let inference_shape = (self.imgsz.0 as u32, self.imgsz.1 as u32);
        // Postprocess time runs from output extraction to the postprocess call.
        let speed = || Speed::new(t_inf - t_pre, t_post - t_inf, now_ms() - t_post);

        // Baked-argmax semantic models output a single `[B, H, W] uint8` class map
        // (ArgMax + Cast folded into the graph).
        let results = if semantic_baked {
            let name = &self.output_names[0];
            let (shape, data) = outputs[name.as_str()]
                .try_extract_tensor::<u8>()
                .map_err(|e| JsError::new(&format!("failed to extract output '{name}': {e}")))?;
            let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            // Drop the leading batch dim so the shape is `[H, W]` (or pass through).
            let img_shape: &[usize] = if shape.len() == 3 {
                &shape[1..]
            } else {
                &shape
            };
            postprocess_semantic_mask(
                data,
                img_shape,
                names,
                orig_img,
                String::new(),
                speed(),
                inference_shape,
            )
        } else {
            // Borrow each output's data directly (no copy) and feed it to the shared
            // f32 postprocessor while `outputs` is still alive.
            let mut views: Vec<(&[f32], Vec<usize>)> = Vec::with_capacity(self.output_names.len());
            for name in &self.output_names {
                let (shape, data) =
                    outputs[name.as_str()]
                        .try_extract_tensor::<f32>()
                        .map_err(|e| {
                            JsError::new(&format!("failed to extract output '{name}': {e}"))
                        })?;
                views.push((data, shape.iter().map(|&d| d as usize).collect()));
            }
            postprocess(
                views,
                self.metadata.task,
                &pre,
                &config,
                names,
                orig_img,
                String::new(),
                speed(),
                inference_shape,
                self.metadata.end2end,
                self.metadata.kpt_shape,
            )
        };
        let payload = JsResults::from_results(&results, self.metadata.task);
        to_js(&payload, "results")
    }
}

/// Per-frame state carried from `preprocess_rgba` to the matching `postprocess`.
struct PendingFrame {
    /// Letterbox geometry (scale/padding/orig_shape) for coordinate scaling.
    pre: PreprocessResult,
    /// Original RGB image (HWC u8), needed for mask compositing.
    orig_img: Array3<u8>,
    /// Preprocess time in ms, for the `speed` breakdown.
    pre_ms: f64,
}

/// A metadata-only YOLO pre/post pipeline for use with an **external** inference
/// engine (e.g. LiteRT.js running a `.tflite` model in JavaScript).
///
/// Unlike [`YoloModel`], it holds no ONNX Runtime session: JavaScript loads and
/// runs the model, while this struct reuses the shared Rust preprocessing and
/// postprocessing so results match every other path. Per frame the flow is
/// [`preprocess_rgba`](Self::preprocess_rgba) → (JS engine inference) →
/// [`postprocess`](Self::postprocess). It assumes a single prediction in flight
/// at a time (as the webcam render loop does).
#[wasm_bindgen]
pub struct YoloPipeline {
    metadata: ModelMetadata,
    imgsz: (usize, usize),
    pending: Option<PendingFrame>,
}

#[wasm_bindgen]
impl YoloPipeline {
    /// Build a pipeline from a single-file `.tflite` model, reading the
    /// Ultralytics metadata (task, class names, `imgsz`, ...) embedded in the
    /// model bytes, the same way [`YoloModel::load_bytes`] reads it from an ONNX
    /// model, so the ONNX and LiteRT paths load identically from one file.
    ///
    /// # Errors
    /// Returns a JS error if the model carries no Ultralytics metadata or it
    /// cannot be parsed.
    #[wasm_bindgen(constructor)]
    pub fn new(tflite: &[u8]) -> Result<YoloPipeline, JsError> {
        let metadata = build_tflite_metadata(tflite)?;
        let imgsz = metadata.imgsz.unwrap_or((DEFAULT_IMGSZ, DEFAULT_IMGSZ));
        Ok(Self {
            metadata,
            imgsz,
            pending: None,
        })
    }

    /// The model's task (`"detect"`, `"segment"`, ...).
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn task(&self) -> String {
        self.metadata.task.as_str().to_owned()
    }

    /// Whether this is an end-to-end (NMS-free) export, e.g. YOLO26. Its head runs
    /// the NMS/top-k with `int64`/`gather_nd` ops that the LiteRT WebGPU delegate
    /// cannot execute, so such models must run on the CPU (wasm) accelerator.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn end2end(&self) -> bool {
        self.metadata.end2end
    }

    /// Class id -> name map (like `model.names`), as a JS object.
    ///
    /// # Errors
    /// Returns a JS error only if serialization fails (not expected).
    #[wasm_bindgen(getter)]
    pub fn names(&self) -> Result<JsValue, JsError> {
        to_js(&*self.metadata.names, "names")
    }

    /// The model input shape as `[1, 3, H, W]` (NCHW), for sizing the engine's
    /// input tensor. Ultralytics LiteRT exports (`ai_edge_torch`) keep the native
    /// NCHW layout, matching the ONNX path.
    #[wasm_bindgen(getter, js_name = inputShape)]
    #[must_use]
    pub fn input_shape(&self) -> Vec<u32> {
        vec![1, 3, self.imgsz.0 as u32, self.imgsz.1 as u32]
    }

    /// Preprocess raw RGBA pixels (e.g. a webcam `ImageData`) into the NCHW f32
    /// input tensor an Ultralytics LiteRT/TFLite model expects, normalized to
    /// `[0, 1]`, the same preprocessing as the ONNX path.
    ///
    /// Returns the tensor as a `Float32Array` (shape `[1, 3, H, W]`); the
    /// letterbox geometry and original image are stashed for the matching
    /// [`postprocess`](Self::postprocess) call.
    ///
    /// # Errors
    /// Returns a JS error if the buffer size is not `width * height * 4`.
    pub fn preprocess_rgba(
        &mut self,
        rgba: Vec<u8>,
        width: u32,
        height: u32,
    ) -> Result<Float32Array, JsError> {
        let t0 = now_ms();
        let expected = (width as usize) * (height as usize) * 4;
        if rgba.len() != expected {
            return Err(JsError::new(&format!(
                "rgba buffer is {} bytes, expected {expected} for {width}x{height}",
                rgba.len()
            )));
        }
        let img = image::RgbaImage::from_raw(width, height, rgba)
            .ok_or_else(|| JsError::new("failed to build image from rgba buffer"))?;
        let dynimg = image::DynamicImage::ImageRgba8(img);

        let (orig_img, pre) = preprocess_image(
            &dynimg,
            self.imgsz,
            self.metadata.stride,
            self.metadata.task,
        )?;
        let data = pre
            .tensor
            .as_slice()
            .ok_or_else(|| JsError::new("preprocessed tensor is not contiguous"))?;
        let out = Float32Array::from(data);

        self.pending = Some(PendingFrame {
            pre,
            orig_img,
            pre_ms: now_ms() - t0,
        });
        Ok(out)
    }

    /// Postprocess an external engine's raw outputs into the standard `Results`.
    ///
    /// `outputs` are the model's output tensors (one for most tasks; two for
    /// segmentation: detection head + mask prototypes). `shapes` encodes each
    /// output's dims flat as `[rank0, dims0..., rank1, dims1...]`. `inference_ms`
    /// is the time the JS engine reported, used only for the `speed` breakdown.
    /// Consumes the frame stashed by the preceding
    /// [`preprocess_rgba`](Self::preprocess_rgba).
    ///
    /// # Errors
    /// Returns a JS error if called before `preprocess_rgba`, if `shapes` is
    /// malformed, or on serialization failure.
    pub fn postprocess(
        &mut self,
        outputs: Vec<Float32Array>,
        shapes: Vec<u32>,
        inference_ms: f64,
        conf: f32,
        iou: f32,
        classes: Option<Vec<u32>>,
    ) -> Result<JsValue, JsError> {
        let t_post = now_ms();
        let pending = self
            .pending
            .take()
            .ok_or_else(|| JsError::new("postprocess called before preprocess_rgba"))?;

        let mut bufs: Vec<Vec<f32>> = outputs.iter().map(|a| a.to_vec()).collect();
        if bufs.is_empty() {
            return Err(JsError::new(
                "external engine returned no output tensors (the model may have failed to run)",
            ));
        }
        let mut shape_vecs = decode_shapes(&shapes, bufs.len())?;

        // Order so the detection head (rank 3, e.g. [1, C, 8400]) is first and the
        // segmentation prototype tensor (rank 4) second, matching the shared
        // postprocessor regardless of the engine's output order.
        if bufs.len() == 2 && shape_vecs[0].len() == 4 {
            bufs.swap(0, 1);
            shape_vecs.swap(0, 1);
        }

        // Ultralytics LiteRT exports (`ai_edge_torch`) emit box (and pose keypoint)
        // coordinates normalized to [0, 1]; the shared postprocess expects
        // model-input pixels. Scale them on the detection head in place.
        denormalize_head(
            &mut bufs[0],
            &shape_vecs[0],
            self.imgsz,
            self.metadata.task,
            self.metadata.kpt_shape,
        );

        let views: Vec<(&[f32], Vec<usize>)> = bufs
            .iter()
            .zip(&shape_vecs)
            .map(|(b, s)| (b.as_slice(), s.clone()))
            .collect();

        let config = make_config(conf, iou, classes);
        let names: Arc<HashMap<usize, String>> = Arc::clone(&self.metadata.names);
        let inference_shape = (self.imgsz.0 as u32, self.imgsz.1 as u32);
        // Postprocess time is unknown until the call returns, so pass 0 here and
        // stamp the real duration onto the result afterwards (as the native path does).
        let speed = Speed::new(pending.pre_ms, inference_ms, 0.0);

        let mut results = postprocess(
            views,
            self.metadata.task,
            &pending.pre,
            &config,
            names,
            pending.orig_img,
            String::new(),
            speed,
            inference_shape,
            self.metadata.end2end,
            self.metadata.kpt_shape,
        );
        results.speed.postprocess = Some(now_ms() - t_post);
        let payload = JsResults::from_results(&results, self.metadata.task);
        to_js(&payload, "results")
    }
}

/// Decode the flat shape encoding `[rank0, dims0..., rank1, dims1...]` into one
/// `Vec<usize>` per output tensor.
fn decode_shapes(flat: &[u32], count: usize) -> Result<Vec<Vec<usize>>, JsError> {
    let mut shapes = Vec::with_capacity(count);
    let mut i = 0;
    while i < flat.len() {
        let rank = flat[i] as usize;
        i += 1;
        if i + rank > flat.len() {
            return Err(JsError::new("malformed output shapes"));
        }
        shapes.push(flat[i..i + rank].iter().map(|&d| d as usize).collect());
        i += rank;
    }
    if shapes.len() != count {
        return Err(JsError::new(&format!(
            "expected {count} output shapes, decoded {}",
            shapes.len()
        )));
    }
    Ok(shapes)
}

/// Scale a normalized detection head's coordinates to model-input pixels in place.
///
/// Some LiteRT exports emit box (and pose keypoint) coordinates normalized to
/// `[0, 1]`, while the shared postprocess expects model-input pixels. Handles both
/// `[1, C, N]` and `[1, N, C]` layouts; a max-coordinate guard makes it a no-op for
/// pixel-coordinate exports and for box-less tasks (classify/semantic). The OBB
/// angle and keypoint confidence channels are left untouched.
fn denormalize_head(
    buf: &mut [f32],
    shape: &[usize],
    imgsz: (usize, usize),
    task: Task,
    kpt_shape: Option<(usize, usize)>,
) {
    if !matches!(task, Task::Detect | Task::Segment | Task::Pose | Task::Obb) {
        return;
    }
    if shape.len() != 3 || buf.is_empty() {
        return;
    }
    let (d1, d2) = (shape[1], shape[2]);
    // Channel-major [1, C, N] when C <= N (e.g. [1, 84, 8400]); else [1, N, C].
    let channel_major = d1 <= d2;
    let (n, c_count) = if channel_major { (d2, d1) } else { (d1, d2) };
    if c_count < 4 || n == 0 {
        return;
    }
    let idx = |c: usize, p: usize| {
        if channel_major {
            c * n + p
        } else {
            p * c_count + c
        }
    };

    // Only scale when coords look normalized, so a pixel-coord export is untouched.
    let mut max_coord = 0.0f32;
    for p in 0..n {
        for c in 0..4 {
            max_coord = max_coord.max(buf[idx(c, p)].abs());
        }
    }
    if max_coord > 2.0 {
        return;
    }

    // cx, w scale by width; cy, h scale by height. imgsz is (height, width).
    let (w, h) = (imgsz.1 as f32, imgsz.0 as f32);
    for (c, &m) in [w, h, w, h].iter().enumerate() {
        for p in 0..n {
            buf[idx(c, p)] *= m;
        }
    }

    // Pose: keypoint x/y (confidence untouched). The keypoint block follows the
    // box + class channels: it occupies the last `nkpt * ndim` channels.
    if task == Task::Pose
        && let Some((nkpt, ndim)) = kpt_shape
        && ndim >= 2
    {
        let start = c_count.saturating_sub(nkpt * ndim);
        for k in 0..nkpt {
            let xc = start + k * ndim;
            let yc = xc + 1;
            if yc < c_count {
                for p in 0..n {
                    buf[idx(xc, p)] *= w;
                    buf[idx(yc, p)] *= h;
                }
            }
        }
    }
}

/// Map any `ort::Error<M>` (the marker `M` records the originating operation) to
/// a JS error.
fn map_ort<M>(e: ort::Error<M>) -> JsError {
    JsError::new(&format!("ort error: {e}"))
}

/// Build a `.map_err` closure that prefixes any displayable error with `context`,
/// so call sites read `.map_err(err_ctx("inference failed"))`.
fn err_ctx<E: std::fmt::Display>(context: &'static str) -> impl FnOnce(E) -> JsError {
    move |e| JsError::new(&format!("{context}: {e}"))
}

/// Serialize a value to a `JsValue`, mapping a failure to a JS error naming `what`.
fn to_js<T: Serialize>(value: &T, what: &str) -> Result<JsValue, JsError> {
    serde_wasm_bindgen::to_value(value)
        .map_err(|e| JsError::new(&format!("failed to serialize {what}: {e}")))
}

/// The Ultralytics pose drawing scheme: skeleton connectivity plus the per-limb
/// and per-keypoint palette colors. Lets the JS annotator draw pose exactly like
/// the native renderer without duplicating any palette.
#[derive(Serialize)]
struct PoseScheme {
    /// Keypoint index pairs connected by limbs.
    skeleton: Vec<[usize; 2]>,
    /// Hex color per keypoint (length 17).
    keypoint_colors: Vec<String>,
    /// Hex color per limb (length 19, aligned with `skeleton`).
    limb_colors: Vec<String>,
}

/// Return the Ultralytics pose skeleton + keypoint/limb colors (constant). The
/// JS annotator calls this once to color pose overlays from the shared palette.
///
/// # Errors
/// Returns a JS error only if serialization fails (not expected).
#[wasm_bindgen]
pub fn pose_palette() -> Result<JsValue, JsError> {
    let scheme = PoseScheme {
        skeleton: SKELETON.to_vec(),
        keypoint_colors: KPT_COLOR_INDICES
            .iter()
            .map(|&i| Color::from_pose_index(i).to_hex())
            .collect(),
        limb_colors: LIMB_COLOR_INDICES
            .iter()
            .map(|&i| Color::from_pose_index(i).to_hex())
            .collect(),
    };
    to_js(&scheme, "pose palette")
}

/// Install a panic hook that logs Rust panics to the browser console.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}
