// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Browser/WebAssembly bindings for `ultralytics-inference`.
//!
//! Runs YOLO ONNX models in the browser on WebGPU via ONNX Runtime Web (bridged
//! by [`ort-web`](https://ort.pyke.io/backends/web)), reusing the core crate's
//! preprocessing/postprocessing so results match the native and Python paths.
//! The published npm package wraps this behind a small `YOLO` class. The whole
//! crate is gated to `wasm32`; elsewhere it compiles to an empty library.
#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use ndarray::Array3;
use ort::ep::WebGPU;
use ort::session::{RunOptions, Session};
use ort::value::Tensor;
use ort_web::sync_outputs;
use serde::Serialize;
use wasm_bindgen::prelude::*;

use ultralytics_inference::metadata::ModelMetadata;
use ultralytics_inference::postprocessing::postprocess;
use ultralytics_inference::preprocessing::{
    preprocess_image_center_crop, preprocess_image_with_precision,
};
use ultralytics_inference::results::{Results, SemanticMask, Speed};
use ultralytics_inference::visualizer::color::Color;
use ultralytics_inference::visualizer::skeleton::{
    KPT_COLOR_INDICES, LIMB_COLOR_INDICES, SKELETON,
};
use ultralytics_inference::{InferenceConfig, Task};

mod onnx_meta;

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

/// Parse the Ultralytics metadata embedded in an ONNX model's `metadata_props`.
///
/// `ort-web` does not implement ONNX session metadata retrieval, so we read the
/// `metadata_props` (key/value pairs such as `task`, `names`, `imgsz`) straight
/// from the model protobuf, rebuild the `key: value` YAML the native path uses,
/// and hand it to the shared parser.
fn build_metadata(model_bytes: &[u8]) -> Result<ModelMetadata, JsError> {
    let props = onnx_meta::parse_metadata_props(model_bytes);
    if props.is_empty() {
        return Err(JsError::new(
            "no metadata found in ONNX model. Export it with Ultralytics \
             (`model.export(format='onnx')`) so the task, class names, and imgsz \
             are embedded.",
        ));
    }
    let combined = props
        .iter()
        .map(|(k, v)| format!("{k}: {v}"))
        .collect::<Vec<_>>()
        .join("\n");
    ModelMetadata::from_yaml_str(&combined).map_err(err_ctx("failed to parse model metadata"))
}

/// A loaded YOLO model ready for inference in the browser.
///
/// Created via [`YoloModel::load_bytes`]; run with [`YoloModel::predict`]. The
/// TypeScript wrapper exposes this as `YOLO`.
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
    /// session on the requested execution provider. Equivalent to Python's
    /// `YOLO('model.onnx')`.
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

        // Original image as HWC u8 (RGB), needed by postprocessing for coordinate scaling.
        let rgb = dynimg.to_rgb8();
        let (w, h) = rgb.dimensions();
        let orig_img = Array3::from_shape_vec((h as usize, w as usize, 3), rgb.into_raw())
            .map_err(err_ctx("failed to build image array"))?;

        // Classification uses center-crop (like Ultralytics); all other tasks
        // use letterbox. Both share the f32 NCHW output.
        let pre = if self.metadata.task == Task::Classify {
            preprocess_image_center_crop(&dynimg, self.imgsz, false)
        } else {
            preprocess_image_with_precision(&dynimg, self.imgsz, self.metadata.stride, false)
        };

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

        // Borrow each output's data directly (no copy, important for the large
        // semantic logits, ~160 MB) and feed it to the shared postprocessor while
        // `outputs` is still alive.
        let t_post = now_ms();
        let mut views: Vec<(&[f32], Vec<usize>)> = Vec::with_capacity(self.output_names.len());
        for name in &self.output_names {
            let (shape, data) = outputs[name.as_str()]
                .try_extract_tensor::<f32>()
                .map_err(|e| JsError::new(&format!("failed to extract output '{name}': {e}")))?;
            views.push((data, shape.iter().map(|&d| d as usize).collect()));
        }

        let mut config = InferenceConfig::new().with_confidence(conf).with_iou(iou);
        if let Some(classes) = classes {
            config = config.with_classes(classes.into_iter().map(|c| c as usize).collect());
        }
        let names: Arc<HashMap<usize, String>> = Arc::clone(&self.metadata.names);
        let inference_shape = (self.imgsz.0 as u32, self.imgsz.1 as u32);
        let t_end = now_ms();
        let speed = Speed::new(t_inf - t_pre, t_post - t_inf, t_end - t_post);

        let results = postprocess(
            views,
            self.metadata.task,
            &pre,
            &config,
            names,
            orig_img,
            String::new(),
            speed,
            inference_shape,
            self.metadata.end2end,
            self.metadata.kpt_shape,
        );
        let payload = JsResults::from_results(&results, self.metadata.task);
        to_js(&payload, "results")
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

/// One detected box, mirroring `Boxes` in the Ultralytics API (pixel `xyxy`).
/// `color` is the Ultralytics palette color for the class (`#rrggbb`).
#[derive(Serialize)]
struct JsBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    conf: f32,
    cls: usize,
    name: String,
    color: String,
}

/// One oriented box (`xywhr` + score), mirroring `Obb`.
#[derive(Serialize)]
struct JsObb {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    angle: f32,
    conf: f32,
    cls: usize,
    name: String,
    color: String,
}

/// Keypoints for one detection (`[[x, y, conf], ...]`), with the instance color.
#[derive(Serialize)]
struct JsKeypoints {
    points: Vec<[f32; 3]>,
    color: String,
}

/// Per-stage timing in ms, mirroring `Speed` (decode+letterbox / inference+sync / decode+NMS).
#[derive(Serialize)]
struct JsSpeed {
    preprocess: f64,
    inference: f64,
    postprocess: f64,
}

/// Classification probabilities, mirroring `Probs`.
#[derive(Serialize)]
struct JsProbs {
    top1: usize,
    top5: Vec<usize>,
    top1conf: f32,
    top5conf: Vec<f32>,
    name: String,
    color: String,
}

/// JS-facing results payload, mirroring the Ultralytics `Results` API. Per-task
/// fields are empty/`null` when unused.
#[derive(Serialize)]
struct JsResults {
    task: String,
    width: u32,
    height: u32,
    boxes: Vec<JsBox>,
    obb: Vec<JsObb>,
    keypoints: Vec<JsKeypoints>,
    probs: Option<JsProbs>,
    /// Segment masks as a translucent colored `RGBA` overlay (`width*height*4`);
    /// empty for other tasks. A `Uint8Array`, drawable straight onto a canvas.
    #[serde(with = "serde_bytes")]
    masks: Vec<u8>,
    /// Semantic class id per pixel, little-endian `u16` (`width*height*2` bytes),
    /// row-major; empty for other tasks. The TS wrapper reinterprets it as a
    /// `Uint16Array`. The IGNORE sentinel `65535` marks background or
    /// class-filtered pixels.
    #[serde(with = "serde_bytes")]
    semantic: Vec<u8>,
    /// Per-stage timing in ms: `{ preprocess, inference, postprocess }`.
    speed: JsSpeed,
}

impl JsResults {
    /// Convert core [`Results`] into the serializable JS payload, labeling it with
    /// the model's declared `task` (the caller already holds it, so there is no
    /// need to guess it back from which result fields are populated).
    fn from_results(r: &Results, task: Task) -> Self {
        // Class id -> display name and palette color, shared by every detection type.
        let name = |c: usize| r.names.get(&c).cloned().unwrap_or_default();
        let hex = |c: usize| Color::from_index(c).to_hex();

        let boxes = r.boxes.as_ref().map_or_else(Vec::new, |b| {
            let (xyxy, conf, cls) = (b.xyxy(), b.conf(), b.cls());
            (0..b.len())
                .map(|i| {
                    let c = cls[i] as usize;
                    JsBox {
                        x1: xyxy[[i, 0]],
                        y1: xyxy[[i, 1]],
                        x2: xyxy[[i, 2]],
                        y2: xyxy[[i, 3]],
                        conf: conf[i],
                        cls: c,
                        name: name(c),
                        color: hex(c),
                    }
                })
                .collect()
        });

        let obb = r.obb.as_ref().map_or_else(Vec::new, |o| {
            let (xywhr, conf, cls) = (o.xywhr(), o.conf(), o.cls());
            (0..conf.len())
                .map(|i| {
                    let c = cls[i] as usize;
                    JsObb {
                        x: xywhr[[i, 0]],
                        y: xywhr[[i, 1]],
                        w: xywhr[[i, 2]],
                        h: xywhr[[i, 3]],
                        angle: xywhr[[i, 4]],
                        conf: conf[i],
                        cls: c,
                        name: name(c),
                        color: hex(c),
                    }
                })
                .collect()
        });

        let keypoints = r.keypoints.as_ref().map_or_else(Vec::new, |k| {
            let (n, npt) = (k.data.shape()[0], k.data.shape()[1]);
            (0..n)
                .map(|i| JsKeypoints {
                    points: (0..npt)
                        .map(|j| [k.data[[i, j, 0]], k.data[[i, j, 1]], k.data[[i, j, 2]]])
                        .collect(),
                    // Skeleton uses the matching detection's color.
                    color: boxes.get(i).map_or_else(|| hex(0), |b| b.color.clone()),
                })
                .collect()
        });

        let probs = r.probs.as_ref().map(|p| JsProbs {
            top1: p.top1(),
            top5: p.top5(),
            top1conf: p.top1conf(),
            top5conf: p.top5conf(),
            name: name(p.top1()),
            color: hex(p.top1()),
        });

        Self {
            task: task.as_str().to_owned(),
            width: r.orig_shape.1,
            height: r.orig_shape.0,
            boxes,
            obb,
            keypoints,
            probs,
            masks: build_mask_overlay(r),
            semantic: r.semantic_mask.as_ref().map_or_else(Vec::new, |sem| {
                let mut bytes = Vec::with_capacity(sem.data.len() * 2);
                for &v in &sem.data {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                bytes
            }),
            speed: JsSpeed {
                preprocess: r.speed.preprocess.unwrap_or(0.0),
                inference: r.speed.inference.unwrap_or(0.0),
                postprocess: r.speed.postprocess.unwrap_or(0.0),
            },
        }
    }
}

/// Write one colored RGBA pixel at `(x, y)` into a `width`-wide buffer.
fn put(buf: &mut [u8], w: usize, x: usize, y: usize, c: Color, a: u8) {
    let i = (y * w + x) * 4;
    buf[i..i + 4].copy_from_slice(&[c.0, c.1, c.2, a]);
}

/// Build a translucent colored RGBA overlay (`width*height*4`) from the segment
/// instance masks or the semantic class map, colored by the class palette.
/// Empty for other tasks or if the mask resolution does not match the image.
fn build_mask_overlay(r: &Results) -> Vec<u8> {
    let (h, w) = (r.orig_shape.0 as usize, r.orig_shape.1 as usize);
    let mut buf = vec![0u8; w * h * 4];

    // Segment: per-instance binary masks, each colored by its detection class.
    if let (Some(masks), Some(boxes)) = (&r.masks, &r.boxes) {
        let (n, mh, mw) = masks.data.dim();
        if mh != h || mw != w {
            return Vec::new();
        }
        let cls = boxes.cls();
        for i in 0..n.min(cls.len()) {
            let c = Color::from_index(cls[i] as usize);
            for y in 0..h {
                for x in 0..w {
                    if masks.data[[i, y, x]] > 0.5 {
                        put(&mut buf, w, x, y, c, 140);
                    }
                }
            }
        }
        return buf;
    }

    // Semantic: per-pixel class map, blended 50/50 like the native renderer.
    if let Some(sem) = &r.semantic_mask {
        if sem.data.dim() != (h, w) {
            return Vec::new();
        }
        for y in 0..h {
            for x in 0..w {
                // Filtered-out classes carry the IGNORE sentinel; leave transparent.
                let cls = sem.data[[y, x]];
                if cls != SemanticMask::IGNORE {
                    put(&mut buf, w, x, y, Color::from_index(cls as usize), 128);
                }
            }
        }
        return buf;
    }

    Vec::new()
}

/// The Ultralytics pose drawing scheme: skeleton connectivity plus the per-limb
/// and per-keypoint palette colors. Lets the JS annotator draw pose exactly like
/// the native/Python renderer without duplicating any palette.
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
