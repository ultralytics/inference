// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Browser/WebAssembly bindings for `ultralytics-inference`.
//!
//! Runs YOLO ONNX models in the browser on WebGPU via ONNX Runtime Web (bridged
//! by [`ort-web`](https://ort.pyke.io/backends/web)), reusing the core crate's
//! preprocessing/postprocessing so results match the native and Python paths.
//! The published npm package wraps this behind a small `YOLO` class. The whole
//! crate is gated to `wasm32`; elsewhere it compiles to an empty library.
#![cfg(target_arch = "wasm32")]

use std::cell::Cell;
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
use ultralytics_inference::results::{Results, Speed};
use ultralytics_inference::visualizer::color::Color;
use ultralytics_inference::visualizer::skeleton::{
    KPT_COLOR_INDICES, LIMB_COLOR_INDICES, SKELETON,
};
use ultralytics_inference::{InferenceConfig, Task};

/// Default inference image size used when a model does not record `imgsz` in its
/// metadata. Mirrors the native crate's fallback.
const DEFAULT_IMGSZ: usize = 640;

/// High-resolution timestamp in milliseconds (`performance.now()`), for the
/// per-stage `speed` breakdown. Falls back to 0 if unavailable.
fn now_ms() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map_or(0.0, |p| p.now())
}

thread_local! {
    /// Whether `ort::set_api` has already installed the ort-web backend. Wasm is
    /// single-threaded, so a thread-local flag is a sufficient one-time guard.
    static API_READY: Cell<bool> = const { Cell::new(false) };
}

/// Install the ort-web backend exactly once for this page.
///
/// Loads either the WebGPU build (`webgpu = true`) or the portable CPU/wasm build
/// (`webgpu = false`, the universal fallback for browsers without WebGPU). The
/// build is fetched from `cdn.pyke.io` unless `ort_base_url` is given to
/// self-host it. Subsequent calls are no-ops.
async fn ensure_backend(ort_base_url: Option<String>, webgpu: bool) -> Result<(), JsError> {
    if API_READY.with(Cell::get) {
        return Ok(());
    }
    let api = match ort_base_url.filter(|s| !s.is_empty()) {
        // Self-hosted: pick the entrypoint + binary for the chosen build. The
        // wrapper (.mjs) name defaults to the binary with `.wasm` -> `.mjs`.
        Some(base) => {
            let dist = if webgpu {
                ort_web::Dist::new(base)
                    .with_script_name("ort.webgpu.min.js")
                    .with_binary_name("ort-wasm-simd-threaded.jsep.wasm")
            } else {
                ort_web::Dist::new(base)
                    .with_script_name("ort.wasm.min.js")
                    .with_binary_name("ort-wasm-simd-threaded.wasm")
            };
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
    .map_err(|e| JsError::new(&format!("failed to initialize ort-web backend: {e}")))?;
    ort::set_api(api);
    API_READY.with(|c| c.set(true));
    Ok(())
}

/// Parse the Ultralytics metadata embedded in an ONNX model's `metadata_props`.
///
/// `ort-web` does not implement ONNX session metadata retrieval, so we read the
/// `metadata_props` (key/value pairs such as `task`, `names`, `imgsz`) straight
/// from the model protobuf, rebuild the `key: value` YAML the native path uses,
/// and hand it to the shared parser.
fn build_metadata(model_bytes: &[u8]) -> Result<ModelMetadata, JsError> {
    let props = parse_metadata_props(model_bytes);
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
    ModelMetadata::from_yaml_str(&combined)
        .map_err(|e| JsError::new(&format!("failed to parse model metadata: {e}")))
}

/// Read a protobuf base-128 varint at `pos`, advancing it. Returns `None` on a
/// truncated/oversized value.
fn read_varint(buf: &[u8], pos: &mut usize) -> Option<u64> {
    let mut result = 0u64;
    let mut shift = 0u32;
    loop {
        let byte = *buf.get(*pos)?;
        *pos += 1;
        result |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Some(result);
        }
        shift += 7;
        if shift >= 64 {
            return None;
        }
    }
}

/// Extract `ModelProto.metadata_props` (field 14, repeated
/// `StringStringEntryProto`) into a key/value map by walking the top-level
/// protobuf fields. Large fields such as the graph are skipped by length without
/// being decoded.
fn parse_metadata_props(buf: &[u8]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut pos = 0;
    while pos < buf.len() {
        let Some(tag) = read_varint(buf, &mut pos) else {
            break;
        };
        let field = tag >> 3;
        match tag & 7 {
            0 => {
                if read_varint(buf, &mut pos).is_none() {
                    break;
                }
            }
            1 => pos += 8,
            5 => pos += 4,
            2 => {
                let Some(len) = read_varint(buf, &mut pos) else {
                    break;
                };
                let len = len as usize;
                let Some(sub) = buf.get(pos..pos + len) else {
                    break;
                };
                pos += len;
                if field == 14
                    && let Some((key, value)) = parse_string_string_entry(sub)
                {
                    map.insert(key, value);
                }
            }
            _ => break,
        }
    }
    map
}

/// Parse a `StringStringEntryProto` (field 1 = key, field 2 = value).
fn parse_string_string_entry(buf: &[u8]) -> Option<(String, String)> {
    let mut pos = 0;
    let mut key = None;
    let mut value = None;
    while pos < buf.len() {
        let tag = read_varint(buf, &mut pos)?;
        let field = tag >> 3;
        match tag & 7 {
            2 => {
                let len = read_varint(buf, &mut pos)? as usize;
                let bytes = buf.get(pos..pos + len)?;
                pos += len;
                let text = String::from_utf8_lossy(bytes).into_owned();
                match field {
                    1 => key = Some(text),
                    2 => value = Some(text),
                    _ => {}
                }
            }
            0 => {
                read_varint(buf, &mut pos)?;
            }
            1 => pos += 8,
            5 => pos += 4,
            _ => return None,
        }
    }
    Some((key?, value.unwrap_or_default()))
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
    /// Active accelerator label (`"WebGPU"` or `"CPU (wasm)"`).
    backend: &'static str,
}

#[wasm_bindgen]
impl YoloModel {
    /// Load a model from raw ONNX bytes (fetched by the JS wrapper).
    ///
    /// Initializes the WebGPU backend on first use, reads the embedded
    /// Ultralytics metadata from the model bytes, then commits an ONNX Runtime
    /// session with the WebGPU execution provider. Equivalent to Python's
    /// `YOLO('model.onnx')`.
    ///
    /// # Errors
    /// Returns a JS error if the backend cannot start, the bytes are not a valid
    /// model, or the model lacks Ultralytics metadata.
    pub async fn load_bytes(
        bytes: Vec<u8>,
        ort_base_url: Option<String>,
        webgpu: bool,
    ) -> Result<YoloModel, JsError> {
        ensure_backend(ort_base_url, webgpu).await?;
        let metadata = build_metadata(&bytes)?;
        let mut builder = Session::builder().map_err(map_ort)?;
        if webgpu {
            // No `error_on_failure`: if the adapter is missing, ORT falls back to
            // its CPU EP rather than aborting.
            builder = builder
                .with_execution_providers([WebGPU::default().build()])
                .map_err(map_ort)?;
        }
        let session = builder
            .commit_from_memory(&bytes)
            .await
            .map_err(|e| JsError::new(&format!("failed to load model from bytes: {e}")))?;
        Self::from_session(session, metadata, webgpu)
    }

    /// The model's task (`"detect"`, `"segment"`, `"pose"`, `"classify"`,
    /// `"obb"`, or `"semantic"`).
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn task(&self) -> String {
        format!("{:?}", self.metadata.task).to_lowercase()
    }

    /// The active accelerator: `"WebGPU"` or `"CPU (wasm)"` (the fallback when the
    /// browser has no WebGPU).
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn backend(&self) -> String {
        self.backend.to_string()
    }

    /// Class id -> name map (like Ultralytics `model.names`), as a JS object.
    ///
    /// # Errors
    /// Returns a JS error only if serialization fails (not expected).
    #[wasm_bindgen(getter)]
    pub fn names(&self) -> Result<JsValue, JsError> {
        serde_wasm_bindgen::to_value(&*self.metadata.names)
            .map_err(|e| JsError::new(&format!("failed to serialize names: {e}")))
    }

    /// Run inference on a single encoded image (JPEG or PNG bytes).
    ///
    /// `conf` and `iou` are the confidence and NMS IoU thresholds; pass the
    /// model defaults (0.25 / 0.7) to match Ultralytics. Returns a plain JS
    /// object whose shape mirrors the Ultralytics `Results` API.
    ///
    /// # Errors
    /// Returns a JS error if the image cannot be decoded or inference fails.
    pub async fn predict(
        &mut self,
        image: Vec<u8>,
        conf: f32,
        iou: f32,
    ) -> Result<JsValue, JsError> {
        let dynimg = image::load_from_memory(&image)
            .map_err(|e| JsError::new(&format!("failed to decode image: {e}")))?;
        self.run(dynimg, conf, iou).await
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
        self.run(image::DynamicImage::ImageRgba8(img), conf, iou)
            .await
    }
}

impl YoloModel {
    /// Finish constructing a model from a committed session and its parsed
    /// metadata by resolving the input size and output names.
    fn from_session(
        session: Session,
        metadata: ModelMetadata,
        webgpu: bool,
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
            backend: if webgpu { "WebGPU" } else { "CPU (wasm)" },
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
    ) -> Result<JsValue, JsError> {
        let t_pre = now_ms();

        // Original image as HWC u8 (RGB), needed by postprocessing for coordinate scaling.
        let rgb = dynimg.to_rgb8();
        let (w, h) = rgb.dimensions();
        let orig_img = Array3::from_shape_vec((h as usize, w as usize, 3), rgb.into_raw())
            .map_err(|e| JsError::new(&format!("failed to build image array: {e}")))?;

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
            .map_err(|e| JsError::new(&format!("inference failed: {e}")))?;
        // Outputs live in the ONNX Runtime wasm context; copy them back to Rust.
        sync_outputs(&mut outputs)
            .await
            .map_err(|e| JsError::new(&format!("failed to sync outputs: {e}")))?;

        // Borrow each output's data directly (no copy — important for the large
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

        let config = InferenceConfig::new().with_confidence(conf).with_iou(iou);
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
        let payload = JsResults::from_results(&results);
        serde_wasm_bindgen::to_value(&payload)
            .map_err(|e| JsError::new(&format!("failed to serialize results: {e}")))
    }
}

/// Map any `ort::Error<M>` (the marker `M` records the originating operation) to
/// a JS error.
fn map_ort<M>(e: ort::Error<M>) -> JsError {
    JsError::new(&format!("ort error: {e}"))
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
    /// Per-stage timing in ms: `{ preprocess, inference, postprocess }`.
    speed: JsSpeed,
}

impl JsResults {
    /// Convert core [`Results`] into the serializable JS payload.
    fn from_results(r: &Results) -> Self {
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
            task: format!("{:?}", task_of(r)).to_lowercase(),
            width: r.orig_shape.1,
            height: r.orig_shape.0,
            boxes,
            obb,
            keypoints,
            probs,
            masks: build_mask_overlay(r),
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
                put(
                    &mut buf,
                    w,
                    x,
                    y,
                    Color::from_index(sem.data[[y, x]] as usize),
                    128,
                );
            }
        }
        return buf;
    }

    Vec::new()
}

/// Infer the task label from which result fields are populated.
fn task_of(r: &Results) -> Task {
    if r.obb.is_some() {
        Task::Obb
    } else if r.keypoints.is_some() {
        Task::Pose
    } else if r.masks.is_some() {
        Task::Segment
    } else if r.probs.is_some() {
        Task::Classify
    } else if r.semantic_mask.is_some() {
        Task::Semantic
    } else {
        Task::Detect
    }
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
    serde_wasm_bindgen::to_value(&scheme)
        .map_err(|e| JsError::new(&format!("failed to serialize pose palette: {e}")))
}

/// Install a panic hook that logs Rust panics to the browser console.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}
