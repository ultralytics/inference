// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Browser/WebAssembly bindings for `ultralytics-inference`.
//!
//! This crate runs Ultralytics YOLO ONNX models in the browser on **WebGPU**.
//! Inference is executed by the official ONNX Runtime Web build, bridged into
//! Rust through [`ort`](https://ort.pyke.io) and its
//! [`ort-web`](https://ort.pyke.io/backends/web) backend. All preprocessing
//! (letterbox + normalize) and postprocessing (NMS, coordinate scaling, mask /
//! keypoint / OBB decoding) is reused verbatim from the core
//! `ultralytics-inference` crate, so results match the native and Python paths.
//!
//! The JavaScript-facing surface is intentionally close to the Ultralytics
//! Python API: construct a model, then `await model.predict(image)`. The thin
//! TypeScript wrapper in the published npm package hides the wasm/memory and
//! WebGPU initialization behind `new YOLO(...)`.
//!
//! The whole crate is gated to `wasm32`; on any other target it compiles to an
//! empty library.
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

use ultralytics_inference::colors::Color;
use ultralytics_inference::metadata::ModelMetadata;
use ultralytics_inference::postprocessing::postprocess;
use ultralytics_inference::preprocessing::preprocess_image_with_precision;
use ultralytics_inference::results::{Results, Speed};
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

/// Install the ort-web WebGPU backend exactly once for this page.
///
/// This fetches the ONNX Runtime Web wasm bundle (from `cdn.pyke.io` by default)
/// and registers it as ort's active API. Subsequent calls are no-ops.
async fn ensure_backend() -> Result<(), JsError> {
    if API_READY.with(Cell::get) {
        return Ok(());
    }
    let api = ort_web::api(ort_web::FEATURE_WEBGPU)
        .await
        .map_err(|e| JsError::new(&format!("failed to initialize ort-web WebGPU backend: {e}")))?;
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
/// Created via [`YoloModel::load_url`] or [`YoloModel::load_bytes`]; run with
/// [`YoloModel::predict`]. The TypeScript wrapper exposes this as `YOLO`.
#[wasm_bindgen]
pub struct YoloModel {
    session: Session,
    metadata: ModelMetadata,
    output_names: Vec<String>,
    /// Network input size as `(height, width)`.
    imgsz: (usize, usize),
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
    pub async fn load_bytes(bytes: Vec<u8>) -> Result<YoloModel, JsError> {
        ensure_backend().await?;
        let metadata = build_metadata(&bytes)?;
        let session = Session::builder()
            .map_err(map_ort)?
            .with_execution_providers([WebGPU::default().build().error_on_failure()])
            .map_err(map_ort)?
            .commit_from_memory(&bytes)
            .await
            .map_err(|e| JsError::new(&format!("failed to load model from bytes: {e}")))?;
        Self::from_session(session, metadata)
    }

    /// The model's task (`"detect"`, `"segment"`, `"pose"`, `"classify"`,
    /// `"obb"`, or `"semantic"`).
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn task(&self) -> String {
        format!("{:?}", self.metadata.task).to_lowercase()
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
        let results = self.run(&image, conf, iou).await?;
        let payload = JsResults::from_results(&results);
        serde_wasm_bindgen::to_value(&payload)
            .map_err(|e| JsError::new(&format!("failed to serialize results: {e}")))
    }
}

impl YoloModel {
    /// Finish constructing a model from a committed session and its parsed
    /// metadata by resolving the input size and output names.
    fn from_session(session: Session, metadata: ModelMetadata) -> Result<Self, JsError> {
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
        })
    }

    /// Core async predict: decode -> preprocess -> `run_async` -> sync -> postprocess.
    ///
    /// The three timed stages mirror the Ultralytics `speed` breakdown:
    /// `preprocess` = JPEG/PNG decode + letterbox/normalize; `inference` =
    /// `run_async` + cross-context output sync; `postprocess` = NMS, decoding,
    /// and coordinate scaling. (Image *download* happens in JS before this and is
    /// not counted here.)
    async fn run(&mut self, image: &[u8], conf: f32, iou: f32) -> Result<Results, JsError> {
        let t_pre = now_ms();
        let dynimg = image::load_from_memory(image)
            .map_err(|e| JsError::new(&format!("failed to decode image: {e}")))?;

        // Original image as HWC u8 (RGB), needed by postprocessing for coordinate scaling.
        let rgb = dynimg.to_rgb8();
        let (w, h) = rgb.dimensions();
        let orig_img = Array3::from_shape_vec((h as usize, w as usize, 3), rgb.into_raw())
            .map_err(|e| JsError::new(&format!("failed to build image array: {e}")))?;

        // Letterbox + normalize using the shared CPU preprocessor (f32 input).
        let pre = preprocess_image_with_precision(&dynimg, self.imgsz, self.metadata.stride, false);

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

        // Collect each output as an owned f32 buffer + shape for the shared postprocessor.
        let t_post = now_ms();
        let mut owned: Vec<(Vec<f32>, Vec<usize>)> = Vec::with_capacity(self.output_names.len());
        for name in &self.output_names {
            let value = &outputs[name.as_str()];
            let (shape, data) = value
                .try_extract_tensor::<f32>()
                .map_err(|e| JsError::new(&format!("failed to extract output '{name}': {e}")))?;
            let shape_vec: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            owned.push((data.to_vec(), shape_vec));
        }
        let views: Vec<(&[f32], Vec<usize>)> = owned
            .iter()
            .map(|(d, s)| (d.as_slice(), s.clone()))
            .collect();

        let config = InferenceConfig::new().with_confidence(conf).with_iou(iou);
        let names: Arc<HashMap<usize, String>> = Arc::clone(&self.metadata.names);
        let inference_shape = (self.imgsz.0 as u32, self.imgsz.1 as u32);
        let t_end = now_ms();
        let speed = Speed::new(t_inf - t_pre, t_post - t_inf, t_end - t_post);

        Ok(postprocess(
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
        ))
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
    confidence: f32,
    class_id: usize,
    class_name: String,
    color: String,
}

/// One oriented box (`xywhr` + score), mirroring `OBB`.
#[derive(Serialize)]
struct JsObb {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    angle: f32,
    confidence: f32,
    class_id: usize,
    class_name: String,
    color: String,
}

/// Keypoints for one detection (`[[x, y, conf], ...]`), with the instance's
/// palette `color`.
#[derive(Serialize)]
struct JsKeypoints {
    points: Vec<[f32; 3]>,
    color: String,
}

/// Classification probabilities (top-5), mirroring `Probs`.
#[derive(Serialize)]
struct JsProbs {
    top1: usize,
    top1_name: String,
    top1_conf: f32,
    top5: Vec<usize>,
    top5_conf: Vec<f32>,
    color: String,
}

/// Per-stage timing in milliseconds, mirroring Ultralytics `results.speed`.
#[derive(Serialize)]
struct JsSpeed {
    /// Image decode + letterbox/normalize.
    preprocess: f64,
    /// `run_async` + cross-context output sync (the GPU inference).
    inference: f64,
    /// NMS, decoding, and coordinate scaling.
    postprocess: f64,
}

/// JS-facing results payload. Fields are populated per task; unused ones are
/// `null` on the JS side.
#[derive(Serialize)]
struct JsResults {
    task: String,
    orig_width: u32,
    orig_height: u32,
    boxes: Vec<JsBox>,
    obb: Vec<JsObb>,
    keypoints: Vec<JsKeypoints>,
    probs: Option<JsProbs>,
    /// Number of instance masks (segment task). Mask pixel data is omitted here
    /// to keep the payload small; request it explicitly in a future API.
    mask_count: usize,
    /// Per-stage timing (ms).
    speed: JsSpeed,
}

impl JsResults {
    /// Convert core [`Results`] into the serializable JS payload.
    fn from_results(r: &Results) -> Self {
        let names = &r.names;
        let name_of = |id: usize| names.get(&id).cloned().unwrap_or_default();

        let mut boxes = Vec::new();
        if let Some(b) = &r.boxes {
            let xyxy = b.xyxy();
            let conf = b.conf();
            let cls = b.cls();
            for i in 0..b.len() {
                let class_id = cls[i] as usize;
                boxes.push(JsBox {
                    x1: xyxy[[i, 0]],
                    y1: xyxy[[i, 1]],
                    x2: xyxy[[i, 2]],
                    y2: xyxy[[i, 3]],
                    confidence: conf[i],
                    class_id,
                    class_name: name_of(class_id),
                    color: Color::from_index(class_id).to_hex(),
                });
            }
        }

        let mut obb = Vec::new();
        if let Some(o) = &r.obb {
            let xywhr = o.xywhr();
            let conf = o.conf();
            let cls = o.cls();
            for i in 0..conf.len() {
                let class_id = cls[i] as usize;
                obb.push(JsObb {
                    x: xywhr[[i, 0]],
                    y: xywhr[[i, 1]],
                    width: xywhr[[i, 2]],
                    height: xywhr[[i, 3]],
                    angle: xywhr[[i, 4]],
                    confidence: conf[i],
                    class_id,
                    class_name: name_of(class_id),
                    color: Color::from_index(class_id).to_hex(),
                });
            }
        }

        let mut keypoints = Vec::new();
        if let Some(k) = &r.keypoints {
            let data = &k.data;
            let (n, npt) = (data.shape()[0], data.shape()[1]);
            for i in 0..n {
                let mut points = Vec::with_capacity(npt);
                for j in 0..npt {
                    points.push([data[[i, j, 0]], data[[i, j, 1]], data[[i, j, 2]]]);
                }
                // Color the skeleton with the matching detection's palette color.
                let color = boxes
                    .get(i)
                    .map_or_else(|| Color::from_index(0).to_hex(), |b| b.color.clone());
                keypoints.push(JsKeypoints { points, color });
            }
        }

        let probs = r.probs.as_ref().map(|p| {
            let top1 = p.top1();
            JsProbs {
                top1,
                top1_name: name_of(top1),
                top1_conf: p.top1conf(),
                top5: p.top5(),
                top5_conf: p.top5conf(),
                color: Color::from_index(top1).to_hex(),
            }
        });

        let mask_count = r
            .masks
            .as_ref()
            .map_or(0, ultralytics_inference::Masks::len);

        Self {
            task: format!("{:?}", task_of(r)).to_lowercase(),
            orig_width: r.orig_shape.1,
            orig_height: r.orig_shape.0,
            boxes,
            obb,
            keypoints,
            probs,
            mask_count,
            speed: JsSpeed {
                preprocess: r.speed.preprocess.unwrap_or(0.0),
                inference: r.speed.inference.unwrap_or(0.0),
                postprocess: r.speed.postprocess.unwrap_or(0.0),
            },
        }
    }
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

/// Install a panic hook that logs Rust panics to the browser console.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}
