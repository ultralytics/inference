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

use ultralytics_inference::metadata::ModelMetadata;
use ultralytics_inference::postprocessing::postprocess;
use ultralytics_inference::preprocessing::preprocess_image_with_precision;
use ultralytics_inference::results::{Results, Speed};
use ultralytics_inference::{InferenceConfig, Task};

/// Default inference image size used when a model does not record `imgsz` in its
/// metadata. Mirrors the native crate's fallback.
const DEFAULT_IMGSZ: usize = 640;

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

/// Read the Ultralytics YAML metadata embedded in an ONNX model's custom
/// metadata properties and parse it with the core crate's parser.
fn parse_metadata(session: &Session) -> Result<ModelMetadata, JsError> {
    let meta = session
        .metadata()
        .map_err(|e| JsError::new(&format!("failed to read model metadata: {e}")))?;
    let mut map: HashMap<String, String> = HashMap::new();
    if let Ok(keys) = meta.custom_keys() {
        for key in keys {
            if let Some(value) = meta.custom(&key) {
                map.insert(key, value);
            }
        }
    }
    ModelMetadata::from_onnx_metadata(&map).map_err(|e| {
        JsError::new(&format!(
            "{e}. Export the model with `model.export(format='onnx')` so the \
             Ultralytics metadata (task, classes, imgsz) is embedded."
        ))
    })
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
    /// Load a model from a URL (served alongside the page or from a CDN).
    ///
    /// Initializes the WebGPU backend on first use, then commits an ONNX Runtime
    /// session with the WebGPU execution provider. Equivalent to Python's
    /// `YOLO('model.onnx')`.
    ///
    /// # Errors
    /// Returns a JS error if the backend cannot start, the model cannot be
    /// fetched/committed, or the model lacks Ultralytics metadata.
    pub async fn load_url(url: String) -> Result<YoloModel, JsError> {
        ensure_backend().await?;
        let session = Session::builder()
            .map_err(map_ort)?
            .with_execution_providers([WebGPU::default().build().error_on_failure()])
            .map_err(map_ort)?
            .commit_from_url(&url)
            .await
            .map_err(|e| JsError::new(&format!("failed to load model from {url}: {e}")))?;
        Self::from_session(session)
    }

    /// Load a model from raw ONNX bytes already in memory (e.g. fetched by JS).
    ///
    /// # Errors
    /// Returns a JS error if the backend cannot start, the bytes are not a valid
    /// model, or the model lacks Ultralytics metadata.
    pub async fn load_bytes(bytes: Vec<u8>) -> Result<YoloModel, JsError> {
        ensure_backend().await?;
        let session = Session::builder()
            .map_err(map_ort)?
            .with_execution_providers([WebGPU::default().build().error_on_failure()])
            .map_err(map_ort)?
            .commit_from_memory(&bytes)
            .await
            .map_err(|e| JsError::new(&format!("failed to load model from bytes: {e}")))?;
        Self::from_session(session)
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
    /// Finish constructing a model from a committed session by parsing metadata
    /// and resolving the input size.
    fn from_session(session: Session) -> Result<Self, JsError> {
        let metadata = parse_metadata(&session)?;
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
    async fn run(&mut self, image: &[u8], conf: f32, iou: f32) -> Result<Results, JsError> {
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
        let speed = Speed::new(0.0, 0.0, 0.0);

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
#[derive(Serialize)]
struct JsBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    confidence: f32,
    class_id: usize,
    class_name: String,
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
}

/// Keypoints for one detection (`[[x, y, conf], ...]`).
#[derive(Serialize)]
struct JsKeypoints {
    points: Vec<[f32; 3]>,
}

/// Classification probabilities (top-5), mirroring `Probs`.
#[derive(Serialize)]
struct JsProbs {
    top1: usize,
    top1_name: String,
    top1_conf: f32,
    top5: Vec<usize>,
    top5_conf: Vec<f32>,
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
                keypoints.push(JsKeypoints { points });
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
