// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#![allow(clippy::float_cmp)]

//! Integration tests for the inference library

use ndarray::Array3;
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::tempdir;
use ultralytics_inference::cli::args::PredictArgs;
use ultralytics_inference::cli::predict::run_prediction;
use ultralytics_inference::task::Task;
use ultralytics_inference::{Boxes, InferenceConfig, Results, Speed};
#[cfg(any(feature = "coreml", feature = "cuda"))]
use ultralytics_inference::{Device, YOLOModel};

/// End-to-end `CoreML` test covering two known regressions:
///
/// 1. **Issue #148 / PR #149**: `GatherElements op: Out of range` during warmup.
///    `CoreML`'s DFL head produces out-of-range gather indices on an all-zeros dummy input.
///
/// 2. **`graph_input_cast_0` crash**: `CoreML` must not rename the ONNX input
///    (e.g. `images`) to an internal cast node that ORT cannot feed. The current
///    `MLProgram` path uses static input shapes and `FastPrediction` to avoid that cast.
#[cfg(feature = "coreml")]
#[test]
#[ignore = "downloads a YOLO model; requires CoreML (macOS only)"]
fn test_coreml_model_loads_and_warms_up() {
    let temp_dir = tempdir().expect("temp dir should be created");
    let model_path = temp_dir.path().join("yolo26n.onnx");

    let config = InferenceConfig::new().with_device(Device::CoreMl);
    let mut model = YOLOModel::load_with_config(model_path.to_string_lossy().as_ref(), config)
        .expect("CoreML model should load");

    // warmup() must succeed: graph_input_cast_0 errors are gone (static-shape MLProgram path)
    // and GatherElements out-of-range is tolerated (issue #148 fix).
    model.warmup().expect("CoreML warmup should not fail");
}

#[test]
#[ignore = "downloads a YOLO model and sample image"]
fn test_run_prediction_e2e() {
    let temp_dir = tempdir().expect("temp dir should be created");
    let model_path = temp_dir.path().join("yolo26n.onnx");

    let args = PredictArgs {
        model: Some(model_path.to_string_lossy().into_owned()),
        task: None,
        source: Some("https://ultralytics.com/images/bus.jpg".to_string()),
        conf: 0.25,
        iou: 0.45,
        max_det: 300,
        imgsz: Some(640),
        rect: false,
        batch: 1,
        half: false,
        save: false,
        save_frames: false,
        save_json: false,
        show: false,
        device: None,
        verbose: false,
        classes: None,
    };

    run_prediction(&args);
}

#[test]
#[ignore = "downloads yolo26n-sem.onnx from GitHub releases; requires network"]
fn test_auto_download_semantic_model() {
    let temp_dir = tempdir().expect("temp dir should be created");
    let model_path = temp_dir.path().join("yolo26n-sem.onnx");
    let result = ultralytics_inference::download::try_download_model(&model_path);
    assert!(
        result.is_ok(),
        "download should succeed: {:?}",
        result.err()
    );
    assert!(
        model_path.exists(),
        "yolo26n-sem.onnx should be present after auto-download"
    );
    let size = std::fs::metadata(&model_path).unwrap().len();
    assert!(
        size > 1_000_000,
        "downloaded file should be a real model (>1 MB), got {size} bytes"
    );
}

#[test]
#[ignore = "downloads a YOLO semantic model and sample image"]
fn test_run_prediction_e2e_semantic() {
    let temp_dir = tempdir().expect("temp dir should be created");
    let model_path = temp_dir.path().join("yolo26n-sem.onnx");

    let args = PredictArgs {
        model: Some(model_path.to_string_lossy().into_owned()),
        task: Some(Task::Semantic),
        source: Some("https://ultralytics.com/images/bus.jpg".to_string()),
        conf: 0.25,
        iou: 0.45,
        max_det: 300,
        imgsz: Some(640),
        rect: false,
        batch: 1,
        half: false,
        save: false,
        save_frames: false,
        save_json: false,
        show: false,
        device: None,
        verbose: false,
        classes: None,
    };

    run_prediction(&args);
}

#[test]
#[ignore = "downloads a YOLO semantic model and sample image; writes class-map PNG to runs/"]
fn test_semantic_save_class_map() {
    let temp_dir = tempdir().expect("temp dir should be created");
    let model_path = temp_dir.path().join("yolo26n-sem.onnx");

    let args = PredictArgs {
        model: Some(model_path.to_string_lossy().into_owned()),
        task: Some(Task::Semantic),
        source: Some("https://ultralytics.com/images/bus.jpg".to_string()),
        conf: 0.25,
        iou: 0.45,
        max_det: 300,
        imgsz: Some(640),
        rect: false,
        batch: 1,
        half: false,
        save: false,
        save_frames: false,
        save_json: true,
        show: false,
        device: None,
        verbose: false,
        classes: None,
    };

    run_prediction(&args);

    // Verify a results/ dir was created under runs/semantic/predict*/
    let runs_dir = std::path::Path::new("runs").join("semantic");
    assert!(
        runs_dir.exists(),
        "runs/semantic/ should exist after save_json=true for semantic task"
    );
    let results_png_exists = std::fs::read_dir(&runs_dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(std::result::Result::ok)
        .any(|predict_dir| predict_dir.path().join("results").join("bus.png").exists());
    assert!(
        results_png_exists,
        "class-map PNG should be written to runs/semantic/predict*/results/bus.png"
    );
}

#[test]
fn test_inference_config_creation() {
    let config = InferenceConfig::default();
    assert_eq!(config.confidence_threshold, 0.25);
    assert_eq!(config.iou_threshold, 0.7);
    assert_eq!(config.max_det, 300);
}

#[test]
fn test_inference_config_builder() {
    let config = InferenceConfig::new()
        .with_confidence(0.5)
        .with_iou(0.7)
        .with_max_det(300);

    assert_eq!(config.confidence_threshold, 0.5);
    assert_eq!(config.iou_threshold, 0.7);
    assert_eq!(config.max_det, 300);
}

#[test]
fn test_inference_config_batch() {
    let config = InferenceConfig::new().with_batch(32);
    assert_eq!(config.batch, Some(32));
}

#[test]
fn test_boxes_creation() {
    // Create boxes data: [x1, y1, x2, y2, conf, cls]
    let data = ndarray::array![
        [10.0f32, 20.0, 30.0, 40.0, 0.95, 0.0],
        [50.0, 60.0, 70.0, 80.0, 0.85, 1.0],
    ];

    let boxes = Boxes::new(data, (480, 640));

    assert_eq!(boxes.len(), 2);
    assert!(!boxes.is_empty());
}

#[test]
fn test_boxes_xyxy() {
    let data = ndarray::array![[10.0f32, 20.0, 30.0, 40.0, 0.95, 0.0],];

    let boxes = Boxes::new(data, (480, 640));
    let xyxy = boxes.xyxy();

    assert_eq!(xyxy[[0, 0]], 10.0);
    assert_eq!(xyxy[[0, 1]], 20.0);
    assert_eq!(xyxy[[0, 2]], 30.0);
    assert_eq!(xyxy[[0, 3]], 40.0);
}

#[test]
fn test_boxes_conf_and_cls() {
    let data = ndarray::array![[10.0f32, 20.0, 30.0, 40.0, 0.95, 2.0],];

    let boxes = Boxes::new(data, (480, 640));

    assert_eq!(boxes.conf()[[0]], 0.95);
    assert_eq!(boxes.cls()[[0]], 2.0);
}

#[test]
fn test_results_creation() {
    let orig_img = Array3::zeros((480, 640, 3));
    let names = Arc::new(HashMap::new());
    let speed = Speed::default();

    let results = Results::new(orig_img, "test.jpg".to_string(), names, speed, (640, 640));

    assert!(results.boxes.is_none());
    assert!(results.masks.is_none());
    assert!(results.keypoints.is_none());
    assert!(results.probs.is_none());
    assert!(results.obb.is_none());
}

#[test]
fn test_results_with_boxes() {
    let orig_img = Array3::zeros((480, 640, 3));
    let names = Arc::new(HashMap::new());
    let speed = Speed::default();

    let mut results = Results::new(orig_img, "test.jpg".to_string(), names, speed, (640, 640));

    let boxes_data = ndarray::array![[10.0f32, 20.0, 30.0, 40.0, 0.95, 0.0],];
    results.boxes = Some(Boxes::new(boxes_data, (480, 640)));

    assert!(results.boxes.is_some());
    assert_eq!(results.boxes.as_ref().unwrap().len(), 1);
    assert!(!results.is_empty());
}

#[test]
fn test_results_is_empty() {
    let orig_img = Array3::zeros((480, 640, 3));
    let names = Arc::new(HashMap::new());
    let speed = Speed::default();

    let results = Results::new(orig_img, "test.jpg".to_string(), names, speed, (640, 640));

    assert!(results.is_empty());
}

#[test]
fn test_speed_timing() {
    let speed = Speed::new(10.0, 20.0, 5.0);

    assert_eq!(speed.preprocess, Some(10.0));
    assert_eq!(speed.inference, Some(20.0));
    assert_eq!(speed.postprocess, Some(5.0));
}

// GPU inference tests. Gated on the `cuda` feature and `#[ignore]`d so they run
// only on the self-hosted GPU CI (via `--include-ignored`), never on hosted
// runners. They auto-download `yolo26n.onnx` and the bus.jpg sample.

#[cfg(feature = "cuda")]
fn load_bus_image() -> String {
    ultralytics_inference::download::download_image("https://ultralytics.com/images/bus.jpg")
        .expect("bus.jpg should download")
}

/// Loading on `Device::Cuda(0)` registers the CUDA execution provider and warms
/// up without error (warmup runs inside `load_with_config`).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires NVIDIA GPU; run on GPU CI"]
fn test_cuda_model_loads_and_warms_up() {
    let config = InferenceConfig::new().with_device(Device::Cuda(0));
    let model = YOLOModel::load_with_config("yolo26n.onnx", config)
        .expect("CUDA model should load and warm up");

    let ep = model.execution_provider();
    assert!(
        ep.to_lowercase().contains("cuda"),
        "expected the CUDA execution provider, got '{ep}'"
    );
}

/// End-to-end CUDA inference on bus.jpg yields at least one detection.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires NVIDIA GPU; run on GPU CI"]
fn test_cuda_predict_bus_has_detections() {
    let config = InferenceConfig::new().with_device(Device::Cuda(0));
    let mut model =
        YOLOModel::load_with_config("yolo26n.onnx", config).expect("CUDA model should load");
    let image = load_bus_image();

    let results = model
        .predict(&image)
        .expect("CUDA prediction should succeed");
    let boxes = results
        .first()
        .and_then(|r| r.boxes.as_ref())
        .expect("bus.jpg should produce boxes on CUDA");
    assert!(
        !boxes.is_empty(),
        "expected at least one detection on bus.jpg"
    );
}

/// CUDA and CPU inference produce equivalent detections on the same image.
/// Both sides use the CPU letterbox preprocess (the CUDA model pins
/// `with_cuda_preprocess(false)`, which the `cuda-preprocess` feature otherwise
/// defaults on) so the comparison isolates the inference engines rather than the
/// resize path; the fused GPU preprocess is covered by
/// `test_cuda_preprocess_on_off_parity`.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires NVIDIA GPU; run on GPU CI"]
fn test_cuda_cpu_detection_parity() {
    let image = load_bus_image();

    let mut cuda_model = YOLOModel::load_with_config(
        "yolo26n.onnx",
        InferenceConfig::new()
            .with_device(Device::Cuda(0))
            .with_cuda_preprocess(false),
    )
    .expect("CUDA model should load");
    let cuda = cuda_model.predict(&image).expect("CUDA prediction");
    let cuda_boxes = cuda
        .first()
        .and_then(|r| r.boxes.as_ref())
        .expect("CUDA boxes");

    let mut cpu_model = YOLOModel::load_with_config(
        "yolo26n.onnx",
        InferenceConfig::new().with_device(Device::Cpu),
    )
    .expect("CPU model should load");
    let cpu = cpu_model.predict(&image).expect("CPU prediction");
    let cpu_boxes = cpu
        .first()
        .and_then(|r| r.boxes.as_ref())
        .expect("CPU boxes");

    // Counts match within one box: NMS can keep/drop a borderline candidate
    // differently under GPU vs CPU rounding.
    let diff = cuda_boxes.len().abs_diff(cpu_boxes.len());
    assert!(
        diff <= 1,
        "detection count mismatch: cuda={}, cpu={}",
        cuda_boxes.len(),
        cpu_boxes.len()
    );

    // Matched leading boxes (sorted by confidence) agree to a couple of pixels.
    for (cb, pb) in cuda_boxes
        .xyxy()
        .outer_iter()
        .zip(cpu_boxes.xyxy().outer_iter())
    {
        for (x, y) in cb.iter().zip(pb.iter()) {
            let d = (x - y).abs();
            assert!(d < 2.0, "box corner differs by {d}px between cuda and cpu");
        }
    }
}

/// The fused GPU preprocess (`cuda-preprocess`) and the CPU preprocess produce
/// equivalent detections through the same CUDA execution provider.
#[cfg(feature = "cuda-preprocess")]
#[test]
#[ignore = "requires NVIDIA GPU; run on GPU CI"]
fn test_cuda_preprocess_on_off_parity() {
    let image = load_bus_image();

    let mut fused = YOLOModel::load_with_config(
        "yolo26n.onnx",
        InferenceConfig::new()
            .with_device(Device::Cuda(0))
            .with_cuda_preprocess(true),
    )
    .expect("model with fused GPU preprocess should load");
    let fused_res = fused.predict(&image).expect("fused prediction");
    let fused_boxes = fused_res
        .first()
        .and_then(|r| r.boxes.as_ref())
        .expect("fused boxes");

    let mut cpu_pre = YOLOModel::load_with_config(
        "yolo26n.onnx",
        InferenceConfig::new()
            .with_device(Device::Cuda(0))
            .with_cuda_preprocess(false),
    )
    .expect("model with CPU preprocess should load");
    let cpu_pre_res = cpu_pre.predict(&image).expect("cpu-preprocess prediction");
    let cpu_pre_boxes = cpu_pre_res
        .first()
        .and_then(|r| r.boxes.as_ref())
        .expect("cpu-preprocess boxes");

    let diff = fused_boxes.len().abs_diff(cpu_pre_boxes.len());
    assert!(
        diff <= 1,
        "detection count mismatch: fused={}, cpu-preprocess={}",
        fused_boxes.len(),
        cpu_pre_boxes.len()
    );
}
