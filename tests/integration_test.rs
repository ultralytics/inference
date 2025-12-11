// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Integration tests for the inference library

use inference::{InferenceConfig, Results, Boxes, Speed};
use ndarray::Array3;
use std::collections::HashMap;

#[test]
fn test_inference_config_creation() {
    let config = InferenceConfig::default();
    assert_eq!(config.confidence_threshold, 0.25);
    assert_eq!(config.iou_threshold, 0.45);
    assert_eq!(config.max_detections, 300);
}

#[test]
fn test_inference_config_builder() {
    let config = InferenceConfig::new()
        .with_confidence(0.5)
        .with_iou(0.7)
        .with_max_detections(100);

    assert_eq!(config.confidence_threshold, 0.5);
    assert_eq!(config.iou_threshold, 0.7);
    assert_eq!(config.max_detections, 100);
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
    let data = ndarray::array![
        [10.0f32, 20.0, 30.0, 40.0, 0.95, 0.0],
    ];

    let boxes = Boxes::new(data, (480, 640));
    let xyxy = boxes.xyxy();

    assert_eq!(xyxy[[0, 0]], 10.0);
    assert_eq!(xyxy[[0, 1]], 20.0);
    assert_eq!(xyxy[[0, 2]], 30.0);
    assert_eq!(xyxy[[0, 3]], 40.0);
}

#[test]
fn test_boxes_conf_and_cls() {
    let data = ndarray::array![
        [10.0f32, 20.0, 30.0, 40.0, 0.95, 2.0],
    ];

    let boxes = Boxes::new(data, (480, 640));

    assert_eq!(boxes.conf()[[0]], 0.95);
    assert_eq!(boxes.cls()[[0]], 2.0);
}

#[test]
fn test_results_creation() {
    let orig_img = Array3::zeros((480, 640, 3));
    let names = HashMap::new();
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
    let names = HashMap::new();
    let speed = Speed::default();

    let mut results = Results::new(orig_img, "test.jpg".to_string(), names, speed, (640, 640));

    let boxes_data = ndarray::array![
        [10.0f32, 20.0, 30.0, 40.0, 0.95, 0.0],
    ];
    results.boxes = Some(Boxes::new(boxes_data, (480, 640)));

    assert!(results.boxes.is_some());
    assert_eq!(results.boxes.as_ref().unwrap().len(), 1);
    assert!(!results.is_empty());
}

#[test]
fn test_results_is_empty() {
    let orig_img = Array3::zeros((480, 640, 3));
    let names = HashMap::new();
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

// Note: Full integration tests with actual YOLO models require:
// - A test ONNX model file
// - Test images
// These will be added in future test modules
