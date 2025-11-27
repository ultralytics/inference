// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Integration tests for the inference library

use inference::{InferenceConfig, DetectionResult};

#[test]
fn test_inference_config_creation() {
    let config = InferenceConfig::default();
    assert_eq!(config.confidence_threshold, 0.25);
    assert_eq!(config.iou_threshold, 0.45);
    assert_eq!(config.max_detections, 300);
}

#[test]
fn test_detection_result_creation() {
    let detection = DetectionResult::new([10.0, 20.0, 30.0, 40.0], 0.95, 0);
    assert_eq!(detection.confidence, 0.95);
    assert_eq!(detection.class_id, 0);
    assert_eq!(detection.center(), (20.0, 30.0));
}

#[test]
fn test_detection_result_area() {
    let detection = DetectionResult::new([0.0, 0.0, 10.0, 20.0], 0.9, 1);
    assert_eq!(detection.area(), 200.0);
}

// Note: These are placeholder tests. Real integration tests will be added
// once ONNX Runtime and image processing dependencies are integrated.
// Future tests will include:
// - Loading actual YOLO models
// - Running inference on test images
// - Validating detection outputs
// - Performance benchmarks
