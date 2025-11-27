// © 2014-2025 Ultralytics Inc. 🚀 All rights reserved. CONFIDENTIAL: Unauthorized use or distribution prohibited.

//! Inference engine and result types

/// Configuration for inference
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Confidence threshold for detections
    pub confidence_threshold: f32,
    /// IoU threshold for NMS
    pub iou_threshold: f32,
    /// Maximum number of detections
    pub max_detections: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        InferenceConfig {
            confidence_threshold: 0.25,
            iou_threshold: 0.45,
            max_detections: 300,
        }
    }
}

/// Detection result from YOLO inference
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// Bounding box coordinates [x1, y1, x2, y2]
    pub bbox: [f32; 4],
    /// Confidence score
    pub confidence: f32,
    /// Class ID
    pub class_id: usize,
    /// Class name (if available)
    pub class_name: Option<String>,
}

impl DetectionResult {
    /// Create a new detection result
    pub fn new(bbox: [f32; 4], confidence: f32, class_id: usize) -> Self {
        DetectionResult {
            bbox,
            confidence,
            class_id,
            class_name: None,
        }
    }

    /// Get the center point of the bounding box
    pub fn center(&self) -> (f32, f32) {
        let x = (self.bbox[0] + self.bbox[2]) / 2.0;
        let y = (self.bbox[1] + self.bbox[3]) / 2.0;
        (x, y)
    }

    /// Get the width and height of the bounding box
    pub fn size(&self) -> (f32, f32) {
        let w = self.bbox[2] - self.bbox[0];
        let h = self.bbox[3] - self.bbox[1];
        (w, h)
    }

    /// Get the area of the bounding box
    pub fn area(&self) -> f32 {
        let (w, h) = self.size();
        w * h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detection_result() {
        let det = DetectionResult::new([10.0, 20.0, 50.0, 80.0], 0.95, 0);
        assert_eq!(det.center(), (30.0, 50.0));
        assert_eq!(det.size(), (40.0, 60.0));
        assert_eq!(det.area(), 2400.0);
    }

    #[test]
    fn test_config_default() {
        let config = InferenceConfig::default();
        assert_eq!(config.confidence_threshold, 0.25);
        assert_eq!(config.iou_threshold, 0.45);
        assert_eq!(config.max_detections, 300);
    }
}
