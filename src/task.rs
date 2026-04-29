// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Task definitions for YOLO models.
//!
//! This module defines the different tasks that YOLO models can perform,
//! along with their associated capabilities and string representations.

use std::fmt;
use std::str::FromStr;

/// YOLO model task types.
///
/// Each task type corresponds to a different computer vision problem
/// that YOLO models can solve. The task type determines the expected
/// model outputs and post-processing steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Task {
    /// Object detection.
    /// Predicts bounding boxes and class labels for objects in an image.
    #[default]
    Detect,
    /// Instance segmentation.
    /// Predicts bounding boxes, class labels, and pixel-level masks for objects.
    Segment,
    /// Pose estimation.
    /// Predicts bounding boxes and skeletal keypoints for objects (e.g., humans).
    Pose,
    /// Image classification.
    /// Predicts class probabilities for the entire image (no localization).
    Classify,
    /// Oriented bounding box detection (OBB).
    /// Predicts rotated bounding boxes for objects, useful for aerial imagery etc.
    Obb,
}

impl Task {
    /// Get the string representation used in ONNX model metadata
    /// (e.g. `"detect"`, `"segment"`).
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Detect => "detect",
            Self::Segment => "segment",
            Self::Pose => "pose",
            Self::Classify => "classify",
            Self::Obb => "obb",
        }
    }

    /// ONNX filename suffix for this task, used to construct `yolo26n{suffix}.onnx`.
    ///
    /// ```
    /// use ultralytics_inference::Task;
    /// assert_eq!(Task::Detect.model_suffix(), "");
    /// assert_eq!(Task::Segment.model_suffix(), "-seg");
    /// ```
    #[must_use]
    pub const fn model_suffix(&self) -> &'static str {
        match self {
            Self::Detect => "",
            Self::Segment => "-seg",
            Self::Pose => "-pose",
            Self::Classify => "-cls",
            Self::Obb => "-obb",
        }
    }

    /// Default nano YOLO26 model filename for this task.
    ///
    /// Used by the CLI to auto-pick a model when `--model` is omitted but `--task` is set.
    ///
    /// ```
    /// use ultralytics_inference::Task;
    /// assert_eq!(Task::Detect.default_model(), "yolo26n.onnx");
    /// assert_eq!(Task::Segment.default_model(), "yolo26n-seg.onnx");
    /// ```
    #[must_use]
    pub fn default_model(&self) -> String {
        format!("yolo26n{}.onnx", self.model_suffix())
    }

    /// Returns `true` when the task outputs bounding boxes — namely Detect, Segment, Pose, and Obb.
    #[must_use]
    pub const fn has_boxes(&self) -> bool {
        matches!(self, Self::Detect | Self::Segment | Self::Pose | Self::Obb)
    }

    /// Returns `true` only for the Segment task, which outputs per-instance segmentation masks.
    #[must_use]
    pub const fn has_masks(&self) -> bool {
        matches!(self, Self::Segment)
    }

    /// Returns `true` only for the Pose task, which outputs skeletal keypoints.
    #[must_use]
    pub const fn has_keypoints(&self) -> bool {
        matches!(self, Self::Pose)
    }

    /// Returns `true` only for the Classify task, which outputs global class probabilities.
    #[must_use]
    pub const fn has_probs(&self) -> bool {
        matches!(self, Self::Classify)
    }

    /// Returns `true` only for the Obb task, which outputs oriented (rotated) bounding boxes.
    #[must_use]
    pub const fn has_obb(&self) -> bool {
        matches!(self, Self::Obb)
    }
}

impl fmt::Display for Task {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Task {
    type Err = TaskParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "detect" | "detection" => Ok(Self::Detect),
            "segment" | "segmentation" => Ok(Self::Segment),
            "pose" | "keypoint" | "keypoints" => Ok(Self::Pose),
            "classify" | "classification" | "cls" => Ok(Self::Classify),
            "obb" | "oriented" => Ok(Self::Obb),
            _ => Err(TaskParseError(s.to_string())),
        }
    }
}

/// Error returned when parsing an invalid task string.
#[derive(Debug, Clone)]
pub struct TaskParseError(String);

impl fmt::Display for TaskParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid task '{}', expected one of: detect, segment, pose, classify, obb",
            self.0
        )
    }
}

impl std::error::Error for TaskParseError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_from_str() {
        assert_eq!("detect".parse::<Task>().unwrap(), Task::Detect);
        assert_eq!("segment".parse::<Task>().unwrap(), Task::Segment);
        assert_eq!("pose".parse::<Task>().unwrap(), Task::Pose);
        assert_eq!("classify".parse::<Task>().unwrap(), Task::Classify);
        assert_eq!("obb".parse::<Task>().unwrap(), Task::Obb);

        // Alternative names
        assert_eq!("detection".parse::<Task>().unwrap(), Task::Detect);
        assert_eq!("segmentation".parse::<Task>().unwrap(), Task::Segment);
        assert_eq!("keypoints".parse::<Task>().unwrap(), Task::Pose);
        assert_eq!("cls".parse::<Task>().unwrap(), Task::Classify);
    }

    #[test]
    fn test_task_display() {
        assert_eq!(Task::Detect.to_string(), "detect");
        assert_eq!(Task::Segment.to_string(), "segment");
    }

    #[test]
    fn test_task_capabilities() {
        assert!(Task::Detect.has_boxes());
        assert!(!Task::Detect.has_masks());
        assert!(Task::Segment.has_masks());
        assert!(Task::Pose.has_keypoints());
        assert!(Task::Classify.has_probs());
        assert!(Task::Obb.has_obb());
    }
}
