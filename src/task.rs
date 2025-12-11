// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Task definitions for YOLO models.
//!
//! This module defines the different tasks that YOLO models can perform,
//! along with their associated configurations.

use std::fmt;
use std::str::FromStr;

/// YOLO model task types.
///
/// Each task type corresponds to a different type of computer vision problem
/// that YOLO models can solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Task {
    /// Object detection - predicts bounding boxes and class labels.
    Detect,
    /// Instance segmentation - predicts masks for each detected object.
    Segment,
    /// Pose estimation - predicts keypoints for detected objects (e.g., human pose).
    Pose,
    /// Image classification - predicts class probabilities for the entire image.
    Classify,
    /// Oriented bounding box detection - predicts rotated bounding boxes.
    Obb,
}

impl Task {
    /// Returns the string representation used in ONNX model metadata.
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

    /// Returns whether this task produces bounding boxes.
    #[must_use]
    pub const fn has_boxes(&self) -> bool {
        matches!(self, Self::Detect | Self::Segment | Self::Pose | Self::Obb)
    }

    /// Returns whether this task produces segmentation masks.
    #[must_use]
    pub const fn has_masks(&self) -> bool {
        matches!(self, Self::Segment)
    }

    /// Returns whether this task produces keypoints.
    #[must_use]
    pub const fn has_keypoints(&self) -> bool {
        matches!(self, Self::Pose)
    }

    /// Returns whether this task produces classification probabilities.
    #[must_use]
    pub const fn has_probs(&self) -> bool {
        matches!(self, Self::Classify)
    }

    /// Returns whether this task produces oriented bounding boxes.
    #[must_use]
    pub const fn has_obb(&self) -> bool {
        matches!(self, Self::Obb)
    }
}

impl fmt::Display for Task {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
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

impl Default for Task {
    fn default() -> Self {
        Self::Detect
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
