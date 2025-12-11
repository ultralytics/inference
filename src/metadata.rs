// © 2014-2025 Ultralytics Inc. 🚀 All rights reserved. CONFIDENTIAL: Unauthorized use or distribution prohibited.

//! ONNX model metadata parsing.
//!
//! This module handles parsing metadata from Ultralytics YOLO ONNX models.
//! The metadata is stored as YAML in the ONNX model's custom metadata properties.

use std::collections::HashMap;

use crate::error::{InferenceError, Result};
use crate::task::Task;

/// Metadata extracted from an Ultralytics YOLO ONNX model.
///
/// This struct contains all the configuration information embedded in the model,
/// including class names, input dimensions, and task type.
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Model description (e.g., "Ultralytics YOLO11n model trained on coco.yaml").
    pub description: String,
    /// Model author.
    pub author: String,
    /// Export date.
    pub date: String,
    /// Ultralytics version used for export.
    pub version: String,
    /// License information.
    pub license: String,
    /// Documentation URL.
    pub docs: String,
    /// The task this model performs.
    pub task: Task,
    /// Model stride (typically 32 for YOLO).
    pub stride: u32,
    /// Batch size the model was exported with.
    pub batch: usize,
    /// Input image size as (height, width).
    pub imgsz: (usize, usize),
    /// Number of input channels (typically 3 for RGB).
    pub channels: usize,
    /// Whether the model uses FP16 (half precision).
    pub half: bool,
    /// Class ID to class name mapping.
    pub names: HashMap<usize, String>,
}

impl ModelMetadata {
    /// Parse metadata from ONNX model custom metadata properties.
    ///
    /// # Arguments
    ///
    /// * `metadata_map` - The custom metadata from the ONNX model session.
    ///
    /// # Errors
    ///
    /// Returns an error if the metadata is missing or malformed.
    pub fn from_onnx_metadata(metadata_map: &HashMap<String, String>) -> Result<Self> {
        // The metadata is typically stored under a single key containing YAML
        // Try common key names used by Ultralytics
        let yaml_str = metadata_map
            .get("metadata")
            .or_else(|| metadata_map.get("model_metadata"))
            .or_else(|| {
                // If no standard key, check if all metadata is in one value
                metadata_map.values().find(|v| v.contains("task:"))
            })
            .ok_or_else(|| {
                InferenceError::ModelLoadError(
                    "No metadata found in ONNX model. Ensure the model was exported with Ultralytics.".to_string()
                )
            })?;

        Self::from_yaml_str(yaml_str)
    }

    /// Parse metadata from a YAML string.
    ///
    /// # Arguments
    ///
    /// * `yaml_str` - The YAML-formatted metadata string.
    ///
    /// # Errors
    ///
    /// Returns an error if the YAML is malformed or missing required fields.
    pub fn from_yaml_str(yaml_str: &str) -> Result<Self> {
        // Parse YAML manually to avoid serde_yaml dependency complexity
        let mut metadata = Self::default();

        for line in yaml_str.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Handle key: value pairs
            if let Some((key, value)) = line.split_once(':') {
                let key = key.trim();
                let value = value.trim().trim_matches('\'').trim_matches('"');

                match key {
                    "description" => metadata.description = value.to_string(),
                    "author" => metadata.author = value.to_string(),
                    "date" => metadata.date = value.to_string(),
                    "version" => metadata.version = value.to_string(),
                    "license" => metadata.license = value.to_string(),
                    "docs" => metadata.docs = value.to_string(),
                    "task" => {
                        metadata.task = value.parse().map_err(|e| {
                            InferenceError::ModelLoadError(format!("Invalid task in metadata: {e}"))
                        })?;
                    }
                    "stride" => {
                        metadata.stride = value.parse().map_err(|_| {
                            InferenceError::ModelLoadError(format!(
                                "Invalid stride value: {value}"
                            ))
                        })?;
                    }
                    "batch" => {
                        metadata.batch = value.parse().map_err(|_| {
                            InferenceError::ModelLoadError(format!("Invalid batch value: {value}"))
                        })?;
                    }
                    "channels" => {
                        metadata.channels = value.parse().map_err(|_| {
                            InferenceError::ModelLoadError(format!(
                                "Invalid channels value: {value}"
                            ))
                        })?;
                    }
                    "half" => {
                        metadata.half = value == "true";
                    }
                    _ => {
                        // Check for class name entries (numeric keys)
                        if let Ok(class_id) = key.trim().parse::<usize>() {
                            metadata.names.insert(class_id, value.to_string());
                        }
                    }
                }
            }
        }

        // Parse imgsz which can be a list like [640, 640]
        if let Some(imgsz_line) = yaml_str.lines().find(|l| l.contains("imgsz:")) {
            metadata.imgsz = Self::parse_imgsz(yaml_str, imgsz_line)?;
        }

        // Parse names block if not already parsed inline
        if metadata.names.is_empty() {
            metadata.names = Self::parse_names_block(yaml_str)?;
        }

        Ok(metadata)
    }

    /// Parse the imgsz field which can be a YAML list.
    fn parse_imgsz(yaml_str: &str, imgsz_line: &str) -> Result<(usize, usize)> {
        // Check if imgsz is on a single line like "imgsz: [640, 640]"
        if let Some(bracket_start) = imgsz_line.find('[') {
            if let Some(bracket_end) = imgsz_line.find(']') {
                let values: Vec<usize> = imgsz_line[bracket_start + 1..bracket_end]
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
                if values.len() >= 2 {
                    return Ok((values[0], values[1]));
                }
            }
        }

        // Check for multi-line YAML list format
        let lines: Vec<&str> = yaml_str.lines().collect();
        let mut imgsz_values = Vec::new();

        for (i, line) in lines.iter().enumerate() {
            if line.contains("imgsz:") {
                // Look at following lines for list items
                for following in lines.iter().skip(i + 1) {
                    let trimmed = following.trim();
                    if trimmed.starts_with('-') {
                        if let Ok(val) = trimmed.trim_start_matches('-').trim().parse::<usize>() {
                            imgsz_values.push(val);
                        }
                    } else if !trimmed.is_empty() && !trimmed.starts_with('#') {
                        break;
                    }
                    if imgsz_values.len() >= 2 {
                        break;
                    }
                }
                break;
            }
        }

        if imgsz_values.len() >= 2 {
            Ok((imgsz_values[0], imgsz_values[1]))
        } else {
            // Default to 640x640
            Ok((640, 640))
        }
    }

    /// Parse the names block from YAML or Python dict format.
    fn parse_names_block(yaml_str: &str) -> Result<HashMap<usize, String>> {
        let mut names = HashMap::new();

        // First, try to find `names: {0: 'person', 1: 'bicycle', ...}` Python dict format
        // This is how Ultralytics stores names in ONNX metadata
        if let Some(start) = yaml_str.find("names:") {
            let after_names = &yaml_str[start + 6..];
            let trimmed = after_names.trim();

            // Check if it's Python dict format (starts with {)
            if trimmed.starts_with('{') {
                if let Some(end) = trimmed.find('}') {
                    let dict_str = &trimmed[1..end];
                    return Self::parse_python_dict(dict_str);
                }
            }
        }

        // Fall back to YAML block format
        let lines: Vec<&str> = yaml_str.lines().collect();
        let mut in_names_block = false;
        let mut names_indent = 0;

        for line in &lines {
            let trimmed = line.trim();

            if trimmed.starts_with("names:") {
                in_names_block = true;
                names_indent = line.len() - line.trim_start().len();
                continue;
            }

            if in_names_block {
                let current_indent = line.len() - line.trim_start().len();

                // Check if we've exited the names block
                if !trimmed.is_empty() && !trimmed.starts_with('#') && current_indent <= names_indent
                {
                    // Only exit if this isn't a class entry
                    if !trimmed.chars().next().is_some_and(|c| c.is_ascii_digit()) {
                        break;
                    }
                }

                // Parse class entries like "0: person" or "  0: person"
                if let Some((key, value)) = trimmed.split_once(':') {
                    if let Ok(class_id) = key.trim().parse::<usize>() {
                        let class_name = value.trim().trim_matches('\'').trim_matches('"');
                        names.insert(class_id, class_name.to_string());
                    }
                }
            }
        }

        Ok(names)
    }

    /// Parse a Python dict string like `0: 'person', 1: 'bicycle'`.
    fn parse_python_dict(dict_str: &str) -> Result<HashMap<usize, String>> {
        let mut names = HashMap::new();

        // Split by comma, but be careful with quotes
        for entry in dict_str.split(',') {
            let entry = entry.trim();
            if let Some((key, value)) = entry.split_once(':') {
                let key = key.trim();
                let value = value.trim().trim_matches('\'').trim_matches('"');
                if let Ok(class_id) = key.parse::<usize>() {
                    names.insert(class_id, value.to_string());
                }
            }
        }

        Ok(names)
    }

    /// Get the number of classes in this model.
    #[must_use]
    pub fn num_classes(&self) -> usize {
        self.names.len()
    }

    /// Get a class name by ID.
    #[must_use]
    pub fn class_name(&self, class_id: usize) -> Option<&str> {
        self.names.get(&class_id).map(String::as_str)
    }
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            description: String::new(),
            author: "Ultralytics".to_string(),
            date: String::new(),
            version: String::new(),
            license: "AGPL-3.0".to_string(),
            docs: "https://docs.ultralytics.com".to_string(),
            task: Task::Detect,
            stride: 32,
            batch: 1,
            imgsz: (640, 640),
            channels: 3,
            half: false,
            names: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_METADATA: &str = r#"
description: Ultralytics YOLO11n model trained on /usr/src/ultralytics/ultralytics/cfg/datasets/coco.yaml
author: Ultralytics
date: '2025-12-11T20:19:45.464021'
version: 8.3.236
license: AGPL-3.0 License (https://ultralytics.com/license)
docs: https://docs.ultralytics.com
stride: 32
task: detect
batch: 1
imgsz:
- 640
- 640
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
channels: 3
"#;

    #[test]
    fn test_parse_metadata() {
        let metadata = ModelMetadata::from_yaml_str(SAMPLE_METADATA).unwrap();

        assert_eq!(metadata.task, Task::Detect);
        assert_eq!(metadata.stride, 32);
        assert_eq!(metadata.batch, 1);
        assert_eq!(metadata.imgsz, (640, 640));
        assert_eq!(metadata.channels, 3);
        assert_eq!(metadata.num_classes(), 4);
        assert_eq!(metadata.class_name(0), Some("person"));
        assert_eq!(metadata.class_name(1), Some("bicycle"));
        assert_eq!(metadata.class_name(2), Some("car"));
        assert_eq!(metadata.class_name(3), Some("motorcycle"));
    }

    #[test]
    fn test_parse_inline_imgsz() {
        let yaml = "task: detect\nimgsz: [640, 640]\nstride: 32";
        let metadata = ModelMetadata::from_yaml_str(yaml).unwrap();
        assert_eq!(metadata.imgsz, (640, 640));
    }

    #[test]
    fn test_default_metadata() {
        let metadata = ModelMetadata::default();
        assert_eq!(metadata.task, Task::Detect);
        assert_eq!(metadata.stride, 32);
        assert_eq!(metadata.imgsz, (640, 640));
    }
}
