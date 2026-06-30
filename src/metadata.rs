// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! ONNX model metadata parsing.
//!
//! This module handles parsing metadata from Ultralytics YOLO ONNX models.
//! The metadata is stored as YAML in the ONNX model's custom metadata properties.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{InferenceError, Result};
use crate::task::Task;

/// Metadata extracted from an Ultralytics YOLO ONNX model.
///
/// This struct contains all the configuration information embedded in the model,
/// including class names, input dimensions, and task type.
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Model description (e.g., "Ultralytics `YOLO11n` model trained on coco.yaml").
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
    pub imgsz: Option<(usize, usize)>,
    /// Number of input channels (typically 3 for RGB).
    pub channels: usize,
    /// Whether the model uses FP16 (half precision).
    pub half: bool,
    /// Class ID to class name mapping.
    pub names: Arc<HashMap<usize, String>>,
    /// Whether the model was exported with end-to-end NMS-free output
    /// (YOLO26-style post-NMS output: `[B, max_det, 6+extra]`).
    pub end2end: bool,
    /// Pose keypoint shape as (`num_keypoints`, `dims`), e.g. (17, 3).
    pub kpt_shape: Option<(usize, usize)>,
}

impl ModelMetadata {
    /// Parse metadata from ONNX model custom metadata properties.
    ///
    /// # Arguments
    ///
    /// * `metadata_map` - The custom metadata from the ONNX model session.
    ///
    /// # Returns
    ///
    /// * A new `ModelMetadata` instance.
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
    /// # Returns
    ///
    /// * A new `ModelMetadata` instance.
    ///
    /// # Errors
    ///
    /// Returns an error if the YAML is malformed or missing required fields.
    pub fn from_yaml_str(yaml_str: &str) -> Result<Self> {
        let mut metadata = Self::default();
        let mut names: HashMap<usize, String> = HashMap::new();

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
                            InferenceError::ModelLoadError(format!("Invalid stride value: {value}"))
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
                        metadata.half = value == "true" || value == "True";
                    }
                    "end2end" => {
                        metadata.end2end = value == "true" || value == "True";
                    }
                    "args" => {
                        // Parse args dict for half flag: {'half': True, ...}
                        if value.contains("'half': True")
                            || value.contains("\"half\": true")
                            || value.contains("'half':True")
                        {
                            metadata.half = true;
                        }
                    }
                    _ => {
                        // Check for class name entries (numeric keys)
                        if let Ok(class_id) = key.trim().parse::<usize>() {
                            names.insert(class_id, value.to_string());
                        }
                    }
                }
            }
        }

        // imgsz is a two-integer list (`[640, 640]` inline or a `- 640` block),
        // parsed with the same helper as kpt_shape.
        metadata.imgsz = Self::parse_int_pair(yaml_str, "imgsz");

        // kpt_shape is a two-integer list (inline `[17, 3]` or a `- 17` block),
        // parsed with the same helper as imgsz.
        metadata.kpt_shape = Self::parse_int_pair(yaml_str, "kpt_shape");

        // Parse names block if not already parsed inline
        if names.is_empty() {
            names = Self::parse_names_block(yaml_str);
        }

        metadata.names = Arc::new(names);
        Ok(metadata)
    }

    /// Parse a two-integer YAML value for `key`, accepting either the inline form
    /// (`key: [a, b]` / `key: a, b`) or a multi-line block list (`key:` then
    /// `- a` / `- b` on following lines). Returns the first two integers.
    fn parse_int_pair(yaml_str: &str, key: &str) -> Option<(usize, usize)> {
        let prefix = format!("{key}:");
        let lines: Vec<&str> = yaml_str.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            let Some(rest) = line.trim_start().strip_prefix(&prefix) else {
                continue;
            };
            // Inline form: [a, b] / (a, b) / a, b
            let rest = rest
                .trim()
                .trim_matches(|c| matches!(c, '[' | ']' | '(' | ')'));
            let inline: Vec<usize> = rest
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            if inline.len() >= 2 {
                return Some((inline[0], inline[1]));
            }
            // Multi-line block list on following `- value` lines.
            let mut vals = Vec::new();
            for following in lines.iter().skip(i + 1) {
                let t = following.trim();
                if let Some(v) = t.strip_prefix('-') {
                    if let Ok(n) = v.trim().parse::<usize>() {
                        vals.push(n);
                    }
                } else if !t.is_empty() && !t.starts_with('#') {
                    break;
                }
                if vals.len() >= 2 {
                    break;
                }
            }
            if vals.len() >= 2 {
                return Some((vals[0], vals[1]));
            }
        }
        None
    }

    /// Parse the names block from YAML or Python dict format.
    fn parse_names_block(yaml_str: &str) -> HashMap<usize, String> {
        let mut names = HashMap::new();

        // First, try to find `names: {0: 'person', 1: 'bicycle', ...}` Python dict format
        // This is how Ultralytics stores names in ONNX metadata
        if let Some(start) = yaml_str.find("names:") {
            let after_names = &yaml_str[start + 6..];
            let trimmed = after_names.trim();

            // Check if it's Python dict format (starts with {)
            if trimmed.starts_with('{')
                && let Some(end) = trimmed.find('}')
            {
                let dict_str = &trimmed[1..end];
                return Self::parse_python_dict(dict_str);
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
                if !trimmed.is_empty()
                    && !trimmed.starts_with('#')
                    && current_indent <= names_indent
                {
                    // Only exit if this isn't a class entry
                    if !trimmed.chars().next().is_some_and(|c| c.is_ascii_digit()) {
                        break;
                    }
                }

                // Parse class entries like "0: person" or "  0: person"
                if let Some((key, value)) = trimmed.split_once(':')
                    && let Ok(class_id) = key.trim().parse::<usize>()
                {
                    let class_name = value.trim().trim_matches('\'').trim_matches('"');
                    names.insert(class_id, class_name.to_string());
                }
            }
        }

        names
    }

    /// Parse a Python dict string like `0: 'person', 1: 'bicycle'`.
    fn parse_python_dict(dict_str: &str) -> HashMap<usize, String> {
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

        names
    }

    /// Get the number of classes in this model.
    ///
    /// # Returns
    ///
    /// * The count of classes.
    #[must_use]
    pub fn num_classes(&self) -> usize {
        self.names.len()
    }

    /// Get a class name by ID.
    ///
    /// # Arguments
    ///
    /// * `class_id` - The numeric identifier for the class.
    ///
    /// # Returns
    ///
    /// * `Some` class name if found, otherwise `None`.
    #[must_use]
    pub fn class_name(&self, class_id: usize) -> Option<&str> {
        self.names.get(&class_id).map(String::as_str)
    }

    /// Extract the model name from the description.
    ///
    /// E.g. "Ultralytics `YOLO11n` model..." -> "`YOLO11n`"
    /// Returns `YOLO` if extraction fails.
    #[must_use]
    pub fn model_name(&self) -> String {
        // Description format: "Ultralytics <MODEL> model..."
        self.description
            .split_whitespace()
            .find(|&word| word.to_lowercase().starts_with("yolo"))
            .unwrap_or("YOLO")
            .to_string()
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
            imgsz: None,
            channels: 3,
            half: false,
            names: Arc::new(HashMap::new()),
            end2end: false,
            kpt_shape: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_METADATA: &str = r"
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
";

    #[test]
    fn test_parse_metadata() {
        let metadata = ModelMetadata::from_yaml_str(SAMPLE_METADATA).unwrap();

        assert_eq!(metadata.task, Task::Detect);
        assert_eq!(metadata.stride, 32);
        assert_eq!(metadata.batch, 1);
        assert_eq!(metadata.imgsz, Some((640, 640)));
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
        assert_eq!(metadata.imgsz, Some((640, 640)));
    }

    #[test]
    fn test_default_metadata() {
        let metadata = ModelMetadata::default();
        assert_eq!(metadata.task, Task::Detect);
        assert_eq!(metadata.stride, 32);
        assert_eq!(metadata.imgsz, None);
    }

    #[test]
    fn test_parse_multiline_kpt_shape_and_imgsz() {
        // Ultralytics `.tflite` metadata.yaml writes imgsz and kpt_shape as
        // multi-line YAML block lists rather than inline `[a, b]`.
        let yaml = "task: pose\nstride: 32\nimgsz:\n- 640\n- 640\nkpt_shape:\n- 17\n- 3\n";
        let metadata = ModelMetadata::from_yaml_str(yaml).unwrap();
        assert_eq!(metadata.task, Task::Pose);
        assert_eq!(metadata.imgsz, Some((640, 640)));
        assert_eq!(metadata.kpt_shape, Some((17, 3)));
    }

    #[test]
    fn test_parse_inline_kpt_shape() {
        let yaml = "task: pose\nkpt_shape: [17, 3]\nstride: 32";
        let metadata = ModelMetadata::from_yaml_str(yaml).unwrap();
        assert_eq!(metadata.kpt_shape, Some((17, 3)));
    }

    #[test]
    fn test_parse_int_pair_inline_and_block() {
        // Inline forms: `[a, b]`, `(a, b)`, `a, b`.
        assert_eq!(
            ModelMetadata::parse_int_pair("kpt_shape: [17, 3]", "kpt_shape"),
            Some((17, 3))
        );
        assert_eq!(
            ModelMetadata::parse_int_pair("kpt_shape: (17, 2)", "kpt_shape"),
            Some((17, 2))
        );
        assert_eq!(
            ModelMetadata::parse_int_pair("foo: 5, 6", "foo"),
            Some((5, 6))
        );
        // Multi-line block list.
        assert_eq!(
            ModelMetadata::parse_int_pair("kpt_shape:\n- 17\n- 3", "kpt_shape"),
            Some((17, 3))
        );
        // A single value, junk, or wrong key yields None.
        assert_eq!(
            ModelMetadata::parse_int_pair("kpt_shape: [17]", "kpt_shape"),
            None
        );
        assert_eq!(ModelMetadata::parse_int_pair("foo: 5", "foo"), None);
        assert_eq!(ModelMetadata::parse_int_pair("other: 1, 2", "foo"), None);
    }

    #[test]
    fn test_python_dict_names() {
        let yaml = "task: detect\nnames: {0: 'person', 1: 'bicycle', 2: 'car'}";
        let m = ModelMetadata::from_yaml_str(yaml).unwrap();
        assert_eq!(m.num_classes(), 3);
        assert_eq!(m.class_name(1), Some("bicycle"));
    }

    #[test]
    fn test_tflite_style_metadata_text() {
        // The `.tflite` reader rebuilds this text (unquoted `names:` block, inline
        // lists) and hands it to `from_yaml_str`, exactly like the ONNX path.
        let text = "task: detect\nimgsz: [640, 640]\nstride: 32\nend2end: false\nnames:\n  0: person\n  1: traffic light\n  2: men's shoe";
        let m = ModelMetadata::from_yaml_str(text).unwrap();
        assert_eq!(m.task, Task::Detect);
        assert_eq!(m.imgsz, Some((640, 640)));
        assert_eq!(m.num_classes(), 3);
        // Names with a space and an apostrophe must survive (no escaping needed).
        assert_eq!(m.class_name(1), Some("traffic light"));
        assert_eq!(m.class_name(2), Some("men's shoe"));
    }

    #[test]
    fn test_pose_kpt_shape_and_half_flags() {
        let yaml = "task: pose\nkpt_shape: [17, 3]\nhalf: true\nend2end: True";
        let m = ModelMetadata::from_yaml_str(yaml).unwrap();
        assert_eq!(m.task, Task::Pose);
        assert_eq!(m.kpt_shape, Some((17, 3)));
        assert!(m.half);
        assert!(m.end2end);

        // half can also arrive embedded in an args dict.
        let yaml2 = "task: detect\nargs: {'half': True, 'imgsz': 640}";
        let m2 = ModelMetadata::from_yaml_str(yaml2).unwrap();
        assert!(m2.half);
    }

    #[test]
    fn test_invalid_fields_error() {
        assert!(ModelMetadata::from_yaml_str("task: notarealtask").is_err());
        assert!(ModelMetadata::from_yaml_str("task: detect\nstride: abc").is_err());
        assert!(ModelMetadata::from_yaml_str("task: detect\nbatch: xyz").is_err());
    }

    #[test]
    fn test_from_onnx_metadata_keys_and_error() {
        // Standard "metadata" key.
        let map = HashMap::from([("metadata".to_string(), SAMPLE_METADATA.to_string())]);
        let m = ModelMetadata::from_onnx_metadata(&map).unwrap();
        assert_eq!(m.task, Task::Detect);

        // Fallback: any value containing "task:".
        let map = HashMap::from([("whatever".to_string(), "task: pose\nstride: 32".to_string())]);
        let m = ModelMetadata::from_onnx_metadata(&map).unwrap();
        assert_eq!(m.task, Task::Pose);

        // No usable metadata -> error.
        let map = HashMap::from([("unrelated".to_string(), "no yaml here".to_string())]);
        assert!(ModelMetadata::from_onnx_metadata(&map).is_err());
    }

    #[test]
    fn test_model_name_extraction() {
        let m = ModelMetadata::from_yaml_str(SAMPLE_METADATA).unwrap();
        assert_eq!(m.model_name(), "YOLO11n");
        // Falls back to "YOLO" when the description has no yolo token.
        let plain = ModelMetadata::default();
        assert_eq!(plain.model_name(), "YOLO");
    }
}
