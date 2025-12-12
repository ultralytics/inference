// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Model downloading utilities.
//!
//! This module provides functionality to automatically download YOLO models
//! from Ultralytics GitHub releases when they are not found locally.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::error::{InferenceError, Result};

/// Default YOLO model name.
pub const DEFAULT_MODEL: &str = "yolo11n.onnx";

/// URL for downloading the default YOLO model.
const DEFAULT_MODEL_URL: &str =
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx";

/// Get the Ultralytics cache directory for storing downloaded models.
///
/// Returns `~/.cache/ultralytics` on Unix or the equivalent on other platforms.
fn get_cache_dir() -> Result<PathBuf> {
    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| InferenceError::ModelLoadError("Could not determine cache directory".into()))?
        .join("ultralytics");

    // Create the directory if it doesn't exist
    if !cache_dir.exists() {
        fs::create_dir_all(&cache_dir).map_err(|e| {
            InferenceError::ModelLoadError(format!("Failed to create cache directory: {e}"))
        })?;
    }

    Ok(cache_dir)
}

/// Download a file from URL to the specified path.
fn download_file(url: &str, dest: &Path) -> Result<()> {
    eprintln!("Downloading {} to {}", url, dest.display());

    let response = ureq::get(url).call().map_err(|e| {
        InferenceError::ModelLoadError(format!("Failed to download model: {e}"))
    })?;

    let mut reader = response.into_body().into_reader();
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut reader, &mut bytes).map_err(|e| {
        InferenceError::ModelLoadError(format!("Failed to read response body: {e}"))
    })?;

    let mut file = fs::File::create(dest).map_err(|e| {
        InferenceError::ModelLoadError(format!("Failed to create file {}: {e}", dest.display()))
    })?;

    file.write_all(&bytes).map_err(|e| {
        InferenceError::ModelLoadError(format!("Failed to write model file: {e}"))
    })?;

    eprintln!("Download complete: {}", dest.display());
    Ok(())
}

/// Attempt to download a model if it matches a known downloadable model.
///
/// Currently supports:
/// - `yolo11n.onnx` - Default YOLO11n model
///
/// Returns the path to the downloaded model, or an error if download fails
/// or the model is not a known downloadable model.
pub fn try_download_model<P: AsRef<Path>>(model_path: P) -> Result<PathBuf> {
    let path = model_path.as_ref();
    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");

    // Check if this is a model we can download
    let url = match filename {
        DEFAULT_MODEL => DEFAULT_MODEL_URL,
        _ => {
            return Err(InferenceError::ModelLoadError(format!(
                "Model file not found: {}. Only '{}' can be auto-downloaded.",
                path.display(),
                DEFAULT_MODEL
            )));
        }
    };

    // Get cache directory and construct destination path
    let cache_dir = get_cache_dir()?;
    let dest_path = cache_dir.join(filename);

    // Check if already cached
    if dest_path.exists() {
        eprintln!("Using cached model: {}", dest_path.display());
        return Ok(dest_path);
    }

    // Download the model
    download_file(url, &dest_path)?;

    Ok(dest_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_cache_dir() {
        let cache_dir = get_cache_dir();
        assert!(cache_dir.is_ok());
        let dir = cache_dir.unwrap();
        assert!(dir.to_string_lossy().contains("ultralytics"));
    }

    #[test]
    fn test_unknown_model_returns_error() {
        let result = try_download_model("unknown_model.onnx");
        assert!(result.is_err());
    }
}
