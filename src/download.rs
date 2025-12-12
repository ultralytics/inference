// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Model downloading utilities.
//!
//! This module provides functionality to automatically download YOLO models
//! from Ultralytics GitHub releases when they are not found locally.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::error::{InferenceError, Result};

/// Default YOLO model name.
pub const DEFAULT_MODEL: &str = "yolo11n.onnx";

/// URL for downloading the default YOLO model.
const DEFAULT_MODEL_URL: &str =
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx";

/// Format bytes as human-readable string (e.g., "10.4MB").
fn format_bytes(bytes: f64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;

    if bytes >= GB {
        format!("{:.1}GB", bytes / GB)
    } else if bytes >= MB {
        format!("{:.1}MB", bytes / MB)
    } else if bytes >= KB {
        format!("{:.1}KB", bytes / KB)
    } else {
        format!("{:.0}B", bytes)
    }
}

/// Format time duration.
fn format_time(seconds: f64) -> String {
    if seconds < 60.0 {
        format!("{seconds:.1}s")
    } else if seconds < 3600.0 {
        let mins = (seconds / 60.0) as u32;
        let secs = seconds % 60.0;
        format!("{mins}:{secs:04.1}")
    } else {
        let hours = (seconds / 3600.0) as u32;
        let mins = ((seconds % 3600.0) / 60.0) as u32;
        let secs = seconds % 60.0;
        format!("{hours}:{mins:02}:{secs:04.1}")
    }
}

/// Generate progress bar string.
fn generate_bar(progress: f64, width: usize) -> String {
    let filled = (progress * width as f64) as usize;
    let partial = progress * width as f64 - filled as f64;

    let mut bar = "━".repeat(filled);
    if filled < width {
        if partial > 0.5 {
            bar.push('╸');
            bar.push_str(&"─".repeat(width - filled - 1));
        } else {
            bar.push_str(&"─".repeat(width - filled));
        }
    }
    bar
}

/// Download a file from URL to the specified path with progress bar.
fn download_file(url: &str, dest: &Path) -> Result<()> {
    let response = ureq::get(url).call().map_err(|e| {
        InferenceError::ModelLoadError(format!("Failed to download model: {e}"))
    })?;

    // Get content length for progress calculation
    let content_length: Option<u64> = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok());

    let total_size = content_length.unwrap_or(0);
    let mut reader = response.into_body().into_reader();
    let mut bytes = Vec::with_capacity(total_size as usize);
    let mut downloaded: u64 = 0;
    let mut buffer = [0u8; 65536]; // 64KB buffer for fewer updates
    let start_time = Instant::now();
    let mut last_update = Instant::now();

    // Progress bar settings
    const BAR_WIDTH: usize = 12;
    const MIN_UPDATE_INTERVAL: f64 = 0.1; // Update at most every 100ms

    // Description for progress bar
    let desc = format!("Downloading {} to '{}'", url, dest.display());

    loop {
        let bytes_read = reader.read(&mut buffer).map_err(|e| {
            InferenceError::ModelLoadError(format!("Failed to read response: {e}"))
        })?;

        if bytes_read == 0 {
            break;
        }

        bytes.extend_from_slice(&buffer[..bytes_read]);
        downloaded += bytes_read as u64;

        // Rate-limit updates
        let now = Instant::now();
        if now.duration_since(last_update).as_secs_f64() < MIN_UPDATE_INTERVAL {
            continue;
        }
        last_update = now;

        // Calculate progress
        let elapsed = start_time.elapsed().as_secs_f64();
        let rate = if elapsed > 0.0 {
            downloaded as f64 / elapsed
        } else {
            0.0
        };

        if total_size > 0 {
            let progress = (downloaded as f64 / total_size as f64).min(1.0);
            let percent = (progress * 100.0) as u8;
            let bar = generate_bar(progress, BAR_WIDTH);
            let rate_str = format!("{}/s", format_bytes(rate));

            eprint!(
                "\r\x1b[K{}: {}% {} {}/{} {} {}",
                desc,
                percent,
                bar,
                format_bytes(downloaded as f64),
                format_bytes(total_size as f64),
                rate_str,
                format_time(elapsed)
            );
        } else {
            eprint!(
                "\r\x1b[K{}: {} {}/s {}",
                desc,
                format_bytes(downloaded as f64),
                format_bytes(rate),
                format_time(elapsed)
            );
        }
        std::io::stderr().flush().ok();
    }

    // Final progress line
    let elapsed = start_time.elapsed().as_secs_f64();
    let rate = if elapsed > 0.0 {
        downloaded as f64 / elapsed
    } else {
        0.0
    };

    if total_size > 0 {
        let bar = generate_bar(1.0, BAR_WIDTH);
        eprintln!(
            "\r\x1b[K{}: 100% {} {} {}/s {}",
            desc,
            bar,
            format_bytes(total_size as f64),
            format_bytes(rate),
            format_time(elapsed)
        );
    } else {
        eprintln!(
            "\r\x1b[K{}: {} {}/s {}",
            desc,
            format_bytes(downloaded as f64),
            format_bytes(rate),
            format_time(elapsed)
        );
    }

    // Write to file
    let mut file = fs::File::create(dest).map_err(|e| {
        InferenceError::ModelLoadError(format!("Failed to create file {}: {e}", dest.display()))
    })?;

    file.write_all(&bytes).map_err(|e| {
        InferenceError::ModelLoadError(format!("Failed to write model file: {e}"))
    })?;

    Ok(())
}

/// Attempt to download a model if it matches a known downloadable model.
///
/// Currently supports:
/// - `yolo11n.onnx` - Default YOLO11n model
///
/// Downloads to the current working directory (or the directory specified in the path).
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

    // Use the path as provided (current directory if just filename)
    let dest_path = path.to_path_buf();

    // Download the model
    download_file(url, &dest_path)?;

    Ok(dest_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unknown_model_returns_error() {
        let result = try_download_model("unknown_model.onnx");
        assert!(result.is_err());
    }
}
