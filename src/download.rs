// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Model downloading utilities.
//!
//! This module provides functionality to automatically download YOLO models
//! from Ultralytics GitHub releases when they are not found locally.

use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::error::{InferenceError, Result};

/// Default YOLO model name.
pub const DEFAULT_MODEL: &str = "yolo11n.onnx";

/// Default YOLO segmentation model name.
pub const DEFAULT_SEGMENT_MODEL: &str = "yolo11n-seg.onnx";

/// URL for downloading the default YOLO model.
const DEFAULT_DETECT_MODEL_URL: &str =
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx";

/// URL for downloading the default YOLO model.
const DEFAULT_SEGMENT_MODEL_URL: &str =
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.onnx";

/// Default YOLO pose model name.
pub const DEFAULT_POSE_MODEL: &str = "yolo11n-pose.onnx";

/// URL for downloading the default YOLO pose model.
const DEFAULT_POSE_MODEL_URL: &str =
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.onnx";

/// Default YOLO OBB model name.
pub const DEFAULT_OBB_MODEL: &str = "yolo11n-obb.onnx";

/// URL for downloading the default YOLO OBB model.
const DEFAULT_OBB_MODEL_URL: &str =
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-obb.onnx";

/// Default YOLO classification model name.
pub const DEFAULT_CLS_MODEL: &str = "yolo11n-cls.onnx";

/// URL for downloading the default YOLO classification model.
const DEFAULT_CLS_MODEL_URL: &str =
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.onnx";

/// URL for downloading the default Bus.jpg image
const DEFAULT_BUS_IMAGE_URL: &str = "https://ultralytics.com/images/bus.jpg";

/// URL for downloading the default Zidane.jpg image
const DEFAULT_ZIDANE_IMAGE_URL: &str = "https://ultralytics.com/images/zidane.jpg";

/// URL for downloading the default Boats.jpg image (for OBB)
const DEFAULT_BOATS_IMAGE_URL: &str = "https://ultralytics.com/images/boats.jpg";

/// Default image URLs for detection, segmentation, pose, and classification tasks
pub const DEFAULT_IMAGES: &[&str] = &[DEFAULT_BUS_IMAGE_URL, DEFAULT_ZIDANE_IMAGE_URL];

/// Default image URL for OBB (Oriented Bounding Box) tasks
pub const DEFAULT_OBB_IMAGE: &str = DEFAULT_BOATS_IMAGE_URL;

/// Connection timeout in seconds.
const CONNECT_TIMEOUT: u64 = 30;

/// Read timeout in seconds.
const READ_TIMEOUT: u64 = 300;

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
///
/// Uses streaming download to a temporary file, then atomic rename to prevent
/// corrupted files from partial downloads.
fn download_file(url: &str, dest: &Path) -> Result<()> {
    // Create ureq agent with timeouts
    let config = ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(CONNECT_TIMEOUT)))
        .timeout_recv_body(Some(Duration::from_secs(READ_TIMEOUT)))
        .build();
    let agent = ureq::Agent::new_with_config(config);

    let response = agent.get(url).call().map_err(|e| {
        let msg = match &e {
            ureq::Error::Timeout(_) => format!("Connection timed out while downloading {url}"),
            ureq::Error::Io(io_err) => {
                format!("Network error downloading {url}: {io_err}")
            }
            _ => format!("Failed to download {url}: {e}"),
        };
        InferenceError::ModelLoadError(msg)
    })?;

    // Get content length for progress calculation
    let content_length: Option<u64> = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s: &str| s.parse().ok());

    let total_size = content_length.unwrap_or(0);

    // Create temp file for atomic download (same directory for atomic rename)
    let temp_path = dest.with_extension("part");

    // Clean up any existing partial download
    let _ = fs::remove_file(&temp_path);

    let temp_file = File::create(&temp_path).map_err(|e| {
        InferenceError::ModelLoadError(format!(
            "Failed to create temp file {}: {e}",
            temp_path.display()
        ))
    })?;
    let mut writer = BufWriter::new(temp_file);

    let mut reader = response.into_body().into_reader();
    let mut downloaded: u64 = 0;
    let mut buffer = [0u8; 65536]; // 64KB buffer
    let start_time = Instant::now();
    let mut last_update = Instant::now();

    // Progress bar settings
    const BAR_WIDTH: usize = 12;
    const MIN_UPDATE_INTERVAL: f64 = 0.1; // Update at most every 100ms

    // Description for progress bar
    let desc = format!("Downloading {} to '{}'", url, dest.display());

    let download_result: std::result::Result<(), InferenceError> = (|| {
        loop {
            let bytes_read = reader.read(&mut buffer).map_err(|e| {
                InferenceError::ModelLoadError(format!("Failed to read from network: {e}"))
            })?;

            if bytes_read == 0 {
                break;
            }

            // Stream directly to file
            writer.write_all(&buffer[..bytes_read]).map_err(|e| {
                InferenceError::ModelLoadError(format!("Failed to write to temp file: {e}"))
            })?;

            downloaded += bytes_read as u64;

            // Rate-limit progress updates
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

        // Flush the writer
        writer.flush().map_err(|e| {
            InferenceError::ModelLoadError(format!("Failed to flush temp file: {e}"))
        })?;

        Ok(())
    })();

    // Handle download failure - clean up temp file
    if let Err(e) = download_result {
        let _ = fs::remove_file(&temp_path);
        return Err(e);
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

    // Atomic rename from temp file to final destination
    fs::rename(&temp_path, dest).map_err(|e| {
        // Clean up temp file on rename failure
        let _ = fs::remove_file(&temp_path);
        InferenceError::ModelLoadError(format!(
            "Failed to move downloaded file to {}: {e}",
            dest.display()
        ))
    })?;

    Ok(())
}

/// Attempt to download a model if it matches a known downloadable model.
///
/// Currently supports:
/// - `yolo11n.onnx` - Default YOLO11n detection model
/// - `yolo11n-seg.onnx` - YOLO11n segmentation model
/// - `yolo11n-pose.onnx` - YOLO11n pose estimation model
/// - `yolo11n-obb.onnx` - YOLO11n oriented bounding box model
/// - `yolo11n-cls.onnx` - YOLO11n classification model
///
/// Downloads to the current working directory (or the directory specified in the path).
///
/// Returns the path to the downloaded model, or an error if download fails
/// or the model is not a known downloadable model.
pub fn try_download_model<P: AsRef<Path>>(model_path: P) -> Result<PathBuf> {
    let path = model_path.as_ref();
    let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

    // Map of supported downloadable models
    let url = match filename {
        DEFAULT_MODEL => DEFAULT_DETECT_MODEL_URL,
        DEFAULT_SEGMENT_MODEL => DEFAULT_SEGMENT_MODEL_URL,
        DEFAULT_POSE_MODEL => DEFAULT_POSE_MODEL_URL,
        DEFAULT_OBB_MODEL => DEFAULT_OBB_MODEL_URL,
        DEFAULT_CLS_MODEL => DEFAULT_CLS_MODEL_URL,
        _ => {
            return Err(InferenceError::ModelLoadError(format!(
                "Model file not found: {}. Auto-download is supported for: yolo11n.onnx, yolo11n-seg.onnx, yolo11n-pose.onnx, yolo11n-obb.onnx, yolo11n-cls.onnx",
                path.display(),
            )));
        }
    };

    // Use the path as provided (current directory if just filename)
    let dest_path = path.to_path_buf();

    // Download the model
    download_file(url, &dest_path)?;

    Ok(dest_path)
}

/// Download an image from a URL to the current directory.
/// Skips download if the file already exists.
///
/// # Arguments
/// * `url` - The URL to download from
///
/// # Returns
/// The path to the downloaded (or existing) file, or an error if download fails
pub fn download_image(url: &str) -> Result<String> {
    let filename = url.rsplit('/').next().unwrap_or("image.jpg");
    let dest_path = Path::new(filename);

    // Skip download if file already exists
    if dest_path.exists() {
        eprintln!("Image already exists: {filename}");
        return Ok(filename.to_string());
    }

    eprintln!("Downloading {url}...");

    download_file(url, dest_path)?;

    Ok(filename.to_string())
}

/// Download multiple images from URLs to the current directory.
/// Returns a vector of paths to the successfully downloaded files.
/// Skips files that already exist.
///
/// # Arguments
/// * `urls` - Slice of URLs to download
pub fn download_images(urls: &[&str]) -> Vec<String> {
    urls.iter()
        .filter_map(|url| download_image(url).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unknown_model_returns_error() {
        let result = try_download_model("unknown_model.onnx");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Auto-download is only supported"));
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500.0), "500B");
        assert_eq!(format_bytes(1024.0), "1.0KB");
        assert_eq!(format_bytes(1048576.0), "1.0MB");
        assert_eq!(format_bytes(1073741824.0), "1.0GB");
    }

    #[test]
    fn test_format_time() {
        assert_eq!(format_time(5.5), "5.5s");
        assert_eq!(format_time(65.0), "1:05.0");
    }

    #[test]
    fn test_generate_bar() {
        assert_eq!(generate_bar(0.0, 10), "──────────");
        assert_eq!(generate_bar(1.0, 10), "━━━━━━━━━━");
        assert_eq!(generate_bar(0.5, 10), "━━━━━─────");
    }
}
