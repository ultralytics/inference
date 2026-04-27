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

const ASSETS_BASE_URL: &str = "https://github.com/ultralytics/assets/releases/download/v8.4.0";

/// Default YOLO detection model name.
pub const DEFAULT_MODEL: &str = "yolo26n.onnx";

/// Default YOLO segmentation model name.
pub const DEFAULT_SEGMENT_MODEL: &str = "yolo26n-seg.onnx";

/// Default YOLO pose model name.
pub const DEFAULT_POSE_MODEL: &str = "yolo26n-pose.onnx";

/// Default YOLO OBB model name.
pub const DEFAULT_OBB_MODEL: &str = "yolo26n-obb.onnx";

/// Default YOLO classification model name.
pub const DEFAULT_CLS_MODEL: &str = "yolo26n-cls.onnx";

const DOWNLOADABLE_MODELS: &[&str] = &[
    // YOLO26
    "yolo26n.onnx",
    "yolo26n-seg.onnx",
    "yolo26n-pose.onnx",
    "yolo26n-obb.onnx",
    "yolo26n-cls.onnx",
    // YOLO11
    "yolo11n.onnx",
    "yolo11n-seg.onnx",
    "yolo11n-pose.onnx",
    "yolo11n-obb.onnx",
    "yolo11n-cls.onnx",
];

const DEFAULT_BUS_IMAGE_URL: &str = "https://ultralytics.com/images/bus.jpg";
const DEFAULT_ZIDANE_IMAGE_URL: &str = "https://ultralytics.com/images/zidane.jpg";
const DEFAULT_BOATS_IMAGE_URL: &str = "https://ultralytics.com/images/boats.jpg";

/// Default image URLs for detection, segmentation, pose, and classification tasks.
pub const DEFAULT_IMAGES: &[&str] = &[DEFAULT_BUS_IMAGE_URL, DEFAULT_ZIDANE_IMAGE_URL];

/// Default image URL for OBB (Oriented Bounding Box) tasks.
pub const DEFAULT_OBB_IMAGE: &str = DEFAULT_BOATS_IMAGE_URL;

const CONNECT_TIMEOUT: u64 = 30;
const READ_TIMEOUT: u64 = 300;
const MAX_RETRIES: u32 = 3;
const RETRY_BASE_DELAY_SECS: u64 = 2;
const BAR_WIDTH: usize = 12;
const MIN_UPDATE_INTERVAL: f64 = 0.1;

fn format_bytes(bytes: f64) -> String {
    for (unit, factor) in [
        ("GB", 1_073_741_824.0_f64),
        ("MB", 1_048_576.0),
        ("KB", 1024.0),
    ] {
        if bytes >= factor {
            return format!("{:.1}{unit}", bytes / factor);
        }
    }
    format!("{bytes:.0}B")
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn format_time(seconds: f64) -> String {
    if seconds < 60.0 {
        format!("{seconds:.1}s")
    } else if seconds < 3600.0 {
        format!("{}:{:04.1}", (seconds / 60.0) as u32, seconds % 60.0)
    } else {
        format!(
            "{}:{:02}:{:04.1}",
            (seconds / 3600.0) as u32,
            ((seconds % 3600.0) / 60.0) as u32,
            seconds % 60.0
        )
    }
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn generate_bar(progress: f64, width: usize) -> String {
    let filled = ((progress * width as f64) as usize).min(width);
    format!("{}{}", "━".repeat(filled), "─".repeat(width - filled))
}

const fn is_transient(e: &ureq::Error) -> bool {
    match e {
        ureq::Error::Timeout(_) | ureq::Error::Io(_) => true,
        ureq::Error::StatusCode(c) => *c >= 500,
        _ => false,
    }
}

/// Download a file from URL to the specified path with progress bar and retry.
///
/// Retries up to `MAX_RETRIES` times on transient failures with exponential backoff.
/// Permanent errors (4xx, filesystem) are returned immediately without retrying.
/// Uses a temp file and atomic rename to prevent corrupted partial downloads.
#[allow(
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::large_stack_arrays,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn download_file(url: &str, dest: &Path) -> Result<()> {
    let mut last_err = InferenceError::ModelLoadError(String::new());

    for attempt in 1..=MAX_RETRIES {
        let attempt_result: std::result::Result<(), (InferenceError, bool)> = (|| {
            let config = ureq::Agent::config_builder()
                .timeout_connect(Some(Duration::from_secs(CONNECT_TIMEOUT)))
                .timeout_recv_body(Some(Duration::from_secs(READ_TIMEOUT)))
                .build();
            let agent = ureq::Agent::new_with_config(config);

            let response = agent.get(url).call().map_err(|e| {
                let msg = match &e {
                    ureq::Error::Timeout(_) => {
                        format!("Connection timed out while downloading {url}")
                    }
                    ureq::Error::Io(io_err) => format!("Network error downloading {url}: {io_err}"),
                    _ => format!("Failed to download {url}: {e}"),
                };
                (InferenceError::ModelLoadError(msg), is_transient(&e))
            })?;

            let total_size: u64 = response
                .headers()
                .get("content-length")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);

            let temp_path = dest.with_file_name(format!(
                "{}.part.{}.{}",
                dest.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("download"),
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .subsec_nanos()
            ));
            let _ = fs::remove_file(&temp_path);

            let mut downloaded: u64 = 0;
            let start_time = Instant::now();
            let desc: String = format!("Downloading {url} to '{}'", dest.display());
            let stream_result: std::result::Result<(), (InferenceError, bool)> = {
                let mut writer = BufWriter::new(File::create(&temp_path).map_err(|e| {
                    (
                        InferenceError::ModelLoadError(format!(
                            "Failed to create temp file {}: {e}",
                            temp_path.display()
                        )),
                        false,
                    )
                })?);
                let mut reader = response.into_body().into_reader();
                let mut buffer = [0u8; 65536];
                let mut last_update = Instant::now();

                (|| {
                    loop {
                        let bytes_read = reader.read(&mut buffer).map_err(|e| {
                            (
                                InferenceError::ModelLoadError(format!(
                                    "Failed to read from network: {e}"
                                )),
                                true,
                            )
                        })?;
                        if bytes_read == 0 {
                            break;
                        }
                        writer.write_all(&buffer[..bytes_read]).map_err(|e| {
                            (
                                InferenceError::ModelLoadError(format!(
                                    "Failed to write to temp file: {e}"
                                )),
                                false,
                            )
                        })?;
                        downloaded += bytes_read as u64;

                        let now = Instant::now();
                        if now.duration_since(last_update).as_secs_f64() < MIN_UPDATE_INTERVAL {
                            continue;
                        }
                        last_update = now;

                        let elapsed = start_time.elapsed().as_secs_f64();
                        let rate = if elapsed > 0.0 {
                            downloaded as f64 / elapsed
                        } else {
                            0.0
                        };
                        if total_size > 0 {
                            let progress = (downloaded as f64 / total_size as f64).min(1.0);
                            let bar = generate_bar(progress, BAR_WIDTH);
                            eprint!(
                                "\r\x1b[K{desc}: {}% {bar} {}/{} {}/s {}",
                                (progress * 100.0) as u8,
                                format_bytes(downloaded as f64),
                                format_bytes(total_size as f64),
                                format_bytes(rate),
                                format_time(elapsed)
                            );
                        } else {
                            eprint!(
                                "\r\x1b[K{desc}: {} {}/s {}",
                                format_bytes(downloaded as f64),
                                format_bytes(rate),
                                format_time(elapsed)
                            );
                        }
                        std::io::stderr().flush().ok();
                    }
                    writer.flush().map_err(|e| {
                        (
                            InferenceError::ModelLoadError(format!(
                                "Failed to flush temp file: {e}"
                            )),
                            false,
                        )
                    })?;
                    Ok(())
                })()
                // writer, reader, buffer, last_update dropped here
            };

            if stream_result.is_err() {
                let _ = fs::remove_file(&temp_path);
            }
            stream_result?;

            let elapsed = start_time.elapsed().as_secs_f64();
            let rate = if elapsed > 0.0 {
                downloaded as f64 / elapsed
            } else {
                0.0
            };
            if total_size > 0 {
                eprintln!(
                    "\r\x1b[K{desc}: 100% {} {} {}/s {}",
                    generate_bar(1.0, BAR_WIDTH),
                    format_bytes(total_size as f64),
                    format_bytes(rate),
                    format_time(elapsed)
                );
            } else {
                eprintln!(
                    "\r\x1b[K{desc}: {} {}/s {}",
                    format_bytes(downloaded as f64),
                    format_bytes(rate),
                    format_time(elapsed)
                );
            }

            fs::rename(&temp_path, dest).map_err(|e| {
                let _ = fs::remove_file(&temp_path);
                (
                    InferenceError::ModelLoadError(format!(
                        "Failed to move downloaded file to {}: {e}",
                        dest.display()
                    )),
                    false,
                )
            })?;

            Ok(())
        })();

        match attempt_result {
            Ok(()) => return Ok(()),
            Err((e, false)) => return Err(e),
            Err((e, true)) => {
                last_err = e;
                if attempt < MAX_RETRIES {
                    let delay = RETRY_BASE_DELAY_SECS * (1 << (attempt - 1));
                    eprintln!(
                        "Download attempt {attempt}/{MAX_RETRIES} failed: {last_err}. Retrying in {delay}s..."
                    );
                    std::thread::sleep(Duration::from_secs(delay));
                }
            }
        }
    }

    Err(last_err)
}

/// Attempt to download a model if it matches a known downloadable model.
///
/// Supports the YOLO26 and YOLO11 nano ONNX models listed in [`DOWNLOADABLE_MODELS`].
/// Every supported file resolves to `{ASSETS_BASE_URL}/{filename}`.
///
/// Returns the path to the downloaded model.
#[allow(clippy::missing_errors_doc)]
pub fn try_download_model<P: AsRef<Path>>(path: P) -> Result<PathBuf> {
    let path = path.as_ref();
    let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

    if !DOWNLOADABLE_MODELS.contains(&filename) {
        return Err(InferenceError::ModelLoadError(format!(
            "Model file not found: {}. Auto-download is supported for: {}",
            path.display(),
            DOWNLOADABLE_MODELS.join(", "),
        )));
    }

    download_file(&format!("{ASSETS_BASE_URL}/{filename}"), path)?;
    Ok(path.to_path_buf())
}

/// Download an image from a URL to the current directory.
/// Skips download if the file already exists.
///
/// # Errors
/// Returns an error if the download fails or file I/O errors occur.
pub fn download_image(url: &str) -> Result<String> {
    let filename = url.rsplit('/').next().unwrap_or("image.jpg");
    let dest = Path::new(filename);

    if !dest.exists() {
        download_file(url, dest)?;
    }

    Ok(dest
        .canonicalize()
        .or_else(|_| std::env::current_dir().map(|p| p.join(filename)))
        .map_or_else(
            |_| filename.to_string(),
            |p| p.to_string_lossy().into_owned(),
        ))
}

/// Download multiple images from URLs to the current directory.
/// Skips files that already exist.
#[must_use]
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
        assert!(err.contains("Auto-download is supported for"));
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500.0), "500B");
        assert_eq!(format_bytes(1024.0), "1.0KB");
        assert_eq!(format_bytes(1_048_576.0), "1.0MB");
        assert_eq!(format_bytes(1_073_741_824.0), "1.0GB");
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
