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

/// YOLO Model families, sizes, and variants supported for auto-download.
const MODEL_FAMILIES: &[&str] = &["yolo26", "yolo11", "yolov8"];
const MODEL_SIZES: &[&str] = &["n", "s", "m", "l", "x"];
const MODEL_VARIANTS: &[&str] = &["", "-seg", "-pose", "-obb", "-cls"];
/// Variants only available for the yolo26 family.
const YOLO26_ONLY_VARIANTS: &[&str] = &["-sem", "-depth"];

/// Every auto-downloadable model filename, as the cross product of family, size, and variant.
///
/// `-sem` and `-depth` are only emitted for the `yolo26` family.
fn downloadable_models() -> Vec<String> {
    MODEL_FAMILIES
        .iter()
        .flat_map(|family| {
            let extra: &[&str] = if *family == "yolo26" {
                YOLO26_ONLY_VARIANTS
            } else {
                &[]
            };
            MODEL_SIZES.iter().flat_map(move |size| {
                MODEL_VARIANTS
                    .iter()
                    .chain(extra.iter())
                    .map(move |variant| format!("{family}{size}{variant}.onnx"))
            })
        })
        .collect()
}

/// Human-readable summary of the supported families, sizes, and variants for error messages.
fn supported_models_help() -> String {
    let variants_display = [
        "detect",
        "-seg",
        "-pose",
        "-obb",
        "-cls",
        "-sem (yolo26 only)",
        "-depth (yolo26 only)",
    ];
    let sizes_display = MODEL_SIZES.join(", ");
    let variants_joined = variants_display.join(", ");

    let family_lines: Vec<String> = MODEL_FAMILIES
        .iter()
        .map(|f| format!("    {f:<8} sizes: [{sizes_display}]  variants: [{variants_joined}]"))
        .collect();

    format!(
        "Auto-download is supported for:\n{}\n\n  Usage:\n    ultralytics-inference predict --model yolo26n\n    ultralytics-inference predict --model yolo26n.onnx",
        family_lines.join("\n")
    )
}

const DEFAULT_BUS_IMAGE_URL: &str = "https://ultralytics.com/images/bus.jpg";
const DEFAULT_ZIDANE_IMAGE_URL: &str = "https://ultralytics.com/images/zidane.jpg";
const DEFAULT_BOATS_IMAGE_URL: &str = "https://ultralytics.com/images/boats.jpg";

/// Default image URLs for detection, segmentation, pose, classification, semantic, and depth tasks.
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

/// Render one progress line, e.g. `Downloading …: 42% ━━━──── 4.2MB/10MB 1.1MB/s 3.8s`.
///
/// The percentage and bar are dropped when the server sent no `content-length`
/// (`total_size == 0`), so the same line serves the in-flight updates and the final one.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn progress_line(downloaded: u64, total_size: u64, elapsed: f64) -> String {
    let rate = if elapsed > 0.0 {
        downloaded as f64 / elapsed
    } else {
        0.0
    };
    let done = format_bytes(downloaded as f64);
    let speed = format_bytes(rate);
    let time = format_time(elapsed);
    if total_size == 0 {
        return format!("  {done} {speed}/s {time}");
    }
    let progress = (downloaded as f64 / total_size as f64).min(1.0);
    format!(
        "  {:>3}% {} {done}/{} {speed}/s {time}",
        (progress * 100.0) as u8,
        generate_bar(progress, BAR_WIDTH),
        format_bytes(total_size as f64),
    )
}

/// Whether a download error is worth retrying: timeouts, I/O errors, and 5xx responses.
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
#[cfg_attr(coverage_nightly, coverage(off))]
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
            eprintln!("Downloading {url} to '{}'", dest.display());
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
                        eprint!("\r\x1b[K{}", progress_line(downloaded, total_size, elapsed));
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
            eprintln!("\r\x1b[K{}", progress_line(downloaded, total_size, elapsed));

            if let Err(e) = fs::rename(&temp_path, dest) {
                let _ = fs::remove_file(&temp_path);
                if dest.exists() {
                    return Ok(());
                }
                return Err((
                    InferenceError::ModelLoadError(format!(
                        "Failed to move downloaded file to {}: {e}",
                        dest.display()
                    )),
                    false,
                ));
            }

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

/// Append `.onnx` to extensionless paths and normalize case-variant extensions (e.g. `.ONNX`).
/// Paths with any other extension (e.g. `.pt`) are returned unchanged.
fn normalize_model_path(path: &Path) -> PathBuf {
    match path.extension().and_then(|e| e.to_str()) {
        None => {
            let stem = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            path.with_file_name(format!("{stem}.onnx"))
        }
        Some(e) if e.eq_ignore_ascii_case("onnx") && e != "onnx" => {
            let stem = path.file_stem().and_then(|n| n.to_str()).unwrap_or("");
            path.with_file_name(format!("{stem}.onnx"))
        }
        _ => path.to_path_buf(),
    }
}

/// Attempt to download a model if it matches a known downloadable model.
///
/// Supports all `YOLO26`, `YOLO11`, and `YOLOv8` ONNX models across sizes (n/s/m/l/x) and
/// task variants (detect, segment, pose, obb, classify, semantic, depth).
/// Every supported file resolves to `{ASSETS_BASE_URL}/{filename}`.
///
/// # Errors
///
/// Returns an error if the model name is not in the supported list, or if the download fails.
pub fn try_download_model<P: AsRef<Path>>(path: P) -> Result<PathBuf> {
    let path = path.as_ref();
    let normalized = normalize_model_path(path);

    let filename = normalized
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");

    let models = downloadable_models();
    if !models.iter().any(|m| m == filename) {
        return Err(InferenceError::ModelLoadError(format!(
            "Model file not found: {}\n\n{}",
            path.display(),
            supported_models_help(),
        )));
    }

    download_file(&format!("{ASSETS_BASE_URL}/{filename}"), &normalized)?;
    Ok(normalized)
}

/// Download an image from a URL to the current directory.
/// Skips download if the file already exists.
///
/// # Errors
/// Returns an error if the download fails or file I/O errors occur.
#[cfg_attr(coverage_nightly, coverage(off))]
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
#[cfg_attr(coverage_nightly, coverage(off))]
pub fn download_images(urls: &[&str]) -> Vec<String> {
    urls.iter()
        .filter_map(|url| download_image(url).ok())
        .collect()
}

/// ONNX Runtime version the `OpenVINO` provider plugin is built for; must match the one `ort` links.
#[cfg(feature = "openvino")]
const ORT_OPENVINO_VERSION: &str = "1.28.0";

/// Release holding the `OpenVINO` provider plugin bundles, one `.tgz` per platform.
#[cfg(feature = "openvino")]
const ORT_OPENVINO_URL: &str = "https://github.com/ultralytics/inference/releases/download/v0.0.11";

#[cfg(feature = "openvino")]
static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Download once, check against its pinned SHA-256 and unpack this platform's `OpenVINO`
/// provider plugin bundle into the user cache, and return the plugin's path.
///
/// # Errors
///
/// Returns an error if the bundle cannot be downloaded, verified, unpacked, or installed.
#[cfg(feature = "openvino")]
#[allow(clippy::too_many_lines)]
#[cfg_attr(coverage_nightly, coverage(off))]
pub(crate) fn openvino_plugin() -> Result<PathBuf> {
    let (target, plugin, bridge, sha256) = match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => (
            "linux-x64",
            "libonnxruntime_providers_openvino.so",
            "libonnxruntime_providers_shared.so",
            "dafeb17cbab258b8c9087d417763a1aafea6d789015b91a85564b5a97713d82d",
        ),
        ("windows", "x86_64") => (
            "windows-x64",
            "onnxruntime_providers_openvino.dll",
            "onnxruntime_providers_shared.dll",
            "f57f68b921eb2925d95ee78c524290f5e72b3ef82687b5f44ab953f99c350bde",
        ),
        (os, arch) => {
            return Err(InferenceError::ModelLoadError(format!(
                "No OpenVINO provider bundle is published for {os}-{arch}"
            )));
        }
    };

    let name = format!("openvino-ep-{ORT_OPENVINO_VERSION}-{target}");
    let cache = dirs::cache_dir()
        .ok_or_else(|| InferenceError::ModelLoadError("No user cache directory".into()))?
        .join("ultralytics-inference");
    let dir = cache.join(&name);
    let plugin_path = dir.join(plugin);
    // Unique per call, so concurrent first loads never share temporary names.
    let nonce = format!(
        "{}.{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );

    if !plugin_path.exists() {
        // Stage in a folder and rename it into place, so an interrupted run never leaves a
        // bundle that looks complete.
        let part = cache.join(format!("{name}.part.{nonce}"));
        let staged = (|| -> Result<()> {
            fs::create_dir_all(&part).map_err(|e| {
                InferenceError::ModelLoadError(format!("Failed to create {}: {e}", part.display()))
            })?;
            let archive = part.join(format!("{name}.tgz"));
            download_file(&format!("{ORT_OPENVINO_URL}/{name}.tgz"), &archive)?;

            let bytes = fs::read(&archive).map_err(|e| {
                InferenceError::ModelLoadError(format!("Failed to read {name}.tgz: {e}"))
            })?;
            let digest = hmac_sha256::Hash::hash(&bytes).iter().fold(
                String::with_capacity(64),
                |mut hex, b| {
                    use std::fmt::Write as _;
                    let _ = write!(hex, "{b:02x}");
                    hex
                },
            );
            if digest != sha256 {
                return Err(InferenceError::ModelLoadError(format!(
                    "{name}.tgz does not match its pinned SHA-256 (got {digest}), so it was not loaded"
                )));
            }

            // Unpack in process from the verified bytes, so no system `tar` is needed.
            let _ = fs::remove_file(&archive);
            unpack_tgz(&bytes, &part).map_err(|e| {
                InferenceError::ModelLoadError(format!("Failed to unpack {name}.tgz: {e}"))
            })?;
            if !part.join(plugin).exists() {
                return Err(InferenceError::ModelLoadError(format!(
                    "{name}.tgz does not contain {plugin}"
                )));
            }
            Ok(())
        })();
        if let Err(e) = staged {
            let _ = fs::remove_dir_all(&part);
            return Err(e);
        }

        // A rename onto a finished bundle fails, so a caller that lost the race drops its own
        // copy and uses the winner's bundle.
        if let Err(e) = fs::rename(&part, &dir) {
            let _ = fs::remove_dir_all(&part);
            if !plugin_path.exists() {
                return Err(InferenceError::ModelLoadError(format!(
                    "Failed to move bundle to {}: {e}",
                    dir.display()
                )));
            }
        }
    }

    // ONNX Runtime loads the provider bridge from the executable's folder, not the plugin's.
    // Copy under a unique name and rename, so a concurrent load never sees a partial file.
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf))
        .ok_or_else(|| {
            InferenceError::ModelLoadError("Cannot find the executable's folder".into())
        })?;
    let bridge_path = exe_dir.join(bridge);
    if !bridge_path.exists() {
        let tmp = exe_dir.join(format!("{bridge}.{nonce}.part"));
        if let Err(e) =
            fs::copy(dir.join(bridge), &tmp).and_then(|_| fs::rename(&tmp, &bridge_path))
        {
            let _ = fs::remove_file(&tmp);
            if !bridge_path.exists() {
                return Err(InferenceError::ModelLoadError(format!(
                    "OpenVINO needs {bridge} next to the executable, but copying it into {} failed: {e}. \
                     Copy {} there yourself.",
                    exe_dir.display(),
                    dir.join(bridge).display()
                )));
            }
        }
    }

    // Windows does not search the plugin's folder for its DLLs, so load OpenVINO and the TBB it
    // links from the bundle first; the plugin then reuses those loaded modules.
    #[cfg(windows)]
    for dll in ["tbb12.dll", "openvino.dll"] {
        ort::util::preload_dylib(dir.join(dll)).map_err(|e| {
            InferenceError::ModelLoadError(format!(
                "Failed to load {dll} from {}: {e}",
                dir.display()
            ))
        })?;
    }
    Ok(plugin_path)
}

/// Unpack a gzipped tar into `dest`, streaming entries to disk. Only directories, regular files
/// and symlinks are accepted, and nothing may point outside `dest`.
#[cfg(feature = "openvino")]
#[cfg_attr(coverage_nightly, coverage(off))]
fn unpack_tgz(archive: &[u8], dest: &Path) -> std::io::Result<()> {
    use std::io::{Error, ErrorKind};
    use std::path::Component;

    // Header fields are NUL-terminated or NUL-padded.
    fn field(bytes: &[u8]) -> std::io::Result<&str> {
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        std::str::from_utf8(&bytes[..end]).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "entry name is not UTF-8")
        })
    }

    let invalid = |msg: &str| Error::new(ErrorKind::InvalidData, msg.to_owned());
    let inside = |path: &Path| {
        path.components()
            .all(|c| matches!(c, Component::Normal(_) | Component::CurDir))
    };

    let mut gz = flate2::read::GzDecoder::new(archive);
    let mut header = [0u8; 512];
    loop {
        gz.read_exact(&mut header)?;
        if header.iter().all(|&b| b == 0) {
            return Ok(());
        }
        let rel = Path::new(field(&header[..100])?);
        if !inside(rel) {
            return Err(invalid("entry path leaves the bundle folder"));
        }
        let size = u64::from_str_radix(field(&header[124..136])?.trim(), 8)
            .map_err(|_| invalid("entry size is not octal"))?;
        let path = dest.join(rel);

        // Entry data is padded to the next 512-byte block.
        let mut entry = (&mut gz).take(size.div_ceil(512) * 512);
        match header[156] {
            b'0' | 0 => {
                std::io::copy(&mut (&mut entry).take(size), &mut File::create(&path)?)?;
            }
            b'5' => fs::create_dir_all(&path)?,
            #[cfg(unix)]
            b'2' => {
                let target = field(&header[157..257])?;
                if Path::new(target).components().count() != 1 || !inside(Path::new(target)) {
                    return Err(invalid("symlink target leaves the bundle folder"));
                }
                std::os::unix::fs::symlink(target, &path)?;
            }
            _ => return Err(invalid("unsupported entry type")),
        }
        std::io::copy(&mut entry, &mut std::io::sink())?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unknown_model_returns_error() {
        let result = try_download_model("unknown_model.onnx");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Model file not found"));
    }

    #[test]
    fn test_normalize_model_path() {
        // no extension -> append .onnx
        assert_eq!(
            normalize_model_path(Path::new("yolo26n")),
            PathBuf::from("yolo26n.onnx")
        );
        // already .onnx -> unchanged
        assert_eq!(
            normalize_model_path(Path::new("yolo26n.onnx")),
            PathBuf::from("yolo26n.onnx")
        );
        // case-variant .ONNX -> normalized to .onnx
        assert_eq!(
            normalize_model_path(Path::new("yolo26n.ONNX")),
            PathBuf::from("yolo26n.onnx")
        );
        // unrelated extension -> preserved as-is (will fail name check later)
        assert_eq!(
            normalize_model_path(Path::new("yolo26n.pt")),
            PathBuf::from("yolo26n.pt")
        );
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
        // >= 3600s switches to the H:MM:SS.s layout.
        assert!(format_time(3661.5).starts_with("1:01:"));
    }

    #[test]
    fn test_generate_bar() {
        assert_eq!(generate_bar(0.0, 10), "──────────");
        assert_eq!(generate_bar(1.0, 10), "━━━━━━━━━━");
        assert_eq!(generate_bar(0.5, 10), "━━━━━─────");
        // progress > 1.0 never exceeds the bar width.
        assert_eq!(generate_bar(2.0, 6), "━━━━━━");
    }

    #[test]
    fn test_progress_line() {
        // With content-length: percentage, bar, transferred/total, and rate.
        let line = progress_line(5 * 1_048_576, 10 * 1_048_576, 2.0);
        assert!(line.contains(" 50% "), "{line}");
        assert!(line.contains("5.0MB/10.0MB"), "{line}");
        assert!(line.contains("2.5MB/s"), "{line}");
        assert!(line.contains('━') && line.contains('─'), "{line}");

        // A server that sent no content-length drops the percentage and the bar.
        let line = progress_line(1024, 0, 1.0);
        assert!(!line.contains('%'), "{line}");
        assert!(!line.contains('━') && !line.contains('─'), "{line}");
        assert_eq!(line.trim(), "1.0KB 1.0KB/s 1.0s");

        // Zero elapsed must not divide by zero, and an overshooting server clamps at 100%.
        assert!(progress_line(1024, 0, 0.0).contains("0B/s"), "zero elapsed");
        assert!(progress_line(20, 10, 1.0).contains("100%"), "overshoot");
    }

    #[test]
    fn test_downloadable_models_membership() {
        let models = downloadable_models();
        assert!(models.iter().any(|m| m == "yolo26n.onnx"));
        assert!(models.iter().any(|m| m == "yolo11n-seg.onnx"));
        assert!(models.iter().any(|m| m == "yolov8n.onnx"));
        // `-sem` and `-depth` are yolo26-only variants.
        assert!(models.iter().any(|m| m == "yolo26n-sem.onnx"));
        assert!(models.iter().any(|m| m == "yolo26n-depth.onnx"));
    }

    #[test]
    fn test_supported_models_help_text() {
        let help = supported_models_help();
        assert!(help.contains("Auto-download is supported for:"));
        assert!(help.contains("yolo26"));
    }
}
