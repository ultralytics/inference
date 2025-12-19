// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Input source handling for YOLO inference.
//!
//! This module provides abstractions for various input sources including
//! images, videos, webcams, and streaming URLs.

use std::path::{Path, PathBuf};

use image::DynamicImage;
use ndarray::Array3;

use crate::error::{InferenceError, Result};

/// Represents different input sources for inference.
#[derive(Debug, Clone)]
pub enum Source {
    /// Path to an image file.
    Image(PathBuf),
    /// In-memory image.
    ImageBuffer(DynamicImage),
    /// Raw HWC u8 array.
    Array(Array3<u8>),
    /// HTTP/HTTPS URL to an image file.
    ImageUrl(String),
    /// List of image paths.
    ImageList(Vec<PathBuf>),
    /// Path to a video file.
    Video(PathBuf),
    /// Webcam device index.
    Webcam(u32),
    /// Streaming URL (RTSP, RTMP, HTTP).
    Stream(String),
    /// Directory containing images.
    Directory(PathBuf),
    /// Glob pattern for images.
    Glob(String),
}

impl Source {
    /// Check if this source is a single image.
    #[must_use]
    pub const fn is_image(&self) -> bool {
        matches!(
            self,
            Self::Image(_) | Self::ImageBuffer(_) | Self::Array(_) | Self::ImageUrl(_)
        )
    }

    /// Check if this source is a video or stream.
    #[must_use]
    pub const fn is_video(&self) -> bool {
        matches!(self, Self::Video(_) | Self::Webcam(_) | Self::Stream(_))
    }

    /// Check if this source is a directory or glob pattern.
    #[must_use]
    pub const fn is_batch(&self) -> bool {
        matches!(
            self,
            Self::Directory(_) | Self::Glob(_) | Self::ImageList(_)
        )
    }

    /// Get the path if this source has one.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        match self {
            Self::Image(p) | Self::Video(p) | Self::Directory(p) => Some(p),
            _ => None,
        }
    }

    /// Check if a URL points to an image based on extension.
    fn is_image_url(url: &str) -> bool {
        let url_lower = url.to_lowercase();
        // Remove query parameters if present
        let path_part = url_lower.split('?').next().unwrap_or(&url_lower);

        std::path::Path::new(path_part)
            .extension()
            .is_some_and(|ext| {
                let s = ext.to_string_lossy();
                s.eq_ignore_ascii_case("jpg")
                    || s.eq_ignore_ascii_case("jpeg")
                    || s.eq_ignore_ascii_case("png")
                    || s.eq_ignore_ascii_case("bmp")
                    || s.eq_ignore_ascii_case("gif")
                    || s.eq_ignore_ascii_case("webp")
                    || s.eq_ignore_ascii_case("tiff")
                    || s.eq_ignore_ascii_case("tif")
            })
    }
}

/// Convert from a string path to Source.
impl From<&str> for Source {
    fn from(s: &str) -> Self {
        // Check for webcam index
        if let Ok(idx) = s.parse::<u32>() {
            return Self::Webcam(idx);
        }

        // Check for HTTP/HTTPS URLs
        if s.starts_with("http://") || s.starts_with("https://") {
            // Check if it's an image URL by extension
            if Self::is_image_url(s) {
                return Self::ImageUrl(s.to_string());
            }
            // Otherwise treat as video stream
            return Self::Stream(s.to_string());
        }

        // Check for streaming URLs
        if s.starts_with("rtsp://") || s.starts_with("rtmp://") {
            return Self::Stream(s.to_string());
        }

        // Check for glob pattern
        if s.contains('*') {
            return Self::Glob(s.to_string());
        }

        let path = PathBuf::from(s);

        // Check if it's a directory
        if path.is_dir() {
            return Self::Directory(path);
        }

        // Check file extension for video
        if let Some(ext) = path.extension() {
            let ext = ext.to_string_lossy().to_lowercase();
            if matches!(
                ext.as_str(),
                "mp4" | "avi" | "mov" | "mkv" | "wmv" | "flv" | "webm" | "m4v" | "mpeg" | "mpg"
            ) {
                return Self::Video(path);
            }
        }

        // Default to image
        Self::Image(path)
    }
}

impl From<String> for Source {
    fn from(s: String) -> Self {
        Self::from(s.as_str())
    }
}

impl From<PathBuf> for Source {
    fn from(path: PathBuf) -> Self {
        Self::from(path.to_string_lossy().as_ref())
    }
}

impl From<&Path> for Source {
    fn from(path: &Path) -> Self {
        Self::from(path.to_string_lossy().as_ref())
    }
}

impl From<DynamicImage> for Source {
    fn from(img: DynamicImage) -> Self {
        Self::ImageBuffer(img)
    }
}

impl From<Array3<u8>> for Source {
    fn from(arr: Array3<u8>) -> Self {
        Self::Array(arr)
    }
}

impl From<u32> for Source {
    fn from(idx: u32) -> Self {
        Self::Webcam(idx)
    }
}

impl From<i32> for Source {
    fn from(idx: i32) -> Self {
        #[allow(clippy::cast_sign_loss)]
        Self::Webcam(idx as u32)
    }
}

/// Metadata about a source frame.
#[derive(Debug, Clone)]
pub struct SourceMeta {
    /// Frame index (0 for single images).
    pub frame_idx: usize,
    /// Total frames (1 for single images, may be unknown for streams).
    pub total_frames: Option<usize>,
    /// Source path or identifier.
    pub path: String,
    /// Frames per second (for video sources).
    pub fps: Option<f32>,
}

impl Default for SourceMeta {
    fn default() -> Self {
        Self {
            frame_idx: 0,
            total_frames: Some(1),
            path: String::new(),
            fps: None,
        }
    }
}

#[cfg(feature = "video")]
use ffmpeg_next as ffmpeg;

/// Iterator over frames from a source.
pub struct SourceIterator {
    source: Source,
    current_frame: usize,
    image_paths: Vec<PathBuf>,
    #[cfg(feature = "video")]
    decoder: Option<video_rs::decode::Decoder>,
    #[cfg(feature = "video")]
    webcam_decoder: Option<(ffmpeg::format::context::Input, ffmpeg::decoder::Video)>,
    #[cfg(feature = "video")]
    webcam_stream_index: usize,
    #[cfg(feature = "video")]
    total_frames: Option<usize>,
    #[cfg(feature = "video")]
    webcam_init_failed: bool,
}

impl SourceIterator {
    /// Create a new source iterator.
    ///
    /// # Errors
    ///
    /// Returns an error if the source cannot be opened.
    pub fn new(source: Source) -> Result<Self> {
        let image_paths = match &source {
            Source::Directory(path) => Self::collect_images_from_dir(path)?,
            Source::Glob(pattern) => Self::collect_images_from_glob(pattern)?,
            Source::Image(path) => vec![path.clone()],
            // URLs are handled separately via next_image_url
            Source::ImageList(paths) => paths.clone(),
            _ => vec![],
        };

        Ok(Self {
            source,
            current_frame: 0,
            image_paths,
            #[cfg(feature = "video")]
            decoder: None,
            #[cfg(feature = "video")]
            webcam_decoder: None,
            #[cfg(feature = "video")]
            webcam_stream_index: 0,
            #[cfg(feature = "video")]
            total_frames: None,
            #[cfg(feature = "video")]
            webcam_init_failed: false,
        })
    }

    /// Collect image paths from a directory.
    fn collect_images_from_dir(dir: &Path) -> Result<Vec<PathBuf>> {
        if !dir.is_dir() {
            return Err(InferenceError::ImageError(format!(
                "Not a directory: {}",
                dir.display()
            )));
        }

        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(InferenceError::IoError)?
            .filter_map(std::result::Result::ok)
            .map(|entry| entry.path())
            .filter(|path| Self::is_image_file(path))
            .collect();

        paths.sort();
        Ok(paths)
    }

    /// Collect image paths from a glob pattern.
    ///
    /// Note: This is a simplified glob implementation that only supports patterns like "dir/*.jpg"
    /// For more complex glob patterns, consider adding the `glob` crate.
    fn collect_images_from_glob(pattern: &str) -> Result<Vec<PathBuf>> {
        // Simple glob: split into directory and extension pattern
        // Supports patterns like "images/*.jpg" or "path/to/dir/*.png"
        if let Some(star_pos) = pattern.find('*') {
            let dir_part = &pattern[..star_pos];
            let dir = if dir_part.is_empty() {
                Path::new(".")
            } else {
                Path::new(dir_part.trim_end_matches('/').trim_end_matches('\\'))
            };

            // Get extension filter from pattern (e.g., "*.jpg" -> "jpg")
            let ext_filter: Option<String> = pattern[star_pos..]
                .strip_prefix("*.")
                .map(str::to_lowercase);

            if !dir.is_dir() {
                return Err(InferenceError::ImageError(format!(
                    "Directory not found: {}",
                    dir.display()
                )));
            }

            let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
                .map_err(InferenceError::IoError)?
                .filter_map(std::result::Result::ok)
                .map(|entry| entry.path())
                .filter(|path| {
                    ext_filter.as_ref().map_or_else(
                        || Self::is_image_file(path),
                        |ext| {
                            path.extension()
                                .is_some_and(|e| e.to_string_lossy().to_lowercase() == *ext)
                        },
                    )
                })
                .collect();

            paths.sort();
            Ok(paths)
        } else {
            // No glob pattern, treat as single file
            Ok(vec![PathBuf::from(pattern)])
        }
    }

    /// Check if a path is an image file based on extension.
    fn is_image_file(path: &Path) -> bool {
        path.extension().is_some_and(|ext| {
            let ext = ext.to_string_lossy().to_lowercase();
            matches!(
                ext.as_str(),
                "jpg" | "jpeg" | "png" | "bmp" | "gif" | "webp" | "tiff" | "tif"
            )
        })
    }

    /// Download an image from a URL.
    fn download_image(url: &str) -> Result<DynamicImage> {
        let mut response = ureq::get(url)
            .call()
            .map_err(|e| InferenceError::ImageError(format!("Failed to download {url}: {e}")))?
            .into_body();

        let bytes = response.read_to_vec().map_err(|e| {
            InferenceError::ImageError(format!("Failed to read response from {url}: {e}"))
        })?;

        image::load_from_memory(&bytes).map_err(|e| {
            InferenceError::ImageError(format!("Failed to decode image from {url}: {e}"))
        })
    }

    /// Get the next image from a URL.
    fn next_image_url(&mut self, url: &str) -> Option<Result<(DynamicImage, SourceMeta)>> {
        if self.current_frame > 0 {
            return None;
        }

        self.current_frame = 1;
        let meta = SourceMeta {
            frame_idx: 0,
            total_frames: Some(1),
            path: url.to_string(),
            fps: None,
        };

        match Self::download_image(url) {
            Ok(img) => Some(Ok((img, meta))),
            Err(e) => Some(Err(e)),
        }
    }

    /// Get the next image from the source.
    fn next_image(&mut self) -> Option<Result<(DynamicImage, SourceMeta)>> {
        if self.current_frame >= self.image_paths.len() {
            return None;
        }

        let path = &self.image_paths[self.current_frame];
        let meta = SourceMeta {
            frame_idx: self.current_frame,
            total_frames: Some(self.image_paths.len()),
            path: path.to_string_lossy().to_string(),
            fps: None,
        };

        self.current_frame += 1;

        match image::open(path) {
            Ok(img) => Some(Ok((img, meta))),
            Err(e) => Some(Err(InferenceError::ImageError(format!(
                "Failed to load {}: {e}",
                path.display()
            )))),
        }
    }

    /// Get the next video frame.
    #[cfg(feature = "video")]
    #[allow(unsafe_code, clippy::too_many_lines)]
    fn next_video_frame(&mut self) -> Option<Result<(DynamicImage, SourceMeta)>> {
        // Handle Webcam separately using native ffmpeg
        if let Source::Webcam(idx) = &self.source {
            if self.webcam_init_failed {
                return None;
            }

            if self.webcam_decoder.is_none() {
                // Initialize webcam
                ffmpeg::init().ok();

                // Get format by name (returns Option<Format>)
                let input_format_name = if cfg!(target_os = "macos") {
                    "avfoundation"
                } else if cfg!(target_os = "linux") {
                    "video4linux2"
                } else if cfg!(target_os = "windows") {
                    "dshow"
                } else {
                    self.webcam_init_failed = true;
                    return Some(Err(InferenceError::VideoError(
                        "Unsupported OS for webcam".to_string(),
                    )));
                };

                // Find input format by name using low-level C API
                let c_name = std::ffi::CString::new(input_format_name).unwrap();
                #[allow(unsafe_code)]
                let ptr = unsafe { ffmpeg_sys_next::av_find_input_format(c_name.as_ptr()) };

                let input_format = if ptr.is_null() {
                    self.webcam_init_failed = true;
                    return Some(Err(InferenceError::VideoError(format!(
                        "Input format '{input_format_name}' not found"
                    ))));
                } else {
                    #[allow(unsafe_code, clippy::ptr_cast_constness)]
                    unsafe {
                        ffmpeg::format::Input::wrap(ptr.cast_mut())
                    }
                };

                // Determine device name based on OS and index
                let device_name = if cfg!(target_os = "macos") {
                    idx.to_string() // Just index for avfoundation
                } else if cfg!(target_os = "linux") {
                    format!("/dev/video{idx}")
                } else if cfg!(target_os = "windows") {
                    format!("video={idx}")
                } else {
                    self.webcam_init_failed = true;
                    return Some(Err(InferenceError::VideoError(
                        "Unsupported OS for webcam device name".to_string(),
                    )));
                };

                // Set explicit framerate to avoid default NTSC mismatch
                let mut options = ffmpeg::Dictionary::new();
                options.set("framerate", "30");

                match ffmpeg::format::open_with(
                    &PathBuf::from(&device_name),
                    &ffmpeg::Format::Input(input_format),
                    options,
                ) {
                    #[allow(clippy::single_match_else)]
                    Ok(ctx) => match ctx {
                        ffmpeg::format::context::Context::Input(ictx) => {
                            let input =
                                ictx.streams()
                                    .best(ffmpeg::media::Type::Video)
                                    .ok_or_else(|| {
                                        InferenceError::VideoError(
                                            "No video stream found in webcam".to_string(),
                                        )
                                    });

                            match input {
                                Ok(stream) => {
                                    let stream_index = stream.index();
                                    self.webcam_stream_index = stream_index;
                                    let context_decoder =
                                        ffmpeg::codec::context::Context::from_parameters(
                                            stream.parameters(),
                                        )
                                        .unwrap();
                                    match context_decoder.decoder().video() {
                                        Ok(decoder) => {
                                            self.webcam_decoder = Some((ictx, decoder));
                                        }
                                        Err(e) => {
                                            self.webcam_init_failed = true;
                                            return Some(Err(InferenceError::VideoError(format!(
                                                "Failed to create webcam decoder: {e}"
                                            ))));
                                        }
                                    }
                                }
                                Err(e) => {
                                    self.webcam_init_failed = true;
                                    return Some(Err(e));
                                }
                            }
                        }
                        ffmpeg::format::context::Context::Output(_) => {
                            self.webcam_init_failed = true;
                            return Some(Err(InferenceError::VideoError(
                                "Opened context is not an input context".to_string(),
                            )));
                        }
                    },
                    Err(e) => {
                        self.webcam_init_failed = true;
                        return Some(Err(InferenceError::VideoError(format!(
                            "Failed to open webcam: {e}"
                        ))));
                    }
                }
            }

            if let Some((ictx, decoder)) = &mut self.webcam_decoder {
                let mut decoded = ffmpeg::util::frame::video::Video::empty();

                // Read packets until we get a full frame
                for (stream, packet) in ictx.packets() {
                    if stream.index() == self.webcam_stream_index
                        && decoder.send_packet(&packet).is_ok()
                        && decoder.receive_frame(&mut decoded).is_ok()
                    {
                        // Convert to DynamicImage
                        // Handle pixel formatting manually or use a helper
                        // For simplicity, we assume RGB24 or BGR24 or similar, likely need swscale

                        // We need a scaler to ensure RGB output
                        let mut rgb_frame = ffmpeg::util::frame::video::Video::empty();
                        let mut scaler = ffmpeg::software::scaling::context::Context::get(
                            decoded.format(),
                            decoded.width(),
                            decoded.height(),
                            ffmpeg::format::Pixel::RGB24,
                            decoded.width(),
                            decoded.height(),
                            ffmpeg::software::scaling::flag::Flags::BILINEAR,
                        )
                        .unwrap();

                        scaler.run(&decoded, &mut rgb_frame).ok();

                        let width = rgb_frame.width();
                        let height = rgb_frame.height();
                        let data = rgb_frame.data(0);
                        let stride = rgb_frame.stride(0);

                        // Tightly packed RGB data
                        let mut rgb_data = Vec::with_capacity((width * height * 3) as usize);
                        for y in 0..height as usize {
                            let row = &data[y * stride..y * stride + (width as usize) * 3];
                            rgb_data.extend_from_slice(row);
                        }

                        let img_buffer =
                            image::RgbImage::from_raw(width, height, rgb_data).unwrap();
                        let img = DynamicImage::ImageRgb8(img_buffer);

                        let meta = SourceMeta {
                            frame_idx: self.current_frame,
                            total_frames: None,
                            path: format!("Webcam {idx}"),
                            fps: None,
                        };
                        self.current_frame += 1;
                        return Some(Ok((img, meta)));
                    }
                }
                return None; // End of stream or error
            }
            return None;
        }

        // Initialize decoder if needed (Video/Stream)
        if self.decoder.is_none() {
            let path_str = match &self.source {
                Source::Video(p) => Some(p.to_string_lossy().to_string()),
                Source::Stream(s) => Some(s.clone()),
                // Webcam handled above
                _ => None,
            };

            if let Some(path_str) = path_str {
                match video_rs::decode::Decoder::new(Path::new(&path_str)) {
                    Ok(d) => {
                        // Calculate total frames from duration and frame rate
                        // Note: limit to Video source only as streams/webcams are infinite
                        #[allow(clippy::collapsible_if)]
                        if let Source::Video(_) = &self.source {
                            if let Ok(duration) = d.duration() {
                                let fps = d.frame_rate();
                                let duration_seconds = duration.as_secs_f64();
                                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                                {
                                    self.total_frames =
                                        Some((duration_seconds * f64::from(fps)) as usize);
                                }
                            }
                        }
                        self.decoder = Some(d);
                    }
                    Err(e) => {
                        // Debug: print error to stderr
                        eprintln!("Debug: Decode failed: {e}");
                        return Some(Err(InferenceError::VideoError(format!(
                            "Failed to create decoder: {e}"
                        ))));
                    }
                }
                // Note: Decoder initialized.
            }
        }

        if let Some(decoder) = &mut self.decoder {
            // Attempt to decode next frame
            match decoder.decode() {
                Ok((_ts, frame)) => {
                    let fps = decoder.frame_rate();
                    let meta = SourceMeta {
                        frame_idx: self.current_frame,
                        total_frames: self.total_frames,
                        path: self
                            .source
                            .path()
                            .map(|p| p.to_string_lossy().to_string())
                            .unwrap_or_default(),
                        fps: Some(fps),
                    };
                    self.current_frame += 1;

                    match video_frame_to_image(&frame) {
                        Ok(img) => Some(Ok((img, meta))),
                        Err(e) => Some(Err(e)),
                    }
                }
                Err(e) => {
                    eprintln!("Debug: Decode failed: {e}");
                    None
                }
            }
        } else {
            None
        }
    }

    #[cfg(not(feature = "video"))]
    #[allow(
        clippy::unused_self,
        clippy::unnecessary_wraps,
        clippy::needless_pass_by_ref_mut
    )]
    fn next_video_frame(&mut self) -> Option<Result<(DynamicImage, SourceMeta)>> {
        Some(Err(InferenceError::FeatureNotEnabled(
            "Video support requires 'video' feature".to_string(),
        )))
    }
}

impl Iterator for SourceIterator {
    type Item = Result<(DynamicImage, SourceMeta)>;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.source {
            Source::Image(_) | Source::Directory(_) | Source::Glob(_) | Source::ImageList(_) => {
                self.next_image()
            }
            Source::ImageUrl(url) => {
                let url = url.clone();
                self.next_image_url(&url)
            }
            Source::ImageBuffer(img) => {
                if self.current_frame == 0 {
                    self.current_frame = 1;
                    let meta = SourceMeta::default();
                    Some(Ok((img.clone(), meta)))
                } else {
                    None
                }
            }
            Source::Array(arr) => {
                if self.current_frame == 0 {
                    self.current_frame = 1;
                    let meta = SourceMeta::default();
                    // Convert array to image
                    match array_to_image(arr) {
                        Ok(img) => Some(Ok((img, meta))),
                        Err(e) => Some(Err(e)),
                    }
                } else {
                    None
                }
            }
            Source::Video(_) | Source::Webcam(_) | Source::Stream(_) => self.next_video_frame(),
        }
    }
}

/// Convert an HWC u8 array to a `DynamicImage`.
fn array_to_image(arr: &Array3<u8>) -> Result<DynamicImage> {
    let shape = arr.shape();
    let height = u32::try_from(shape[0])
        .map_err(|_| InferenceError::ImageError("Image height exceeds u32::MAX".to_string()))?;
    let width = u32::try_from(shape[1])
        .map_err(|_| InferenceError::ImageError("Image width exceeds u32::MAX".to_string()))?;

    let mut rgb_data = Vec::with_capacity((height * width * 3) as usize);
    for y in 0..height as usize {
        for x in 0..width as usize {
            rgb_data.push(arr[[y, x, 0]]);
            rgb_data.push(arr[[y, x, 1]]);
            rgb_data.push(arr[[y, x, 2]]);
        }
    }

    let img_buffer = image::RgbImage::from_raw(width, height, rgb_data).ok_or_else(|| {
        InferenceError::ImageError("Failed to create image from array".to_string())
    })?;

    Ok(DynamicImage::ImageRgb8(img_buffer))
}

#[cfg(feature = "video")]
/// Convert a `video_rs` Frame (ndarray 0.16) to `DynamicImage`.
fn video_frame_to_image(arr: &video_rs::Frame) -> Result<DynamicImage> {
    let shape = arr.shape();
    let height = u32::try_from(shape[0])
        .map_err(|_| InferenceError::ImageError("Image height exceeds u32::MAX".to_string()))?;
    let width = u32::try_from(shape[1])
        .map_err(|_| InferenceError::ImageError("Image width exceeds u32::MAX".to_string()))?;

    let mut rgb_data = Vec::with_capacity((height * width * 3) as usize);
    for y in 0..height as usize {
        for x in 0..width as usize {
            rgb_data.push(arr[[y, x, 0]]);
            rgb_data.push(arr[[y, x, 1]]);
            rgb_data.push(arr[[y, x, 2]]);
        }
    }

    let img_buffer = image::RgbImage::from_raw(width, height, rgb_data).ok_or_else(|| {
        InferenceError::ImageError("Failed to create image from video frame".to_string())
    })?;

    Ok(DynamicImage::ImageRgb8(img_buffer))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_from_string() {
        assert!(matches!(Source::from("image.jpg"), Source::Image(_)));
        assert!(matches!(Source::from("video.mp4"), Source::Video(_)));
        assert!(matches!(
            Source::from("rtsp://example.com"),
            Source::Stream(_)
        ));
        assert!(matches!(Source::from("0"), Source::Webcam(0)));
        assert!(matches!(Source::from("*.jpg"), Source::Glob(_)));
    }

    #[test]
    fn test_source_checks() {
        let img = Source::Image(PathBuf::from("test.jpg"));
        assert!(img.is_image());
        assert!(!img.is_video());

        let vid = Source::Video(PathBuf::from("test.mp4"));
        assert!(!vid.is_image());
        assert!(vid.is_video());

        let dir = Source::Directory(PathBuf::from("./images"));
        assert!(dir.is_batch());
    }
}
