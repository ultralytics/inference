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
    pub fn is_image(&self) -> bool {
        matches!(self, Self::Image(_) | Self::ImageBuffer(_) | Self::Array(_))
    }

    /// Check if this source is a video or stream.
    #[must_use]
    pub const fn is_video(&self) -> bool {
        matches!(self, Self::Video(_) | Self::Webcam(_) | Self::Stream(_))
    }

    /// Check if this source is a directory or glob pattern.
    #[must_use]
    pub const fn is_batch(&self) -> bool {
        matches!(self, Self::Directory(_) | Self::Glob(_))
    }

    /// Get the path if this source has one.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        match self {
            Self::Image(p) | Self::Video(p) | Self::Directory(p) => Some(p),
            _ => None,
        }
    }
}

/// Convert from a string path to Source.
impl From<&str> for Source {
    fn from(s: &str) -> Self {
        // Check for webcam index
        if let Ok(idx) = s.parse::<u32>() {
            return Self::Webcam(idx);
        }

        // Check for streaming URLs
        if s.starts_with("rtsp://")
            || s.starts_with("rtmp://")
            || s.starts_with("http://")
            || s.starts_with("https://")
        {
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

/// Iterator over frames from a source.
pub struct SourceIterator {
    source: Source,
    current_frame: usize,
    image_paths: Vec<PathBuf>,
    // TODO: Add video capture handle when video support is implemented
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
            _ => vec![],
        };

        Ok(Self {
            source,
            current_frame: 0,
            image_paths,
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
            .map_err(|e| InferenceError::IoError(e))?
            .filter_map(|entry| entry.ok())
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
                .map(|s| s.to_lowercase());

            if !dir.is_dir() {
                return Err(InferenceError::ImageError(format!(
                    "Directory not found: {}",
                    dir.display()
                )));
            }

            let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
                .map_err(InferenceError::IoError)?
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| {
                    if let Some(ref ext) = ext_filter {
                        path.extension()
                            .map(|e| e.to_string_lossy().to_lowercase() == *ext)
                            .unwrap_or(false)
                    } else {
                        Self::is_image_file(path)
                    }
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
        path.extension()
            .map(|ext| {
                let ext = ext.to_string_lossy().to_lowercase();
                matches!(
                    ext.as_str(),
                    "jpg" | "jpeg" | "png" | "bmp" | "gif" | "webp" | "tiff" | "tif"
                )
            })
            .unwrap_or(false)
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

    /// Get the next frame from video/webcam (placeholder).
    fn next_video_frame(&mut self) -> Option<Result<(DynamicImage, SourceMeta)>> {
        // TODO: Implement video frame extraction
        // This requires ffmpeg-next or similar library
        None
    }
}

impl Iterator for SourceIterator {
    type Item = Result<(DynamicImage, SourceMeta)>;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.source {
            Source::Image(_) | Source::Directory(_) | Source::Glob(_) => self.next_image(),
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

/// Convert an HWC u8 array to a DynamicImage.
fn array_to_image(arr: &Array3<u8>) -> Result<DynamicImage> {
    let shape = arr.shape();
    let (height, width) = (shape[0] as u32, shape[1] as u32);

    let mut rgb_data = Vec::with_capacity((height * width * 3) as usize);
    for y in 0..height as usize {
        for x in 0..width as usize {
            rgb_data.push(arr[[y, x, 0]]);
            rgb_data.push(arr[[y, x, 1]]);
            rgb_data.push(arr[[y, x, 2]]);
        }
    }

    let img_buffer = image::RgbImage::from_raw(width, height, rgb_data)
        .ok_or_else(|| InferenceError::ImageError("Failed to create image from array".to_string()))?;

    Ok(DynamicImage::ImageRgb8(img_buffer))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_from_string() {
        assert!(matches!(Source::from("image.jpg"), Source::Image(_)));
        assert!(matches!(Source::from("video.mp4"), Source::Video(_)));
        assert!(matches!(Source::from("rtsp://example.com"), Source::Stream(_)));
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
