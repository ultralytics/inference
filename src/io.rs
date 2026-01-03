// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! I/O utilities for saving results including video encoding.

#[cfg(feature = "video")]
use video_rs::{Encoder, Time, encode::Settings as EncoderSettings};

use crate::error::{InferenceError, Result};
use std::path::{Path, PathBuf};

#[cfg(feature = "video")]
use std::sync::Once;

#[cfg(feature = "video")]
static INIT: Once = Once::new();

/// Initialize global video logging configuration.
///
/// ensuring `video-rs` is initialized and `FFmpeg` logs are silenced
/// (only errors are shown). safe to call multiple times.
#[allow(clippy::missing_const_for_fn)]
pub fn init_logging() {
    #[cfg(feature = "video")]
    INIT.call_once(|| {
        if let Err(e) = video_rs::init() {
            eprintln!("Failed to initialize video-rs: {e}");
        }

        #[cfg(feature = "ffmpeg-next")]
        ffmpeg_next::log::set_level(ffmpeg_next::log::Level::Error);
    });
}

/// A wrapper around `video-rs` encoder to simplify video saving.
#[cfg(feature = "video")]
pub struct VideoWriter {
    encoder: Encoder,
    frame_duration: Time,
    position: Time,
    width: usize,
    height: usize,
}

#[cfg(feature = "video")]
impl VideoWriter {
    /// Create a new `VideoWriter`.
    ///
    /// # Arguments
    ///
    /// * `path` - Output video path (e.g., "output.mp4").
    /// * `width` - Video width.
    /// * `height` - Video height.
    /// * `fps` - Frames per second.
    ///
    /// # Errors
    ///
    /// Returns an error if the encoder cannot be initialized.
    pub fn new<P: AsRef<Path>>(path: P, width: usize, height: usize, fps: f32) -> Result<Self> {
        let output_path = path.as_ref().to_path_buf();

        // Ensure parent directory exists
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                InferenceError::IoError(format!(
                    "Failed to create directory {}: {e}",
                    parent.display()
                ))
            })?;
        }

        let settings = EncoderSettings::preset_h264_yuv420p(width, height, false);
        let encoder = Encoder::new(output_path.as_path(), settings).map_err(|e| {
            InferenceError::VideoError(format!("Failed to create video encoder: {e}"))
        })?;

        // Calculate frame duration
        // video-rs uses a rational time base.
        // We can approximate by converting seconds to Time.
        let seconds_per_frame = 1.0 / f64::from(fps);
        let frame_duration = Time::from_secs_f64(seconds_per_frame);

        Ok(Self {
            encoder,
            frame_duration,
            position: Time::zero(),
            width,
            height,
        })
    }

    /// Write a frame to the video.
    ///
    /// # Arguments
    ///
    /// * `frame` - Input frame as `DynamicImage`.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding fails or frame dimensions don't match.
    pub fn write_frame(&mut self, frame: &image::DynamicImage) -> Result<()> {
        let img_buffer = frame.to_rgb8();
        let width = img_buffer.width() as usize;
        let height = img_buffer.height() as usize;

        if width != self.width || height != self.height {
            return Err(InferenceError::VideoError(format!(
                "Frame dimensions {}x{} do not match video dimensions {}x{}",
                width, height, self.width, self.height
            )));
        }

        let raw = img_buffer.into_raw();

        #[cfg(feature = "ndarray_0_16")]
        let frame_array = ndarray_0_16::Array3::from_shape_vec((height, width, 3), raw)
            .map_err(|e| InferenceError::VideoError(e.to_string()))?;

        #[cfg(not(feature = "ndarray_0_16"))]
        let frame_array = ndarray::Array3::from_shape_vec((height, width, 3), raw)
            .map_err(|e| InferenceError::VideoError(e.to_string()))?;

        self.encoder
            .encode(&frame_array, self.position)
            .map_err(|e| InferenceError::VideoError(format!("Failed to encode frame: {e}")))?;

        self.position = self.position.aligned_with(self.frame_duration).add();
        Ok(())
    }

    /// Finish writing the video.
    ///
    /// Calling this explicitly is optional as `drop` will also clean up,
    /// but this allows catching errors.
    /// # Errors
    ///
    /// Returns an error if the encoder fails to finish.
    pub fn finish(mut self) -> Result<()> {
        self.encoder.finish().map_err(|e| {
            InferenceError::VideoError(format!("Failed to finish video encoding: {e}"))
        })
    }
}

/// Helper struct to handle saving inference results to video or disk.
///
/// This consolidates logic for deciding whether to save as a video file
/// or individual frames, and manages the `VideoWriter` state.
pub struct SaveResults {
    save_dir: PathBuf,
    #[allow(dead_code)]
    save_frames: bool,
    #[cfg(feature = "video")]
    video_writer: Option<VideoWriter>,
}

impl SaveResults {
    /// Create a new `SaveResults`.
    ///
    /// # Arguments
    ///
    /// * `save_dir` - Directory to save results.
    /// * `save_frames` - If true, force saving individual frames even for video sources.
    #[must_use]
    pub fn new(save_dir: PathBuf, save_frames: bool) -> Self {
        init_logging();

        Self {
            save_dir,
            #[allow(unused)]
            save_frames,
            #[cfg(feature = "video")]
            video_writer: None,
        }
    }

    /// Save an annotated frame.
    ///
    /// Decides automatically whether to append to a video or save as an image file
    /// based on the source type and configuration.
    ///
    /// # Arguments
    ///
    /// * `is_video` - Whether the source is a video/stream.
    /// * `meta` - Source metadata (path, frame index, fps).
    /// * `annotated` - The annotated image to save.
    ///
    /// # Errors
    ///
    /// Returns an error if saving the image or video frame fails.
    pub fn save(
        &mut self,
        is_video: bool,
        meta: &crate::source::SourceMeta,
        annotated: &image::DynamicImage,
    ) -> Result<()> {
        init_logging();

        #[cfg(feature = "video")]
        let save_as_video = is_video && !self.save_frames;
        #[cfg(not(feature = "video"))]
        let save_as_video = false;

        if save_as_video {
            #[cfg(feature = "video")]
            {
                // Video saving logic
                if self.video_writer.is_none() {
                    let filename = Path::new(&meta.path)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy();

                    let output_name = Path::new(filename.as_ref())
                        .with_extension("mp4")
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();

                    let save_path = self.save_dir.join(output_name);
                    let width = annotated.width() as usize;
                    let height = annotated.height() as usize;
                    let fps = meta.fps.unwrap_or(30.0);

                    // Ensure directory exists
                    if let Some(parent) = save_path.parent()
                        && !parent.exists()
                    {
                        std::fs::create_dir_all(parent)
                            .map_err(|e| InferenceError::IoError(e.to_string()))?;
                    }

                    self.video_writer = Some(VideoWriter::new(save_path, width, height, fps)?);
                }

                if let Some(writer) = &mut self.video_writer {
                    writer.write_frame(annotated)?;
                }
            }
        } else {
            // Image saving logic
            let filename = if is_video {
                format!(
                    "{}_{}.jpg",
                    Path::new(&meta.path)
                        .file_stem()
                        .unwrap_or_default()
                        .to_string_lossy(),
                    meta.frame_idx
                )
            } else {
                Path::new(&meta.path)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string()
            };

            let save_path = self.save_dir.join(filename);

            // Ensure directory exists (might be redundant but safe)
            if !self.save_dir.exists() {
                std::fs::create_dir_all(&self.save_dir)
                    .map_err(|e| InferenceError::IoError(e.to_string()))?;
            }

            annotated
                .save(&save_path)
                .map_err(|e| InferenceError::ImageError(e.to_string()))?;
        }
        Ok(())
    }

    /// Finish any active video writing.
    ///
    /// # Errors
    ///
    /// Returns an error if the video writer fails to finish.
    pub fn finish(self) -> Result<()> {
        #[cfg(feature = "video")]
        if let Some(writer) = self.video_writer {
            writer.finish()?;
        }
        Ok(())
    }
}
