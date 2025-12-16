// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Image viewer for displaying inference results.

use image::DynamicImage;
use minifb::{Key, Window, WindowOptions};

use crate::error::{InferenceError, Result};

/// A simple image viewer using minifb.
pub struct Viewer {
    window: Window,
    pub width: usize,
    pub height: usize,
    buffer: Vec<u32>,
}

impl Viewer {
    /// Create a new viewer window.
    pub fn new(title: &str, width: usize, height: usize) -> Result<Self> {
        let mut window = Window::new(
            title,
            width,
            height,
            WindowOptions {
                resize: true,
                ..WindowOptions::default()
            },
        )
        .map_err(|e| InferenceError::VisualizerError(format!("Failed to create window: {}", e)))?;

        // Limit update rate
        window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));

        Ok(Self {
            window,
            width,
            height,
            buffer: Vec::new(),
        })
    }

    /// Update the window with a new image.
    ///
    /// The image is resized to fit the window if necessary.
    pub fn update(&mut self, image: &DynamicImage) -> Result<bool> {
        if !self.window.is_open()
            || self.window.is_key_down(Key::Escape)
            || self.window.is_key_down(Key::Q)
        {
            return Ok(false);
        }

        let (img_width, img_height) = (image.width() as usize, image.height() as usize);

        // Resize buffer if needed
        let num_pixels = img_width * img_height;
        if self.buffer.len() != num_pixels {
            self.buffer.resize(num_pixels, 0);
        }

        // Convert image to BGRA format expected by minifb (u32 per pixel: 0x00RRGGBB)
        let rgb = image.to_rgb8();
        for (i, pixel) in rgb.pixels().enumerate() {
            let r = pixel[0] as u32;
            let g = pixel[1] as u32;
            let b = pixel[2] as u32;
            // Pack as 0x00RRGGBB
            self.buffer[i] = (r << 16) | (g << 8) | b;
        }

        // Update dimensions if changed
        if self.width != img_width || self.height != img_height {
            self.width = img_width;
            self.height = img_height;
        }

        self.window
            .update_with_buffer(&self.buffer, self.width, self.height)
            .map_err(|e| {
                InferenceError::VisualizerError(format!("Failed to update window: {}", e))
            })?;

        Ok(true)
    }

    /// Wait for a specified duration while keeping the window responsive.
    pub fn wait(&mut self, duration: std::time::Duration) -> Result<bool> {
        // If buffer is empty, we can't really update, just return
        if self.buffer.is_empty() {
            return Ok(true);
        }

        let start = std::time::Instant::now();
        while start.elapsed() < duration {
            if !self.window.is_open()
                || self.window.is_key_down(Key::Escape)
                || self.window.is_key_down(Key::Q)
            {
                return Ok(false);
            }
            // Use update_with_buffer to ensure the image persists
            // minifb handles frame limiting, so this loop won't spin 100% CPU indiscriminately
            // if limit_update_rate is set (which it is).
            let _ = self
                .window
                .update_with_buffer(&self.buffer, self.width, self.height);
        }
        Ok(true)
    }
}
