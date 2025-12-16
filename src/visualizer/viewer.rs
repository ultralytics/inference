// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Image viewer for displaying inference results.

use image::DynamicImage;
use minifb::{Key, Window, WindowOptions};

use crate::error::{InferenceError, Result};

/// A simple image viewer using minifb.
pub struct Viewer {
    window: Window,
    width: usize,
    height: usize,
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

        // Convert image to BGRA format expected by minifb (u32 per pixel: 0x00RRGGBB)
        // Note: minifb expects 0x00RRGGBB for 24-bit color, effectively RGB
        let mut buffer: Vec<u32> = Vec::with_capacity(img_width * img_height);

        let rgb = image.to_rgb8();
        for pixel in rgb.pixels() {
            let r = pixel[0] as u32;
            let g = pixel[1] as u32;
            let b = pixel[2] as u32;
            // Pack as 0x00RRGGBB
            let color = (r << 16) | (g << 8) | b;
            buffer.push(color);
        }

        // If window size matches image size, just update
        if self.width != img_width || self.height != img_height {
            // Recreate window if dimensions changed significantly?
            // For now, simpler to just assume window matches image or handle resize later
            // Actually minifb handles content resize if we update with buffer dimensions
            self.width = img_width;
            self.height = img_height;
        }

        self.window
            .update_with_buffer(&buffer, img_width, img_height)
            .map_err(|e| {
                InferenceError::VisualizerError(format!("Failed to update window: {}", e))
            })?;

        Ok(true)
    }
}
