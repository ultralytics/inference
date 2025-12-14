// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Image preprocessing for YOLO inference.
//!
//! This module handles all image preprocessing operations needed before
//! running YOLO model inference, including resizing, padding, and normalization.

use fast_image_resize::{images::Image, ResizeAlg, ResizeOptions, Resizer};
use half::f16;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, RgbImage};
use ndarray::{Array3, Array4};

/// Default letterbox padding color (gray).
pub const LETTERBOX_COLOR: [u8; 3] = [114, 114, 114];

/// Tensor data that can be either FP32 or FP16.
#[derive(Debug, Clone)]
pub enum TensorData {
    /// 32-bit floating point tensor.
    Float32(Array4<f32>),
    /// 16-bit floating point tensor.
    Float16(Array4<f16>),
}

impl TensorData {
    /// Get the shape of the tensor.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Float32(t) => t.shape(),
            Self::Float16(t) => t.shape(),
        }
    }
}

/// Result of preprocessing an image, containing the tensor and transform info.
#[derive(Debug, Clone)]
pub struct PreprocessResult {
    /// Preprocessed image tensor in NCHW format, normalized to [0, 1].
    pub tensor: Array4<f32>,
    /// Preprocessed FP16 tensor (if requested).
    pub tensor_f16: Option<Array4<f16>>,
    /// Original image dimensions (height, width).
    pub orig_shape: (u32, u32),
    /// Scale factors applied (scale_y, scale_x).
    pub scale: (f32, f32),
    /// Padding applied (pad_top, pad_left).
    pub padding: (f32, f32),
}

/// Preprocess an image for YOLO inference.
///
/// Performs letterbox resizing, BGR to RGB conversion (if needed),
/// normalization to [0, 1], and conversion to NCHW tensor format.
///
/// # Arguments
///
/// * `image` - Input image.
/// * `target_size` - Target size as (height, width).
/// * `stride` - Model stride for padding alignment (typically 32).
///
/// # Returns
///
/// Preprocessed tensor and transform information for post-processing.
#[must_use]
pub fn preprocess_image(
    image: &DynamicImage,
    target_size: (usize, usize),
    stride: u32,
) -> PreprocessResult {
    preprocess_image_with_precision(image, target_size, stride, false)
}

/// Preprocess an image for YOLO inference with optional FP16 output.
///
/// # Arguments
///
/// * `image` - Input image.
/// * `target_size` - Target size as (height, width).
/// * `stride` - Model stride for padding alignment (typically 32).
/// * `half` - If true, also generate FP16 tensor for FP16 models.
///
/// # Returns
///
/// Preprocessed tensor and transform information for post-processing.
#[must_use]
pub fn preprocess_image_with_precision(
    image: &DynamicImage,
    target_size: (usize, usize),
    stride: u32,
    half: bool,
) -> PreprocessResult {
    let (orig_width, orig_height) = image.dimensions();
    let orig_shape = (orig_height, orig_width);

    // Calculate letterbox dimensions
    let (new_width, new_height, pad_left, pad_top, scale) =
        calculate_letterbox_params(orig_width, orig_height, target_size, stride);

    // Perform letterbox resize
    let letterboxed = letterbox_image(image, new_width, new_height, pad_left, pad_top, target_size);

    // Convert to normalized NCHW tensor (always compute FP32 for postprocessing)
    let tensor = image_to_tensor(&letterboxed);

    // Optionally compute FP16 tensor directly from image (avoiding FP32 round-trip)
    let tensor_f16 = if half {
        Some(image_to_tensor_f16(&letterboxed))
    } else {
        None
    };

    PreprocessResult {
        tensor,
        tensor_f16,
        orig_shape,
        scale,
        padding: (pad_top as f32, pad_left as f32),
    }
}

/// Calculate letterbox parameters for resizing.
///
/// # Arguments
///
/// * `orig_width` - Original image width.
/// * `orig_height` - Original image height.
/// * `target_size` - Target size as (height, width).
/// * `stride` - Model stride for alignment.
///
/// # Returns
///
/// Tuple of (new_width, new_height, pad_left, pad_top, (scale_y, scale_x)).
fn calculate_letterbox_params(
    orig_width: u32,
    orig_height: u32,
    target_size: (usize, usize),
    stride: u32,
) -> (u32, u32, u32, u32, (f32, f32)) {
    let (target_h, target_w) = (target_size.0 as f32, target_size.1 as f32);
    let (orig_h, orig_w) = (orig_height as f32, orig_width as f32);

    // Calculate scale to fit within target while maintaining aspect ratio
    let scale = (target_h / orig_h).min(target_w / orig_w);

    // New dimensions after scaling
    let new_w = (orig_w * scale).round() as u32;
    let new_h = (orig_h * scale).round() as u32;

    // Calculate padding to center the image
    let pad_w = (target_size.1 as u32).saturating_sub(new_w);
    let pad_h = (target_size.0 as u32).saturating_sub(new_h);

    let pad_left = pad_w / 2;
    let pad_top = pad_h / 2;

    // Scale factors for coordinate conversion back to original
    let scale_x = new_w as f32 / orig_w;
    let scale_y = new_h as f32 / orig_h;

    (new_w, new_h, pad_left, pad_top, (scale_y, scale_x))
}

/// Apply letterbox transformation to an image.
///
/// Resizes the image maintaining aspect ratio and adds padding.
/// Uses SIMD-accelerated resizing via fast_image_resize.
fn letterbox_image(
    image: &DynamicImage,
    new_width: u32,
    new_height: u32,
    pad_left: u32,
    pad_top: u32,
    target_size: (usize, usize),
) -> RgbImage {
    // Convert to RGB8
    let src_rgb = image.to_rgb8();
    
    // Resize using image::imageops (Triangle = Bilinear)
    let resized_rgb = image::imageops::resize(
        &src_rgb,
        new_width,
        new_height,
        image::imageops::FilterType::Triangle,
    );

    // Create output image with letterbox color
    let mut output: RgbImage = ImageBuffer::from_pixel(
        target_size.1 as u32,
        target_size.0 as u32,
        Rgb(LETTERBOX_COLOR),
    );

    // Copy resized image onto output
    image::imageops::overlay(
        &mut output,
        &resized_rgb,
        i64::from(pad_left),
        i64::from(pad_top),
    );

    output
}

/// Convert an RGB image to a normalized NCHW tensor (FP32).
///
/// # Arguments
///
/// * `image` - RGB image to convert.
///
/// # Returns
///
/// Array4 with shape (1, 3, H, W) and values in [0, 1].
fn image_to_tensor(image: &RgbImage) -> Array4<f32> {
    let (width, height) = image.dimensions();
    let (w, h) = (width as usize, height as usize);
    let pixels = image.as_raw();

    // Pre-allocate tensor
    let mut tensor = Array4::zeros((1, 3, h, w));

    // Get mutable slices for each channel for faster access
    let (r_slice, rest) = tensor.as_slice_mut().unwrap().split_at_mut(h * w);
    let (g_slice, b_slice) = rest.split_at_mut(h * w);

    // Convert HWC interleaved RGB to CHW planar format
    for (i, chunk) in pixels.chunks_exact(3).enumerate() {
        r_slice[i] = f32::from(chunk[0]) / 255.0;
        g_slice[i] = f32::from(chunk[1]) / 255.0;
        b_slice[i] = f32::from(chunk[2]) / 255.0;
    }

    tensor
}

/// Convert an RGB image to a normalized NCHW tensor (FP16).
///
/// Converts directly from u8 to f16, avoiding intermediate f32 conversion.
///
/// # Arguments
///
/// * `image` - RGB image to convert.
///
/// # Returns
///
/// Array4 with shape (1, 3, H, W) and f16 values in [0, 1].
fn image_to_tensor_f16(image: &RgbImage) -> Array4<f16> {
    let (width, height) = image.dimensions();
    let (w, h) = (width as usize, height as usize);
    let pixels = image.as_raw();

    // Pre-allocate tensor
    let mut tensor = Array4::from_elem((1, 3, h, w), f16::ZERO);

    // Get mutable slices for each channel for faster access
    let (r_slice, rest) = tensor.as_slice_mut().unwrap().split_at_mut(h * w);
    let (g_slice, b_slice) = rest.split_at_mut(h * w);

    // Precompute 1/255 as f16 for direct conversion
    let scale = f16::from_f32(1.0 / 255.0);

    // Convert HWC interleaved RGB to CHW planar format directly to f16
    for (i, chunk) in pixels.chunks_exact(3).enumerate() {
        r_slice[i] = f16::from_f32(f32::from(chunk[0])) * scale;
        g_slice[i] = f16::from_f32(f32::from(chunk[1])) * scale;
        b_slice[i] = f16::from_f32(f32::from(chunk[2])) * scale;
    }

    tensor
}

/// Convert a raw HWC u8 array to a normalized NCHW tensor.
///
/// # Arguments
///
/// * `image` - HWC array with shape (H, W, C) and u8 values.
///
/// # Returns
///
/// Array4 with shape (1, C, H, W) and values in [0, 1].
#[must_use]
pub fn array_to_tensor(image: &Array3<u8>) -> Array4<f32> {
    let shape = image.shape();
    let (height, width, channels) = (shape[0], shape[1], shape[2]);

    let mut tensor = Array4::zeros((1, channels, height, width));

    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                tensor[[0, c, y, x]] = f32::from(image[[y, x, c]]) / 255.0;
            }
        }
    }

    tensor
}

/// Convert a `DynamicImage` to an HWC ndarray.
#[must_use]
pub fn image_to_array(image: &DynamicImage) -> Array3<u8> {
    let rgb = image.to_rgb8();
    let (width, height) = rgb.dimensions();
    let pixels = rgb.into_raw();

    // Create array directly from raw pixels (already in HWC format)
    Array3::from_shape_vec((height as usize, width as usize, 3), pixels)
        .expect("Failed to create array from image pixels")
}

/// Scale coordinates from model output space back to original image space.
///
/// # Arguments
///
/// * `coords` - Coordinates in model space (after letterbox).
/// * `scale` - Scale factors (scale_y, scale_x) from preprocessing.
/// * `padding` - Padding (pad_top, pad_left) from preprocessing.
///
/// # Returns
///
/// Coordinates in original image space.
#[must_use]
pub fn scale_coords(coords: &[f32; 4], scale: (f32, f32), padding: (f32, f32)) -> [f32; 4] {
    let (scale_y, scale_x) = scale;
    let (pad_top, pad_left) = padding;

    [
        (coords[0] - pad_left) / scale_x, // x1
        (coords[1] - pad_top) / scale_y,  // y1
        (coords[2] - pad_left) / scale_x, // x2
        (coords[3] - pad_top) / scale_y,  // y2
    ]
}

/// Clip coordinates to image bounds.
///
/// # Arguments
///
/// * `coords` - Box coordinates [x1, y1, x2, y2].
/// * `shape` - Image shape (height, width).
///
/// # Returns
///
/// Clipped coordinates.
#[must_use]
pub fn clip_coords(coords: &[f32; 4], shape: (u32, u32)) -> [f32; 4] {
    let (h, w) = (shape.0 as f32, shape.1 as f32);
    [
        coords[0].clamp(0.0, w),
        coords[1].clamp(0.0, h),
        coords[2].clamp(0.0, w),
        coords[3].clamp(0.0, h),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_letterbox_params_square() {
        let (new_w, new_h, pad_left, pad_top, _scale) =
            calculate_letterbox_params(640, 640, (640, 640), 32);

        assert_eq!(new_w, 640);
        assert_eq!(new_h, 640);
        assert_eq!(pad_left, 0);
        assert_eq!(pad_top, 0);
    }

    #[test]
    fn test_letterbox_params_wide() {
        let (new_w, new_h, _, _, _) = calculate_letterbox_params(1280, 720, (640, 640), 32);

        // Wide image should be scaled down with height padded
        assert!(new_w <= 640);
        assert!(new_h <= 640);
    }

    #[test]
    fn test_scale_coords() {
        let coords = [100.0, 100.0, 200.0, 200.0];
        let scale = (1.0, 1.0);
        let padding = (10.0, 10.0);

        let scaled = scale_coords(&coords, scale, padding);

        assert!((scaled[0] - 90.0).abs() < 1e-6);
        assert!((scaled[1] - 90.0).abs() < 1e-6);
        assert!((scaled[2] - 190.0).abs() < 1e-6);
        assert!((scaled[3] - 190.0).abs() < 1e-6);
    }

    #[test]
    fn test_clip_coords() {
        let coords = [-10.0, -20.0, 700.0, 500.0];
        let clipped = clip_coords(&coords, (480, 640));

        assert!((clipped[0] - 0.0).abs() < 1e-6);
        assert!((clipped[1] - 0.0).abs() < 1e-6);
        assert!((clipped[2] - 640.0).abs() < 1e-6);
        assert!((clipped[3] - 480.0).abs() < 1e-6);
    }
}
