// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Image preprocessing for YOLO inference.
//!
//! This module handles all image preprocessing operations needed before
//! running YOLO model inference, including resizing, padding, and normalization.

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
    /// Scale factors applied (`scale_y`, `scale_x`).
    pub scale: (f32, f32),
    /// Padding applied (`pad_top`, `pad_left`).
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

    let tensor = image_to_tensor(&letterboxed);

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
        #[allow(clippy::cast_precision_loss)]
        padding: (pad_top as f32, pad_left as f32),
    }
}

/// Calculate letterbox parameters for resizing.
///
/// Computes new dimensions and padding to fit the image within the target size while maintaining aspect ratio.
///
/// # Arguments
///
/// * `orig_width` - Original image width.
/// * `orig_height` - Original image height.
/// * `target_size` - Target size as (height, width).
/// * `stride` - Model stride for alignment (unused in calculation but kept for API compatibility).
///
/// # Returns
///
/// Tuple containing:
/// 1. `new_width`: Scaled width.
/// 2. `new_height`: Scaled height.
/// 3. `pad_left`: Left padding.
/// 4. `pad_top`: Top padding.
/// 5. `(scale_y, scale_x)`: Scale factors.
fn calculate_letterbox_params(
    orig_width: u32,
    orig_height: u32,
    target_size: (usize, usize),
    _stride: u32,
) -> (u32, u32, u32, u32, (f32, f32)) {
    #[allow(clippy::cast_precision_loss)]
    let (target_h, target_w) = (target_size.0 as f32, target_size.1 as f32);
    #[allow(clippy::cast_precision_loss)]
    let (orig_h, orig_w) = (orig_height as f32, orig_width as f32);

    // Calculate scale to fit within target while maintaining aspect ratio
    let scale = (target_h / orig_h).min(target_w / orig_w);

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let new_w = (orig_w * scale).round() as u32;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let new_h = (orig_h * scale).round() as u32;

    // Calculate padding to center the image (matching Python Ultralytics default)
    #[allow(clippy::cast_possible_truncation)]
    let pad_w = (target_size.1 as u32).saturating_sub(new_w);
    #[allow(clippy::cast_possible_truncation)]
    let pad_h = (target_size.0 as u32).saturating_sub(new_h);

    // Center alignment: divide padding equally on both sides
    let pad_left = pad_w / 2;
    let pad_top = pad_h / 2;

    // Scale factors for coordinate conversion back to original
    #[allow(clippy::cast_precision_loss)]
    let scale_x = new_w as f32 / orig_w;
    #[allow(clippy::cast_precision_loss)]
    let scale_y = new_h as f32 / orig_h;

    (new_w, new_h, pad_left, pad_top, (scale_y, scale_x))
}
/// Apply letterbox transformation to an image.
///
/// Resizes the image maintaining aspect ratio and adds padding usually to center it.
/// Uses SIMD-accelerated resizing via `fast_image_resize`.
///
/// # Arguments
///
/// * `image` - Source dynamic image.
/// * `new_width` - Target width after scaling (before padding).
/// * `new_height` - Target height after scaling (before padding).
/// * `pad_left` - Padding to add on the left.
/// * `pad_top` - Padding to add on the top.
/// * `target_size` - Final output dimensions (height, width).
///
/// # Returns
///
/// `RgbImage` padded and resized to `target_size`.
fn letterbox_image(
    image: &DynamicImage,
    new_width: u32,
    new_height: u32,
    pad_left: u32,
    pad_top: u32,
    target_size: (usize, usize),
) -> RgbImage {
    use fast_image_resize::{PixelType, ResizeAlg, ResizeOptions, Resizer, images::Image};

    let src_rgb = image.to_rgb8();
    let (src_w, src_h) = src_rgb.dimensions();

    let src_image = Image::from_vec_u8(src_w, src_h, src_rgb.into_raw(), PixelType::U8x3)
        .expect("Failed to create source image");

    let mut dst_image = Image::new(new_width, new_height, PixelType::U8x3);

    let mut resizer = Resizer::new();
    let options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(
        fast_image_resize::FilterType::Lanczos3,
    ));
    resizer
        .resize(&src_image, &mut dst_image, Some(&options))
        .expect("Failed to resize image");

    // Create output image with letterbox color
    #[allow(clippy::cast_possible_truncation)]
    let mut output: RgbImage = ImageBuffer::from_pixel(
        target_size.1 as u32,
        target_size.0 as u32,
        Rgb(LETTERBOX_COLOR),
    );

    let resized_rgb: RgbImage = ImageBuffer::from_raw(new_width, new_height, dst_image.into_vec())
        .expect("Failed to create resized image buffer");

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

    let mut tensor = Array4::zeros((1, 3, h, w));

    // Get mutable slices for each channel for faster access
    let (r_slice, rest) = tensor.as_slice_mut().unwrap().split_at_mut(h * w);
    let (g_slice, b_slice) = rest.split_at_mut(h * w);

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

    let mut tensor = Array4::from_elem((1, 3, h, w), f16::ZERO);

    let (r_slice, rest) = tensor.as_slice_mut().unwrap().split_at_mut(h * w);
    let (g_slice, b_slice) = rest.split_at_mut(h * w);

    // Precompute 1/255 as f16 for direct conversion
    let scale = f16::from_f32(1.0 / 255.0);

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
///
/// # Panics
///
/// Panics if the array cannot be created from the image pixels (e.g. dimension mismatch).
#[must_use]
pub fn image_to_array(image: &DynamicImage) -> Array3<u8> {
    let rgb = image.to_rgb8();
    let (width, height) = rgb.dimensions();
    let pixels = rgb.into_raw();

    Array3::from_shape_vec((height as usize, width as usize, 3), pixels)
        .expect("Failed to create array from image pixels")
}

/// Scale coordinates from model output space back to original image space.
///
/// # Arguments
///
/// * `coords` - Coordinates in model space (after letterbox).
/// * `scale` - Scale factors (`scale_y`, `scale_x`) from preprocessing.
/// * `padding` - Padding (`pad_top`, `pad_left`) from preprocessing.
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
pub const fn clip_coords(coords: &[f32; 4], shape: (u32, u32)) -> [f32; 4] {
    #[allow(clippy::cast_precision_loss)]
    let (h, w) = (shape.0 as f32, shape.1 as f32);
    [
        coords[0].clamp(0.0, w),
        coords[1].clamp(0.0, h),
        coords[2].clamp(0.0, w),
        coords[3].clamp(0.0, h),
    ]
}

/// Preprocess an image for YOLO classification (Center Crop).
///
/// Resizes the image so the shortest side matches `target_size`, then center crops.
///
/// # Arguments
///
/// * `image` - Input image.
/// * `target_size` - Target size as (height, width).
/// * `half` - If true, also generate FP16 tensor.
///
/// # Returns
///
/// Preprocessed tensor and transform information.
#[must_use]
pub fn preprocess_image_center_crop(
    image: &DynamicImage,
    target_size: (usize, usize),
    half: bool,
) -> PreprocessResult {
    let (orig_width, orig_height) = image.dimensions();
    let orig_shape = (orig_height, orig_width);

    // Perform center crop resize
    let (cropped, scale) = center_crop_image(image, target_size);

    // Convert to normalized NCHW tensor
    let tensor = image_to_tensor(&cropped);

    // Optionally compute FP16 tensor
    let tensor_f16 = if half {
        Some(image_to_tensor_f16(&cropped))
    } else {
        None
    };

    // For classification, we don't need complex coordinate mapping back to original
    // But we provide approximate scale/padding to satisfy strict types if needed.
    // In classification, we rarely map bounding boxes back, so these are less critical.
    let padding = (0.0, 0.0);

    PreprocessResult {
        tensor,
        tensor_f16,
        orig_shape,
        scale,
        padding,
    }
}

/// Resize and center crop image.
///
/// Resizes the image such that the shortest side equals the target dimension,
/// maintaining aspect ratio, then crops the center `target_size`.
///
/// # Arguments
///
/// * `image` - Source dynamic image.
/// * `target_size` - Desired output dimensions (height, width).
///
/// # Returns
///
/// Tuple containing:
/// 1. `cropped`: The processed `RgbImage`.
/// 2. `scale`: Scale factors applied (same for x and y).
#[allow(clippy::similar_names)]
fn center_crop_image(image: &DynamicImage, target_size: (usize, usize)) -> (RgbImage, (f32, f32)) {
    use fast_image_resize::{PixelType, ResizeAlg, ResizeOptions, Resizer, images::Image};

    let (src_w, src_h) = image.dimensions();
    #[allow(clippy::cast_possible_truncation)]
    let (target_h, target_w) = (target_size.0 as u32, target_size.1 as u32);

    // Calculate scale to "cover" the target area
    // scale = max(target_w / src_w, target_h / src_h)
    #[allow(clippy::cast_precision_loss)]
    let scale_x = target_w as f32 / src_w as f32;
    #[allow(clippy::cast_precision_loss)]
    let scale_y = target_h as f32 / src_h as f32;
    let scale = scale_x.max(scale_y);

    let (new_w, new_h) = if scale_x >= scale_y {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        (target_w, (src_h as f32 * scale_x) as u32)
    } else {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        ((src_w as f32 * scale_y) as u32, target_h)
    };

    // Resize first
    let src_rgb = image.to_rgb8();
    let src_image = Image::from_vec_u8(src_w, src_h, src_rgb.into_raw(), PixelType::U8x3)
        .expect("Failed to create source image");

    // Valid dimensions check
    let safe_new_w = new_w.max(1);
    let safe_new_h = new_h.max(1);

    let mut dst_image = Image::new(safe_new_w, safe_new_h, PixelType::U8x3);

    let mut resizer = Resizer::new();
    let options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(
        fast_image_resize::FilterType::Bilinear,
    ));
    resizer
        .resize(&src_image, &mut dst_image, Some(&options))
        .expect("Failed to resize image");

    // Convert back to RgbImage to crop
    let resized_buffer = dst_image.into_vec();
    let resized_rgb = RgbImage::from_raw(safe_new_w, safe_new_h, resized_buffer)
        .expect("Failed to create resized buffer");

    // Calculate crop offsets using Banker's Rounding (to match Python round())
    #[allow(clippy::cast_precision_loss)]
    let crop_x_float = (new_w.saturating_sub(target_w)) as f32 / 2.0;
    #[allow(clippy::cast_precision_loss)]
    let crop_y_float = (new_h.saturating_sub(target_h)) as f32 / 2.0;

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let crop_x = bankers_round(crop_x_float) as u32;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let crop_y = bankers_round(crop_y_float) as u32;

    let cropped =
        image::imageops::crop_imm(&resized_rgb, crop_x, crop_y, target_w, target_h).to_image();

    (cropped, (scale, scale))
}

/// Round float to nearest integer, rounding half to even (Banker's Rounding).
/// This matches Python's `round()` behavior.
fn bankers_round(v: f32) -> f32 {
    let n = v.floor();
    let d = v - n;
    if (d - 0.5).abs() < 1e-6 {
        if n % 2.0 == 0.0 { n } else { n + 1.0 }
    } else {
        v.round()
    }
}

#[allow(clippy::similar_names)]
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
