// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Image annotation utilities.
//!
//! This module provides functions for drawing bounding boxes, labels, and other
//! annotations on images based on inference results.

use crate::results::Results;
use crate::visualizer::color::{COLORS, POSE_COLORS};
use crate::visualizer::skeleton::{KPT_COLOR_INDICES, LIMB_COLOR_INDICES, SKELETON};
use ab_glyph::{Font, FontRef, PxScale, ScaleFont};
use image::{DynamicImage, Rgb};
use imageproc::drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use std::fs::{self, File};
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};

/// Assets URL for downloading fonts
const ASSETS_URL: &str = "https://github.com/ultralytics/assets/releases/download/v0.0.0";

/// Return the RGB color assigned to a class ID.
///
/// Cycles through the 20-entry Ultralytics palette so every class gets a
/// distinct, repeating color regardless of how many classes the model has.
#[must_use]
pub const fn get_class_color(class_id: usize) -> Rgb<u8> {
    let color = COLORS[class_id % COLORS.len()];
    Rgb(color)
}

/// Return the next unused run directory under `base` with the given `prefix`.
///
/// Tries `<base>/<prefix>` first, then `<base>/<prefix>2`, `<base>/<prefix>3`, and so on
/// until a path that does not yet exist is found.
#[must_use]
pub fn find_next_run_dir(base: &str, prefix: &str) -> String {
    crate::io::find_next_run_dir(base, prefix)
}

/// Load an image from `path`, working around a zune-jpeg stride bug for JPEG files.
///
/// JPEG files are decoded with `jpeg_decoder` directly to avoid a stride mismatch
/// that can corrupt images decoded through the default image crate backend.
/// All other formats fall through to `image::open`.
///
/// # Errors
///
/// Returns an [`image::ImageError`] if the file cannot be opened or decoded.
#[allow(clippy::missing_errors_doc)]
pub fn load_image(path: &str) -> image::ImageResult<DynamicImage> {
    let path_obj = Path::new(path);
    let ext = path_obj
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase);

    if let Some("jpg" | "jpeg") = ext.as_deref()
        && let Ok(file) = File::open(path)
    {
        let mut decoder = jpeg_decoder::Decoder::new(BufReader::new(file));
        if let Ok(pixels) = decoder.decode()
            && let Some(metadata) = decoder.info()
        {
            let width = u32::from(metadata.width);
            let height = u32::from(metadata.height);
            match metadata.pixel_format {
                jpeg_decoder::PixelFormat::RGB24 => {
                    if let Some(buffer) = image::ImageBuffer::from_raw(width, height, pixels) {
                        return Ok(DynamicImage::ImageRgb8(buffer));
                    }
                }
                jpeg_decoder::PixelFormat::L8 => {
                    if let Some(buffer) = image::ImageBuffer::from_raw(width, height, pixels) {
                        return Ok(DynamicImage::ImageLuma8(buffer));
                    }
                }
                _ => {}
            }
        }
    }
    // Fallback
    image::open(path)
}
/// Return the local path to `font`, downloading it from the Ultralytics asset CDN if absent.
///
/// Fonts are cached in the platform config directory (`~/.config/Ultralytics/` on Linux,
/// `~/Library/Application Support/Ultralytics/` on macOS). Returns `None` if the directory
/// cannot be created or the download fails.
#[must_use]
pub fn check_font(font: &str) -> Option<PathBuf> {
    let font_name = Path::new(font).file_name()?.to_string_lossy();
    let config_dir = dirs::config_dir()?.join("Ultralytics");
    let font_path = config_dir.join(font_name.as_ref());

    if font_path.exists() {
        return Some(font_path);
    }

    // Create config directory if it doesn't exist
    if let Err(e) = fs::create_dir_all(&config_dir) {
        eprintln!("Failed to create config directory: {e}");
        return None;
    }

    // Download font
    let url = format!("{ASSETS_URL}/{font_name}");
    println!("Downloading {url} to {}", font_path.display());

    match ureq::get(&url).call() {
        Ok(response) => {
            let mut file = match File::create(&font_path) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("Failed to create font file: {e}");
                    return None;
                }
            };

            let mut reader = response.into_body().into_reader();
            if let Err(e) = io::copy(&mut reader, &mut file) {
                eprintln!("Failed to download font: {e}");
                // Try to remove partial file
                let _ = fs::remove_file(&font_path);
                return None;
            }

            Some(font_path)
        }
        Err(e) => {
            eprintln!("Failed to download font from {url}: {e}");
            None
        }
    }
}

/// Overlay inference results onto `image` and return the annotated copy.
///
/// Draws whichever result types are present: semantic segmentation mask, detection boxes
/// with instance masks, pose skeletons, oriented bounding boxes, and classification
/// probabilities. The font (Arial or Arial Unicode for non-ASCII names) is downloaded
/// on first use and cached locally.
///
/// # Arguments
///
/// * `image` - Source image to annotate.
/// * `result` - Inference results produced by the model.
/// * `top_k` - Maximum number of classification labels to show. Defaults to 5 when `None`.
#[must_use]
pub fn annotate_image(
    image: &DynamicImage,
    result: &Results,
    top_k: Option<usize>,
) -> DynamicImage {
    let mut img = image.to_rgb8();

    let font_name = if result.boxes.is_some() && result.names.values().any(|n| !n.is_ascii()) {
        "Arial.Unicode.ttf"
    } else {
        "Arial.ttf"
    };

    // Load font
    let font_path = check_font(font_name);
    let font_data = font_path.and_then(|path| {
        let mut file = File::open(path).ok()?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).ok()?;
        Some(buffer)
    });

    let font = font_data
        .as_ref()
        .and_then(|data| FontRef::try_from_slice(data).ok());

    // Draw all annotations using helpers
    draw_semantic_mask(&mut img, result);
    draw_detection(&mut img, result, font.as_ref());
    draw_pose(&mut img, result, None, None, None);
    draw_obb(&mut img, result, font.as_ref());
    draw_classification(&mut img, result, font.as_ref(), top_k.unwrap_or(5));

    DynamicImage::ImageRgb8(img)
}

/// Draw a thick line segment between two points using Bresenham's algorithm.
///
/// # Arguments
///
/// * `img` - Target image to draw on.
/// * `x1`, `y1` - Start point coordinates.
/// * `x2`, `y2` - End point coordinates.
/// * `color` - RGB color of the line.
/// * `thickness` - Line width in pixels; 1 draws a single-pixel line.
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
fn draw_line_segment(
    img: &mut image::RgbImage,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    color: Rgb<u8>,
    thickness: i32,
) {
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap
    )]
    let (width, height) = img.dimensions();

    // Bresenham's line algorithm with thickness
    let dx = (x2 - x1).abs();
    let dy = (y2 - y1).abs();
    let sx = if x1 < x2 { 1.0 } else { -1.0 };
    let sy = if y1 < y2 { 1.0 } else { -1.0 };
    let mut err = dx - dy;

    let mut x = x1;
    let mut y = y1;

    loop {
        // Draw a thick point
        let half_t = thickness / 2;
        for tx in -half_t..=half_t {
            for ty in -half_t..=half_t {
                let px = (x as i32 + tx).max(0).min(width as i32 - 1) as u32;
                let py = (y as i32 + ty).max(0).min(height as i32 - 1) as u32;
                img.put_pixel(px, py, color);
            }
        }

        if (x - x2).abs() < 1.0 && (y - y2).abs() < 1.0 {
            break;
        }

        let e2 = 2.0 * err;
        if e2 > -dy {
            err -= dy;
            x += sx;
        }
        if e2 < dx {
            err += dx;
            y += sy;
        }
    }
}

/// Draw a filled circle centered at `(cx, cy)` with the given `radius`.
///
/// Pixels outside the image bounds are silently skipped.
///
/// # Arguments
///
/// * `img` - Target image to draw on.
/// * `cx`, `cy` - Center of the circle in pixel coordinates.
/// * `radius` - Radius in pixels.
/// * `color` - RGB fill color.
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
fn draw_filled_circle(img: &mut image::RgbImage, cx: i32, cy: i32, radius: i32, color: Rgb<u8>) {
    let (width, height) = img.dimensions();

    for y in (cy - radius)..=(cy + radius) {
        for x in (cx - radius)..=(cx + radius) {
            let dx = x - cx;
            let dy = y - cy;
            if dx * dx + dy * dy <= radius * radius
                && x >= 0
                && y >= 0
                && x < width as i32
                && y < height as i32
            {
                img.put_pixel(x as u32, y as u32, color);
            }
        }
    }
}

/// Overlay semantic segmentation class colors at 50% alpha onto `img`.
///
/// Each pixel is colorized using `COLORS[class_id % COLORS.len()]` blended 50/50 with
/// the original pixel. A no-op when `result.semantic_mask` is `None`.
fn draw_semantic_mask(img: &mut image::RgbImage, result: &Results) {
    let Some(ref semantic_mask) = result.semantic_mask else {
        return;
    };
    let (width, height) = img.dimensions();
    let w = width as usize;
    let mask_data = semantic_mask
        .data
        .as_slice()
        .expect("semantic mask must be contiguous");
    let pixels = img.as_flat_samples_mut();
    let buf = pixels.samples;
    let n_colors = COLORS.len();
    for y in 0..height as usize {
        let mask_row = y * w;
        let img_row = y * w * 3;
        for x in 0..w {
            let class_id = mask_data[mask_row + x] as usize;
            let color = COLORS[class_id % n_colors];
            let p = img_row + x * 3;
            buf[p] = (buf[p] / 2).saturating_add(color[0] / 2);
            buf[p + 1] = (buf[p + 1] / 2).saturating_add(color[1] / 2);
            buf[p + 2] = (buf[p + 2] / 2).saturating_add(color[2] / 2);
        }
    }
}

/// Draw a class label with a filled background rectangle, avoiding overlap with previously placed labels.
///
/// The label is positioned above `(anchor_x, anchor_y)`. If that position would clip
/// the top edge, the label falls back to `top_fallback_y`. Up to 10 downward shifts
/// are attempted to avoid collision with rectangles already in `occupied`.
///
/// # Arguments
///
/// * `img` - Target image to draw on.
/// * `font` - Loaded font used for measuring and rendering glyphs.
/// * `occupied` - Running list of already-placed label rectangles; updated in place.
/// * `label` - Text to render inside the background box.
/// * `color` - Background fill color; text color is chosen automatically for contrast.
/// * `font_scale` - Font size in pixels.
/// * `anchor_x` - Horizontal anchor: left edge of the associated bounding box.
/// * `anchor_y` - Vertical anchor: top edge of the associated bounding box.
/// * `top_fallback_y` - Y position to use when the label would clip above the image.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::too_many_arguments
)]
fn draw_label(
    img: &mut image::RgbImage,
    font: &FontRef,
    occupied: &mut Vec<Rect>,
    label: &str,
    color: Rgb<u8>,
    font_scale: f32,
    anchor_x: i32,
    anchor_y: i32,
    top_fallback_y: i32,
) {
    let (width, height) = img.dimensions();
    let scale = PxScale::from(font_scale);
    let scaled_font = font.as_scaled(scale);
    let text_w = label
        .chars()
        .map(|c| scaled_font.h_advance(scaled_font.glyph_id(c)))
        .sum::<f32>()
        .ceil() as i32;
    let text_h = scale.y.ceil() as i32;

    let mut text_x = anchor_x;
    let mut text_y = anchor_y - text_h;

    if text_y < 0 {
        text_y = top_fallback_y;
    }
    if text_x < 0 {
        text_x = 0;
    }
    if text_x + text_w >= width as i32 {
        text_x = width as i32 - text_w - 1;
    }
    if text_y + text_h >= height as i32 {
        text_y = height as i32 - text_h - 1;
    }

    let mut attempts = 0;
    let max_attempts = 10;
    let mut current_rect = Rect::at(text_x, text_y).of_size(text_w as u32, text_h as u32);

    while attempts < max_attempts {
        if !occupied
            .iter()
            .any(|existing| rect_intersect(&current_rect, existing))
        {
            break;
        }

        text_y += text_h;
        if text_y + text_h >= height as i32 {
            text_y = anchor_y - text_h;
            if text_y < 0 {
                text_y = top_fallback_y;
            }
            text_x += 10;
            if text_x + text_w >= width as i32 {
                break;
            }
        }

        current_rect = Rect::at(text_x, text_y).of_size(text_w as u32, text_h as u32);
        attempts += 1;
    }

    occupied.push(current_rect);

    if text_x >= 0
        && text_y >= 0
        && text_x + text_w < width as i32
        && text_y + text_h < height as i32
    {
        draw_filled_rect_mut(img, current_rect, color);
        let text_color = get_text_color(color);
        draw_text_mut(img, text_color, text_x, text_y, scale, font, label);
    }
}

/// Draw detection boxes, optional instance masks, and class labels onto `img`.
///
/// Instance masks are blended at 30% opacity before boxes are drawn on top.
/// Labels are positioned above each box; overlap between labels is resolved
/// by shifting downward up to 10 times.
///
/// # Arguments
///
/// * `img` - Target image to draw on.
/// * `result` - Inference results; only `boxes` and `masks` fields are read.
/// * `font` - Optional loaded font; labels are skipped when `None`.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::manual_clamp,
    clippy::too_many_lines
)]
fn draw_detection(img: &mut image::RgbImage, result: &Results, font: Option<&FontRef>) {
    let (width, height) = img.dimensions();

    // Calculate dynamic scale factor based on image size (reference 640x640)
    let max_dim = width.max(height) as f32;
    let scale_factor = (max_dim / 640.0).max(1.0);

    // Scale thickness and font size
    let thickness = (1.0 * scale_factor).round().max(1.0) as i32;
    let font_scale = (11.0 * scale_factor).max(10.0); // Min font size 10

    if let Some(ref boxes) = result.boxes {
        let xyxy = boxes.xyxy();
        let conf = boxes.conf();
        let cls = boxes.cls();

        // Create an overlay image for masks to handle overlaps correctly
        let mut overlay = img.clone();
        let mut mask_present = false;

        // Draw masks onto the overlay
        if let Some(ref masks) = result.masks {
            let mask_n = masks.data.dim().0;

            for i in 0..boxes.len() {
                if i >= mask_n {
                    break;
                }

                let class_id = cls[i] as usize;
                let color = get_class_color(class_id);
                let (r, g, b) = (color.0[0], color.0[1], color.0[2]);

                mask_present = true;

                let x1 = xyxy[[i, 0]].max(0.0).min(width as f32) as u32;
                let y1 = xyxy[[i, 1]].max(0.0).min(height as f32) as u32;
                let x2 = xyxy[[i, 2]].max(0.0).min(width as f32) as u32;
                let y2 = xyxy[[i, 3]].max(0.0).min(height as f32) as u32;

                for y in y1..y2 {
                    for x in x1..x2 {
                        if masks.data[[i, y as usize, x as usize]] > 0.5 {
                            let pixel = overlay.get_pixel_mut(x, y);
                            pixel.0[0] = r;
                            pixel.0[1] = g;
                            pixel.0[2] = b;
                        }
                    }
                }
            }
        }

        // Blend overlay with original image
        if mask_present {
            let alpha = 0.3;
            for y in 0..height {
                for x in 0..width {
                    let p_img = img.get_pixel_mut(x, y);
                    let p_overlay = overlay.get_pixel(x, y);

                    p_img.0[0] = f32::from(p_overlay.0[0])
                        .mul_add(alpha, f32::from(p_img.0[0]) * (1.0 - alpha))
                        as u8;
                    p_img.0[1] = f32::from(p_overlay.0[1])
                        .mul_add(alpha, f32::from(p_img.0[1]) * (1.0 - alpha))
                        as u8;
                    p_img.0[2] = f32::from(p_overlay.0[2])
                        .mul_add(alpha, f32::from(p_img.0[2]) * (1.0 - alpha))
                        as u8;
                }
            }
        }

        // Keep track of occupied label areas to avoid overlap
        let mut labels_rects: Vec<Rect> = Vec::new();

        // Draw boxes and labels
        for i in 0..boxes.len() {
            let class_id = cls[i] as usize;
            let confidence = conf[i];

            let mut x1 = xyxy[[i, 0]].round() as i32;
            let mut y1 = xyxy[[i, 1]].round() as i32;
            let mut x2 = xyxy[[i, 2]].round() as i32;
            let mut y2 = xyxy[[i, 3]].round() as i32;

            if x1 > x2 {
                std::mem::swap(&mut x1, &mut x2);
            }
            if y1 > y2 {
                std::mem::swap(&mut y1, &mut y2);
            }

            x1 = x1.max(0).min(width as i32 - 1);
            y1 = y1.max(0).min(height as i32 - 1);
            x2 = x2.max(0).min(width as i32 - 1);
            y2 = y2.max(0).min(height as i32 - 1);

            if x2 <= x1 || y2 <= y1 {
                continue;
            }

            let color = get_class_color(class_id);

            // Draw box
            for t in 0..thickness {
                let tx1 = (x1 + t).min(x2);
                let ty1 = (y1 + t).min(y2);
                let tx2 = (x2 - t).max(tx1);
                let ty2 = (y2 - t).max(ty1);
                if tx2 > tx1 && ty2 > ty1 {
                    let rect = Rect::at(tx1, ty1).of_size((tx2 - tx1) as u32, (ty2 - ty1) as u32);
                    draw_hollow_rect_mut(img, rect, color);
                }
            }

            // Draw label
            let class_name = result.names.get(&class_id).map_or("object", String::as_str);
            let label = format!(" {class_name} {confidence:.2} ");

            if let Some(f) = font {
                draw_label(
                    img,
                    f,
                    &mut labels_rects,
                    &label,
                    color,
                    font_scale,
                    x1,
                    y1,
                    y1,
                );
            }
        }
    }
}

/// Return black or white, choosing whichever contrasts better with `bg`.
///
/// Uses the ITU-R BT.601 luminance formula. Backgrounds with luminance above 150
/// get black text; darker backgrounds get white text.
fn get_text_color(bg: Rgb<u8>) -> Rgb<u8> {
    let r = f32::from(bg.0[0]);
    let g = f32::from(bg.0[1]);
    let b = f32::from(bg.0[2]);

    // Calculate luminance (standard formula)
    let luminance = 0.114_f32.mul_add(b, 0.299_f32.mul_add(r, 0.587 * g));

    // Threshold usually around 128-186. Using 150 for better safety on mid-tones.
    if luminance > 150.0 {
        Rgb([0, 0, 0]) // Black text for light backgrounds
    } else {
        Rgb([255, 255, 255]) // White text for dark backgrounds
    }
}
/// Return `true` if rectangles `r1` and `r2` overlap.
fn rect_intersect(r1: &Rect, r2: &Rect) -> bool {
    let r1_left = r1.left();
    let r1_right = r1.right();
    let r1_top = r1.top();
    let r1_bottom = r1.bottom();

    let r2_left = r2.left();
    let r2_right = r2.right();
    let r2_top = r2.top();
    let r2_bottom = r2.bottom();

    !(r2_left >= r1_right || r2_right <= r1_left || r2_top >= r1_bottom || r2_bottom <= r1_top)
}

/// Draw pose estimation results (skeleton and keypoints)
///
/// # Arguments
///
/// * `img` - The image to draw on
/// * `result` - The inference results containing keypoints
/// * `skeleton` - Optional custom skeleton structure (pairs of keypoint indices).
///                If `None`, uses the default human pose skeleton from `SKELETON`.
/// * `limb_colors` - Optional custom color indices for limbs.
///                   If `None`, uses the default from `LIMB_COLOR_INDICES`.
/// * `kpt_colors` - Optional custom color indices for keypoints.
///                  If `None`, uses the default from `KPT_COLOR_INDICES`.
///
/// # Examples
///
/// ```ignore
/// // Use default human pose configuration
/// draw_pose(&mut img, result, None, None, None);
///
/// // Use custom skeleton for animals
/// const ANIMAL_SKELETON: [[usize; 2]; 10] = [...];
/// const ANIMAL_LIMB_COLORS: [usize; 10] = [0, 0, 9, 9, ...];
/// const ANIMAL_KPT_COLORS: [usize; 15] = [16, 16, 0, 0, ...];
/// draw_pose(
///     &mut img,
///     result,
///     Some(&ANIMAL_SKELETON),
///     Some(&ANIMAL_LIMB_COLORS),
///     Some(&ANIMAL_KPT_COLORS),
/// );
/// ```
#[allow(
    clippy::doc_overindented_list_items,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap
)]
fn draw_pose(
    img: &mut image::RgbImage,
    result: &Results,
    skeleton: Option<&[[usize; 2]]>,
    limb_colors: Option<&[usize]>,
    kpt_colors: Option<&[usize]>,
) {
    let (width, height) = img.dimensions();

    // Calculate dynamic scale factor based on image size (reference 640x640)
    let max_dim = width.max(height) as f32;
    let scale_factor = (max_dim / 640.0).max(1.0);

    // Scale thickness and radius
    let thickness = (1.0 * scale_factor).round().max(1.0) as i32;
    let radius = (3.0 * scale_factor).round() as i32;

    if let Some(ref keypoints) = result.keypoints {
        // Use provided parameters or defaults
        let skeleton = skeleton.unwrap_or(&SKELETON);
        let limb_colors = limb_colors.unwrap_or(&LIMB_COLOR_INDICES);
        let kpt_colors = kpt_colors.unwrap_or(&KPT_COLOR_INDICES);

        let kpt_data = &keypoints.data;
        let n_persons = kpt_data.shape()[0];
        let n_kpts = kpt_data.shape()[1];

        for person_idx in 0..n_persons {
            for (limb_idx, &[kpt_a, kpt_b]) in skeleton.iter().enumerate() {
                if kpt_a >= n_kpts || kpt_b >= n_kpts {
                    continue;
                }

                let x1 = kpt_data[[person_idx, kpt_a, 0]];
                let y1 = kpt_data[[person_idx, kpt_a, 1]];
                let conf1 = kpt_data[[person_idx, kpt_a, 2]];
                let x2 = kpt_data[[person_idx, kpt_b, 0]];
                let y2 = kpt_data[[person_idx, kpt_b, 1]];
                let conf2 = kpt_data[[person_idx, kpt_b, 2]];

                if conf1 > 0.5 && conf2 > 0.5 {
                    let color_idx = limb_colors[limb_idx % limb_colors.len()];
                    let color = Rgb(POSE_COLORS[color_idx]);
                    draw_line_segment(img, x1, y1, x2, y2, color, thickness);
                }
            }

            for kpt_idx in 0..n_kpts {
                let x = kpt_data[[person_idx, kpt_idx, 0]];
                let y = kpt_data[[person_idx, kpt_idx, 1]];
                let conf = kpt_data[[person_idx, kpt_idx, 2]];

                if conf > 0.5 && x >= 0.0 && y >= 0.0 && x < width as f32 && y < height as f32 {
                    let color_idx = kpt_colors[kpt_idx % kpt_colors.len()];
                    let color = Rgb(POSE_COLORS[color_idx]);
                    draw_filled_circle(img, x as i32, y as i32, radius, color);
                }
            }
        }
    }
}

/// Draw oriented bounding boxes and class labels onto `img`.
///
/// Each box is drawn as four line segments connecting its rotated corners.
/// Labels are positioned at the first corner of each box.
///
/// # Arguments
///
/// * `img` - Target image to draw on.
/// * `result` - Inference results; only the `obb` field is read.
/// * `font` - Optional loaded font; labels are skipped when `None`.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::manual_clamp
)]
fn draw_obb(img: &mut image::RgbImage, result: &Results, font: Option<&FontRef>) {
    let (width, height) = img.dimensions();

    // Calculate dynamic scale factor based on image size (reference 640x640)
    let max_dim = width.max(height) as f32;
    let scale_factor = (max_dim / 640.0).max(1.0);

    // Scale thickness and font size
    let thickness = (1.0 * scale_factor).round().max(1.0) as i32;
    let font_scale = (11.0 * scale_factor).max(10.0); // Min font size 10

    if let Some(ref obb) = result.obb {
        let corners = obb.xyxyxyxy();
        let conf = obb.conf();
        let cls = obb.cls();

        // Keep track of occupied label areas to avoid overlap
        let mut labels_rects: Vec<Rect> = Vec::new();

        for i in 0..obb.len() {
            let class_id = cls[i] as usize;
            let color = get_class_color(class_id);

            for j in 0..4 {
                let next_j = (j + 1) % 4;
                let x1 = corners[[i, j, 0]];
                let y1 = corners[[i, j, 1]];
                let x2 = corners[[i, next_j, 0]];
                let y2 = corners[[i, next_j, 1]];
                draw_line_segment(img, x1, y1, x2, y2, color, thickness);
            }

            let class_name = result.names.get(&class_id).map_or("object", String::as_str);
            let label = format!(" {} {:.2} ", class_name, conf[i]);

            if let Some(f) = font {
                draw_label(
                    img,
                    f,
                    &mut labels_rects,
                    &label,
                    color,
                    font_scale,
                    corners[[i, 0, 0]] as i32,
                    corners[[i, 0, 1]] as i32,
                    0,
                );
            }
        }
    }
}
/// Blend a solid color rectangle over a region of `img` at the given `alpha` opacity.
///
/// Pixels outside the image bounds are silently skipped. `alpha` is clamped to `[0.0, 1.0]`.
///
/// # Arguments
///
/// * `img` - Target image to blend onto.
/// * `x`, `y` - Top-left corner of the rectangle in pixel coordinates.
/// * `w`, `h` - Width and height of the rectangle in pixels.
/// * `color` - Blend color.
/// * `alpha` - Opacity; 0.0 is fully transparent, 1.0 is fully opaque.
#[allow(
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::manual_clamp
)]
fn draw_transparent_rect(
    img: &mut image::RgbImage,
    x: i32,
    y: i32,
    w: u32,
    h: u32,
    color: Rgb<u8>,
    alpha: f32,
) {
    let (width, height) = img.dimensions();
    let alpha = alpha.max(0.0).min(1.0);
    let inv_alpha = 1.0 - alpha;

    let r = f32::from(color[0]);
    let g = f32::from(color[1]);
    let b = f32::from(color[2]);

    for dy in 0..h {
        let py = y + dy as i32;
        if py < 0 || py >= height as i32 {
            continue;
        }

        for dx in 0..w {
            let px = x + dx as i32;
            if px < 0 || px >= width as i32 {
                continue;
            }

            let pixel = img.get_pixel_mut(px as u32, py as u32);
            let current = pixel.0;

            let new_r = f32::from(current[0]).mul_add(inv_alpha, r * alpha) as u8;
            let new_g = f32::from(current[1]).mul_add(inv_alpha, g * alpha) as u8;
            let new_b = f32::from(current[2]).mul_add(inv_alpha, b * alpha) as u8;

            *pixel = Rgb([new_r, new_g, new_b]);
        }
    }
}

/// Draw top-k classification probabilities onto `img` with a semi-transparent backing box.
///
/// Classes with a score below 0.01 are omitted. The list is rendered at the top-left
/// corner with a 40% opacity black background to keep text readable over any image content.
///
/// # Arguments
///
/// * `img` - Target image to draw on.
/// * `result` - Inference results; only the `probs` field is read.
/// * `font` - Optional loaded font; output is skipped when `None`.
/// * `top_k` - Maximum number of class entries to display.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::manual_clamp
)]
fn draw_classification(
    img: &mut image::RgbImage,
    result: &Results,
    font: Option<&FontRef>,
    top_k: usize,
) {
    if let Some(ref probs) = result.probs {
        let top_indices = probs.top_k(top_k);
        let (width, _height) = img.dimensions();

        if let Some(ref f) = font {
            // Adaptive font scale based on image width
            let scale_factor = (width as f32 / 600.0).max(0.6).min(2.0);
            let base_size = 30.0;
            let scale = PxScale::from(base_size * scale_factor);
            let line_height = (scale.y * 1.2) as i32;

            let x_pos = (20.0 * scale_factor) as i32;
            let mut y_pos = (20.0 * scale_factor) as i32;

            // Calculate max text width for background box
            let mut max_width = 0;
            let mut entries = Vec::new();

            for &class_id in &top_indices {
                let score = probs.data[class_id];
                if score < 0.01 {
                    continue;
                }

                let class_name = result.names.get(&class_id).map_or("class", String::as_str);

                let label = format!("{class_name} {score:.2}");
                entries.push(label);
            }

            // Basic approximation: chars * scale * 0.5 (average char width)
            for label in &entries {
                let w = (label.len() as f32 * scale.x * 0.5) as u32;
                if w > max_width {
                    max_width = w;
                }
            }

            // Draw background for all entries with padding
            let box_height = (entries.len() as i32 * line_height) + 10;
            let box_width = max_width + 20;

            if !entries.is_empty() {
                draw_transparent_rect(
                    img,
                    x_pos - 5,
                    y_pos - 5,
                    box_width,
                    box_height as u32,
                    Rgb([0, 0, 0]),
                    0.4, // 40% opacity black tint
                );
            }

            for label in entries {
                // Draw text (white)
                draw_text_mut(img, Rgb([255, 255, 255]), x_pos, y_pos, scale, f, &label);

                y_pos += line_height;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::results::Speed;
    use ndarray::Array3;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn test_get_class_color() {
        let c = get_class_color(0);
        assert_eq!(c, Rgb([4, 42, 255]));
    }

    #[test]
    fn test_draw_line_segment() {
        let mut img = image::RgbImage::new(100, 100);
        let color = Rgb([255, 0, 0]);
        // Draw horizontal line
        draw_line_segment(&mut img, 10.0, 10.0, 50.0, 10.0, color, 1);

        // Check a pixel in the middle
        assert_eq!(*img.get_pixel(30, 10), color);
        // Check pixel outside
        assert_eq!(*img.get_pixel(30, 20), Rgb([0, 0, 0]));
    }

    #[test]
    fn test_annotate_image_empty() {
        let img = DynamicImage::new_rgb8(100, 100);

        let orig_img = Array3::<u8>::zeros((100, 100, 3));
        let path = "test.jpg".to_string();
        let names = Arc::new(HashMap::new());
        let speed = Speed::new(0.0, 0.0, 0.0);

        // Correct constructor matching src/results.rs:101
        let results = Results::new(orig_img, path, names, speed, (640, 640));

        let annotated = annotate_image(&img, &results, None);
        // Should return same dimensions
        assert_eq!(annotated.width(), 100);
        assert_eq!(annotated.height(), 100);
    }
}
