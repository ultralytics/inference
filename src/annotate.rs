// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Image annotation utilities.
//!
//! This module provides functions for drawing bounding boxes, labels, and other
//! annotations on images based on inference results.

use crate::results::Results;
use ab_glyph::{Font, FontRef, PxScale, ScaleFont};
use image::{DynamicImage, Rgb};
use imageproc::drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use std::fs::{self, File};
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};

/// Assets URL for downloading fonts
const ASSETS_URL: &str = "https://github.com/ultralytics/assets/releases/download/v0.0.0";

/// Ultralytics Color Palette
pub const COLORS: [[u8; 3]; 20] = [
    [4, 42, 255],    // #042aff
    [11, 219, 235],  // #0bdbeb
    [243, 243, 243], // #f3f3f3
    [0, 223, 183],   // #00dfb7
    [17, 31, 104],   // #111f68
    [255, 111, 221], // #ff6fdd
    [255, 68, 79],   // #ff444f
    [204, 237, 0],   // #cced00
    [0, 243, 68],    // #00f344
    [189, 0, 255],   // #bd00ff
    [0, 180, 255],   // #00b4ff
    [221, 0, 186],   // #dd00ba
    [0, 255, 255],   // #00ffff
    [38, 192, 0],    // #26c000
    [1, 255, 179],   // #01ffb3
    [125, 36, 255],  // #7d24ff
    [123, 0, 104],   // #7b0068
    [255, 27, 108],  // #ff1b6c
    [252, 109, 47],  // #fc6d2f
    [162, 255, 11],  // #a2ff0b
];

/// Ultralytics Pose Color Palette
pub const POSE_COLORS: [[u8; 3]; 20] = [
    [255, 128, 0],   // #ff8000
    [255, 153, 51],  // #ff9933
    [255, 178, 102], // #ffb266
    [230, 230, 0],   // #e6e600
    [255, 153, 255], // #ff99ff
    [153, 204, 255], // #99ccff
    [255, 102, 255], // #ff66ff
    [255, 51, 255],  // #ff33ff
    [102, 178, 255], // #66b2ff
    [51, 153, 255],  // #3399ff
    [255, 153, 153], // #ff9999
    [255, 102, 102], // #ff6666
    [255, 51, 51],   // #ff3333
    [153, 255, 153], // #99ff99
    [102, 255, 102], // #66ff66
    [51, 255, 51],   // #33ff33
    [0, 255, 0],     // #00ff00
    [0, 0, 255],     // #0000ff
    [255, 0, 0],     // #ff0000
    [255, 255, 255], // #ffffff
];

/// Get color for a class ID
pub fn get_class_color(class_id: usize) -> Rgb<u8> {
    let color = COLORS[class_id % COLORS.len()];
    Rgb(color)
}

/// Find the next available run directory (predict, predict2, predict3, etc.)
pub fn find_next_run_dir(base: &str, prefix: &str) -> String {
    let base_path = Path::new(base);

    // First try without number
    let first = base_path.join(prefix);
    if !first.exists() {
        return first.to_string_lossy().to_string();
    }

    // Try with incrementing numbers
    for i in 2.. {
        let numbered = base_path.join(format!("{prefix}{i}"));
        if !numbered.exists() {
            return numbered.to_string_lossy().to_string();
        }
    }

    // Fallback (should never reach here)
    base_path.join(prefix).to_string_lossy().to_string()
}

/// Load image helper to bypass zune-jpeg stride issues
pub fn load_image(path: &str) -> image::ImageResult<DynamicImage> {
    let path_obj = Path::new(path);
    let ext = path_obj
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase());

    if let Some("jpg") | Some("jpeg") = ext.as_deref() {
        if let Ok(file) = File::open(path) {
            let mut decoder = jpeg_decoder::Decoder::new(BufReader::new(file));
            if let Ok(pixels) = decoder.decode() {
                if let Some(metadata) = decoder.info() {
                    let width = metadata.width as u32;
                    let height = metadata.height as u32;
                    match metadata.pixel_format {
                        jpeg_decoder::PixelFormat::RGB24 => {
                            if let Some(buffer) =
                                image::ImageBuffer::from_raw(width, height, pixels)
                            {
                                return Ok(DynamicImage::ImageRgb8(buffer));
                            }
                        }
                        jpeg_decoder::PixelFormat::L8 => {
                            if let Some(buffer) =
                                image::ImageBuffer::from_raw(width, height, pixels)
                            {
                                return Ok(DynamicImage::ImageLuma8(buffer));
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    // Fallback
    image::open(path)
}
/// Check if font exists locally or download it
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

/// Helper to check if a string contains non-ASCII characters
fn is_ascii(s: &str) -> bool {
    s.is_ascii()
}
/// Annotate an image with detection boxes and labels
pub fn annotate_image(image: &DynamicImage, result: &Results, top_k: Option<usize>) -> DynamicImage {
    let mut img = image.to_rgb8();
    let (width, height) = img.dimensions();

    // Check if any class name is non-ASCII to select font
    let mut use_unicode_font = false;
    if result.boxes.is_some() {
        for name in result.names.values() {
            if !is_ascii(name) {
                use_unicode_font = true;
                break;
            }
        }
    }

    let font_name = if use_unicode_font {
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
    draw_detection(&mut img, result, font.as_ref());
    draw_pose(&mut img, result);
    draw_obb(&mut img, result, font.as_ref());
    draw_classification(&mut img, result, font.as_ref(), top_k.unwrap_or(5));

    DynamicImage::ImageRgb8(img)
}

/// Draw a line segment on an image
fn draw_line_segment(
    img: &mut image::RgbImage,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    color: Rgb<u8>,
    thickness: i32,
) {
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

/// Draw a filled circle on an image
fn draw_filled_circle(img: &mut image::RgbImage, cx: i32, cy: i32, radius: i32, color: Rgb<u8>) {
    let (width, height) = img.dimensions();

    for y in (cy - radius)..=(cy + radius) {
        for x in (cx - radius)..=(cx + radius) {
            let dx = x - cx;
            let dy = y - cy;
            if dx * dx + dy * dy <= radius * radius {
                if x >= 0 && y >= 0 && x < width as i32 && y < height as i32 {
                    img.put_pixel(x as u32, y as u32, color);
                }
            }
        }
    }
}


/// Draw object detection results (boxes and masks)
fn draw_detection(img: &mut image::RgbImage, result: &Results, font: Option<&FontRef>) {
    let (width, height) = img.dimensions();
    
    if let Some(ref boxes) = result.boxes {
        let xyxy = boxes.xyxy();
        let conf = boxes.conf();
        let cls = boxes.cls();

        // Create an overlay image for masks to handle overlaps correctly
        let mut overlay = img.clone();
        let mut mask_present = false;

        // Draw masks onto the overlay
        if let Some(ref masks) = result.masks {
            let (mask_n, _mask_h, _mask_w) = masks.data.dim();

            for i in 0..boxes.len() {
                if i >= mask_n { break; }

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

                    p_img.0[0] = (p_overlay.0[0] as f32 * alpha + p_img.0[0] as f32 * (1.0 - alpha)) as u8;
                    p_img.0[1] = (p_overlay.0[1] as f32 * alpha + p_img.0[1] as f32 * (1.0 - alpha)) as u8;
                    p_img.0[2] = (p_overlay.0[2] as f32 * alpha + p_img.0[2] as f32 * (1.0 - alpha)) as u8;
                }
            }
        }

        // Draw boxes and labels
        for i in 0..boxes.len() {
            let class_id = cls[i] as usize;
            let confidence = conf[i];

            let mut x1 = xyxy[[i, 0]].round() as i32;
            let mut y1 = xyxy[[i, 1]].round() as i32;
            let mut x2 = xyxy[[i, 2]].round() as i32;
            let mut y2 = xyxy[[i, 3]].round() as i32;

            if x1 > x2 { std::mem::swap(&mut x1, &mut x2); }
            if y1 > y2 { std::mem::swap(&mut y1, &mut y2); }

            x1 = x1.max(0).min(width as i32 - 1);
            y1 = y1.max(0).min(height as i32 - 1);
            x2 = x2.max(0).min(width as i32 - 1);
            y2 = y2.max(0).min(height as i32 - 1);

            if x2 <= x1 || y2 <= y1 { continue; }

            let color = get_class_color(class_id);

            // Draw box
            let thickness = 3;
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
            let class_name = result.names.get(&class_id).map(String::as_str).unwrap_or("object");
            let label = format!(" {class_name} {:.2} ", confidence);

            if let Some(ref f) = font {
                let scale = PxScale::from(24.0);
                let scaled_font = f.as_scaled(scale);
                let mut text_w = 0.0;
                for c in label.chars() {
                    text_w += scaled_font.h_advance(scaled_font.glyph_id(c));
                }
                let text_w = text_w.ceil() as i32;
                let text_h = scale.y.ceil() as i32;

                let text_y = if y1 > text_h { y1 - text_h } else { y1 };
                let text_x = x1;

                if text_x >= 0 && text_y >= 0 && text_x + text_w < width as i32 && text_y + text_h < height as i32 {
                    let rect = Rect::at(text_x, text_y).of_size(text_w as u32, text_h as u32);
                    draw_filled_rect_mut(img, rect, color);
                    draw_text_mut(img, Rgb([255, 255, 255]), text_x, text_y, scale, f, &label);
                }
            }
        }
    }
}

/// Draw pose estimation results (skeleton and keypoints)
fn draw_pose(img: &mut image::RgbImage, result: &Results) {
    let (width, height) = img.dimensions();

    if let Some(ref keypoints) = result.keypoints {
        const SKELETON: [[usize; 2]; 19] = [
            [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
            [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
            [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
            [1, 3], [2, 4], [3, 5], [4, 6]
        ];

        const LIMB_COLORS: [[u8; 3]; 19] = [
            [255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0],
            [255, 153, 255], [153, 204, 255], [255, 102, 255], [255, 51, 255],
            [102, 178, 255], [51, 153, 255], [255, 153, 153], [255, 102, 102],
            [255, 51, 51], [153, 255, 153], [102, 255, 102], [51, 255, 51],
            [0, 255, 0], [0, 0, 255], [255, 0, 0]
        ];

        let kpt_data = &keypoints.data;
        let n_persons = kpt_data.shape()[0];
        let n_kpts = kpt_data.shape()[1];

        for person_idx in 0..n_persons {
            for (limb_idx, &[kpt_a, kpt_b]) in SKELETON.iter().enumerate() {
                if kpt_a >= n_kpts || kpt_b >= n_kpts { continue; }

                let x1 = kpt_data[[person_idx, kpt_a, 0]];
                let y1 = kpt_data[[person_idx, kpt_a, 1]];
                let conf1 = kpt_data[[person_idx, kpt_a, 2]];
                let x2 = kpt_data[[person_idx, kpt_b, 0]];
                let y2 = kpt_data[[person_idx, kpt_b, 1]];
                let conf2 = kpt_data[[person_idx, kpt_b, 2]];

                if conf1 > 0.5 && conf2 > 0.5 {
                    let color = Rgb(LIMB_COLORS[limb_idx % LIMB_COLORS.len()]);
                    draw_line_segment(img, x1, y1, x2, y2, color, 2);
                }
            }

            for kpt_idx in 0..n_kpts {
                let x = kpt_data[[person_idx, kpt_idx, 0]];
                let y = kpt_data[[person_idx, kpt_idx, 1]];
                let conf = kpt_data[[person_idx, kpt_idx, 2]];

                if conf > 0.5 && x >= 0.0 && y >= 0.0 && x < width as f32 && y < height as f32 {
                    let color = Rgb(POSE_COLORS[kpt_idx % POSE_COLORS.len()]);
                    draw_filled_circle(img, x as i32, y as i32, 5, color);
                }
            }
        }
    }
}

/// Draw oriented bounding boxes (OBB)
fn draw_obb(img: &mut image::RgbImage, result: &Results, font: Option<&FontRef>) {
    let (width, height) = img.dimensions();

    if let Some(ref obb) = result.obb {
        let corners = obb.xyxyxyxy();
        let conf = obb.conf();
        let cls = obb.cls();

        for i in 0..obb.len() {
            let class_id = cls[i] as usize;
            let color = get_class_color(class_id);

            for j in 0..4 {
                let next_j = (j + 1) % 4;
                let x1 = corners[[i, j, 0]];
                let y1 = corners[[i, j, 1]];
                let x2 = corners[[i, next_j, 0]];
                let y2 = corners[[i, next_j, 1]];
                draw_line_segment(img, x1, y1, x2, y2, color, 3);
            }

            let class_name = result.names.get(&class_id).map(String::as_str).unwrap_or("object");
            let label = format!(" {} {:.2} ", class_name, conf[i]);

            if let Some(ref f) = font {
                let scale = PxScale::from(24.0);
                let scaled_font = f.as_scaled(scale);
                let mut text_w = 0.0;
                for c in label.chars() {
                    text_w += scaled_font.h_advance(scaled_font.glyph_id(c));
                }
                let text_w = text_w.ceil() as i32;
                let text_h = scale.y.ceil() as i32;

                let text_x = corners[[i, 0, 0]] as i32;
                let text_y = (corners[[i, 0, 1]] as i32 - text_h).max(0);

                if text_x >= 0 && text_y >= 0 && text_x + text_w < width as i32 && text_y + text_h < height as i32 {
                    let rect = Rect::at(text_x, text_y).of_size(text_w as u32, text_h as u32);
                    draw_filled_rect_mut(img, rect, color);
                    draw_text_mut(img, Rgb([255, 255, 255]), text_x, text_y, scale, f, &label);
                }
            }
        }
    }
}
/// Draw a transparent rectangle on an image
fn draw_transparent_rect(img: &mut image::RgbImage, x: i32, y: i32, w: u32, h: u32, color: Rgb<u8>, alpha: f32) {
    let (width, height) = img.dimensions();
    let alpha = alpha.max(0.0).min(1.0);
    let inv_alpha = 1.0 - alpha;
    
    let r = color[0] as f32;
    let g = color[1] as f32;
    let b = color[2] as f32;

    for dy in 0..h {
        let py = y + dy as i32;
        if py < 0 || py >= height as i32 { continue; }
        
        for dx in 0..w {
            let px = x + dx as i32;
            if px < 0 || px >= width as i32 { continue; }
            
            let pixel = img.get_pixel_mut(px as u32, py as u32);
            let current = pixel.0;
            
            let new_r = (current[0] as f32 * inv_alpha + r * alpha) as u8;
            let new_g = (current[1] as f32 * inv_alpha + g * alpha) as u8;
            let new_b = (current[2] as f32 * inv_alpha + b * alpha) as u8;
            
            *pixel = Rgb([new_r, new_g, new_b]);
        }
    }
}

/// Draw classification results
fn draw_classification(
    img: &mut image::RgbImage,
    result: &Results,
    font: Option<&FontRef>,
    top_k: usize,
) {
    if let Some(ref probs) = result.probs {
        let top_indices = probs.top_k(top_k);
        let (width, height) = img.dimensions();

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
                if score < 0.01 { continue; }
                
                let class_name = result
                    .names
                    .get(&class_id)
                    .map(String::as_str)
                    .unwrap_or("class");
                
                let label = format!("{} {:.2}", class_name, score);
                entries.push(label);
            }
            
            // Basic approximation: chars * scale * 0.5 (average char width)
            for label in &entries {
                let w = (label.len() as f32 * scale.x * 0.5) as u32;
                if w > max_width { max_width = w; }
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
                draw_text_mut(
                    img,
                    Rgb([255, 255, 255]),
                    x_pos,
                    y_pos,
                    scale,
                    f,
                    &label,
                );
                
                y_pos += line_height;
            }
        }
}
}

