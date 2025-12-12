// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

use crate::results::Results;
use ab_glyph::{FontRef, PxScale};
use image::{DynamicImage, Rgb};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
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
pub fn annotate_image(image: &DynamicImage, result: &Results) -> DynamicImage {
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

    if let Some(ref boxes) = result.boxes {
        let xyxy = boxes.xyxy();
        let conf = boxes.conf();
        let cls = boxes.cls();

        for i in 0..boxes.len() {
            let class_id = cls[i] as usize;
            let confidence = conf[i];

            // Get box coordinates and clamp to image bounds
            let mut x1 = xyxy[[i, 0]].round() as i32;
            let mut y1 = xyxy[[i, 1]].round() as i32;
            let mut x2 = xyxy[[i, 2]].round() as i32;
            let mut y2 = xyxy[[i, 3]].round() as i32;

            // Ensure x1 < x2 and y1 < y2
            if x1 > x2 {
                std::mem::swap(&mut x1, &mut x2);
            }
            if y1 > y2 {
                std::mem::swap(&mut y1, &mut y2);
            }

            // Clamp to image bounds
            x1 = x1.max(0).min(width as i32 - 1);
            y1 = y1.max(0).min(height as i32 - 1);
            x2 = x2.max(0).min(width as i32 - 1);
            y2 = y2.max(0).min(height as i32 - 1);

            // Skip invalid boxes
            if x2 <= x1 || y2 <= y1 {
                continue;
            }

            let color = get_class_color(class_id);

            // Draw bounding box with fixed thickness
            let thickness = 3;
            for t in 0..thickness {
                let tx1 = (x1 + t).min(x2);
                let ty1 = (y1 + t).min(y2);
                let tx2 = (x2 - t).max(tx1);
                let ty2 = (y2 - t).max(ty1);
                if tx2 > tx1 && ty2 > ty1 {
                    let rect = Rect::at(tx1, ty1).of_size((tx2 - tx1) as u32, (ty2 - ty1) as u32);
                    draw_hollow_rect_mut(&mut img, rect, color);
                }
            }

            // Draw label
            let class_name = result
                .names
                .get(&class_id)
                .map(String::as_str)
                .unwrap_or("object");
            let label = format!("{} {:.2}", class_name, confidence);

            if let Some(ref f) = font {
                let scale = PxScale::from(16.0);
                // Position text above box if there's room, otherwise below
                let text_y = if y1 > 20 { y1 - 20 } else { y2 + 5 };
                let text_x = x1.max(0);
                // Only draw text if within bounds
                if text_x >= 0 && text_y >= 0 && text_x < width as i32 && text_y < height as i32 {
                    draw_text_mut(&mut img, color, text_x, text_y, scale, f, &label);
                }
            }
        }
    }

    DynamicImage::ImageRgb8(img)
}
