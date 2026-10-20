// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Image annotation utilities.
//!
//! This module provides functions for drawing bounding boxes, labels, and other
//! annotations on images based on inference results.

#[allow(clippy::wildcard_imports)] // native: rayon prelude; wasm: sequential shims
use crate::parallel::*;
use crate::results::Results;

/// Pixels handed to one task when blending a full-frame overlay across cores.
const BLEND_CHUNK: usize = 4096;
use crate::visualizer::color::{COLORS, Color, Colormap, DEPTH_ALPHA, DepthViz};
use crate::visualizer::skeleton::{KPT_COLOR_INDICES, LIMB_COLOR_INDICES, SKELETON};
use ab_glyph::{Font, FontRef, PxScale, ScaleFont};
use image::{DynamicImage, Rgb};
use imageproc::drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

/// Assets URL for downloading fonts
const ASSETS_URL: &str = "https://github.com/ultralytics/assets/releases/download/v0.0.0";

/// Drawing options for [`annotate_image_with_options`].
///
/// The keypoint fields follow the Ultralytics `Results.plot` and `Annotator.kpts` arguments.
/// Like Ultralytics, a 17-keypoint (COCO-Pose) model is drawn with the pose palette and the
/// human skeleton, and any other keypoint layout with one palette color per keypoint and no
/// lines; set [`skeleton`](Self::skeleton) to connect the keypoints of a custom model.
#[derive(Debug, Clone, PartialEq)]
pub struct AnnotateOptions {
    /// Maximum number of classification labels to show.
    pub top_k: usize,
    /// Depth colormap; only depth results use it.
    pub colormap: Colormap,
    /// Depth normalization; only depth results use it.
    pub depth_viz: DepthViz,
    /// Depth overlay opacity in `0.0..=1.0` (see [`annotate_image_with`]).
    pub depth_alpha: f32,
    /// Keypoint radius in pixels (`kpt_radius`).
    pub kpt_radius: i32,
    /// Draw the lines between keypoints (`kpt_line`).
    pub kpt_line: bool,
    /// Minimum confidence for drawing a keypoint or a line (`conf_thres`).
    pub kpt_conf: f32,
    /// One color for every keypoint and line (`kpt_color`). `None` uses the palette.
    pub kpt_color: Option<Color>,
    /// Pairs of keypoint indices to connect. `None` uses the human skeleton for 17-keypoint
    /// models and draws no lines for other layouts.
    pub skeleton: Option<Vec<[usize; 2]>>,
}

impl Default for AnnotateOptions {
    fn default() -> Self {
        Self {
            top_k: 5,
            colormap: Colormap::default(),
            depth_viz: DepthViz::default(),
            depth_alpha: DEPTH_ALPHA,
            kpt_radius: 5,
            kpt_line: true,
            kpt_conf: 0.25,
            kpt_color: None,
            skeleton: None,
        }
    }
}

/// Return the RGB color assigned to a class ID.
///
/// Cycles through the 20-entry Ultralytics palette so every class gets a
/// distinct, repeating color regardless of how many classes the model has.
#[must_use]
pub const fn get_class_color(class_id: usize) -> Rgb<u8> {
    let color = COLORS[class_id % COLORS.len()];
    Rgb(color)
}

/// Dynamic render scale derived from image size (reference 640x640).
///
/// Returns the scale factor and the base line thickness. Callers derive the
/// font size or marker radius from the factor as needed.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn render_scale(width: u32, height: u32) -> (f32, i32) {
    let max_dim = width.max(height) as f32;
    let scale_factor = (max_dim / 640.0).max(1.0);
    let thickness = (1.0 * scale_factor).round().max(1.0) as i32;
    (scale_factor, thickness)
}

/// Load and decode the image at `path` by its contents.
///
/// # Errors
///
/// Returns an [`image::ImageError`] if the file cannot be opened or decoded.
pub fn load_image(path: &str) -> image::ImageResult<DynamicImage> {
    crate::source::open_image(Path::new(path))
}
/// Return the local path to `font`, downloading it from the Ultralytics asset CDN if absent.
///
/// Fonts are cached in the platform config directory (`~/.config/Ultralytics/` on Linux,
/// `~/Library/Application Support/Ultralytics/` on macOS). Returns `None` if the directory
/// cannot be created or the download fails.
#[must_use]
pub fn check_font(font: &str) -> Option<PathBuf> {
    let font_path = font_path(font)?;
    if font_path.exists() {
        return Some(font_path);
    }

    // Create config directory if it doesn't exist
    if let Err(e) = fs::create_dir_all(font_path.parent()?) {
        eprintln!("Failed to create config directory: {e}");
        return None;
    }

    let url = format!("{ASSETS_URL}/{}", font_path.file_name()?.to_string_lossy());
    if let Err(e) = crate::download::download_file(&url, &font_path) {
        eprintln!("Failed to download font: {e}");
        return None;
    }
    Some(font_path)
}

/// Where `check_font` keeps `font`: the Ultralytics folder in the user config directory.
fn font_path(font: &str) -> Option<PathBuf> {
    Some(
        dirs::config_dir()?
            .join("Ultralytics")
            .join(Path::new(font).file_name()?),
    )
}

/// The label font, read once per process: every annotated frame needs it, and the Unicode
/// face non-ASCII class names pick is about 23 MB. `None` while it is not available.
///
/// A failed download is retried at most once a minute, with calls arriving during an attempt
/// waiting for it: retrying on every frame would wait out the download's retries and backoff
/// each time (about 6 s per frame without network). A font file that appears on disk is read
/// on the next call.
#[allow(clippy::significant_drop_tightening)] // the lock is held so others wait for the download
fn font_bytes(unicode: bool) -> Option<&'static [u8]> {
    type Font = (OnceLock<Vec<u8>>, Mutex<Option<Instant>>);
    static ARIAL: Font = (OnceLock::new(), Mutex::new(None));
    static ARIAL_UNICODE: Font = (OnceLock::new(), Mutex::new(None));
    let ((bytes, last_download), name) = if unicode {
        (&ARIAL_UNICODE, "Arial.Unicode.ttf")
    } else {
        (&ARIAL, "Arial.ttf")
    };
    if let Some(bytes) = bytes.get() {
        return Some(bytes);
    }
    let path = font_path(name)?;
    let mut last = last_download.lock().unwrap_or_else(PoisonError::into_inner);
    if !path.exists() && last.is_none_or(|t| t.elapsed() >= Duration::from_mins(1)) {
        *last = Some(Instant::now());
        let _ = check_font(name);
    }
    let data = fs::read(path).ok()?;
    Some(bytes.get_or_init(|| data))
}

/// Overlay inference results onto `image` and return the annotated copy.
///
/// Draws whichever result types are present: detection boxes with instance masks,
/// semantic segmentation mask, classification probabilities, pose skeletons, and
/// oriented bounding boxes. The font (Arial or Arial Unicode for non-ASCII names) is downloaded
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
    annotate_image_with_options(
        image,
        result,
        &AnnotateOptions {
            top_k: top_k.unwrap_or(5),
            ..AnnotateOptions::default()
        },
    )
}

/// Annotate an image like [`annotate_image`], but with an explicit depth `colormap`,
/// normalization `viz`, and overlay opacity `depth_alpha`.
///
/// Only depth results use `colormap`/`viz`/`depth_alpha`; every other task ignores them.
/// `depth_alpha` is the depth-overlay opacity in `0.0..=1.0`:
/// [`DEPTH_ALPHA`] (`0.6`) is the [`annotate_image`] default and blends it over the image;
/// `1.0` renders the full colorized map, `0.0` shows only the source. See [`annotate_image`]
/// for the general behavior.
#[must_use]
pub fn annotate_image_with(
    image: &DynamicImage,
    result: &Results,
    top_k: Option<usize>,
    colormap: Colormap,
    viz: DepthViz,
    depth_alpha: f32,
) -> DynamicImage {
    annotate_image_with_options(
        image,
        result,
        &AnnotateOptions {
            top_k: top_k.unwrap_or(5),
            colormap,
            depth_viz: viz,
            depth_alpha,
            ..AnnotateOptions::default()
        },
    )
}

/// Annotate an image like [`annotate_image`], with every drawing option in `options`.
///
/// # Example
///
/// ```no_run
/// # use ultralytics_inference::annotate::{AnnotateOptions, annotate_image_with_options};
/// # fn demo(image: &image::DynamicImage, result: &ultralytics_inference::Results) {
/// // Connect the keypoints of a custom 4-keypoint model and draw them a little larger.
/// let options = AnnotateOptions {
///     skeleton: Some(vec![[0, 1], [1, 2], [2, 3]]),
///     kpt_radius: 6,
///     ..AnnotateOptions::default()
/// };
/// let annotated = annotate_image_with_options(image, result, &options);
/// # }
/// ```
#[must_use]
pub fn annotate_image_with_options(
    image: &DynamicImage,
    result: &Results,
    options: &AnnotateOptions,
) -> DynamicImage {
    let mut img = image.to_rgb8();

    let unicode = result.boxes.is_some() && result.names.values().any(|n| !n.is_ascii());
    let font = font_bytes(unicode).and_then(|data| FontRef::try_from_slice(data).ok());

    draw_semantic_mask(&mut img, result);
    draw_detection(&mut img, result, font.as_ref());
    draw_pose(&mut img, result, options);
    draw_obb(&mut img, result, font.as_ref());
    draw_classification(&mut img, result, font.as_ref(), options.top_k);

    draw_depth_map(
        &mut img,
        result,
        options.colormap,
        options.depth_viz,
        options.depth_alpha,
    );

    DynamicImage::ImageRgb8(img)
}

/// Draw the colorized depth map over `img` in place at opacity `alpha`
/// (`(1 - alpha) * image + alpha * depth`): `1.0` is the full colorized map, lower blends.
///
/// No-op when the result carries no depth map or the map's resolution disagrees with the
/// image. Colorization (`colormap` + normalization `viz`) is delegated to
/// [`DepthMap::colorize`](crate::results::DepthMap::colorize).
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn draw_depth_map(
    img: &mut image::RgbImage,
    result: &Results,
    colormap: Colormap,
    viz: DepthViz,
    alpha: f32,
) {
    let Some(depth) = result.depth.as_ref() else {
        return;
    };
    let (width, height) = img.dimensions();
    // Depth map is stored at original image resolution; bail if it disagrees with the image.
    if depth.data.shape() != [height as usize, width as usize] {
        return;
    }
    let alpha = alpha.clamp(0.0, 1.0);
    // `colorize` returns row-major RGB, the same order the sample buffer walks. Blending a
    // full frame is per-pixel independent, so it splits across cores like the colorize
    // itself; `crate::parallel` is sequential on wasm.
    let heat = depth.colorize(colormap, viz);
    img.as_flat_samples_mut()
        .samples
        .par_chunks_mut(BLEND_CHUNK * 3)
        .enumerate()
        .for_each(|(c, dst)| {
            for (k, channel) in dst.iter_mut().enumerate() {
                let i = c * BLEND_CHUNK * 3 + k;
                let h = heat[i / 3][i % 3];
                *channel = f32::from(*channel)
                    .mul_add(1.0 - alpha, f32::from(h) * alpha)
                    .round() as u8;
            }
        });
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
        // Draw a thick point, skipping pixels outside the image so off-frame parts are clipped
        let half_t = thickness / 2;
        for tx in -half_t..=half_t {
            for ty in -half_t..=half_t {
                let (px, py) = (x as i32 + tx, y as i32 + ty);
                if px >= 0 && py >= 0 && px < width as i32 && py < height as i32 {
                    img.put_pixel(px as u32, py as u32, color);
                }
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
    let w = img.width() as usize;
    let mask_data = semantic_mask
        .data
        .as_slice()
        .expect("semantic mask must be contiguous");
    let n_colors = COLORS.len();
    // Rows are independent, so blend them across cores; `crate::parallel` is sequential on wasm.
    // The chunk size is at least 1 because a zero-width image has no rows to blend.
    img.as_flat_samples_mut()
        .samples
        .par_chunks_mut((w * 3).max(1))
        .enumerate()
        .for_each(|(y, row)| {
            let (pixels, _) = row.as_chunks_mut::<3>();
            for (px, &raw) in pixels.iter_mut().zip(&mask_data[y * w..(y + 1) * w]) {
                if raw == crate::results::SemanticMask::IGNORE {
                    continue; // filtered-out class: leave the original pixel
                }
                let color = COLORS[raw as usize % n_colors];
                for (channel, tint) in px.iter_mut().zip(color) {
                    *channel = (*channel / 2).saturating_add(tint / 2);
                }
            }
        });
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

    // A missing or degenerate font can produce a zero-width label. imageproc's
    // Rect::of_size panics on a zero dimension, and an empty label has nothing
    // to draw, so bail out early.
    if text_w <= 0 || text_h <= 0 {
        return;
    }

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

    // Dynamic scale factor and thickness based on image size (reference 640x640)
    let (scale_factor, thickness) = render_scale(width, height);
    let font_scale = (11.0 * scale_factor).max(10.0); // Min font size 10

    if let Some(ref boxes) = result.boxes {
        let xyxy = boxes.xyxy();
        let conf = boxes.conf();
        let cls = boxes.cls();

        // Overlay copy so overlapping masks compose against the original pixels. Only the
        // mask path needs it, so a box-only result (detect/pose/obb) never pays for it.
        let mut overlay = result.masks.is_some().then(|| img.clone());
        let mut mask_present = false;

        // Draw masks onto the overlay
        if let (Some(masks), Some(overlay)) = (result.masks.as_ref(), overlay.as_mut()) {
            let mask_n = masks.data.dim().0;

            for i in 0..boxes.len() {
                if i >= mask_n {
                    break;
                }

                let class_id = cls[i] as usize;
                let color = get_class_color(class_id);

                mask_present = true;

                let x1 = xyxy[[i, 0]].max(0.0).min(width as f32) as u32;
                let y1 = xyxy[[i, 1]].max(0.0).min(height as f32) as u32;
                let x2 = xyxy[[i, 2]].max(0.0).min(width as f32) as u32;
                let y2 = xyxy[[i, 3]].max(0.0).min(height as f32) as u32;

                // Row slices instead of per-pixel `[[i, y, x]]` and `get_pixel_mut` indexing.
                let plane = masks.data.index_axis(ndarray::Axis(0), i);
                let (w, x1, x2) = (width as usize, x1 as usize, x2 as usize);
                let samples = overlay.as_flat_samples_mut().samples;
                for y in y1 as usize..y2 as usize {
                    let hits = plane.slice(ndarray::s![y, x1..x2]);
                    let row = &mut samples[(y * w + x1) * 3..(y * w + x2) * 3];
                    let (pixels, _) = row.as_chunks_mut::<3>();
                    for (m, px) in hits.iter().zip(pixels) {
                        if *m > 0.5 {
                            *px = color.0;
                        }
                    }
                }
            }
        }

        // Blend overlay with original image. Walking the two sample buffers directly keeps
        // this a linear pass; `get_pixel_mut` recomputed the offset for all 300k+ pixels.
        if let Some(overlay) = overlay.filter(|_| mask_present) {
            let alpha = 0.3;
            let overlay = overlay.as_raw();
            img.as_flat_samples_mut()
                .samples
                .par_chunks_mut(BLEND_CHUNK * 3)
                .enumerate()
                .for_each(|(c, dst)| {
                    let src = &overlay[c * BLEND_CHUNK * 3..];
                    for (px, src) in dst.iter_mut().zip(src) {
                        *px = f32::from(*src).mul_add(alpha, f32::from(*px) * (1.0 - alpha)) as u8;
                    }
                });
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

/// Draw pose keypoints and the lines between them, styled by the keypoint fields of `options`.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap
)]
fn draw_pose(img: &mut image::RgbImage, result: &Results, options: &AnnotateOptions) {
    let (width, height) = img.dimensions();

    // Dynamic scale factor and thickness based on image size (reference 640x640)
    let (_, thickness) = render_scale(width, height);

    if let Some(ref keypoints) = result.keypoints {
        let kpt_data = &keypoints.data;
        let n_persons = kpt_data.shape()[0];
        let n_kpts = kpt_data.shape()[1];
        // Like Ultralytics, a point is drawn when its score reaches `kpt_conf`; a 2-channel
        // export carries no score and marks a missing point with (0, 0) instead.
        let has_conf = kpt_data.shape()[2] > 2;
        let visible = |p: usize, k: usize| {
            if has_conf {
                kpt_data[[p, k, 2]] >= options.kpt_conf
            } else {
                kpt_data[[p, k, 0]] != 0.0 || kpt_data[[p, k, 1]] != 0.0
            }
        };

        // Like Ultralytics, only the 17-keypoint COCO-Pose layout gets the pose palette and the
        // human skeleton; other layouts take one palette color per keypoint and no lines.
        let coco = n_kpts == KPT_COLOR_INDICES.len();
        let skeleton: &[[usize; 2]] = match &options.skeleton {
            Some(skeleton) => skeleton,
            None if coco => &SKELETON,
            None => &[],
        };
        // Ultralytics passes these RGB entries to OpenCV, which draws into a BGR image, so its
        // pictures show them with red and blue swapped; swap them too so the colors match.
        let palette = |i: usize| {
            let Color(r, g, b) = Color::from_index(i);
            Color(b, g, r)
        };
        let kpt_color = |k: usize| {
            options.kpt_color.unwrap_or_else(|| {
                if coco {
                    Color::from_pose_index(KPT_COLOR_INDICES[k])
                } else {
                    palette(k)
                }
            })
        };
        let limb_color = |l: usize| {
            options.kpt_color.unwrap_or_else(|| {
                if coco && options.skeleton.is_none() {
                    Color::from_pose_index(LIMB_COLOR_INDICES[l])
                } else {
                    palette(l)
                }
            })
        };

        for person_idx in 0..n_persons {
            if options.kpt_line {
                for (limb_idx, &[kpt_a, kpt_b]) in skeleton.iter().enumerate() {
                    if kpt_a >= n_kpts
                        || kpt_b >= n_kpts
                        || !visible(person_idx, kpt_a)
                        || !visible(person_idx, kpt_b)
                    {
                        continue;
                    }
                    let x1 = kpt_data[[person_idx, kpt_a, 0]];
                    let y1 = kpt_data[[person_idx, kpt_a, 1]];
                    let x2 = kpt_data[[person_idx, kpt_b, 0]];
                    let y2 = kpt_data[[person_idx, kpt_b, 1]];
                    let color = limb_color(limb_idx);
                    let color = Rgb([color.0, color.1, color.2]);
                    draw_line_segment(img, x1, y1, x2, y2, color, thickness);
                }
            }

            for kpt_idx in 0..n_kpts {
                if visible(person_idx, kpt_idx) {
                    let x = kpt_data[[person_idx, kpt_idx, 0]];
                    let y = kpt_data[[person_idx, kpt_idx, 1]];
                    let color = kpt_color(kpt_idx);
                    let color = Rgb([color.0, color.1, color.2]);
                    draw_filled_circle(img, x as i32, y as i32, options.kpt_radius, color);
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
    clippy::cast_possible_wrap
)]
fn draw_obb(img: &mut image::RgbImage, result: &Results, font: Option<&FontRef>) {
    let (width, height) = img.dimensions();

    // Dynamic scale factor and thickness based on image size (reference 640x640)
    let (scale_factor, thickness) = render_scale(width, height);
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
    let tint = color.0.map(|c| f32::from(c) * alpha);

    // Clamp the rectangle to the image once, then blend whole row slices.
    let (x0, x1) = (
        x.clamp(0, width as i32),
        (x + w as i32).clamp(0, width as i32),
    );
    let (y0, y1) = (
        y.clamp(0, height as i32),
        (y + h as i32).clamp(0, height as i32),
    );
    let width = width as usize;
    let samples = img.as_flat_samples_mut().samples;
    for py in y0 as usize..y1 as usize {
        let row = &mut samples[(py * width + x0 as usize) * 3..(py * width + x1 as usize) * 3];
        let (pixels, _) = row.as_chunks_mut::<3>();
        for pixel in pixels {
            for (channel, tint) in pixel.iter_mut().zip(tint) {
                *channel = f32::from(*channel).mul_add(inv_alpha, tint) as u8;
            }
        }
    }
}

/// Draw top-k classification probabilities onto `img` with a semi-transparent backing box.
///
/// The list is rendered at the top-left corner with a 40% opacity black background to keep text readable over any image content.
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
    clippy::cast_possible_wrap
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
            let scale_factor = (width as f32 / 600.0).clamp(0.6, 2.0);
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

    use crate::results::{Boxes, DepthMap, Keypoints, Obb, Probs, SemanticMask};
    use ndarray::array;

    fn base_results(names: HashMap<usize, String>) -> Results {
        Results::new(
            Array3::<u8>::zeros((128, 128, 3)),
            "img.jpg".to_string(),
            Arc::new(names),
            Speed::default(),
            (128, 128),
        )
    }

    #[test]
    fn test_render_scale_grows_with_image() {
        let (small_font, small_thick) = render_scale(64, 64);
        let (big_font, big_thick) = render_scale(2000, 2000);
        assert!(big_font >= small_font);
        assert!(big_thick >= small_thick);
        assert!(small_thick >= 1);
    }

    #[test]
    fn test_get_class_color_wraps() {
        // Indexing beyond the palette wraps rather than panicking.
        let _ = get_class_color(0);
        let _ = get_class_color(1000);
    }

    #[test]
    fn test_annotate_detection_draws() {
        let names = HashMap::from([(0usize, "person".to_string())]);
        let mut r = base_results(names);
        r.boxes = Some(Boxes::new(
            array![[10.0, 10.0, 80.0, 90.0, 0.95, 0.0]],
            (128, 128),
        ));
        let img = DynamicImage::new_rgb8(128, 128);
        let out = annotate_image(&img, &r, None);
        assert_eq!(out.width(), 128);
    }

    #[test]
    fn test_annotate_detection_unicode_label() {
        // Non-ASCII class name selects the unicode font branch.
        let names = HashMap::from([(0usize, "人".to_string())]);
        let mut r = base_results(names);
        r.boxes = Some(Boxes::new(
            array![[5.0, 5.0, 60.0, 60.0, 0.8, 0.0]],
            (128, 128),
        ));
        let out = annotate_image(&DynamicImage::new_rgb8(128, 128), &r, None);
        assert_eq!(out.height(), 128);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_annotate_pose_draws() {
        let mut r = base_results(HashMap::from([(0usize, "person".to_string())]));
        // One pose, 17 COCO keypoints with (x, y, conf), and the (x, y) layout of exports
        // without a visibility channel.
        for dims in [3, 2] {
            let mut kpt = Array3::<f32>::zeros((1, 17, dims));
            for k in 0..17 {
                kpt[[0, k, 0]] = 20.0 + k as f32;
                kpt[[0, k, 1]] = 30.0 + k as f32;
                if dims > 2 {
                    kpt[[0, k, 2]] = 0.9;
                }
            }
            r.keypoints = Some(Keypoints::new(kpt, (128, 128)));
            let out = annotate_image(&DynamicImage::new_rgb8(128, 128), &r, None);
            assert_ne!(
                out,
                DynamicImage::new_rgb8(128, 128),
                "nothing drawn for {dims} dims"
            );
        }
    }

    #[test]
    fn test_annotate_custom_keypoints() {
        let mut r = base_results(HashMap::from([(0usize, "tiger".to_string())]));
        // A 12-keypoint layout gets no lines by default; a custom skeleton connects points.
        let mut kpt = Array3::<f32>::zeros((1, 12, 3));
        kpt.slice_mut(ndarray::s![0, .., 2]).fill(0.9);
        kpt[[0, 0, 0]] = 20.0;
        kpt[[0, 0, 1]] = 64.0;
        kpt[[0, 1, 0]] = 100.0;
        kpt[[0, 1, 1]] = 64.0;
        r.keypoints = Some(Keypoints::new(kpt, (128, 128)));
        let img = DynamicImage::new_rgb8(128, 128);
        let mid = |options: &AnnotateOptions| {
            annotate_image_with_options(&img, &r, options)
                .to_rgb8()
                .get_pixel(60, 64)
                .0
        };
        assert_eq!(mid(&AnnotateOptions::default()), [0, 0, 0]);
        let options = AnnotateOptions {
            skeleton: Some(vec![[0, 1]]),
            kpt_color: Some(Color(255, 0, 0)),
            ..AnnotateOptions::default()
        };
        assert_eq!(mid(&options), [255, 0, 0]);
        let no_lines = AnnotateOptions {
            kpt_line: false,
            ..options
        };
        assert_eq!(mid(&no_lines), [0, 0, 0]);
    }

    #[test]
    fn test_annotate_obb_draws() {
        let mut r = base_results(HashMap::from([(0usize, "ship".to_string())]));
        r.obb = Some(Obb::new(
            array![[64.0, 64.0, 40.0, 20.0, 0.5, 0.9, 0.0]],
            (128, 128),
        ));
        let out = annotate_image(&DynamicImage::new_rgb8(128, 128), &r, None);
        assert_eq!(out.width(), 128);
    }

    #[test]
    fn test_annotate_classification_draws() {
        let mut r = base_results(HashMap::from([
            (0usize, "cat".to_string()),
            (1, "dog".to_string()),
        ]));
        r.probs = Some(Probs::new(array![0.2, 0.8]));
        let out = annotate_image(&DynamicImage::new_rgb8(128, 128), &r, Some(2));
        assert_eq!(out.width(), 128);
    }

    #[test]
    fn test_annotate_many_overlapping_boxes() {
        // Overlapping boxes exercise the label-collision avoidance and rect_intersect paths.
        let names = HashMap::from([(0usize, "person".to_string()), (1, "car".to_string())]);
        let mut r = base_results(names);
        r.boxes = Some(Boxes::new(
            array![
                [10.0, 10.0, 70.0, 70.0, 0.95, 0.0],
                [12.0, 12.0, 72.0, 72.0, 0.90, 1.0],
                [14.0, 8.0, 74.0, 68.0, 0.85, 0.0],
                [0.0, 0.0, 40.0, 40.0, 0.80, 1.0]
            ],
            (128, 128),
        ));
        let out = annotate_image(&DynamicImage::new_rgb8(128, 128), &r, None);
        assert_eq!(out.width(), 128);
    }

    #[test]
    fn test_annotate_classification_topk() {
        let names: HashMap<usize, String> = (0..6).map(|i| (i, format!("class{i}"))).collect();
        let mut r = base_results(names);
        r.probs = Some(Probs::new(array![0.05, 0.05, 0.1, 0.2, 0.3, 0.3]));
        // top_k larger than typical to drive the top-k label loop.
        let out = annotate_image(&DynamicImage::new_rgb8(200, 200), &r, Some(5));
        assert_eq!(out.height(), 200);
    }

    #[test]
    fn test_annotate_semantic_draws() {
        let mut r = base_results(HashMap::from([
            (0usize, "bg".to_string()),
            (1, "road".to_string()),
        ]));
        // The semantic mask is a per-pixel class map matching the image size.
        let mut mask_data = ndarray::Array2::<u16>::zeros((128, 128));
        mask_data[[10, 10]] = 1;
        r.semantic_mask = Some(SemanticMask::new(mask_data, (128, 128)));
        let out = annotate_image(&DynamicImage::new_rgb8(128, 128), &r, None);
        assert_eq!(out.width(), 128);
    }

    #[test]
    fn test_annotate_depth_blends_in_place() {
        let mut r = base_results(HashMap::new());
        let mut depth = ndarray::Array2::<f32>::zeros((128, 128));
        depth[[10, 10]] = 5.0;
        depth[[20, 20]] = 2.0;
        r.depth = Some(DepthMap::new(depth, (128, 128)));
        let out = annotate_image(&DynamicImage::new_rgb8(128, 128), &r, None);
        // Depth blends over the image, so the output keeps the input size.
        assert_eq!(out.width(), 128);
        assert_eq!(out.height(), 128);
        // A valid pixel picks up the colormap, so it no longer matches the black input.
        assert_ne!(out.to_rgb8().get_pixel(10, 10).0, [0, 0, 0]);
    }
}
