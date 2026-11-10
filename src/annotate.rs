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
use crate::visualizer::color::{Color, Colormap, DEPTH_ALPHA, DepthViz};
use crate::visualizer::skeleton::{KPT_COLOR_INDICES, LIMB_COLOR_INDICES, SKELETON};
use ab_glyph::{Font, FontRef, GlyphId, PxScale, ScaleFont, point};
use harfrust::{Buffer, Direction, Script, ShapeOptions, ShapePlan, ShapePlanKey, ShaperFont};
use image::{DynamicImage, Rgb};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::ops::Range;
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
#[allow(clippy::struct_excessive_bools)] // the Ultralytics `plot` switches
pub struct AnnotateOptions {
    /// Maximum number of classification labels to show.
    pub top_k: usize,
    /// Depth colormap; only depth results use it.
    pub colormap: Colormap,
    /// Depth normalization; only depth results use it.
    pub depth_viz: DepthViz,
    /// Depth overlay opacity in `0.0..=1.0` (see [`annotate_image_with`]).
    pub depth_alpha: f32,
    /// Box and outline width in pixels, which also sizes box labels (`line_width`). `None` takes
    /// 0.3% of the mean image side, at least 2, like Ultralytics.
    pub line_width: Option<u32>,
    /// Draw boxes and oriented boxes with their labels (`show_boxes`).
    pub show_boxes: bool,
    /// Put the class name in box labels (`show_labels`).
    pub show_labels: bool,
    /// Put the confidence in box labels (`show_conf`).
    pub show_conf: bool,
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
            line_width: None,
            show_boxes: true,
            show_labels: true,
            show_conf: true,
            kpt_radius: 5,
            kpt_line: true,
            kpt_conf: 0.25,
            kpt_color: None,
            skeleton: None,
        }
    }
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
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)]
pub fn annotate_image_with_options(
    image: &DynamicImage,
    result: &Results,
    options: &AnnotateOptions,
) -> DynamicImage {
    let mut img = image.to_rgb8();
    let (width, height) = img.dimensions();
    let side = f64::from(width) + f64::from(height);
    // Ultralytics sums the image shape with its 3 channels for the line width.
    let lw = options.line_width.map_or_else(
        || {
            (f64::midpoint(side, 3.0) * 0.003)
                .round_ties_even()
                .max(2.0) as i32
        },
        |lw| lw.min(i32::MAX as u32) as i32,
    );

    let box_labels = options.show_boxes
        && (options.show_labels || options.show_conf)
        && (result.boxes.is_some() || result.obb.is_some());
    let unicode =
        (box_labels || result.probs.is_some()) && result.names.values().any(|n| !n.is_ascii());
    // Like Ultralytics, classification and non-ASCII names use a TrueType font at 3.5% of the mean
    // side (PIL sizes are em sizes, `PxScale` is the line height), and other box labels are sized
    // to match the OpenCV Hershey text of the line width.
    let pil = unicode || result.probs.is_some();
    let font = font_bytes(unicode).and_then(|data| {
        let glyphs = FontRef::try_from_slice(data).ok()?;
        let em = (side / 2.0 * 0.035).round_ties_even().max(12.0) as f32;
        let scale = if pil {
            em * glyphs.height_unscaled() / glyphs.units_per_em()?
        } else {
            10.5 * lw as f32
        };
        Some(LabelFont {
            glyphs,
            face: harfrust::Font::new(data, 0)?,
            plans: RefCell::default(),
            scale: PxScale::from(scale),
            pil,
        })
    });

    draw_semantic_mask(&mut img, result);
    draw_detection(&mut img, result, font.as_ref(), options, lw);
    draw_pose(&mut img, result, options, lw);
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

/// Draw a line from `a` to `b` like `OpenCV`'s thick lines: every pixel within half the
/// `thickness`, rounded up, of the segment, so ends and corners come out round. A thickness of 1
/// draws a one-pixel line.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn draw_line(img: &mut image::RgbImage, a: [i32; 2], b: [i32; 2], color: Rgb<u8>, thickness: i32) {
    let half = if thickness > 1 {
        (thickness + 1) / 2
    } else {
        0
    };
    let reach = half as f32 + 0.5;
    let (width, height) = (img.width() as f32, img.height() as f32);
    let [ax, ay, bx, by] = [a[0], a[1], b[0], b[1]].map(|v| v as f32);
    let (dx, dy) = (bx - ax, by - ay);
    let len2 = dx.mul_add(dx, dy * dy);
    let rows = (ay.min(by) - reach).ceil().max(0.0) as u32
        ..((ay.max(by) + reach).floor() + 1.0).min(height) as u32;
    for py in rows {
        let fy = py as f32 - ay;
        // Columns within reach of the infinite line on this row, then of the segment itself.
        let (mut lo, mut hi) = (ax.min(bx) - reach, ax.max(bx) + reach);
        if dy != 0.0 {
            let (center, spread) = (fy * dx / dy, reach * len2.sqrt() / dy.abs());
            lo = lo.max(ax + center - spread);
            hi = hi.min(ax + center + spread);
        }
        for px in lo.ceil().max(0.0) as u32..(hi.floor() + 1.0).min(width) as u32 {
            let fx = px as f32 - ax;
            let t = if len2 > 0.0 {
                (fx.mul_add(dx, fy * dy) / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let (ex, ey) = (t.mul_add(-dx, fx), t.mul_add(-dy, fy));
            if ex.mul_add(ex, ey * ey) <= reach * reach {
                img.put_pixel(px, py, color);
            }
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
#[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
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
/// Each pixel is colorized with its class color ([`Color::for_class`]) blended 50/50 with
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
                let Color(r, g, b) = Color::for_class(&result.colors, raw as usize);
                for (channel, tint) in px.iter_mut().zip([r, g, b]) {
                    *channel = (*channel / 2).saturating_add(tint / 2);
                }
            }
        });
}

/// The label font: `harfrust` shapes text with it, reusing a plan per direction and script, and
/// `ab_glyph` draws the shaped glyphs.
struct LabelFont {
    glyphs: FontRef<'static>,
    face: harfrust::Font,
    plans: RefCell<Vec<ShapePlan>>,
    /// Text size.
    scale: PxScale,
    /// Size labels by their text like Ultralytics PIL text, instead of like its `OpenCV` text.
    pil: bool,
}

/// The runs of `text` in drawing order from left to right, as byte range, direction and script:
/// `unicode-bidi` splits it by direction, and each run splits by script.
fn runs(text: &str) -> Vec<(Range<usize>, Direction, Option<Script>)> {
    let bidi = unicode_bidi::BidiInfo::new(text, None);
    let mut out = Vec::new();
    for paragraph in &bidi.paragraphs {
        let (levels, runs) = bidi.visual_runs(paragraph, paragraph.range.clone());
        for run in runs {
            let mut pieces = vec![(run.clone(), None)];
            for (i, c) in text[run.clone()].char_indices() {
                let script = harfrust::unicode::script(c.into());
                // Spaces, digits, punctuation and marks take the script around them.
                if matches!(script, Script::COMMON | Script::INHERITED | Script::UNKNOWN) {
                    continue;
                }
                match pieces.last_mut() {
                    Some((_, current @ None)) => *current = Some(script),
                    Some((range, Some(current))) if *current != script => {
                        range.end = run.start + i;
                        pieces.push((run.start + i..run.end, Some(script)));
                    }
                    _ => {}
                }
            }
            let direction = if levels[run.start].is_rtl() {
                pieces.reverse();
                Direction::RightToLeft
            } else {
                Direction::LeftToRight
            };
            out.extend(
                pieces
                    .into_iter()
                    .map(|(range, script)| (range, direction, script)),
            );
        }
    }
    out
}

/// Shape `text` and lay it out from left to right: each glyph with its pen position in pixels,
/// and the advance width of the label. Each run is shaped in logical order with the text around
/// it as context, which joins Arabic letters, places marks and mirrors brackets.
fn layout(font: &LabelFont, scale: PxScale, text: &str) -> (Vec<(GlyphId, f64, f64)>, f64) {
    let px = f64::from(font.glyphs.as_scaled(scale).h_scale_factor());
    let shaper = ShaperFont::new(&font.face);
    let mut buffer = Buffer::new();
    let mut glyphs = Vec::new();
    let mut caret = 0.0;
    for (range, direction, script) in runs(text) {
        buffer.clear();
        buffer.push_str(&text[range.clone()]);
        buffer.set_pre_context(&text[..range.start]);
        buffer.set_post_context(&text[range.end..]);
        buffer.set_direction(direction);
        buffer.set_script(script);
        let key = ShapePlanKey::new(&font.face, script, direction);
        let mut plans = font.plans.borrow_mut();
        if !plans.iter().any(|plan| key.matches(plan)) {
            plans.push(ShapePlan::new(&font.face, direction, script, None, &[]));
        }
        let options = ShapeOptions::new().plan(plans.iter().find(|plan| key.matches(plan)));
        if harfrust::shape(&shaper, &mut buffer, options).is_err() {
            continue;
        }
        for (info, pos) in buffer.glyph_infos().iter().zip(buffer.glyph_positions()) {
            if let Ok(id) = u16::try_from(info.glyph_id) {
                let (dx, dy) = (f64::from(pos.x_offset), f64::from(pos.y_offset));
                glyphs.push((GlyphId(id), dx.mul_add(px, caret), -dy * px));
            }
            caret = f64::from(pos.x_advance).mul_add(px, caret);
        }
    }
    (glyphs, caret)
}

/// Glyph coverage rasterized once per frame, keyed by glyph, font size and the glyph's offset in
/// its label: labels such as `person 0.87` and `person 0.86` share every glyph but the last.
type GlyphCache = HashMap<(u16, u32, u32, u32), Option<(i32, i32, usize, Vec<f32>)>>;

/// Draw the `glyphs` of a [`layout`] at `(x, y)`, taking each glyph's coverage from `cache` when
/// an earlier label already rasterized it.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
fn draw_text(
    img: &mut image::RgbImage,
    color: Rgb<u8>,
    (x, y): (i32, i32),
    scale: PxScale,
    font: &FontRef,
    glyphs: &[(GlyphId, f64, f64)],
    cache: &mut GlyphCache,
) {
    let ascent = font.as_scaled(scale).ascent();
    let (width, height) = (i64::from(img.width()), i64::from(img.height()));
    for &(id, pen_x, pen_y) in glyphs {
        let at = point(pen_x as f32, pen_y as f32 + ascent);
        let key = (id.0, scale.y.to_bits(), at.x.to_bits(), at.y.to_bits());
        let glyph = cache.entry(key).or_insert_with(|| {
            let outlined = font.outline_glyph(id.with_scale_and_position(scale, at))?;
            let bounds = outlined.px_bounds();
            let w = bounds.width() as usize;
            let mut coverage = vec![0.0; w * bounds.height() as usize];
            outlined.draw(|gx, gy, v| coverage[gy as usize * w + gx as usize] = v);
            Some((bounds.min.x as i32, bounds.min.y as i32, w, coverage))
        });
        let Some((min_x, min_y, w, coverage)) = glyph else {
            continue;
        };
        for (row, line) in coverage.chunks((*w).max(1)).enumerate() {
            let py = i64::from(y + *min_y) + row as i64;
            for (col, &v) in line.iter().enumerate() {
                let px = i64::from(x + *min_x) + col as i64;
                if v > 0.0 && (0..width).contains(&px) && (0..height).contains(&py) {
                    let pixel = img.get_pixel_mut(px as u32, py as u32);
                    let v = v.clamp(0.0, 1.0);
                    *pixel = imageproc::pixelops::weighted_sum(*pixel, color, 1.0 - v, v);
                }
            }
        }
    }
}

/// The width and height of a [`layout`] like the box PIL measures for text: its advance and its
/// lowest glyph pixel below the top of the line.
#[allow(clippy::cast_possible_truncation)]
fn text_size(font: &LabelFont, glyphs: &[(GlyphId, f64, f64)], advance: f64) -> (i32, i32) {
    let ascent = font.glyphs.as_scaled(font.scale).ascent();
    let bottom = glyphs
        .iter()
        .filter_map(|&(id, x, y)| {
            let at = point(x as f32, y as f32 + ascent);
            font.glyphs
                .outline_glyph(id.with_scale_and_position(font.scale, at))
        })
        .fold(0.0, |bottom, glyph| glyph.px_bounds().max.y.max(bottom));
    (advance.ceil() as i32, bottom as i32)
}

/// Draw `label` on a `color` box at the top-left corner `(x, y)` of a shape like the Ultralytics
/// annotator: above the shape when it fits, else inside it, and moved left to stay in the image.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn draw_label(
    img: &mut image::RgbImage,
    font: &LabelFont,
    cache: &mut GlyphCache,
    label: &str,
    color: Rgb<u8>,
    lw: i32,
    (x, y): (i32, i32),
) {
    let (glyphs, advance) = layout(font, font.scale, label);
    // OpenCV text of line width `lw` is `9 * lw` tall, and Ultralytics pads it by 3.
    let (w, h) = if font.pil {
        text_size(font, &glyphs, advance)
    } else {
        (advance.round() as i32, 9 * lw + 3)
    };
    let outside = y >= h;
    let x = x.min(img.width() as i32 - w);
    let top = if outside { y - h } else { y };
    draw_transparent_rect(img, x, top, w + 1, h + 1, color, 1.0);
    // PIL text hangs from the top of the box; OpenCV text sits on a baseline 2 px above the box
    // bottom, or 1 px when inside.
    let ascent = font.glyphs.as_scaled(font.scale).ascent().round() as i32;
    let text_y = match (font.pil, outside) {
        (true, _) => top,
        (false, true) => y - 2 - ascent,
        (false, false) => y + h - 1 - ascent,
    };
    let text = Color(color[0], color[1], color[2]).text_color();
    let text = Rgb([text.0, text.1, text.2]);
    // OpenCV strokes its text `lw - 1` px thick, bolder than Arial, so the glyphs are drawn
    // again shifted by up to half the line width.
    let stroke = if font.pil { 1 } else { (lw + 1) / 2 };
    for (dx, dy) in (0..stroke).flat_map(|dx| (0..stroke).map(move |dy| (dx, dy))) {
        let at = (x + dx - (stroke - 1) / 2, text_y + dy - (stroke - 1) / 2);
        draw_text(img, text, at, font.scale, &font.glyphs, &glyphs, cache);
    }
}

/// Draw instance masks, then boxes or oriented boxes with their labels like the Ultralytics
/// annotator, from the last box to the first.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap
)]
fn draw_detection(
    img: &mut image::RgbImage,
    result: &Results,
    font: Option<&LabelFont>,
    options: &AnnotateOptions,
    lw: i32,
) {
    draw_masks(img, result);
    if !options.show_boxes {
        return;
    }
    // Each shape as its corners, class, confidence and track ID.
    let shapes: Vec<_> = if let Some(ref obb) = result.obb {
        let (c, cls, conf, id) = (obb.xyxyxyxy(), obb.cls(), obb.conf(), obb.id());
        (0..obb.len())
            .map(|i| {
                let corners = (0..4).map(|j| [c[[i, j, 0]], c[[i, j, 1]]]).collect();
                (corners, cls[i], conf[i], id.map(|id| id[i]))
            })
            .collect()
    } else if let Some(ref boxes) = result.boxes {
        let (b, cls, conf, id) = (boxes.xyxy(), boxes.cls(), boxes.conf(), boxes.id());
        (0..boxes.len())
            .map(|i| {
                let [x1, y1, x2, y2] = [b[[i, 0]], b[[i, 1]], b[[i, 2]], b[[i, 3]]];
                let corners = vec![[x1, y1], [x2, y1], [x2, y2], [x1, y2]];
                (corners, cls[i], conf[i], id.map(|id| id[i]))
            })
            .collect()
    } else {
        return;
    };

    let mut cache = GlyphCache::new();
    for (corners, cls, conf, id) in shapes.into_iter().rev() {
        let class_id = cls as usize;
        let Color(r, g, b) = Color::for_class(&result.colors, class_id);
        let color = Rgb([r, g, b]);
        let points: Vec<[i32; 2]> = corners.iter().map(|&[x, y]| [x as i32, y as i32]).collect();
        for (k, &a) in points.iter().enumerate() {
            draw_line(img, a, points[(k + 1) % points.len()], color, lw);
        }

        let name = result.names.get(&class_id).map_or("object", String::as_str);
        let name = id.map_or_else(|| name.to_string(), |id| format!("id:{} {name}", id as i64));
        let label = match (options.show_labels, options.show_conf) {
            (true, true) => format!("{name} {conf:.2}"),
            (true, false) => name,
            (false, true) => format!("{conf:.2}"),
            (false, false) => continue,
        };
        if let Some(font) = font {
            let corner = |k: usize| points.iter().map(|p| p[k]).min().unwrap_or(0).max(0);
            draw_label(
                img,
                font,
                &mut cache,
                &label,
                color,
                lw,
                (corner(0), corner(1)),
            );
        }
    }
}

/// Blend the instance masks over `img` at 50% opacity in their class colors.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn draw_masks(img: &mut image::RgbImage, result: &Results) {
    let (width, height) = img.dimensions();
    if let Some(ref boxes) = result.boxes {
        let xyxy = boxes.xyxy();
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
                let color = Color::for_class(&result.colors, class_id);
                let color = Rgb([color.0, color.1, color.2]);

                mask_present = true;

                // Original-size masks only fill their box; input-size ones are scaled to the
                // image first like the Ultralytics annotator, which can spill past the box.
                let plane = masks.data.index_axis(ndarray::Axis(0), i);
                let w = width as usize;
                let scaled;
                let (region, [x, y]) = if plane.dim() == (height as usize, w) {
                    let side = |v: f32, n: u32| v.max(0.0).min(n as f32).ceil() as usize;
                    let (x1, y1) = (side(xyxy[[i, 0]], width), side(xyxy[[i, 1]], height));
                    let (x2, y2) = (side(xyxy[[i, 2]], width), side(xyxy[[i, 3]], height));
                    (
                        plane.slice_move(ndarray::s![y1..y2.max(y1), x1..x2.max(x1)]),
                        [x1, y1],
                    )
                } else {
                    scaled = crate::postprocessing::scale_mask(&plane, (height, width));
                    (scaled.0.view(), scaled.1)
                };

                // Row slices instead of per-pixel `[[i, y, x]]` and `get_pixel_mut` indexing.
                let samples = overlay.as_flat_samples_mut().samples;
                for (dy, hits) in region.outer_iter().enumerate() {
                    let start = ((y + dy) * w + x) * 3;
                    let row = &mut samples[start..start + hits.len() * 3];
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
            let alpha = 0.5;
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
    }
}

/// Draw pose keypoints and the lines between them like the Ultralytics annotator, styled by the
/// keypoint fields of `options`, with lines half the line width `lw`.
#[allow(clippy::cast_possible_truncation)]
fn draw_pose(img: &mut image::RgbImage, result: &Results, options: &AnnotateOptions, lw: i32) {
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
        // human skeleton; other layouts take one class palette color per keypoint and no lines,
        // and a custom skeleton colors each line like its start keypoint.
        let coco = n_kpts == KPT_COLOR_INDICES.len();
        let skeleton: &[[usize; 2]] = match &options.skeleton {
            Some(skeleton) => skeleton,
            None if coco => &SKELETON,
            None => &[],
        };
        let kpt_color = |k: usize| {
            options.kpt_color.unwrap_or_else(|| {
                if coco {
                    Color::from_pose_index(KPT_COLOR_INDICES[k])
                } else {
                    Color::from_index(k)
                }
            })
        };
        let limb_color = |l: usize, start: usize| match options.kpt_color {
            None if coco && options.skeleton.is_none() => {
                Color::from_pose_index(LIMB_COLOR_INDICES[l])
            }
            _ => kpt_color(start),
        };

        let point = |p: usize, k: usize| [kpt_data[[p, k, 0]] as i32, kpt_data[[p, k, 1]] as i32];

        for person_idx in (0..n_persons).rev() {
            for kpt_idx in 0..n_kpts {
                if visible(person_idx, kpt_idx) {
                    let [x, y] = point(person_idx, kpt_idx);
                    let c = kpt_color(kpt_idx);
                    draw_filled_circle(img, x, y, options.kpt_radius, Rgb([c.0, c.1, c.2]));
                }
            }
            if options.kpt_line {
                for (limb_idx, &[kpt_a, kpt_b]) in skeleton.iter().enumerate() {
                    if kpt_a >= n_kpts
                        || kpt_b >= n_kpts
                        || !visible(person_idx, kpt_a)
                        || !visible(person_idx, kpt_b)
                    {
                        continue;
                    }
                    let (a, b) = (point(person_idx, kpt_a), point(person_idx, kpt_b));
                    if a.into_iter().chain(b).min() < Some(0) {
                        continue;
                    }
                    let c = limb_color(limb_idx, kpt_a);
                    draw_line(img, a, b, Rgb([c.0, c.1, c.2]), (lw + 1) / 2);
                }
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
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::manual_clamp
)]
fn draw_transparent_rect(
    img: &mut image::RgbImage,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    color: Rgb<u8>,
    alpha: f32,
) {
    let (width, height) = img.dimensions();
    let alpha = alpha.max(0.0).min(1.0);
    let inv_alpha = 1.0 - alpha;
    let tint = color.0.map(|c| f32::from(c) * alpha);

    // Clamp the rectangle to the image once, then blend whole row slices.
    let (x0, x1) = (x.clamp(0, width as i32), (x + w).clamp(0, width as i32));
    let (y0, y1) = (y.clamp(0, height as i32), (y + h).clamp(0, height as i32));
    let width = width as usize;
    let samples = img.as_flat_samples_mut().samples;
    for py in y0 as usize..y1 as usize {
        let row = &mut samples[(py * width + x0 as usize) * 3..(py * width + x1 as usize) * 3];
        let (pixels, _) = row.as_chunks_mut::<3>();
        for pixel in pixels {
            for (channel, tint) in pixel.iter_mut().zip(tint) {
                *channel = f32::from(*channel).mul_add(inv_alpha, tint + 0.5) as u8;
            }
        }
    }
}

/// Draw the top-k classification probabilities like the Ultralytics annotator: one `name score`
/// line per class from 3% of the height in from the top-left corner, each on a gray box at half
/// opacity.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn draw_classification(
    img: &mut image::RgbImage,
    result: &Results,
    font: Option<&LabelFont>,
    top_k: usize,
) {
    let (Some(probs), Some(font)) = (result.probs.as_ref(), font) else {
        return;
    };
    let x = (f64::from(img.height()) * 0.03).round_ties_even() as i32;
    let mut y = x;
    let mut cache = GlyphCache::new();
    for class_id in probs.top_k(top_k) {
        let name = result
            .names
            .get(&class_id)
            .map_or_else(|| class_id.to_string(), Clone::clone);
        let text = format!("{name} {:.2}", probs.data[class_id]);
        let (glyphs, advance) = layout(font, font.scale, &text);
        let (w, h) = text_size(font, &glyphs, advance);
        let gray = Rgb([64, 64, 64]);
        draw_transparent_rect(img, x, y, w + 2, h + 2, gray, 128.0 / 255.0);
        let white = Rgb([255, 255, 255]);
        draw_text(
            img,
            white,
            (x, y),
            font.scale,
            &font.glyphs,
            &glyphs,
            &mut cache,
        );
        y += h;
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
    fn test_layout_orders_and_shapes_rtl_labels() {
        // Each run's text in drawing order, right-to-left runs reversed.
        let visual = |text: &str| -> String {
            let runs = runs(text)
                .into_iter()
                .map(|(range, direction, _)| match direction {
                    Direction::RightToLeft => text[range].chars().rev().collect(),
                    _ => text[range].to_string(),
                });
            runs.collect()
        };
        assert_eq!(visual(" person 0.73 "), " person 0.73 ");
        assert_eq!(visual(" כלב (dog) 0.64 "), " 0.64 )dog( בלכ ");
        assert_eq!(visual("COVID مريض 0.55"), "COVID 0.55 ضيرم");
        // Hebrew and Arabic in one right-to-left run are shaped as separate scripts.
        assert_eq!(runs("שלום سلام").len(), 2);

        // With the font cached: Arabic letters join, lam-alef joins across a vowel, and shin and
        // sin dots differ.
        let Some(data) = font_bytes(true) else {
            return;
        };
        let font = LabelFont {
            glyphs: FontRef::try_from_slice(data).unwrap(),
            face: harfrust::Font::new(data, 0).unwrap(),
            plans: RefCell::default(),
            scale: PxScale::from(20.0),
            pil: true,
        };
        let ids = |text| -> Vec<u16> {
            let (glyphs, _) = layout(&font, PxScale::from(20.0), text);
            glyphs.iter().map(|g| g.0.0).collect()
        };
        let cmap =
            |text: &str| -> Vec<u16> { text.chars().map(|c| font.glyphs.glyph_id(c).0).collect() };
        assert_eq!(ids("شخص"), cmap("\u{FEBA}\u{FEA8}\u{FEB7}"));
        assert_eq!(ids("\u{644}\u{64E}\u{627}"), cmap("\u{64E}\u{FEFB}"));
        assert_eq!(
            ids("\u{5E9}\u{5C1} \u{5E9}\u{5C2}"),
            cmap("\u{FB2B} \u{FB2A}")
        );
    }

    #[test]
    fn test_draw_line_is_as_thick_as_opencv() {
        let mut img = image::RgbImage::new(100, 100);
        let color = Rgb([255, 0, 0]);
        // OpenCV covers 1, 3, 5 and 5 rows for thickness 1 to 4.
        for (thickness, rows) in [(1, 1), (2, 3), (3, 5), (4, 5)] {
            img.fill(0);
            draw_line(&mut img, [10, 50], [60, 50], color, thickness);
            let drawn = (0..100).filter(|&y| *img.get_pixel(30, y) == color).count();
            assert_eq!(drawn, rows, "thickness {thickness}");
        }
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
        // A custom skeleton line takes its start keypoint's class palette color, in true RGB.
        let options = AnnotateOptions {
            skeleton: Some(vec![[0, 1]]),
            ..AnnotateOptions::default()
        };
        assert_eq!(mid(&options), [4, 42, 255]);
        let options = AnnotateOptions {
            kpt_color: Some(Color(255, 0, 0)),
            ..options
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
