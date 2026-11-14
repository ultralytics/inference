// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! JS-facing results payload and its mapping from the core [`Results`].
//!
//! Mirrors the Ultralytics `Results` API field-for-field so the TypeScript
//! wrapper sees a plain serializable object. Per-task fields are empty/`null`
//! when unused.

use ndarray::{ArrayView1, ArrayView2, CowArray, Ix2, s};
use serde::Serialize;

use ultralytics_inference::Task;
use ultralytics_inference::results::{Results, SemanticMask};
use ultralytics_inference::visualizer::color::{Color, Colormap, DepthViz};

/// One detected box, mirroring `Boxes` in the Ultralytics API (pixel `xyxy`).
/// `color` is the class color (`#rrggbb`).
#[derive(Serialize)]
struct JsBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    /// Track id, only when tracking.
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<u32>,
    conf: f32,
    cls: usize,
    name: String,
    color: String,
}

/// One oriented box (`xywhr` + score), mirroring `Obb`.
#[derive(Serialize)]
struct JsObb {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    angle: f32,
    /// Track id, only when tracking.
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<u32>,
    conf: f32,
    cls: usize,
    name: String,
    color: String,
}

/// Keypoints for one detection (`[[x, y, conf], ...]`), with the instance color.
#[derive(Serialize)]
struct JsKeypoints {
    points: Vec<[f32; 3]>,
    color: String,
}

/// Per-stage timing in ms, mirroring `Speed` (preprocess / inference+sync / postprocess+serialization).
#[derive(Serialize)]
struct JsSpeed {
    preprocess: f64,
    inference: f64,
    postprocess: f64,
}

/// Classification probabilities, mirroring `Probs`.
#[derive(Serialize)]
struct JsProbs {
    top1: usize,
    top5: Vec<usize>,
    top1conf: f32,
    top5conf: Vec<f32>,
    name: String,
    /// Display names for each `top5` class, so `annotate` can draw the full list.
    top5names: Vec<String>,
    color: String,
}

/// JS-facing results payload, mirroring the Ultralytics `Results` API. Per-task
/// fields are empty/`null` when unused.
#[derive(Serialize)]
pub(crate) struct JsResults {
    task: String,
    width: u32,
    height: u32,
    boxes: Vec<JsBox>,
    obb: Vec<JsObb>,
    keypoints: Vec<JsKeypoints>,
    probs: Option<JsProbs>,
    /// Segment or semantic masks as a translucent colored `RGBA` overlay
    /// (`width*height*4`); empty for other tasks. A `Uint8Array`, drawable straight onto a canvas.
    #[serde(with = "serde_bytes")]
    masks: Vec<u8>,
    /// Semantic class id per pixel, little-endian `u16` (`width*height*2` bytes),
    /// row-major; empty for other tasks. The TS wrapper reinterprets it as a
    /// `Uint16Array`. The IGNORE sentinel `65535` marks background or
    /// class-filtered pixels.
    #[serde(with = "serde_bytes")]
    semantic_mask: Vec<u8>,
    /// Depth map as an opaque `RGBA` overlay (`width*height*4`), colorized with the
    /// requested colormap over valid (`>0`) pixels; empty for other tasks. A
    /// `Uint8Array`, drawable straight onto a canvas to blend over the frame.
    #[serde(with = "serde_bytes")]
    depth: Vec<u8>,
    /// Depth range `[min, max]` in meters over valid pixels; `None` for other tasks.
    depth_range: Option<[f32; 2]>,
    /// Per-stage timing in ms: `{ preprocess, inference, postprocess }`.
    speed: JsSpeed,
}

impl JsResults {
    /// Convert core [`Results`] into the serializable JS payload, labeling it with
    /// the model's declared `task` (the caller already holds it, so there is no
    /// need to guess it back from which result fields are populated).
    pub(crate) fn from_results(
        r: &Results,
        task: Task,
        colormap: Colormap,
        viz: DepthViz,
        overlay: Vec<u8>,
    ) -> Self {
        // Class id -> display name and class color, shared by every detection type.
        let name = |c: usize| r.names.get(&c).cloned().unwrap_or_default();
        let hex = |c: usize| Color::for_class(&r.colors, c).to_hex();

        let boxes = r.boxes.as_ref().map_or_else(Vec::new, |b| {
            let (xyxy, id, conf, cls) = (b.xyxy(), b.id(), b.conf(), b.cls());
            (0..b.len())
                .map(|i| {
                    let c = cls[i] as usize;
                    JsBox {
                        x1: xyxy[[i, 0]],
                        y1: xyxy[[i, 1]],
                        x2: xyxy[[i, 2]],
                        y2: xyxy[[i, 3]],
                        id: id.map(|id| id[i] as u32),
                        conf: conf[i],
                        cls: c,
                        name: name(c),
                        color: hex(c),
                    }
                })
                .collect()
        });

        let obb = r.obb.as_ref().map_or_else(Vec::new, |o| {
            let (xywhr, id, conf, cls) = (o.xywhr(), o.id(), o.conf(), o.cls());
            (0..conf.len())
                .map(|i| {
                    let c = cls[i] as usize;
                    JsObb {
                        x: xywhr[[i, 0]],
                        y: xywhr[[i, 1]],
                        w: xywhr[[i, 2]],
                        h: xywhr[[i, 3]],
                        angle: xywhr[[i, 4]],
                        id: id.map(|id| id[i] as u32),
                        conf: conf[i],
                        cls: c,
                        name: name(c),
                        color: hex(c),
                    }
                })
                .collect()
        });

        let keypoints = r.keypoints.as_ref().map_or_else(Vec::new, |k| {
            let (n, npt) = (k.data.shape()[0], k.data.shape()[1]);
            // A 2-channel export carries no visibility score and marks a missing point with
            // (0, 0), like Ultralytics, so only that point counts as not seen.
            let has_conf = k.data.shape()[2] > 2;
            (0..n)
                .map(|i| JsKeypoints {
                    points: (0..npt)
                        .map(|j| {
                            let (x, y) = (k.data[[i, j, 0]], k.data[[i, j, 1]]);
                            let conf = if has_conf {
                                k.data[[i, j, 2]]
                            } else if x != 0.0 || y != 0.0 {
                                1.0
                            } else {
                                0.0
                            };
                            [x, y, conf]
                        })
                        .collect(),
                    // Skeleton uses the matching detection's color.
                    color: boxes.get(i).map_or_else(|| hex(0), |b| b.color.clone()),
                })
                .collect()
        });

        let probs = r.probs.as_ref().map(|p| JsProbs {
            top1: p.top1(),
            top5: p.top5(),
            top1conf: p.top1conf(),
            top5conf: p.top5conf(),
            name: name(p.top1()),
            top5names: p.top5().into_iter().map(name).collect(),
            color: hex(p.top1()),
        });

        Self {
            task: task.as_str().to_owned(),
            width: r.orig_shape.1,
            height: r.orig_shape.0,
            boxes,
            obb,
            keypoints,
            probs,
            masks: if overlay.is_empty() {
                build_semantic_overlay(r)
            } else {
                overlay
            },
            semantic_mask: r.semantic_mask.as_ref().map_or_else(Vec::new, |sem| {
                let mut bytes = Vec::with_capacity(sem.data.len() * 2);
                for &v in &sem.data {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                bytes
            }),
            depth: build_depth_overlay(r, colormap, viz),
            depth_range: r
                .depth
                .as_ref()
                .and_then(|d| Some([d.min_depth()?, d.max_depth()?])),
            speed: JsSpeed {
                preprocess: r.speed.preprocess.unwrap_or(0.0),
                inference: r.speed.inference.unwrap_or(0.0),
                postprocess: r.speed.postprocess.unwrap_or(0.0),
            },
        }
    }
}

/// Write one colored RGBA pixel at `(x, y)` into a `width`-wide buffer.
fn put(buf: &mut [u8], w: usize, x: usize, y: usize, c: Color, a: u8) {
    let i = (y * w + x) * 4;
    buf[i..i + 4].copy_from_slice(&[c.0, c.1, c.2, a]);
}

/// The part of one instance mask inside its box, borrowed while drawing it straight away and
/// copied only to wait for tracking to pick the rows to draw.
pub(crate) struct InstanceMask<'a> {
    /// Image `(height, width)`.
    size: (usize, usize),
    /// Top left `(x, y)` of the box.
    corner: (usize, usize),
    inside: CowArray<'a, f32, Ix2>,
    color: Color,
}

impl<'a> InstanceMask<'a> {
    /// The part of `mask` inside the box of `box_data`, in its class color (`colors` being the
    /// model's `colors` metadata).
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub(crate) fn new(
        mask: ArrayView2<'a, f32>,
        box_data: ArrayView1<'_, f32>,
        colors: &[Color],
    ) -> Self {
        let (h, w) = mask.dim();
        let x1 = (box_data[2].max(0.0) as usize).saturating_add(1).min(w);
        let y1 = (box_data[3].max(0.0) as usize).saturating_add(1).min(h);
        let x0 = (box_data[0].max(0.0) as usize).min(x1);
        let y0 = (box_data[1].max(0.0) as usize).min(y1);
        Self {
            size: (h, w),
            corner: (x0, y0),
            inside: mask.slice_move(s![y0..y1, x0..x1]).into(),
            color: Color::for_class(colors, box_data[5] as usize),
        }
    }

    /// The mask with its own copy of the pixels, to draw after tracking.
    pub(crate) fn into_owned(self) -> InstanceMask<'static> {
        InstanceMask {
            inside: self.inside.into_owned().into(),
            ..self
        }
    }

    /// Composite the mask into the browser overlay, over the masks drawn before it.
    pub(crate) fn draw(&self, overlay: &mut Vec<u8>) {
        let ((h, w), (x0, y0)) = (self.size, self.corner);
        if overlay.is_empty() {
            overlay.resize(h * w * 4, 0);
        }
        for ((y, x), &v) in self.inside.indexed_iter() {
            if v > 0.5 {
                put(overlay, w, x0 + x, y0 + y, self.color, 128);
            }
        }
    }
}

/// Build a translucent colored RGBA overlay (`width*height*4`) from the semantic
/// class map, in the class colors.
/// Empty for other tasks or if the mask resolution does not match the image.
fn build_semantic_overlay(r: &Results) -> Vec<u8> {
    let (h, w) = (r.orig_shape.0 as usize, r.orig_shape.1 as usize);

    // Semantic: per-pixel class map, blended 50/50 like the native renderer.
    if let Some(sem) = &r.semantic_mask {
        if sem.data.dim() != (h, w) {
            return Vec::new();
        }
        let mut buf = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                // Filtered-out classes carry the IGNORE sentinel; leave transparent.
                let cls = sem.data[[y, x]];
                if cls != SemanticMask::IGNORE {
                    let color = Color::for_class(&r.colors, cls as usize);
                    put(&mut buf, w, x, y, color, 128);
                }
            }
        }
        return buf;
    }

    Vec::new()
}

/// Colorize the depth map into an RGBA overlay (`width*height*4`) with the given
/// `colormap` and normalization `viz`; invalid pixels are black. Empty when the result has
/// no depth map or its resolution does not match the image.
///
/// Fully opaque (`alpha = 255`): the blend opacity is applied by the consumer at draw
/// time (`annotate` composites at `0.6` to match native, the demo exposes a slider), so
/// this stays the raw colorized map and can be shown at full opacity.
fn build_depth_overlay(r: &Results, colormap: Colormap, viz: DepthViz) -> Vec<u8> {
    let Some(depth) = &r.depth else {
        return Vec::new();
    };
    let (h, w) = (r.orig_shape.0 as usize, r.orig_shape.1 as usize);
    if depth.data.dim() != (h, w) {
        return Vec::new();
    }
    let mut buf = vec![0u8; w * h * 4];
    // `buf` is exactly `w * h` RGBA pixels, so the trailing partial chunk is always empty.
    let (pixels, _) = buf.as_chunks_mut::<4>();
    for (px, rgb) in pixels.iter_mut().zip(depth.colorize(colormap, viz)) {
        *px = [rgb[0], rgb[1], rgb[2], 255];
    }
    buf
}
