// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! JS-facing results payload and its mapping from the core [`Results`].
//!
//! Mirrors the Ultralytics `Results` API field-for-field so the TypeScript
//! wrapper sees a plain serializable object. Per-task fields are empty/`null`
//! when unused. The only entry points the bindings need are [`JsResults`] and
//! [`JsResults::from_results`]; every other type here is an internal detail of
//! that mapping.

use serde::Serialize;

use ultralytics_inference::Task;
use ultralytics_inference::results::{Results, SemanticMask};
use ultralytics_inference::visualizer::color::{Color, inferno};

/// One detected box, mirroring `Boxes` in the Ultralytics API (pixel `xyxy`).
/// `color` is the Ultralytics palette color for the class (`#rrggbb`).
#[derive(Serialize)]
struct JsBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
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

/// Per-stage timing in ms, mirroring `Speed` (decode+letterbox / inference+sync / decode+NMS).
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
    /// Segment masks as a translucent colored `RGBA` overlay (`width*height*4`);
    /// empty for other tasks. A `Uint8Array`, drawable straight onto a canvas.
    #[serde(with = "serde_bytes")]
    masks: Vec<u8>,
    /// Semantic class id per pixel, little-endian `u16` (`width*height*2` bytes),
    /// row-major; empty for other tasks. The TS wrapper reinterprets it as a
    /// `Uint16Array`. The IGNORE sentinel `65535` marks background or
    /// class-filtered pixels.
    #[serde(with = "serde_bytes")]
    semantic_mask: Vec<u8>,
    /// Depth map as an opaque `RGBA` image (`width*height*4`), colorized with the
    /// INFERNO colormap over valid (`>0`) pixels; empty for other tasks. A
    /// `Uint8Array`, drawable straight onto a canvas.
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
    pub(crate) fn from_results(r: &Results, task: Task) -> Self {
        // Class id -> display name and palette color, shared by every detection type.
        let name = |c: usize| r.names.get(&c).cloned().unwrap_or_default();
        let hex = |c: usize| Color::from_index(c).to_hex();

        let boxes = r.boxes.as_ref().map_or_else(Vec::new, |b| {
            let (xyxy, conf, cls) = (b.xyxy(), b.conf(), b.cls());
            (0..b.len())
                .map(|i| {
                    let c = cls[i] as usize;
                    JsBox {
                        x1: xyxy[[i, 0]],
                        y1: xyxy[[i, 1]],
                        x2: xyxy[[i, 2]],
                        y2: xyxy[[i, 3]],
                        conf: conf[i],
                        cls: c,
                        name: name(c),
                        color: hex(c),
                    }
                })
                .collect()
        });

        let obb = r.obb.as_ref().map_or_else(Vec::new, |o| {
            let (xywhr, conf, cls) = (o.xywhr(), o.conf(), o.cls());
            (0..conf.len())
                .map(|i| {
                    let c = cls[i] as usize;
                    JsObb {
                        x: xywhr[[i, 0]],
                        y: xywhr[[i, 1]],
                        w: xywhr[[i, 2]],
                        h: xywhr[[i, 3]],
                        angle: xywhr[[i, 4]],
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
            (0..n)
                .map(|i| JsKeypoints {
                    points: (0..npt)
                        .map(|j| [k.data[[i, j, 0]], k.data[[i, j, 1]], k.data[[i, j, 2]]])
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
            masks: build_mask_overlay(r),
            semantic_mask: r.semantic_mask.as_ref().map_or_else(Vec::new, |sem| {
                let mut bytes = Vec::with_capacity(sem.data.len() * 2);
                for &v in &sem.data {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                bytes
            }),
            depth: build_depth_overlay(r),
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

/// Build a translucent colored RGBA overlay (`width*height*4`) from the segment
/// instance masks or the semantic class map, colored by the class palette.
/// Empty for other tasks or if the mask resolution does not match the image.
fn build_mask_overlay(r: &Results) -> Vec<u8> {
    let (h, w) = (r.orig_shape.0 as usize, r.orig_shape.1 as usize);
    let mut buf = vec![0u8; w * h * 4];

    // Segment: per-instance binary masks, each colored by its detection class.
    if let (Some(masks), Some(boxes)) = (&r.masks, &r.boxes) {
        let (n, mh, mw) = masks.data.dim();
        if mh != h || mw != w {
            return Vec::new();
        }
        let cls = boxes.cls();
        for i in 0..n.min(cls.len()) {
            let c = Color::from_index(cls[i] as usize);
            for y in 0..h {
                for x in 0..w {
                    if masks.data[[i, y, x]] > 0.5 {
                        put(&mut buf, w, x, y, c, 140);
                    }
                }
            }
        }
        return buf;
    }

    // Semantic: per-pixel class map, blended 50/50 like the native renderer.
    if let Some(sem) = &r.semantic_mask {
        if sem.data.dim() != (h, w) {
            return Vec::new();
        }
        for y in 0..h {
            for x in 0..w {
                // Filtered-out classes carry the IGNORE sentinel; leave transparent.
                let cls = sem.data[[y, x]];
                if cls != SemanticMask::IGNORE {
                    put(&mut buf, w, x, y, Color::from_index(cls as usize), 128);
                }
            }
        }
        return buf;
    }

    Vec::new()
}

/// Colorize the depth map into an opaque RGBA image (`width*height*4`) with the INFERNO
/// colormap, min/max-normalized over valid (`>0`) pixels; invalid pixels are black. Empty
/// when the result has no depth map or its resolution does not match the image.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
fn build_depth_overlay(r: &Results) -> Vec<u8> {
    let Some(depth) = &r.depth else {
        return Vec::new();
    };
    let (h, w) = (r.orig_shape.0 as usize, r.orig_shape.1 as usize);
    if depth.data.dim() != (h, w) {
        return Vec::new();
    }
    let (vmin, vmax) = match (depth.min_depth(), depth.max_depth()) {
        (Some(lo), Some(hi)) if hi > lo => (lo, hi),
        (Some(lo), _) => (lo, lo + 1e-6),
        _ => (0.0, 1.0),
    };
    let inv = 1.0 / (vmax - vmin);
    let mut buf = vec![0u8; w * h * 4];
    for (px, &d) in buf.chunks_exact_mut(4).zip(depth.data.iter()) {
        if d > 0.0 {
            let c = inferno((d - vmin) * inv);
            px.copy_from_slice(&[c[0], c[1], c[2], 255]);
        } else {
            px[3] = 255; // opaque black for invalid pixels
        }
    }
    buf
}
