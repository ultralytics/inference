// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! `ReID` appearance features from an ONNX embedding model like `yolo26n-reid.onnx`, the
//! Ultralytics `ReID` encoder for exported models.

use std::path::Path;

use ndarray::{Array3, Array4, Axis};
use ort::session::Session;
use ort::value::{TensorRef, ValueType};

use crate::error::{InferenceError, Result};
use crate::model::YOLOModel;
#[allow(clippy::wildcard_imports)]
use crate::parallel::*;
use crate::utils::xywh_to_xyxy;

/// The `ReID` model and the crop size it takes.
pub(super) struct Reid {
    session: Session,
    /// Crop `(height, width)`: 224 square unless the model fixes it, like Ultralytics.
    size: (usize, usize),
    /// Batch size of a model exported with a fixed one.
    batch: Option<usize>,
}

impl Reid {
    /// Load `path` (`auto` is `yolo26n-reid.onnx`) on the device of `detector`.
    pub(super) fn new(detector: &YOLOModel, path: &str) -> Result<Self> {
        let path = if path == "auto" {
            "yolo26n-reid.onnx"
        } else {
            path
        };
        let session = detector.companion_session(Path::new(path))?;
        let dims: Vec<Option<usize>> = match session.inputs().first().map(ort::value::Outlet::dtype)
        {
            Some(ValueType::Tensor { shape, .. }) if shape.len() == 4 => shape
                .iter()
                .map(|&d| usize::try_from(d).ok().filter(|&d| d > 0))
                .collect(),
            _ => vec![None; 4],
        };
        let h = dims[2].unwrap_or(224);
        let (size, batch) = ((h, dims[3].unwrap_or(h)), dims[0]);
        Ok(Self {
            session,
            size,
            batch,
        })
    }

    /// One feature per `[x, y, w, h]` box of `img` (RGB, HWC), `None` where the crop is empty.
    pub(super) fn embed(
        &mut self,
        img: &Array3<u8>,
        boxes: &[[f32; 4]],
    ) -> Result<Vec<Option<Vec<f32>>>> {
        let crops: Vec<Option<[usize; 4]>> = boxes.iter().map(|b| crop(b, img.dim())).collect();
        let valid: Vec<[usize; 4]> = crops.iter().flatten().copied().collect();
        let n = valid.len();
        if n == 0 {
            return Ok(vec![None; boxes.len()]);
        }
        let (h, w) = self.size;
        let mut batch = Array4::<f32>::zeros((n, 3, h, w));
        let rgb = img.as_standard_layout();
        let rgb = (rgb.as_slice().unwrap_or_default(), img.dim().1 * 3);
        resize(
            rgb,
            &valid,
            (h, w),
            batch.as_slice_mut().unwrap_or_default(),
        );
        // A fixed-batch model runs in chunks, the last one padded with its last crop.
        let bs = self.batch.unwrap_or(n);
        let mut feats = Vec::with_capacity(n);
        for start in (0..n).step_by(bs) {
            if bs == n {
                feats.extend(self.run(&batch)?);
            } else {
                let chunk = Array4::from_shape_fn((bs, 3, h, w), |(k, c, y, x)| {
                    batch[[(start + k).min(n - 1), c, y, x]]
                });
                feats.extend(self.run(&chunk)?.into_iter().take(n - start));
            }
        }
        let mut feats = feats.into_iter();
        Ok(crops.iter().map(|c| c.and_then(|_| feats.next())).collect())
    }

    /// Run a batch, returning one feature per image.
    fn run(&mut self, batch: &Array4<f32>) -> Result<Vec<Vec<f32>>> {
        let fail = |e: ort::Error| InferenceError::InferenceError(format!("ReID: {e}"));
        let images = batch.len_of(Axis(0)).max(1);
        let input = TensorRef::from_array_view(batch.view()).map_err(fail)?;
        let outputs = self.session.run(ort::inputs![input]).map_err(fail)?;
        let (_, data) = outputs[0].try_extract_tensor::<f32>().map_err(fail)?;
        let dim = data.len() / images;
        Ok(data.chunks(dim.max(1)).map(<[f32]>::to_vec).collect())
    }
}

/// Crop `[x1, y1, x2, y2]` of a center box in an image of `(height, width, _)`, grown by 2%
/// and 10 pixels like Ultralytics `save_one_box`, or `None` when it is empty.
fn crop(&[x, y, w, h]: &[f32; 4], (ih, iw, _): (usize, usize, usize)) -> Option<[usize; 4]> {
    let [x1, y1, x2, y2] = xywh_to_xyxy(x, y, w, h);
    let (cx, cy) = ((x1 + x2) / 2.0, (y1 + y2) / 2.0);
    let (w, h) = ((x2 - x1) * 1.02 + 10.0, (y2 - y1) * 1.02 + 10.0);
    // Truncate toward zero, then clip to the image.
    let clip = |v: f32, n: usize| (v.trunc() as i64).clamp(0, n as i64) as usize;
    let [x1, y1, x2, y2] = xywh_to_xyxy(cx, cy, w, h);
    let b = [clip(x1, iw), clip(y1, ih), clip(x2, iw), clip(y2, ih)];
    (b[2] > b[0] && b[3] > b[1]).then_some(b)
}

/// Resize each `[x1, y1, x2, y2]` crop of an RGB image (bytes, row stride) into `out`, one
/// CHW `h x w` image of 0-1 values per crop, bilinear like
/// `torch.nn.functional.interpolate(align_corners=False)`. Rows run in parallel.
fn resize(
    (rgb, stride): (&[u8], usize),
    crops: &[[usize; 4]],
    (h, w): (usize, usize),
    out: &mut [f32],
) {
    let unit: [f32; 256] = std::array::from_fn(|v| v as f32 / 255.0);
    // Source pixels and weight of each output row and column.
    let taps = |n: usize, m: usize| -> Vec<(usize, usize, f32)> {
        let scale = n as f32 / m as f32;
        (0..m)
            .map(|i| {
                let f = scale.mul_add(i as f32 + 0.5, -0.5).max(0.0);
                let i0 = (f as usize).min(n - 1);
                (i0, (i0 + 1).min(n - 1), f - i0 as f32)
            })
            .collect()
    };
    let taps: Vec<_> = crops
        .iter()
        .map(|&[x1, y1, x2, y2]| (taps(y2 - y1, h), taps(x2 - x1, w)))
        .collect();
    out.par_chunks_mut(w).enumerate().for_each(|(r, out)| {
        let (k, c, y) = (r / (3 * h), r / h % 3, r % h);
        let ([x1, y1, ..], (ys, xs)) = (crops[k], &taps[k]);
        let (y0, y1b, ly) = ys[y];
        let row = |y: usize| &rgb[(y1 + y) * stride + x1 * 3..];
        let (r0, r1) = (row(y0), row(y1b));
        for (o, &(x0, x1, lx)) in out.iter_mut().zip(xs) {
            let p = |r: &[u8], x: usize| unit[usize::from(r[x * 3 + c])];
            let top = (1.0 - lx).mul_add(p(r0, x0), lx * p(r0, x1));
            let bottom = (1.0 - lx).mul_add(p(r1, x0), lx * p(r1, x1));
            *o = (1.0 - ly).mul_add(top, ly * bottom);
        }
    });
}
