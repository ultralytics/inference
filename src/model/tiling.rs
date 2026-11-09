// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Tiled prediction, the Rust port of the Ultralytics `tile` argument.

// Pixel coordinates and tile counts are far below the f32 and u32 limits.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp
)]

use image::{DynamicImage, GenericImageView};
use ndarray::{Array, Array2, Array3, Axis, RemoveAxis, aview1, s};

use super::YOLOModel;
use crate::error::Result;
use crate::inference::InferenceConfig;
use crate::results::{Boxes, DepthMap, Keypoints, Masks, Obb, Probs, Results, SemanticMask, Speed};
use crate::task::Task;
use crate::utils::{nms_per_class, nms_rotated_per_class};

/// A tile window as `[x1, y1, x2, y2]` in source pixels.
type Window = [u32; 4];

/// Windows of `th` x `tw` covering an `h` x `w` image with 20% overlap and the last window aligned
/// to the edge.
fn tile_windows(h: u32, w: u32, (th, tw): (u32, u32)) -> Vec<Window> {
    let starts = |len: u32, size: u32| -> Vec<u32> {
        let last = len.saturating_sub(size);
        let step = ((8 * size + 5) / 10).max(1); // round(size * 0.8)
        (0..last).step_by(step as usize).chain([last]).collect()
    };
    let xs = starts(w, tw);
    starts(h, th)
        .into_iter()
        .flat_map(|y| {
            xs.iter()
                .map(move |&x| [x, y, (x + tw).min(w), (y + th).min(h)])
        })
        .collect()
}

/// `(box, score, class)` for the `idx` rows, whose first `K` columns are the box.
fn tuples<const K: usize>(data: &Array2<f32>, idx: &[usize]) -> Vec<([f32; K], f32, usize)> {
    idx.iter()
        .map(|&i| {
            (
                std::array::from_fn(|j| data[[i, j]]),
                data[[i, K]],
                data[[i, K + 1]] as usize,
            )
        })
        .collect()
}

/// Per-class NMS over the `idx` rows, returning the kept indices best first.
fn nms(data: &Array2<f32>, idx: &[usize], rotated: bool, c: &InferenceConfig) -> Vec<usize> {
    let kept = if rotated {
        nms_rotated_per_class(&tuples::<5>(data, idx), c.iou_threshold, c.max_det)
    } else {
        nms_per_class(&tuples::<4>(data, idx), c.iou_threshold, c.max_det)
    };
    kept.into_iter().map(|k| idx[k]).collect()
}

/// Merge tile detections already offset to source coordinates, as `ops.merge_tiles` does.
///
/// A box touching an interior window edge is a cut fragment when another window sees it clear of
/// its own interior edges. Fragments are dropped before NMS and restored afterwards when no kept
/// box covers more than half of them. Returns the kept row indices, best first.
fn merge_rows(
    data: &Array2<f32>,
    wins: &[Window],
    all: &[Window],
    shape: (u32, u32),
    rotated: bool,
    config: &InferenceConfig,
) -> Vec<usize> {
    let border = [0.0, 0.0, shape.1 as f32, shape.0 as f32];
    let ext = if rotated {
        Obb::new(data.clone(), shape).xyxy()
    } else {
        data.slice(s![.., ..4]).to_owned()
    };
    let ext: Vec<[f32; 4]> = ext
        .rows()
        .into_iter()
        .map(|e| [e[0], e[1], e[2], e[3]])
        .collect();
    let win_f = |w: &Window| w.map(|v| v as f32);
    // Whether edge `k` of extent `e` lies more than 2 pixels inside edge `k` of window `r`.
    let inside = |e: &[f32; 4], r: [f32; 4], k: usize| {
        if k < 2 {
            r[k] + 2.0 < e[k]
        } else {
            r[k] - 2.0 > e[k]
        }
    };
    let fragment: Vec<bool> = ext
        .iter()
        .zip(wins)
        .map(|(e, w)| {
            let own = win_f(w);
            (0..4).any(|k| !inside(e, own, k) && own[k] != border[k])
                && all
                    .iter()
                    .map(win_f)
                    .any(|r| (0..4).all(|k| inside(e, r, k) || r[k] == border[k]))
        })
        .collect();
    let (lone, valid): (Vec<usize>, Vec<usize>) = (0..ext.len()).partition(|&i| fragment[i]);
    let mut keep = nms(data, &valid, rotated, config);
    if lone.is_empty() {
        return keep;
    }
    let area = |e: [f32; 4]| (e[2] - e[0]).max(0.0) * (e[3] - e[1]).max(0.0);
    let uncovered = |&i: &usize| {
        let a = ext[i];
        !keep.iter().any(|&k| {
            let b = ext[k];
            area([
                a[0].max(b[0]),
                a[1].max(b[1]),
                a[2].min(b[2]),
                a[3].min(b[3]),
            ]) > area(a) / 2.0
        })
    };
    let restored: Vec<usize> = lone.into_iter().filter(uncovered).collect();
    keep.extend(restored);
    nms(data, &keep, rotated, config)
}

/// Offset every tile's boxes, masks, pose keypoints or oriented boxes to source coordinates, merge
/// them with [`merge_rows`] and store the result in `out`.
fn merge_detections(
    out: &mut Results,
    tiles: &[Results],
    windows: &[Window],
    rotated: bool,
    config: &InferenceConfig,
) {
    let shape = out.orig_shape;
    let (mut parts, mut kpts, mut masks, mut wins) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (t, win) in tiles.iter().zip(windows) {
        // Pose emits empty keypoints with no boxes when a tile finds nothing.
        if let Some(k) = &t.keypoints {
            let mut k = k.data.clone();
            let mut xy = k.slice_mut(s![.., .., ..2]);
            xy += &aview1(&[win[0] as f32, win[1] as f32]);
            kpts.push(k);
        }
        let data = if rotated {
            t.obb.as_ref().map(|o| &o.data)
        } else {
            t.boxes.as_ref().map(|b| &b.data)
        };
        let Some(mut d) = data.cloned() else { continue };
        // `xyxy` moves both corners and `xywhr` its center.
        for k in 0..if rotated { 2 } else { 4 } {
            d.column_mut(k).mapv_inplace(|v| v + win[k % 2] as f32);
        }
        if let Some(m) = &t.masks {
            masks.extend(m.data.outer_iter());
        }
        wins.extend(std::iter::repeat_n(*win, d.nrows()));
        parts.push(d);
    }
    let data = stack(&parts).unwrap_or_else(|| Array2::zeros((0, if rotated { 7 } else { 6 })));
    let keep = merge_rows(&data, &wins, windows, shape, rotated, config);
    let merged = data.select(Axis(0), &keep);
    if rotated {
        out.obb = Some(Obb::new(merged, shape));
    } else {
        out.boxes = Some(Boxes::new(merged, shape));
    }
    out.keypoints = stack(&kpts).map(|k| Keypoints::new(k.select(Axis(0), &keep), shape));
    // Each kept mask goes at its window origin in a source-sized stack.
    out.masks = tiles.iter().any(|t| t.masks.is_some()).then(|| {
        let mut data = Array3::zeros((keep.len(), shape.0 as usize, shape.1 as usize));
        for (o, &k) in keep.iter().enumerate() {
            let (m, [x, y, ..]) = (&masks[k], wins[k].map(|v| v as usize));
            data.slice_mut(s![o, y..y + m.nrows(), x..x + m.ncols()])
                .assign(m);
        }
        Masks::new(data, shape)
    });
}

/// Join tile arrays along their first axis, `None` when there are none.
fn stack<D: RemoveAxis>(parts: &[Array<f32, D>]) -> Option<Array<f32, D>> {
    let views: Vec<_> = parts.iter().map(|p| p.view()).collect();
    ndarray::concatenate(Axis(0), &views).ok()
}

/// Stitch dense per-pixel tile outputs. Every window is weighted by its distance to the window
/// edge, so a class map keeps the most central tile and a depth map blends the overlaps.
fn merge_dense(out: &mut Results, tiles: &[Results], windows: &[Window], semantic: bool) {
    let (h, w) = (out.orig_shape.0 as usize, out.orig_shape.1 as usize);
    let (mut classes, mut depth, mut weights) = (
        Array2::<u16>::zeros((h, w)),
        Array2::<f32>::zeros((h, w)),
        Array2::<f32>::zeros((h, w)),
    );
    for (t, win) in tiles.iter().zip(windows) {
        let [x1, y1, x2, y2] = win.map(|v| v as usize);
        let (th, tw) = (y2 - y1, x2 - x1);
        let weight = |y: usize, x: usize| (y + 1).min(th - y).min((x + 1).min(tw - x)) as f32;
        for y in 0..th {
            for x in 0..tw {
                let (wt, at) = (weight(y, x), [y1 + y, x1 + x]);
                if semantic {
                    if let Some(m) = &t.semantic_mask
                        && wt > weights[at]
                    {
                        classes[at] = m.data[[y, x]];
                        weights[at] = wt;
                    }
                } else if let Some(d) = &t.depth {
                    depth[at] = d.data[[y, x]].mul_add(wt, depth[at]);
                    weights[at] += wt;
                }
            }
        }
    }
    if semantic {
        out.semantic_mask = Some(SemanticMask::new(classes, out.orig_shape));
    } else {
        depth.zip_mut_with(&weights, |d, w| *d /= *w);
        out.depth = Some(DepthMap::new(depth, out.orig_shape));
    }
}

/// Merge the tile results of `task` into `out`, which holds the source image, as
/// `Results.merge_tiles` does in Ultralytics. Detection, segmentation, pose and oriented boxes go
/// through [`merge_detections`], semantic and depth maps through [`merge_dense`], and
/// classification averages the tile probabilities.
fn merge_tiles(
    out: &mut Results,
    task: Task,
    tiles: &[Results],
    windows: &[Window],
    config: &InferenceConfig,
) {
    match task {
        Task::Semantic | Task::Depth => {
            merge_dense(out, tiles, windows, task == Task::Semantic);
        }
        Task::Classify => {
            let sum = tiles
                .iter()
                .filter_map(|t| t.probs.as_ref())
                .map(|p| p.data.clone());
            out.probs = sum
                .reduce(|a, b| a + b)
                .map(|s| Probs::new(s / tiles.len() as f32));
        }
        _ => merge_detections(out, tiles, windows, task == Task::Obb, config),
    }
}

impl YOLOModel {
    /// Run inference on [`tile_windows`] of a large image and merge them back into one result in
    /// source coordinates with [`merge_tiles`], like the Ultralytics `tile` argument. Windows are
    /// the configured `tile` square, or the inference size when it is `0`.
    ///
    /// # Errors
    ///
    /// Returns an error if inference fails.
    pub(super) fn predict_tiled(
        &mut self,
        image: &DynamicImage,
        path: String,
    ) -> Result<Vec<Results>> {
        let (th, tw) = self.imgsz();
        let size = match self.config.tile.unwrap_or_default() {
            0 => (th as u32, tw as u32),
            tile => (tile, tile),
        };
        let (w, h) = image.dimensions();
        let windows = tile_windows(h, w, size);
        if windows == [[0, 0, w, h]] {
            let mut results = self.predict_internal(&[image], &[path])?;
            return Ok(results.pop().unwrap_or_default());
        }

        let batch = self.config.batch.unwrap_or(1).max(1);
        let mut tiles: Vec<Results> = Vec::with_capacity(windows.len());
        for chunk in windows.chunks(batch) {
            let crops: Vec<_> = chunk
                .iter()
                .map(|&[x1, y1, x2, y2]| image.crop_imm(x1, y1, x2 - x1, y2 - y1))
                .collect();
            let refs: Vec<_> = crops.iter().collect();
            let results = self.predict_internal(&refs, &vec![path.clone(); crops.len()])?;
            tiles.extend(results.into_iter().flatten());
        }

        let mut out = tiles[0].clone();
        out.orig_img = crate::preprocessing::image_to_array(image);
        out.orig_shape = (h, w);
        out.path = path;
        let total = |f: fn(&Speed) -> Option<f64>| tiles.iter().map(|t| f(&t.speed)).sum();
        out.speed = Speed {
            preprocess: total(|s| s.preprocess),
            inference: total(|s| s.inference),
            postprocess: total(|s| s.postprocess),
        };

        merge_tiles(&mut out, self.metadata.task, &tiles, &windows, &self.config);
        Ok(vec![out])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, arr3};

    #[test]
    fn test_tile_windows_match_the_documented_counts() {
        // Ultralytics tiling guide counts: 1920x1080 at 640 and 3840x2160 at 1280 both run 8.
        assert_eq!(tile_windows(1080, 1920, (640, 640)).len(), 8);
        assert_eq!(tile_windows(2160, 3840, (1280, 1280)).len(), 8);
        // The last window is aligned to the edge and the others step by round(0.8 * size).
        let w = tile_windows(1080, 1920, (640, 640));
        assert_eq!(w[0], [0, 0, 640, 640]);
        assert_eq!(w[1], [512, 0, 1152, 640]);
        assert_eq!(w[3], [1280, 0, 1920, 640]);
        assert_eq!(w[7], [1280, 440, 1920, 1080]);
        // A non-square inference size steps each axis by its own side, as Ultralytics does.
        let w = tile_windows(1080, 1920, (384, 640));
        assert_eq!(
            (w.len(), w[1], w[4]),
            (16, [512, 0, 1152, 384], [0, 307, 640, 691])
        );
        // An image no larger than the window gets one window.
        assert_eq!(tile_windows(300, 500, (640, 640)), [[0, 0, 500, 300]]);
        assert_eq!(tile_windows(640, 640, (640, 640)), [[0, 0, 640, 640]]);
    }

    fn blank(h: u32, w: u32) -> Results {
        Results::new(
            Array3::zeros((h as usize, w as usize, 3)),
            String::new(),
            std::sync::Arc::new(std::collections::HashMap::new()),
            crate::results::Speed::default(),
            (h, w),
        )
    }

    #[test]
    fn test_dense_tiles_merge_by_distance_to_the_edge() {
        // Two 5x5 windows overlapping on columns 3 and 4 of an 8-wide image. Column 3 is nearer the
        // center of the first window and column 4 of the second.
        let windows = [[0, 0, 5, 5], [3, 0, 8, 5]];
        let mut tiles = [blank(5, 5), blank(5, 5)];
        for (t, v) in tiles.iter_mut().zip([1u16, 4]) {
            t.semantic_mask = Some(SemanticMask::new(Array2::from_elem((5, 5), v), (5, 5)));
            t.depth = Some(DepthMap::new(
                Array2::from_elem((5, 5), f32::from(v)),
                (5, 5),
            ));
        }
        let mut out = blank(5, 8);
        let config = InferenceConfig::default();
        merge_tiles(&mut out, Task::Semantic, &tiles, &windows, &config);
        merge_tiles(&mut out, Task::Depth, &tiles, &windows, &config);
        // A class map keeps each pixel's most central tile, so each tile wins its own side.
        let map = out.semantic_mask.unwrap().data;
        assert_eq!(map.row(2).to_vec(), [1, 1, 1, 1, 4, 4, 4, 4]);
        // Depth blends: column 3 weighs 2 and 1, (1 * 2 + 4 * 1) / 3 = 2, and column 4 gives 3.
        let depth = out.depth.unwrap().data;
        assert_eq!(
            depth.row(2).to_vec(),
            [1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0]
        );
    }

    #[test]
    fn test_merge_tiles_offsets_every_task_and_drops_cut_fragments() {
        // Two 100x100 windows overlapping on x = 50..100 of a 150x100 image. The first window
        // sees an object cut by its right edge (x = 100) and the second sees it whole, so only
        // the second copy is kept, moved by its window origin. Alone, the cut copy stays.
        let windows = [[0, 0, 100, 100], [50, 0, 150, 100]];
        let config = InferenceConfig::default();
        let mut tiles = [blank(100, 100), blank(100, 100)];
        for (t, cx) in tiles.iter_mut().zip([80.0, 35.0]) {
            let row = [[cx - 20.0, 10.0, cx + 20.0, 90.0, 0.9, 0.0]];
            t.boxes = Some(Boxes::new(arr2(&row), (100, 100)));
            t.masks = Some(Masks::new(Array3::ones((1, 100, 100)), (100, 100)));
            t.probs = Some(Probs::new(arr1(&[cx, 1.0])));
            t.keypoints = Some(Keypoints::new(arr3(&[[[cx, 50.0, 1.0]]]), (100, 100)));
            let row = [[cx, 50.0, 40.0, 80.0, 0.0, 0.9, 0.0]];
            t.obb = Some(Obb::new(arr2(&row), (100, 100)));
        }
        let merge = |task, n: usize| {
            let mut out = blank(100, 150);
            merge_tiles(&mut out, task, &tiles[..n], &windows[..n], &config);
            out
        };
        let boxes = |r: Results| r.boxes.unwrap().data.into_raw_vec_and_offset().0;
        assert_eq!(
            boxes(merge(Task::Detect, 2)),
            [65.0, 10.0, 105.0, 90.0, 0.9, 0.0]
        );
        assert_eq!(
            boxes(merge(Task::Detect, 1)),
            [60.0, 10.0, 100.0, 90.0, 0.9, 0.0]
        );
        let masks = merge(Task::Segment, 2).masks.unwrap().data;
        assert_eq!(
            (masks.shape(), masks[[0, 0, 49]], masks.sum()),
            (&[1, 100, 150][..], 0.0, 1e4)
        );
        assert_eq!(
            merge(Task::Classify, 2).probs.unwrap().data,
            arr1(&[57.5, 1.0])
        );
        let pose = merge(Task::Pose, 2).keypoints.unwrap().data;
        assert_eq!(pose, arr3(&[[[85.0, 50.0, 1.0]]]));
        // A pose tile that finds nothing has no boxes but still an empty keypoints array.
        let mut empty = [blank(100, 100)];
        empty[0].keypoints = Some(Keypoints::new(Array3::zeros((0, 1, 3)), (100, 100)));
        let mut out = blank(100, 150);
        merge_tiles(&mut out, Task::Pose, &empty, &windows[..1], &config);
        assert_eq!(out.keypoints.unwrap().data.shape(), [0, 1, 3]);
        let obb = merge(Task::Obb, 2).obb.unwrap().data;
        assert_eq!(obb, arr2(&[[85.0, 50.0, 40.0, 80.0, 0.0, 0.9, 0.0]]));
    }

    /// Bus photo at `tile=320` against Ultralytics 8.4.207 (`xyxy`, confidence, class), keeping
    /// the boxes above 0.6 and a loose confidence check, because the JPEG decoders differ slightly
    /// and a box cut by a window edge can come from either window. A tile
    /// larger than the image must equal plain prediction. Skips offline, like the batch test.
    #[test]
    fn test_predict_tiled_matches_ultralytics_on_the_bus_photo() {
        const GOLDEN: [(f32, f32, f32, f32, f32, f32); 12] = [
            (58.4, 398.7, 202.6, 575.5, 0.94, 0.0),
            (256.1, 406.4, 343.6, 575.5, 0.92, 0.0),
            (225.2, 512.0, 319.8, 831.9, 0.89, 0.0),
            (55.7, 512.5, 202.4, 832.0, 0.88, 0.0),
            (245.7, 760.1, 305.7, 859.4, 0.83, 0.0),
            (0.2, 549.3, 79.3, 832.0, 0.81, 0.0),
            (676.0, 512.2, 810.0, 832.0, 0.76, 0.0),
            (490.0, 256.2, 803.7, 574.2, 0.71, 5.0),
            (256.9, 256.1, 575.5, 573.3, 0.70, 5.0),
            (0.0, 760.2, 34.2, 872.9, 0.68, 0.0),
            (669.6, 823.9, 740.7, 879.7, 0.62, 31.0),
            (490.6, 249.2, 789.8, 319.8, 0.62, 2.0),
        ];
        let _guard = crate::model::SESSION_BUILD_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let path = std::path::Path::new("yolo26n.onnx");
        let mut model = match YOLOModel::load(path) {
            Ok(model) => model,
            Err(_) if !path.exists() => return,
            Err(e) => panic!("yolo26n.onnx is present but failed to load: {e}"),
        };
        let image = image::open("assets/bus.jpg").unwrap();
        let name = || "bus.jpg".to_string();

        model.config.tile = Some(320);
        let tiled = model.predict_tiled(&image, name()).unwrap().remove(0);
        let found = tiled.boxes.unwrap().data;
        assert!(found.nrows().abs_diff(20) <= 3, "{} boxes", found.nrows());
        for (x1, y1, x2, y2, conf, cls) in GOLDEN {
            let hit = found.rows().into_iter().any(|r| {
                r[5] == cls
                    && (r[4] - conf).abs() < 0.25
                    && [x1, y1, x2, y2]
                        .iter()
                        .zip(r.iter())
                        .all(|(a, b)| (a - b).abs() < 4.0)
            });
            assert!(hit, "no match for class {cls} box {x1} {y1} {x2} {y2}");
        }

        // `tile = 0` uses the 640 inference size.
        model.config.tile = Some(0);
        let auto = model.predict_tiled(&image, name()).unwrap().remove(0);
        model.config.tile = Some(640);
        let sized = model.predict_tiled(&image, name()).unwrap().remove(0);
        assert_eq!(auto.boxes.unwrap().data, sized.boxes.unwrap().data);

        model.config.tile = Some(5000);
        let wide = model.predict_tiled(&image, name()).unwrap().remove(0);
        model.config.tile = None;
        let plain = model.predict_image(&image, name()).unwrap().remove(0);
        assert_eq!(plain.boxes.unwrap().data, wide.boxes.unwrap().data);
    }
}
