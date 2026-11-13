// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Multi-object tracking, ported from the Ultralytics trackers.
//!
//! `ByteTrack`, `BoT-SORT`, `OC-SORT`, `Deep OC-SORT`, `FastTrack` and `TrackTrack`, with sparse
//! optical flow camera motion compensation and `ReID` appearance features from an ONNX model.
//!
//! [`YOLOModel::track`] runs it like `model.track(persist=True)` in Ultralytics: it keeps the
//! tracked detections, replaces their boxes with the Kalman filter estimate and adds a track id
//! to every row, which [`Boxes::id`] and [`Obb::id`] then return.
//!
//! ```no_run
//! use ultralytics_inference::YOLOModel;
//!
//! let mut model = YOLOModel::load("yolo26n.onnx")?;
//! for frame in ["frame1.jpg", "frame2.jpg"] {
//!     let results = model.track(frame, "bytetrack.yaml")?;
//!     println!("{:?}", results[0].boxes.as_ref().and_then(|b| b.id()));
//! }
//! # Ok::<(), ultralytics_inference::InferenceError>(())
//! ```
//!
//! A [`Tracker`] does the same for your own [`Results`]: feed it each frame's in order.
//!
//! ```no_run
//! use ultralytics_inference::{Tracker, TrackerConfig, YOLOModel};
//!
//! let mut model = YOLOModel::load("yolo26n.onnx")?;
//! let mut tracker = Tracker::new(TrackerConfig::load("bytetrack.yaml")?, &model)?;
//! for frame in ["frame1.jpg", "frame2.jpg"] {
//!     let mut results = model.predict(frame)?;
//!     tracker.update(&mut results[0])?;
//! }
//! # Ok::<(), ultralytics_inference::InferenceError>(())
//! ```

// Track math mixes f32 results with f64 Kalman states as Ultralytics mixes float32 and float64,
// keeps the short matrix names of the papers and OpenCV, and avoids fused multiply-adds and
// midpoints that would round differently from them.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::manual_midpoint,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::suboptimal_flops,
    clippy::suspicious_operation_groupings
)]

mod bot_sort;
mod byte_tracker;
mod config;
mod deep_oc_sort;
mod fast_tracker;
mod gmc;
mod kalman_filter;
mod matching;
mod oc_sort;
#[cfg(not(target_arch = "wasm32"))]
mod reid;
mod track_tracker;

use std::collections::VecDeque;
use std::f32::consts::PI;

use ndarray::{Array2, Array3, Axis};

use crate::error::Result;
#[cfg(not(target_arch = "wasm32"))]
use crate::model::YOLOModel;
use crate::results::{Boxes, Keypoints, Masks, Obb, Results};
use crate::utils::{calculate_iou, calculate_probiou};
pub use config::{GmcMethod, TrackerConfig, TrackerType};

type Mat8 = [[f64; 8]; 8];
/// A detection row: center box, angle of an oriented box, score and class.
type Row = ([f32; 4], Option<f32>, f32, f32);
/// A 2x3 affine camera motion from the previous frame to the current one.
type Warp = [[f64; 3]; 2];
/// The `ReID` features of `[x, y, w, h]` boxes of an image, `None` where a crop is empty. A
/// function, so the tracker builds where no `ReID` model runs, like the browser.
type Embed =
    Box<dyn FnMut(&Array3<u8>, &[[f32; 4]]) -> Result<Vec<Option<Vec<f32>>>> + Send + Sync>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum State {
    /// Started but not confirmed yet (`TrackTrack`).
    #[default]
    New,
    Tracked,
    Lost,
    Removed,
}

/// One track, or one detection before it is matched or starts a track. Holds the state every
/// tracker needs; each tracker only uses its own fields.
#[derive(Debug, Clone, Default)]
struct Track {
    kind: TrackerType,
    /// Kalman state: the measurement and its velocity.
    mean: [f64; 8],
    cov: Mat8,
    state: State,
    /// Confirmed, so it shows in the output.
    activated: bool,
    id: u64,
    score: f32,
    cls: f32,
    /// Rotation of an oriented box.
    angle: Option<f32>,
    /// Row of the latest matched detection in its frame's results.
    idx: usize,
    frame_id: u32,
    start_frame: u32,
    /// Matched frames since the track started (`TrackTrack`).
    tracklet_len: u32,
    /// Unit appearance feature of the latest detection, and its moving average.
    curr_feat: Option<Vec<f32>>,
    smooth_feat: Option<Vec<f32>>,
    /// Recent observed `xyxy` boxes by frame, oldest first (`OC-SORT`, `TrackTrack`).
    observations: VecDeque<(u32, [f32; 4])>,
    /// Latest observed box, `-1` before any (`OC-SORT`).
    last_observation: [f32; 4],
    /// Unit motion direction of the box center (`OC-SORT`).
    velocity: Option<[f32; 2]>,
    /// Kalman state after the latest observation (`OC-SORT`).
    saved: Option<([f64; 8], Mat8)>,
    /// Kalman states after recent updates, oldest first (`FastTrack`).
    history: VecDeque<([f64; 8], Mat8)>,
    /// `FastTrack` occlusion state.
    not_matched: u32,
    occluded: bool,
    occluded_len: u32,
    last_occluded_frame: u32,
    recently_occluded: bool,
    /// Score before the latest update (`TrackTrack`).
    prev_score: f32,
    /// Unit motion direction of each box corner (`TrackTrack`).
    corner_velocity: [[f32; 2]; 4],
}

impl Track {
    /// A detection from row `idx` of its frame's results, with its appearance feature.
    fn new(
        kind: TrackerType,
        (mut xywh, mut angle, score, cls): Row,
        idx: usize,
        feat: Option<&[f32]>,
    ) -> Self {
        // Keep oriented boxes long side first, so equal boxes always give equal measurements.
        if let Some(a) = angle.as_mut()
            && xywh[2] < xywh[3]
        {
            xywh.swap(2, 3);
            *a += PI / 2.0;
        }
        let (mean, cov) = kind.initiate(xywh.map(f64::from));
        let mut track = Self {
            kind,
            mean,
            cov,
            score,
            cls,
            angle,
            idx,
            last_observation: [-1.0; 4],
            prev_score: score,
            ..Self::default()
        };
        if let Some(f) = feat {
            track.update_features(f, 1.0);
        }
        track
    }

    /// Take the frame, score, class, row and angle of a matched detection.
    fn take_detection(&mut self, det: &Self, frame: u32) {
        self.frame_id = frame;
        self.score = det.score;
        self.cls = det.cls;
        self.idx = det.idx;
        // Of the angles that give the same box, keep the one nearest the track, so it never flips.
        if let (Some(a), Some(b)) = (self.angle.as_mut(), det.angle) {
            *a = PI.mul_add(-((b - *a) / PI).round_ties_even(), b);
        }
    }

    /// The detection box as a Kalman measurement.
    const fn measurement(&self) -> [f64; 4] {
        [self.mean[0], self.mean[1], self.mean[2], self.mean[3]]
    }

    /// Remember an observed box, keeping the latest `keep`.
    fn record(&mut self, frame: u32, xyxy: [f32; 4], keep: usize) {
        self.last_observation = xyxy;
        self.observations.push_back((frame, xyxy));
        while self.observations.len() > keep {
            self.observations.pop_front();
        }
    }

    /// Box as `[x_center, y_center, width, height]` from the Kalman state.
    fn xywh(&self) -> [f64; 4] {
        let [x, y, a, h] = self.measurement();
        [x, y, if self.kind.xyah() { a * h } else { a }, h]
    }

    /// Box as `[x1, y1, x2, y2]`, computed like `STrack.xyxy` in Ultralytics.
    fn xyxy(&self) -> [f32; 4] {
        let [x, y, w, h] = self.xywh();
        let (x1, y1) = (x - w / 2.0, y - h / 2.0);
        [x1, y1, x1 + w, y1 + h].map(|v| v as f32)
    }

    /// `[x, y, w, h, angle]` of an oriented track.
    fn xywha(&self) -> Option<[f32; 5]> {
        let [x, y, w, h] = self.xywh().map(|v| v as f32);
        self.angle.map(|a| [x, y, w, h, a])
    }

    /// `IoU` with another track, the probabilistic `IoU` for oriented boxes.
    fn iou(&self, other: &Self) -> f32 {
        match (self.xywha(), other.xywha()) {
            (Some(a), Some(b)) => calculate_probiou(&a, &b),
            _ => calculate_iou(&self.xyxy(), &other.xyxy()),
        }
    }
}

fn center(b: &[f32; 4]) -> [f32; 2] {
    [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]
}

/// The track at arena index `t` and the detection at `d`, which comes after every track.
const fn pair(all: &mut [Track], t: usize, d: usize) -> (&mut Track, &Track) {
    let (tracks, dets) = all.split_at_mut(d);
    (&mut tracks[t], &dets[0])
}

/// Keep rows `idx` of every per-detection field of `result`.
fn keep_rows(result: &mut Results, idx: &[usize]) {
    let rows = |a: &Array2<f32>| a.select(Axis(0), idx);
    if let Some(b) = &mut result.boxes {
        *b = Boxes::new(rows(&b.data), b.orig_shape);
    }
    if let Some(o) = &mut result.obb {
        *o = Obb::new(rows(&o.data), o.orig_shape);
    }
    if let Some(m) = &mut result.masks {
        *m = Masks::new(m.data.select(Axis(0), idx), m.orig_shape);
    }
    if let Some(k) = &mut result.keypoints {
        *k = Keypoints::new(k.data.select(Axis(0), idx), k.orig_shape);
    }
}

/// Multi-object tracker, the Ultralytics `BYTETracker` and the trackers built on it.
///
/// Call [`update`](Self::update) once per frame in order and [`reset`](Self::reset) before
/// a new video.
pub struct Tracker {
    config: TrackerConfig,
    tracked: Vec<Track>,
    lost: Vec<Track>,
    frame_id: u32,
    last_id: u64,
    gmc: Option<gmc::Gmc>,
    reid: Option<Embed>,
}

impl Tracker {
    /// Create a tracker for the detections of `model`, loading the `ReID` model of `config` on
    /// the same device when it matches by appearance.
    ///
    /// # Errors
    ///
    /// Returns an error if the `ReID` model cannot be downloaded or loaded.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(config: TrackerConfig, model: &YOLOModel) -> Result<Self> {
        let reid = (config.tracker_type.gmc_reid() && config.with_reid)
            .then(|| reid::Reid::new(model, &config.model))
            .transpose()?
            .map(|mut r| {
                Box::new(move |img: &Array3<u8>, b: &[[f32; 4]]| r.embed(img, b)) as Embed
            });
        Ok(Self::with_reid(config, reid))
    }

    /// A tracker without the `ReID` model of `with_reid`, which [`Tracker::new`] loads, so it
    /// matches by motion alone. The browser build has only this one.
    #[must_use]
    pub fn without_reid(config: TrackerConfig) -> Self {
        Self::with_reid(config, None)
    }

    fn with_reid(config: TrackerConfig, reid: Option<Embed>) -> Self {
        let gmc = config.tracker_type.gmc_reid() && config.gmc_method == GmcMethod::SparseOptFlow;
        Self {
            config,
            tracked: Vec::new(),
            lost: Vec::new(),
            frame_id: 0,
            last_id: 0,
            gmc: gmc.then(gmc::Gmc::default),
            reid,
        }
    }

    /// Forget every track and the previous frame, and restart the ids at 1.
    pub fn reset(&mut self) {
        let reid = self.reid.take();
        *self = Self::with_reid(self.config.clone(), reid);
    }

    /// Track the detections of the next frame.
    ///
    /// Keeps the shown tracks of `result` in track order, with their Kalman filter boxes
    /// clipped to the image and their track ids. New tracks show once confirmed. Results of
    /// classification, semantic segmentation or depth are left as they are.
    ///
    /// Returns the rows of `result` before the update that its rows now are, for data kept
    /// outside it, or `None` when `result` is unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error if the `ReID` model fails.
    pub fn update(&mut self, result: &mut Results) -> Result<Option<Vec<usize>>> {
        let kind = self.config.tracker_type;
        let boxes = result.boxes.as_ref().map(|b| (b.xywh(), b.conf(), b.cls()));
        let boxes = boxes
            .or_else(|| (result.obb.as_ref()).map(|o| (o.xywhr().to_owned(), o.conf(), o.cls())));
        if boxes.is_none()
            && (result.probs.is_some() || result.semantic_mask.is_some() || result.depth.is_some())
        {
            return Ok(None);
        }
        // A frame without detections still ages the tracks.
        let rows: Vec<Row> = boxes.map_or_else(Vec::new, |(r, conf, cls)| {
            let angle = |i: usize| (r.ncols() > 4).then(|| r[[i, 4]]);
            (0..conf.len())
                .map(|i| ([0, 1, 2, 3].map(|k| r[[i, k]]), angle(i), conf[i], cls[i]))
                .collect()
        });
        self.frame_id += 1;
        let warp = self.gmc.as_mut().map(|g| g.apply(&result.orig_img));

        // High and low score rows; `TrackTrack` keeps degenerate boxes, the others drop them
        // since an aspect ratio of a zero height is infinite.
        let c = &self.config;
        let (mut high, mut low) = (Vec::new(), Vec::new());
        for (i, (b, _, score, _)) in rows.iter().enumerate() {
            if kind != TrackerType::TrackTrack && !(b[2] > 0.0 && b[3] > 0.0) {
                continue;
            }
            if *score >= c.track_high_thresh {
                high.push(i);
            } else if *score > c.track_low_thresh {
                low.push(i);
            }
        }
        // Appearance features of the rows that can match by appearance.
        let embedded = if kind == TrackerType::TrackTrack {
            high.len()
        } else {
            high.len() + low.len()
        };
        let order: Vec<usize> = high.iter().chain(&low).copied().collect();
        let crops: Vec<[f32; 4]> = order[..embedded].iter().map(|&i| rows[i].0).collect();
        let feats = match &mut self.reid {
            Some(embed) => embed(&result.orig_img, &crops)?,
            None => Vec::new(),
        };
        let det = |(k, &i): (usize, &usize)| {
            Track::new(kind, rows[i], i, feats.get(k).and_then(Option::as_deref))
        };
        let dets: Vec<Track> = order.iter().enumerate().map(det).collect();
        if kind == TrackerType::TrackTrack {
            self.step_tracktrack(dets, high.len(), warp);
        } else {
            self.step(dets, high.len(), warp);
        }
        Ok(self.write(result))
    }

    /// Keep the rows of the tracks shown this frame in `result`, with their boxes and ids, and
    /// return them.
    fn write(&self, result: &mut Results) -> Option<Vec<usize>> {
        let shown: Vec<&Track> = self
            .tracked
            .iter()
            .filter(|t| t.activated && t.frame_id == self.frame_id)
            .collect();
        if shown.is_empty() {
            if self.tracked.iter().all(|t| t.activated) {
                return None;
            }
            keep_rows(result, &[]); // hide new tracks until they are confirmed
            return Some(Vec::new());
        }
        let idx: Vec<usize> = shown.iter().map(|t| t.idx).collect();
        keep_rows(result, &idx);
        let (h, w) = (result.orig_shape.0 as f32, result.orig_shape.1 as f32);
        let tail = |t: &Track| [t.id as f32, t.score, t.cls];
        if let Some(b) = &mut result.boxes {
            let rows = shown.iter().flat_map(|t| {
                let xyxy = t.xyxy().into_iter().zip([w, h, w, h]);
                xyxy.map(|(v, max)| v.clamp(0.0, max)).chain(tail(t))
            });
            let data = Array2::from_shape_vec((shown.len(), 7), rows.collect());
            *b = Boxes::new(data.unwrap_or_default(), b.orig_shape);
        } else if let Some(o) = &mut result.obb {
            let rows = shown
                .iter()
                .flat_map(|t| t.xywha().unwrap_or_default().into_iter().chain(tail(t)));
            let data = Array2::from_shape_vec((shown.len(), 8), rows.collect());
            *o = Obb::new(data.unwrap_or_default(), o.orig_shape);
        }
        Some(idx)
    }

    /// One arena of the tracked tracks, the lost tracks and `dets`, with the first two counts.
    fn arena(&mut self, dets: Vec<Track>) -> (Vec<Track>, usize, usize) {
        let (n_t, n_l) = (self.tracked.len(), self.lost.len());
        let mut all = std::mem::take(&mut self.tracked);
        all.append(&mut self.lost);
        all.extend(dets);
        (all, n_t, n_l)
    }

    const fn next_id(&mut self) -> u64 {
        self.last_id += 1;
        self.last_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn frame(rows: &[[f32; 6]]) -> Results {
        let mut r = Results::new(
            ndarray::Array3::zeros((480, 640, 3)),
            String::new(),
            Arc::new(HashMap::new()),
            crate::Speed::default(),
            (480, 640),
        );
        let data = Array2::from_shape_vec((rows.len(), 6), rows.concat()).unwrap();
        r.boxes = Some(Boxes::new(data, (480, 640)));
        r
    }

    #[test]
    fn test_every_tracker_keeps_ids_across_frames() {
        for kind in [
            TrackerType::ByteTrack,
            TrackerType::BotSort,
            TrackerType::OcSort,
            TrackerType::DeepOcSort,
            TrackerType::FastTrack,
            TrackerType::TrackTrack,
        ] {
            let mut tracker = Tracker::without_reid(kind.into());
            let mut ids = Vec::new();
            for step in 0..6u8 {
                let dx = f32::from(step) * 4.0;
                let mut r = frame(&[
                    [10.0 + dx, 10.0, 60.0 + dx, 90.0, 0.9, 0.0],
                    [300.0 - dx, 200.0, 360.0 - dx, 300.0, 0.8, 2.0],
                ]);
                tracker.update(&mut r).unwrap();
                let boxes = r.boxes.unwrap();
                ids.push(boxes.id().unwrap().to_vec());
                assert!((boxes.xyxy()[[0, 0]] - (10.0 + dx)).abs() < 1.0, "{kind:?}");
            }
            assert!(ids.iter().all(|i| i == &[1.0, 2.0]), "{kind:?}: {ids:?}");
        }
    }

    #[test]
    fn test_tracker_hides_new_tracks_until_confirmed() {
        let mut tracker = Tracker::without_reid(TrackerType::ByteTrack.into());
        let mut r = frame(&[]);
        tracker.update(&mut r).unwrap();
        let mut r = frame(&[[10.0, 10.0, 60.0, 90.0, 0.9, 0.0]]);
        tracker.update(&mut r).unwrap();
        assert!(r.boxes.unwrap().is_empty());
        let mut r = frame(&[[11.0, 10.0, 61.0, 90.0, 0.9, 0.0]]);
        tracker.update(&mut r).unwrap();
        assert_eq!(r.boxes.unwrap().id().unwrap().to_vec(), vec![1.0]);
    }
}
