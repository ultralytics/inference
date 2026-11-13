// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! `TrackTrack`: one association of all tracks with all detections over a combined cost of
//! height-modulated `IoU`, appearance, confidence and corner motion, matched iteratively with a
//! falling threshold, and new tracks only where no track already is.
//!
//! The loose NMS pass Ultralytics adds for NMS models (`dets_del`) is not run, which changes
//! nothing for NMS-free models like `YOLO26`.

use ndarray::{Array2, ArrayView1, s};

use super::matching::matrix;
use super::{State, Track, Tracker, Warp, pair};
use crate::utils::calculate_iou;

/// Frames over which corner motion is measured.
const DELTA_T: u32 = 3;

/// Unit direction of each corner (top left, bottom left, top right, bottom right) from `b` to `a`.
fn corner_directions(a: &[f32; 4], b: &[f32; 4]) -> [[f32; 2]; 4] {
    [(0, 1), (0, 3), (2, 1), (2, 3)].map(|(x, y)| {
        let (dx, dy) = (a[x] - b[x], a[y] - b[y]);
        let norm = dx.hypot(dy) + 1e-5;
        [dx / norm, dy / norm]
    })
}

impl Track {
    fn tt_activate(&mut self, id: u64, frame: u32, min_track_len: u32) {
        self.id = id;
        self.record(frame, self.xyxy(), DELTA_T as usize + 1);
        self.tracklet_len = 0;
        self.state = if min_track_len <= 1 {
            State::Tracked
        } else {
            State::New
        };
        self.activated = frame == 1 || self.state == State::Tracked;
        (self.frame_id, self.start_frame) = (frame, frame);
    }

    /// Correct a track with its matched detection, `reactivate` when the track was lost. A new
    /// track is confirmed after `min_track_len` matched frames.
    fn tt_update(&mut self, det: &Self, frame: u32, min_track_len: u32, reactivate: bool) {
        self.prev_score = self.score;
        self.kf_update(det.measurement(), Some(det.score));
        if reactivate {
            self.record(frame, self.xyxy(), DELTA_T as usize + 1);
            self.tracklet_len = 0;
        } else {
            self.tracklet_len += 1;
            let b = det.xyxy();
            self.record(frame, b, DELTA_T as usize + 1);
            let mut v = [[0.0; 2]; 4];
            for dt in 1..=DELTA_T {
                let d = corner_directions(&b, &self.history_box(frame, dt));
                for (v, d) in v.iter_mut().zip(d) {
                    v[0] += d[0] / dt as f32;
                    v[1] += d[1] / dt as f32;
                }
            }
            self.corner_velocity = v.map(|[x, y]| [x / DELTA_T as f32, y / DELTA_T as f32]);
        }
        self.score = det.score;
        if let Some(f) = &det.curr_feat {
            self.update_features(f, 0.95 + 0.05 * (1.0 - self.score));
        }
        if reactivate || self.state == State::Tracked || self.tracklet_len + 1 >= min_track_len {
            (self.state, self.activated) = (State::Tracked, true);
        }
        self.take_detection(det, frame);
    }

    /// The box observed `dt` frames before `frame`, else the latest one.
    fn history_box(&self, frame: u32, dt: u32) -> [f32; 4] {
        let target = frame.checked_sub(dt);
        self.observations
            .iter()
            .find(|(f, _)| Some(*f) == target)
            .or_else(|| self.observations.back())
            .map_or_else(|| self.xyxy(), |o| o.1)
    }
}

/// Match rows to columns by mutual nearest neighbors under `thresh`, lowering `thresh` by
/// `step` each round until no pair is left.
fn iterative_assign(
    mut cost: Array2<f64>,
    mut thresh: f64,
    step: f64,
) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    let argmin = |v: ArrayView1<f64>| (0..v.len()).fold(0, |m, i| if v[i] < v[m] { i } else { m });
    let (rows, cols) = cost.dim();
    let mut matches = Vec::new();
    while !cost.is_empty() {
        let nearest_det: Vec<usize> = cost.rows().into_iter().map(argmin).collect();
        let nearest_track: Vec<usize> = cost.columns().into_iter().map(argmin).collect();
        let new: Vec<(usize, usize)> = (0..rows)
            .map(|t| (t, nearest_det[t]))
            .filter(|&(t, d)| nearest_track[d] == t && cost[[t, d]] < thresh)
            .collect();
        if new.is_empty() {
            break;
        }
        for &(t, d) in &new {
            cost.row_mut(t).fill(f64::INFINITY);
            cost.column_mut(d).fill(f64::INFINITY);
        }
        matches.extend(new);
        thresh -= step;
    }
    let u_rows = (0..rows)
        .filter(|r| !matches.iter().any(|m| m.0 == *r))
        .collect();
    let u_cols = (0..cols)
        .filter(|c| !matches.iter().any(|m| m.1 == *c))
        .collect();
    (matches, u_rows, u_cols)
}

impl Tracker {
    /// `TrackTrack` cost between arena indices.
    fn tt_cost(&self, all: &[Track], tracks: &[usize], dets: &[usize]) -> Array2<f64> {
        let c = &self.config;
        matrix(tracks, dets, |t, d| {
            let (t, d) = (&all[t], &all[d]);
            let (a, b) = (t.xyxy(), d.xyxy());
            let iou = calculate_iou(&a, &b);
            let h_over = a[3].min(b[3]) - a[1].max(b[1]);
            let h_union = a[3].max(b[3]) - a[1].min(b[1]);
            let hmiou = 1.0 - (h_over / (h_union + 1e-9)).clamp(0.0, 1.0) * iou;
            let feats = (
                t.smooth_feat.as_ref().or(t.curr_feat.as_ref()),
                &d.curr_feat,
            );
            let mut cost = match (self.reid.is_some(), feats) {
                (true, (Some(a), Some(b))) => {
                    let cos =
                        (1.0 - a.iter().zip(b).map(|(a, b)| a * b).sum::<f32>()).clamp(0.0, 1.0);
                    c.iou_weight * hmiou + c.reid_weight * cos
                }
                _ => hmiou,
            };
            cost += c.conf_weight * (2.0f32.mul_add(t.score, -t.prev_score) - d.score).abs();
            if iou > 0.1 {
                let dirs = corner_directions(&b, &t.history_box(self.frame_id, DELTA_T));
                let angle: f32 = t
                    .corner_velocity
                    .iter()
                    .zip(dirs)
                    .map(|(v, d)| v[0].mul_add(d[0], v[1] * d[1]).clamp(-1.0, 1.0).acos())
                    .sum();
                cost += c.angle_weight * angle / 4.0 / std::f32::consts::PI * d.score;
            } else {
                cost = 1.0;
            }
            f64::from(cost.clamp(0.0, 1.0))
        })
    }

    /// Track-aware initialization: which of `dets` may start a track, skipping those on top of
    /// an `active` track or a higher-scoring detection.
    fn track_aware(&self, all: &[Track], active: &[usize], dets: &[usize]) -> Vec<bool> {
        let c = &self.config;
        let mut allow: Vec<bool> = dets
            .iter()
            .map(|&d| all[d].score > c.new_track_thresh)
            .collect();
        let iou = |a: usize, b: usize| calculate_iou(&all[a].xyxy(), &all[b].xyxy());
        for (ok, &d) in allow.iter_mut().zip(dets) {
            *ok &= active.iter().all(|&t| iou(d, t) <= c.tai_thr);
        }
        let mut order: Vec<usize> = (0..dets.len()).collect();
        order.sort_by(|&a, &b| all[dets[a]].score.total_cmp(&all[dets[b]].score));
        for &i in order.iter().rev() {
            if allow[i] {
                for j in (0..dets.len()).filter(|&j| j != i) {
                    if iou(dets[i], dets[j]) > c.tai_thr {
                        allow[j] = false;
                    }
                }
            }
        }
        allow
    }

    /// One `TRACKTRACK.update` step.
    #[allow(clippy::too_many_lines)]
    pub(super) fn step_tracktrack(&mut self, dets: Vec<Track>, n_h: usize, warp: Option<Warp>) {
        let c = self.config.clone();
        let (frame, step) = (self.frame_id, f64::from(c.reduce_step));
        let (mut all, n_t, n_l) = self.arena(dets);
        let dets: Vec<usize> = (n_t + n_l..all.len()).collect();
        let unconfirmed: Vec<usize> = (0..n_t)
            .filter(|&i| all[i].state != State::Tracked)
            .collect();
        let pool: Vec<usize> = (0..n_t + n_l)
            .filter(|&i| i >= n_t || all[i].state == State::Tracked)
            .collect();
        for &i in pool.iter().chain(&unconfirmed) {
            if let Some(h) = &warp {
                all[i].warp(h);
            }
            all[i].predict();
        }
        let (mut activated, mut refind, mut lost) = (Vec::new(), Vec::new(), Vec::new());

        // All tracks with all detections, low-score ones costing more.
        let mut cost = self.tt_cost(&all, &pool, &dets);
        cost.slice_mut(s![.., n_h..])
            .mapv_inplace(|v| (v + f64::from(c.penalty_p)).clamp(0.0, 1.0));
        let (matches, u_pool, u_dets) = iterative_assign(cost, f64::from(c.match_thresh), step);
        for (t, d) in matches {
            let (track, det) = pair(&mut all, pool[t], dets[d]);
            let reactivate = track.state != State::Tracked;
            track.tt_update(det, frame, c.min_track_len, reactivate);
            (if reactivate {
                &mut refind
            } else {
                &mut activated
            })
            .push(pool[t]);
        }
        for t in u_pool.into_iter().map(|t| pool[t]) {
            if all[t].state != State::Lost {
                all[t].state = State::Lost;
                lost.push(t);
            }
        }

        // Unconfirmed tracks with the high-score detections left; unmatched ones are removed.
        let left: Vec<usize> = u_dets
            .into_iter()
            .filter(|&d| d < n_h)
            .map(|d| dets[d])
            .collect();
        let cost = self.tt_cost(&all, &unconfirmed, &left);
        let (matches, u_unconfirmed, u_left) =
            iterative_assign(cost, f64::from(c.match_thresh), step);
        for (t, d) in matches {
            let (track, det) = pair(&mut all, unconfirmed[t], left[d]);
            track.tt_update(det, frame, c.min_track_len, false);
            activated.push(unconfirmed[t]);
        }
        for t in u_unconfirmed {
            all[unconfirmed[t]].state = State::Removed;
        }
        let mut left: Vec<usize> = u_left.into_iter().map(|d| left[d]).collect();

        // Optionally rebind long-lost tracks with a looser threshold.
        if c.lost_match_thr > 0.0 {
            let still_lost: Vec<usize> = pool
                .iter()
                .copied()
                .filter(|i| all[*i].state == State::Lost && !lost.contains(i))
                .collect();
            let cost = self.tt_cost(&all, &still_lost, &left);
            let (matches, _, u_left) = iterative_assign(cost, f64::from(c.lost_match_thr), step);
            for (t, d) in matches {
                let (track, det) = pair(&mut all, still_lost[t], left[d]);
                track.tt_update(det, frame, c.min_track_len, true);
                refind.push(still_lost[t]);
            }
            left = u_left.into_iter().map(|d| left[d]).collect();
        }

        let active: Vec<usize> = (0..n_t)
            .filter(|&i| all[i].state == State::Tracked)
            .chain(activated.iter().copied())
            .collect();
        for (d, ok) in left.iter().zip(self.track_aware(&all, &active, &left)) {
            if ok {
                let id = self.next_id();
                all[*d].tt_activate(id, frame, c.min_track_len);
                activated.push(*d);
            }
        }
        self.merge(all, n_t, n_l, (activated, refind, lost));
    }
}
