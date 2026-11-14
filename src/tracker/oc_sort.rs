// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! `OC-SORT`: `ByteTrack` with motion direction costs, observation-centric recovery and
//! re-update.

use std::f32::consts::PI;

use ndarray::Array2;

use super::matching::{assign, fuse, matrix};
use super::{State, Track, Tracker, center};
use crate::utils::{calculate_iou, calculate_probiou};

impl Track {
    /// Unit direction from the observation `delta_t` frames back (or the oldest) to the latest.
    pub(super) fn direction(&self, delta_t: u32) -> Option<[f32; 2]> {
        if self.observations.len() < 2 {
            return None;
        }
        let &(last, current) = self.observations.back()?;
        let prev = self
            .observations
            .iter()
            .rev()
            .find(|(f, _)| *f + delta_t < last + 1)
            .or_else(|| self.observations.front().filter(|(f, _)| *f != last))?
            .1;
        let [cx, cy] = center(&current);
        let [px, py] = center(&prev);
        let (dx, dy) = (cx - px, cy - py);
        let norm = dx.hypot(dy);
        Some(if norm < 1e-6 {
            [0.0; 2]
        } else {
            [dx / norm, dy / norm]
        })
    }

    /// Observation-centric re-update: before re-activating a lost track, replay its Kalman
    /// filter over boxes interpolated from its last observation to `new_xyxy`.
    pub(super) fn reupdate(&mut self, new_xyxy: [f32; 4], frame: u32) {
        let (Some((mean, cov)), Some(&(last_frame, last))) = (self.saved, self.observations.back())
        else {
            return;
        };
        let gap = frame - last_frame;
        if gap <= 1 {
            return;
        }
        (self.mean, self.cov) = (mean, cov);
        for t in 1..gap {
            let alpha = t as f32 / gap as f32;
            let b: [f32; 4] =
                std::array::from_fn(|k| (1.0 - alpha) * last[k] + alpha * new_xyxy[k]);
            let (w, h) = (b[2] - b[0], b[3] - b[1]);
            let xywh = [b[0] + w / 2.0, b[1] + h / 2.0, w, h].map(f64::from);
            self.kf_predict();
            self.kf_update(self.kind.measure(xywh), None);
        }
        self.kf_predict();
    }
}

impl Tracker {
    /// `OC-SORT` motion direction cost: the angle between each track's motion and the direction
    /// from its observation `delta_t` frames back to each detection, over pi.
    pub(super) fn direction_cost(
        &self,
        all: &[Track],
        tracks: &[usize],
        dets: &[usize],
    ) -> Array2<f64> {
        let delta_t = self.config.delta_t;
        matrix(tracks, dets, |t, d| {
            let t = &all[t];
            let Some([vx, vy]) = t.velocity.filter(|_| t.last_observation[0] >= 0.0) else {
                return 0.0;
            };
            let prev = t
                .observations
                .iter()
                .find(|(f, _)| *f + delta_t >= self.frame_id && *f < self.frame_id)
                .map_or(t.last_observation, |o| o.1);
            let ([px, py], [dx, dy]) = (center(&prev), center(&all[d].xyxy()));
            let (dx, dy) = (dx - px, dy - py);
            let norm = dx.hypot(dy);
            if norm > 1e-6 {
                f64::from(((dx * vx + dy * vy) / norm).clamp(-1.0, 1.0).acos() / PI)
            } else {
                0.0
            }
        })
    }

    /// Observation-centric recovery: match the tracks of `pool` and the detections of `high`
    /// left unmatched (`u_pool`, `u_high`) by the last observed boxes, tracked tracks first,
    /// with `apply`. Returns those still unmatched.
    pub(super) fn recover(
        &self,
        all: &mut [Track],
        (pool, high): (&[usize], &[usize]),
        u_pool: &[usize],
        u_high: &[usize],
        apply: &mut impl FnMut(&mut [Track], usize, usize, bool),
    ) -> (Vec<usize>, Vec<usize>) {
        let c = &self.config;
        let (tracked, other): (Vec<usize>, Vec<usize>) = u_pool
            .iter()
            .partition(|&&t| all[pool[t]].state == State::Tracked);
        let mut dets: Vec<usize> = u_high.iter().map(|&d| high[d]).collect();
        let mut left: Vec<usize> = (0..dets.len()).collect();
        let mut u_pool = Vec::new();
        for group in [&tracked, &other] {
            let tracks: Vec<usize> = group.iter().map(|&t| pool[t]).collect();
            let iou = matrix(&tracks, &dets, |t, d| {
                let (t, d) = (&all[t], &all[d]);
                let iou = match (t.xywha(), d.xywha()) {
                    (Some(a), Some(b)) => calculate_probiou(&a, &b),
                    _ if t.last_observation[0] >= 0.0 => {
                        calculate_iou(&t.last_observation, &d.xyxy())
                    }
                    _ => calculate_iou(&t.xyxy(), &d.xyxy()),
                };
                f64::from(1.0 - iou)
            });
            let cost = fuse(&iou, c.fuse_score, all, &dets);
            let cost = self.with_appearance(cost, &iou, all, &tracks, &dets);
            let (matches, u_t, u_d) = assign(&cost, f64::from(c.match_thresh));
            for (t, d) in matches {
                apply(all, tracks[t], dets[d], true);
            }
            u_pool.extend(u_t.into_iter().map(|t| group[t]));
            dets = u_d.iter().map(|&d| dets[d]).collect();
            left = u_d.iter().map(|&d| left[d]).collect();
        }
        (u_pool, left.into_iter().map(|d| u_high[d]).collect())
    }
}
