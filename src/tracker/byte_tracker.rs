// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! `ByteTrack`: two association stages, high then low score detections, which `BoT-SORT`,
//! `OC-SORT`, `Deep OC-SORT` and `FastTrack` build on.

use ndarray::Array2;

use super::matching::{assign, fuse, matrix};
use super::{State, Track, Tracker, TrackerConfig, TrackerType, pair};
use crate::utils::calculate_iou;

impl Track {
    /// Start a track with a new id.
    pub(super) fn activate(&mut self, id: u64, frame: u32, c: &TrackerConfig) {
        self.id = id;
        self.state = State::Tracked;
        self.activated = frame == 1;
        (self.frame_id, self.start_frame) = (frame, frame);
        if self.kind.ocsort() {
            self.record(frame, self.xyxy(), c.delta_t as usize + 2);
            self.saved = Some((self.mean, self.cov));
        } else if self.kind == TrackerType::FastTrack {
            self.push_history(c);
        }
    }

    /// Correct a track with its matched detection, `reactivate` when the track was lost.
    pub(super) fn update(&mut self, det: &Self, frame: u32, c: &TrackerConfig, reactivate: bool) {
        if self.kind.ocsort() {
            self.record(frame, det.xyxy(), c.delta_t as usize + 2);
        }
        self.kf_update(det.measurement());
        self.take_detection(det, frame);
        (self.state, self.activated) = (State::Tracked, true);
        if self.kind.ocsort() {
            self.saved = Some((self.mean, self.cov));
            self.velocity = self.direction(c.delta_t);
        } else if self.kind == TrackerType::FastTrack {
            self.push_history(c);
            (self.occluded, self.not_matched, self.occluded_len) = (false, 0, 0);
            if reactivate {
                self.recently_occluded = false;
            }
        }
    }
}

impl Tracker {
    /// Matching cost of `BYTETracker.get_dists` and its overrides, between arena indices.
    fn dists(&self, all: &[Track], tracks: &[usize], dets: &[usize]) -> Array2<f64> {
        let c = &self.config;
        let iou = matrix(tracks, dets, |t, d| f64::from(1.0 - all[t].iou(&all[d])));
        let mut cost = fuse(&iou, c.fuse_score, all, dets);
        if c.tracker_type.ocsort() {
            cost.zip_mut_with(&self.direction_cost(all, tracks, dets), |a, b| {
                *a += f64::from(c.inertia) * b;
            });
        }
        cost
    }

    /// One `BYTETracker.update` step.
    #[allow(clippy::too_many_lines)]
    pub(super) fn step(&mut self, dets: Vec<Track>, n_h: usize) {
        let c = self.config.clone();
        let (kind, frame) = (c.tracker_type, self.frame_id);
        // One arena: tracked tracks, lost tracks, then high and low score detections.
        let (mut all, n_t, n_l) = self.arena(dets);
        let high: Vec<usize> = (n_t + n_l..n_t + n_l + n_h).collect();
        let low: Vec<usize> = (n_t + n_l + n_h..all.len()).collect();
        let unconfirmed: Vec<usize> = (0..n_t).filter(|&i| !all[i].activated).collect();
        let pool: Vec<usize> = (0..n_t + n_l)
            .filter(|&i| i >= n_t || all[i].activated)
            .collect();
        for &i in &pool {
            all[i].predict();
        }
        let (mut activated, mut refind, mut lost) = (Vec::new(), Vec::new(), Vec::new());
        // Correct track `t` with detection `d`: activated if it was tracked, found again if lost.
        let mut apply = |all: &mut [Track], t: usize, d: usize, oru: bool| {
            let (track, det) = pair(all, t, d);
            if track.state == State::Tracked {
                track.update(det, frame, &c, false);
                activated.push(t);
            } else {
                if oru {
                    track.reupdate(det.xyxy(), frame);
                }
                track.update(det, frame, &c, true);
                refind.push(t);
            }
        };

        // First association: confirmed and lost tracks with the high-score detections.
        let (matches, u_pool, u_high) =
            assign(&self.dists(&all, &pool, &high), f64::from(c.match_thresh));
        for (t, d) in matches {
            apply(&mut all, pool[t], high[d], false);
        }
        let (u_pool, u_high) = if kind.ocsort() && !u_high.is_empty() {
            self.recover(&mut all, (&pool, &high), &u_pool, &u_high, &mut apply)
        } else {
            (u_pool, u_high)
        };

        // Second association: tracked tracks still unmatched with the low-score detections, by
        // IoU alone, since weighing low scores in would push the costs over the threshold.
        let rest: Vec<usize> = u_pool
            .iter()
            .map(|&t| pool[t])
            .filter(|&i| all[i].state == State::Tracked)
            .collect();
        let u_rest: Vec<usize> = if kind.ocsort() && !c.use_byte {
            rest
        } else {
            let iou = matrix(&rest, &low, |t, d| f64::from(1.0 - all[t].iou(&all[d])));
            let (matches, u_rest, _) = assign(&iou, 0.5);
            for (t, d) in matches {
                apply(&mut all, rest[t], low[d], false);
            }
            u_rest.into_iter().map(|t| rest[t]).collect()
        };
        if kind == TrackerType::FastTrack {
            self.occlusions(&mut all, &u_rest, &activated, &mut lost);
        } else {
            for t in u_rest {
                all[t].state = State::Lost;
                lost.push(t);
            }
        }

        // Unconfirmed tracks with the high-score detections left; unmatched ones are removed.
        let left: Vec<usize> = u_high.iter().map(|&d| high[d]).collect();
        let (matches, u_unconfirmed, u_left) = assign(&self.dists(&all, &unconfirmed, &left), 0.7);
        for (t, d) in matches {
            let (track, det) = pair(&mut all, unconfirmed[t], left[d]);
            track.update(det, frame, &c, false);
            activated.push(unconfirmed[t]);
        }
        for t in u_unconfirmed {
            all[unconfirmed[t]].state = State::Removed;
        }

        // The rest start new tracks; FastTrack skips those on top of an active track.
        let mut active = self.suppressing(&all, n_t, &activated, &refind);
        for d in u_left.into_iter().map(|d| left[d]) {
            let b = all[d].xyxy();
            let covered = |a: &Vec<[f32; 4]>| {
                a.iter()
                    .any(|a| calculate_iou(&b, a) >= c.init_iou_suppress)
            };
            if all[d].score < c.new_track_thresh || active.as_ref().is_some_and(covered) {
                continue;
            }
            let id = self.next_id();
            all[d].activate(id, frame, &c);
            activated.push(d);
            if let Some(a) = &mut active {
                a.push(b);
            }
        }
        self.merge(all, n_t, n_l, (activated, refind, lost));
    }

    /// End-of-frame bookkeeping (`merge_track_pools`): tracked are the still tracked, then the
    /// activated and found again; lost are the still lost, then the newly lost. Of a tracked
    /// and a lost track on the same object, the younger one goes.
    pub(super) fn merge(
        &mut self,
        mut all: Vec<Track>,
        n_t: usize,
        n_l: usize,
        (activated, refind, lost_new): (Vec<usize>, Vec<usize>, Vec<usize>),
    ) {
        // Lost tracks expire after `track_buffer` frames, recently occluded `FastTrack` ones
        // after `occ_reappear_window`.
        let (c, frame) = (&self.config, self.frame_id);
        for t in &mut all[n_t..n_t + n_l] {
            let recent =
                t.recently_occluded && frame - t.last_occluded_frame <= c.occ_reappear_window;
            if !recent && frame - t.frame_id > c.track_buffer {
                t.state = State::Removed;
            }
        }
        let mut in_tracked = vec![false; all.len()];
        let mut tracked = Vec::new();
        for i in (0..n_t)
            .filter(|&i| all[i].state == State::Tracked)
            .chain(activated)
            .chain(refind)
        {
            if !std::mem::replace(&mut in_tracked[i], true) {
                tracked.push(i);
            }
        }
        let lost: Vec<usize> = (n_t..n_t + n_l)
            .filter(|&i| !in_tracked[i])
            .chain(lost_new)
            .filter(|&i| all[i].state != State::Removed)
            .collect();
        let (mut keep_t, mut keep_l) = (vec![true; tracked.len()], vec![true; lost.len()]);
        for (p, &a) in tracked.iter().enumerate() {
            for (q, &b) in lost.iter().enumerate() {
                if 1.0 - all[a].iou(&all[b]) < 0.15 {
                    let age = |t: &Track| t.frame_id - t.start_frame;
                    if age(&all[a]) > age(&all[b]) {
                        keep_l[q] = false;
                    } else {
                        keep_t[p] = false;
                    }
                }
            }
        }
        let mut slots: Vec<Option<Track>> = all.into_iter().map(Some).collect();
        let mut take = |idx: Vec<usize>, keep: Vec<bool>| -> Vec<Track> {
            idx.into_iter()
                .zip(keep)
                .filter(|&(_, k)| k)
                .filter_map(|(i, _)| slots[i].take())
                .collect()
        };
        self.tracked = take(tracked, keep_t);
        self.lost = take(lost, keep_l);
    }
}
