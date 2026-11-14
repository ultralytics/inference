// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! `FastTrack`: `ByteTrack` with occlusion handling.

use super::{State, Track, Tracker, TrackerConfig, TrackerType};

impl Track {
    pub(super) fn push_history(&mut self, c: &TrackerConfig) {
        self.history.push_back((self.mean, self.cov));
        if self.history.len() > c.reset_velocity_offset_occ.max(c.reset_pos_offset_occ) + 4 {
            self.history.pop_front();
        }
    }
}

impl Tracker {
    /// `FastTrack` occlusion handling of the tracked tracks left unmatched: a track mostly
    /// covered by an active one rewinds its Kalman state and keeps going for a while.
    pub(super) fn occlusions(
        &self,
        all: &mut [Track],
        unmatched: &[usize],
        activated: &[usize],
        lost: &mut Vec<usize>,
    ) {
        let c = &self.config;
        let frame = self.frame_id;
        let active: Vec<usize> = activated
            .iter()
            .copied()
            .filter(|&i| all[i].activated && !all[i].occluded)
            .collect();
        // Largest fraction of each unmatched box covered by another active track.
        let cover: Vec<f32> = unmatched
            .iter()
            .map(|&u| {
                let b = all[u].xyxy();
                let area = (b[2] - b[0]) * (b[3] - b[1]);
                active
                    .iter()
                    .filter(|&&a| all[a].id != all[u].id)
                    .map(|&a| {
                        let a = all[a].xyxy();
                        let w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
                        let h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
                        w * h / (area + 1e-7)
                    })
                    .fold(0.0, f32::max)
            })
            .collect();
        for (&i, cover) in unmatched.iter().zip(cover) {
            let t = &mut all[i];
            t.not_matched += 1;
            if cover > c.occ_cover_thresh && !t.occluded && t.state == State::Tracked {
                (t.occluded, t.recently_occluded) = (true, true);
                (t.occluded_len, t.last_occluded_frame) = (1, frame);
                let n = t.history.len();
                if n > 0 {
                    // Python's `hist[-k]`, where `hist[-0]` is the oldest entry.
                    if n >= c.reset_velocity_offset_occ {
                        let prev = t.history[(n - c.reset_velocity_offset_occ) % n].0;
                        t.mean[4..].copy_from_slice(&prev[4..]);
                    }
                    if n >= c.reset_pos_offset_occ {
                        let (prev, cov) = t.history[(n - c.reset_pos_offset_occ) % n];
                        t.mean[..4].copy_from_slice(&prev[..4]);
                        t.cov = cov;
                    }
                    t.mean[3] *= c.enlarge_bbox_occ;
                    t.mean[4..]
                        .iter_mut()
                        .for_each(|v| *v *= c.dampen_motion_occ);
                }
            } else if t.occluded {
                t.occluded_len += 1;
            }
            if t.recently_occluded && frame - t.last_occluded_frame > c.occ_reappear_window {
                t.recently_occluded = false;
            }
            if t.not_matched > 2 && (!t.occluded || t.occluded_len > c.active_occ_to_lost_thresh) {
                t.state = State::Lost;
                lost.push(i);
            }
        }
    }

    /// Boxes of the active tracks that new `FastTrack` tracks may not start on, or `None` when
    /// nothing is suppressed.
    pub(super) fn suppressing(
        &self,
        all: &[Track],
        n_t: usize,
        activated: &[usize],
        refind: &[usize],
    ) -> Option<Vec<[f32; 4]>> {
        let c = &self.config;
        if c.tracker_type != TrackerType::FastTrack || c.init_iou_suppress >= 1.0 {
            return None;
        }
        let shown = activated
            .iter()
            .chain(refind)
            .filter(|&&i| all[i].activated);
        let tracked = (0..n_t).filter(|&i| all[i].state == State::Tracked);
        Some(
            shown
                .copied()
                .chain(tracked)
                .map(|i| all[i].xyxy())
                .collect(),
        )
    }
}
