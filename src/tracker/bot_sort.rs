// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! `BoT-SORT` appearance matching, which `Deep OC-SORT` and `TrackTrack` reuse: a moving
//! average of each track's `ReID` features, and costs lowered to the appearance distance of
//! pairs that look alike and overlap enough.

use ndarray::Array2;

use super::matching::matrix;
use super::{Track, Tracker};

impl Track {
    /// Blend a feature into the appearance moving average with weight `alpha` on the old one.
    /// A zero feature carries no appearance and is skipped.
    pub(super) fn update_features(&mut self, feat: &[f32], alpha: f32) {
        let Some(feat) = unit(feat) else {
            return;
        };
        let smooth = self.smooth_feat.as_ref().map_or_else(
            || Some(feat.clone()),
            |s| {
                let blend: Vec<f32> = s
                    .iter()
                    .zip(&feat)
                    .map(|(s, f)| alpha * s + (1.0 - alpha) * f)
                    .collect();
                unit(&blend)
            },
        );
        (self.curr_feat, self.smooth_feat) = (Some(feat), smooth);
    }
}

/// Dot product of two features.
pub(super) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// `v` scaled to unit length, or `None` for a zero vector.
fn unit(v: &[f32]) -> Option<Vec<f32>> {
    let norm = dot(v, v).sqrt();
    (f64::from(norm) >= 1e-12).then(|| v.iter().map(|x| x / norm).collect())
}

impl Tracker {
    /// Lower the cost to the appearance cost of pairs that look alike and overlap enough.
    pub(super) fn with_appearance(
        &self,
        mut cost: Array2<f64>,
        iou: &Array2<f64>,
        all: &[Track],
        tracks: &[usize],
        dets: &[usize],
    ) -> Array2<f64> {
        if self.reid.is_none() || cost.is_empty() {
            return cost;
        }
        let c = &self.config;
        let emb = matrix(tracks, dets, |t, d| {
            let (Some(a), Some(b)) = (&all[t].smooth_feat, &all[d].curr_feat) else {
                return 1.0;
            };
            let norms = dot(a, a).sqrt() * dot(b, b).sqrt();
            f64::from((1.0 - dot(a, b) / norms.max(f32::EPSILON)).max(0.0) / 2.0)
        });
        for ((cost, emb), iou) in cost.iter_mut().zip(&emb).zip(iou) {
            let far = *emb > f64::from(1.0 - c.appearance_thresh)
                || *iou > f64::from(1.0 - c.proximity_thresh);
            *cost = cost.min(if far { 1.0 } else { *emb });
        }
        cost
    }
}
