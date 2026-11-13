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
        let Some(feat) = unit(feat.iter().copied()) else {
            return;
        };
        let smooth = self.smooth_feat.as_ref().map_or_else(
            || Some(feat.clone()),
            |s| {
                unit(
                    s.iter()
                        .zip(&feat)
                        .map(|(s, f)| alpha * s + (1.0 - alpha) * f),
                )
            },
        );
        (self.curr_feat, self.smooth_feat) = (Some(feat), smooth);
    }
}

/// `v` scaled to unit length, or `None` for a zero vector.
fn unit(v: impl Iterator<Item = f32> + Clone) -> Option<Vec<f32>> {
    let norm = v.clone().map(|x| x * x).sum::<f32>().sqrt();
    (f64::from(norm) >= 1e-12).then(|| v.map(|x| x / norm).collect())
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
            let dot: f32 = a.iter().zip(b).map(|(a, b)| a * b).sum();
            let norms = a.iter().map(|v| v * v).sum::<f32>().sqrt()
                * b.iter().map(|v| v * v).sum::<f32>().sqrt();
            f64::from((1.0 - dot / norms.max(f32::EPSILON)).max(0.0) / 2.0)
        });
        for ((cost, emb), iou) in cost.iter_mut().zip(&emb).zip(iou) {
            let far = *emb > f64::from(1.0 - c.appearance_thresh)
                || *iou > f64::from(1.0 - c.proximity_thresh);
            *cost = cost.min(if far { 1.0 } else { *emb });
        }
        cost
    }
}
