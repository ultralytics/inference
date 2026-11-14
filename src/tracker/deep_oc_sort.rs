// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! `Deep OC-SORT`: `OC-SORT` with camera motion compensation and `ReID` features that adapt
//! to the detection confidence.

use super::TrackerConfig;

/// Weight of a track's old appearance feature when a detection scoring `score` updates it, so
/// confident detections change it more.
pub(super) fn feature_alpha(c: &TrackerConfig, score: f32) -> f32 {
    let t = c.track_high_thresh;
    if score > t {
        let trust = (score - t) / (1.0 - t).max(1e-9);
        c.alpha_fixed_emb + (1.0 - c.alpha_fixed_emb) * (1.0 - trust)
    } else {
        1.0
    }
}
