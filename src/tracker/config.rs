// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Tracker settings, the Ultralytics tracker YAML configs.

use std::path::Path;
use std::str::FromStr;

use crate::error::{InferenceError, Result};

/// Tracking algorithm, the `tracker_type` of the Ultralytics tracker configs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrackerType {
    /// `ByteTrack`: two association stages, high then low score detections.
    ByteTrack,
    /// `BoT-SORT`, the Ultralytics default: `ByteTrack` with a center-width-height Kalman state,
    /// camera motion compensation and optional `ReID`.
    #[default]
    BotSort,
    /// `OC-SORT`: `ByteTrack` with motion direction costs, observation-centric recovery and
    /// re-update.
    OcSort,
    /// `Deep OC-SORT`: `OC-SORT` with optional camera motion compensation and `ReID`.
    DeepOcSort,
    /// `FastTrack`: `ByteTrack` with occlusion handling.
    FastTrack,
    /// `TrackTrack`: iterative association over a combined cost, with track-aware initialization.
    TrackTrack,
}

impl FromStr for TrackerType {
    type Err = String;

    /// Parse a tracker name like `bytetrack`, also accepting the Ultralytics config names like
    /// `bytetrack.yaml`.
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.trim_end_matches(".yaml").to_ascii_lowercase().as_str() {
            "bytetrack" => Ok(Self::ByteTrack),
            "botsort" => Ok(Self::BotSort),
            "ocsort" => Ok(Self::OcSort),
            "deepocsort" => Ok(Self::DeepOcSort),
            "fasttrack" => Ok(Self::FastTrack),
            "tracktrack" => Ok(Self::TrackTrack),
            _ => Err(format!(
                "Unknown tracker '{s}', expected bytetrack, botsort, ocsort, deepocsort, fasttrack, tracktrack or a YAML file"
            )),
        }
    }
}

impl TrackerType {
    /// Whether the Kalman state is center, aspect ratio and height (otherwise width and height).
    pub(super) const fn xyah(self) -> bool {
        matches!(
            self,
            Self::ByteTrack | Self::OcSort | Self::DeepOcSort | Self::FastTrack
        )
    }

    pub(super) const fn ocsort(self) -> bool {
        matches!(self, Self::OcSort | Self::DeepOcSort)
    }

    /// Whether it can compensate camera motion and match by appearance.
    pub(super) const fn gmc_reid(self) -> bool {
        matches!(self, Self::BotSort | Self::DeepOcSort | Self::TrackTrack)
    }
}

/// Camera motion compensation method, the `gmc_method` of the Ultralytics tracker configs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GmcMethod {
    /// No compensation.
    None,
    /// Corners tracked by pyramidal Lucas-Kanade optical flow, fitted with a RANSAC similarity.
    SparseOptFlow,
}

impl FromStr for GmcMethod {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "sparseOptFlow" => Ok(Self::SparseOptFlow),
            "none" | "None" => Ok(Self::None),
            _ => Err(format!(
                "Unsupported gmc_method '{s}', expected sparseOptFlow or none"
            )),
        }
    }
}

/// Tracker settings, the keys of the Ultralytics tracker YAML configs.
///
/// [`TrackerConfig::from`] a [`TrackerType`] gives the values of its Ultralytics config, and
/// [`TrackerConfig::load`] reads a config file. Each tracker only reads its own keys.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackerConfig {
    /// Tracking algorithm.
    pub tracker_type: TrackerType,
    /// Detections scoring at least this take part in the first association.
    pub track_high_thresh: f32,
    /// Detections scoring above this and below `track_high_thresh` take part in the second.
    pub track_low_thresh: f32,
    /// Unmatched detections scoring at least this start a new track.
    pub new_track_thresh: f32,
    /// Frames a lost track is kept for.
    pub track_buffer: u32,
    /// Highest matching cost of the first association.
    pub match_thresh: f32,
    /// Weigh the `IoU` by the detection score when matching.
    pub fuse_score: bool,
    /// Camera motion compensation (`BoT-SORT`, `Deep OC-SORT`, `TrackTrack`).
    pub gmc_method: GmcMethod,
    /// Match by appearance too (`BoT-SORT`, `Deep OC-SORT`, `TrackTrack`).
    pub with_reid: bool,
    /// `ReID` ONNX model, downloaded when it is a known Ultralytics model. `auto` takes
    /// `yolo26n-reid.onnx`, since an ONNX detector has no features to reuse.
    pub model: String,
    /// Lowest `IoU` for an appearance match.
    pub proximity_thresh: f32,
    /// Lowest appearance similarity for an appearance match.
    pub appearance_thresh: f32,
    /// Frames over which `OC-SORT` measures the motion direction.
    pub delta_t: u32,
    /// Weight of the `OC-SORT` motion direction cost.
    pub inertia: f32,
    /// Run the `ByteTrack` low-score association in `OC-SORT`.
    pub use_byte: bool,
    /// `Deep OC-SORT` appearance moving average weight.
    pub alpha_fixed_emb: f32,
    /// `FastTrack`: history frames back to restore the velocity from when an occlusion starts.
    pub reset_velocity_offset_occ: usize,
    /// `FastTrack`: history frames back to restore the position from when an occlusion starts.
    pub reset_pos_offset_occ: usize,
    /// `FastTrack`: height scale of an occluded box.
    pub enlarge_bbox_occ: f64,
    /// `FastTrack`: velocity scale of an occluded box.
    pub dampen_motion_occ: f64,
    /// `FastTrack`: occluded frames before a track is lost.
    pub active_occ_to_lost_thresh: u32,
    /// `FastTrack`: covered fraction of a track that counts as occluded.
    pub occ_cover_thresh: f32,
    /// `FastTrack`: frames a recently occluded lost track is kept for.
    pub occ_reappear_window: u32,
    /// `FastTrack`: skip new tracks overlapping an active one by this `IoU` (1.0 disables).
    pub init_iou_suppress: f32,
    /// `TrackTrack`: matching cost for lost tracks in a second pass, 0 disables it.
    pub lost_match_thr: f32,
    /// `TrackTrack`: weight of the height-modulated `IoU` cost.
    pub iou_weight: f32,
    /// `TrackTrack`: weight of the appearance cost.
    pub reid_weight: f32,
    /// `TrackTrack`: weight of the confidence cost.
    pub conf_weight: f32,
    /// `TrackTrack`: weight of the corner angle cost.
    pub angle_weight: f32,
    /// `TrackTrack`: extra cost of low-score detections.
    pub penalty_p: f32,
    /// `TrackTrack`: threshold step of each association round.
    pub reduce_step: f32,
    /// `TrackTrack`: `IoU` that suppresses a new track.
    pub tai_thr: f32,
    /// `TrackTrack`: matched frames before a track shows.
    pub min_track_len: u32,
}

impl Default for TrackerConfig {
    /// The `botsort.yaml` values, and the other trackers' values for the other keys.
    fn default() -> Self {
        Self {
            tracker_type: TrackerType::BotSort,
            track_high_thresh: 0.25,
            track_low_thresh: 0.1,
            new_track_thresh: 0.25,
            track_buffer: 30,
            match_thresh: 0.8,
            fuse_score: true,
            gmc_method: GmcMethod::SparseOptFlow,
            with_reid: false,
            model: "auto".into(),
            proximity_thresh: 0.5,
            appearance_thresh: 0.8,
            delta_t: 3,
            inertia: 0.2,
            use_byte: false,
            alpha_fixed_emb: 0.95,
            reset_velocity_offset_occ: 5,
            reset_pos_offset_occ: 3,
            enlarge_bbox_occ: 1.1,
            dampen_motion_occ: 0.5,
            active_occ_to_lost_thresh: 10,
            occ_cover_thresh: 0.7,
            occ_reappear_window: 40,
            init_iou_suppress: 0.7,
            lost_match_thr: 0.0,
            iou_weight: 0.5,
            reid_weight: 0.5,
            conf_weight: 0.1,
            angle_weight: 0.05,
            penalty_p: 0.2,
            reduce_step: 0.05,
            tai_thr: 0.55,
            min_track_len: 3,
        }
    }
}

impl From<TrackerType> for TrackerConfig {
    /// The values of the Ultralytics config of `tracker_type`.
    fn from(tracker_type: TrackerType) -> Self {
        let base = Self {
            tracker_type,
            ..Self::default()
        };
        match tracker_type {
            TrackerType::DeepOcSort => Self {
                track_high_thresh: 0.3,
                new_track_thresh: 0.3,
                gmc_method: GmcMethod::None,
                appearance_thresh: 0.9,
                ..base
            },
            TrackerType::TrackTrack => Self {
                track_high_thresh: 0.6,
                track_low_thresh: 0.25,
                new_track_thresh: 0.7,
                match_thresh: 0.7,
                ..base
            },
            _ => base,
        }
    }
}

/// Set the config fields named like the YAML keys, ignoring other keys like Ultralytics does.
macro_rules! set_keys {
    ($config:ident, $key:ident, $value:ident; $($field:ident),*) => {
        match $key {
            $(stringify!($field) => {
                $config.$field = $value.parse().map_err(|e| format!("{}: {e}", $key))?;
            })*
            _ => {}
        }
    };
}

impl TrackerConfig {
    /// Load a tracker by name (`botsort`, `bytetrack.yaml`, ...) or from a YAML file with the keys
    /// of the Ultralytics tracker configs. Keys a file leaves out keep the values of its
    /// `tracker_type`.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown tracker name, an unreadable file or a bad value.
    pub fn load(tracker: &str) -> Result<Self> {
        let path = Path::new(tracker);
        if !path.is_file() {
            let kind = tracker.parse::<TrackerType>();
            return kind.map(Self::from).map_err(InferenceError::ConfigError);
        }
        std::fs::read_to_string(path)
            .map_err(|e| e.to_string())
            .and_then(|text| Self::from_yaml(&text))
            .map_err(|e| InferenceError::ConfigError(format!("Tracker '{tracker}': {e}")))
    }

    /// Parse the flat `key: value  # comment` lines of an Ultralytics tracker config.
    fn from_yaml(text: &str) -> std::result::Result<Self, String> {
        let pairs: Vec<(&str, &str)> = text
            .lines()
            .filter_map(|line| line.split('#').next()?.split_once(':'))
            .map(|(k, v)| (k.trim(), v.trim().trim_matches(['"', '\''])))
            .collect();
        let kind = pairs
            .iter()
            .find(|(k, _)| *k == "tracker_type")
            .ok_or("missing tracker_type")?;
        let mut config = Self::from(kind.1.parse::<TrackerType>()?);
        for (key, value) in pairs {
            let value = match value {
                "True" => "true",
                "False" => "false",
                v => v,
            };
            set_keys!(config, key, value;
                track_high_thresh, track_low_thresh, new_track_thresh, track_buffer, match_thresh,
                fuse_score, gmc_method, with_reid, model, proximity_thresh, appearance_thresh,
                delta_t, inertia, use_byte, alpha_fixed_emb, reset_velocity_offset_occ,
                reset_pos_offset_occ, enlarge_bbox_occ, dampen_motion_occ,
                active_occ_to_lost_thresh, occ_cover_thresh, occ_reappear_window, init_iou_suppress,
                lost_match_thr, iou_weight, reid_weight, conf_weight, angle_weight, penalty_p,
                reduce_step, tai_thr, min_track_len);
        }
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracker_config_parsing() {
        assert_eq!("bytetrack.yaml".parse(), Ok(TrackerType::ByteTrack));
        assert_eq!("BoTSORT".parse(), Ok(TrackerType::BotSort));
        assert!("sort".parse::<TrackerType>().is_err());
        let c = TrackerConfig::from_yaml(
            "tracker_type: deepocsort # comment\ntrack_buffer: 60\nwith_reid: True\nmodel: auto\n",
        )
        .unwrap();
        assert_eq!(c.tracker_type, TrackerType::DeepOcSort);
        assert_eq!((c.track_buffer, c.with_reid), (60, true));
        assert!((c.appearance_thresh - 0.9).abs() < f32::EPSILON);
        assert!(TrackerConfig::from_yaml("gmc_method: orb").is_err());
        assert!(TrackerConfig::from_yaml("tracker_type: botsort\ngmc_method: orb").is_err());
    }
}
