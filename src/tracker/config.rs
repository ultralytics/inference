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
    /// `BoT-SORT`, the Ultralytics default: `ByteTrack` with a center-width-height Kalman state.
    #[default]
    BotSort,
}

impl FromStr for TrackerType {
    type Err = String;

    /// Parse a tracker name like `bytetrack`, also accepting the Ultralytics config names like
    /// `bytetrack.yaml`.
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.trim_end_matches(".yaml").to_ascii_lowercase().as_str() {
            "bytetrack" => Ok(Self::ByteTrack),
            "botsort" => Ok(Self::BotSort),
            _ => Err(format!(
                "Unknown tracker '{s}', expected bytetrack, botsort or a YAML file"
            )),
        }
    }
}

impl TrackerType {
    /// Whether the Kalman state is center, aspect ratio and height (otherwise width and height).
    pub(super) const fn xyah(self) -> bool {
        matches!(self, Self::ByteTrack)
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
        }
    }
}

impl From<TrackerType> for TrackerConfig {
    /// The values of the Ultralytics config of `tracker_type`.
    fn from(tracker_type: TrackerType) -> Self {
        Self {
            tracker_type,
            ..Self::default()
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
        let parsed = if path.is_file() {
            std::fs::read_to_string(path)
                .map_err(InferenceError::Io)
                .and_then(|text| Self::from_yaml(&text).map_err(InferenceError::ConfigError))
        } else {
            tracker
                .parse::<TrackerType>()
                .map(Self::from)
                .map_err(InferenceError::ConfigError)
        };
        parsed.map_err(|e| InferenceError::ConfigError(format!("Tracker '{tracker}': {e}")))
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
                fuse_score);
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
            "tracker_type: bytetrack # comment\ntrack_buffer: 60\nfuse_score: False\n",
        )
        .unwrap();
        assert_eq!(c.tracker_type, TrackerType::ByteTrack);
        assert_eq!((c.track_buffer, c.fuse_score), (60, false));
        assert!((c.track_high_thresh - 0.25).abs() < f32::EPSILON);
        assert!(TrackerConfig::from_yaml("track_buffer: 60").is_err());
    }
}
