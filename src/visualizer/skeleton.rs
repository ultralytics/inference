// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Pose skeleton definition.
//!
//! Holds the COCO-Pose keypoint connections used to draw the skeleton lines
//! between detected keypoints, plus the per-limb/keypoint palette indices.

/// COCO-Pose dataset skeleton structure (pairs of keypoint indices).
pub const SKELETON: [[usize; 2]; 19] = [
    [15, 13],
    [13, 11],
    [16, 14],
    [14, 12],
    [11, 12],
    [5, 11],
    [6, 12],
    [5, 6],
    [5, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [1, 2],
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
];

/// Limb color indices into `POSE_COLORS` (arms=blue, legs=orange, face=green).
pub const LIMB_COLOR_INDICES: [usize; 19] = [
    0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16,
];

/// Keypoint color indices into `POSE_COLORS`.
pub const KPT_COLOR_INDICES: [usize; 17] = [16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0];
