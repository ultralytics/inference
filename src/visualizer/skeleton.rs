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

#[cfg(test)]
mod tests {
    use super::*;

    /// COCO-Pose uses 17 keypoints; every skeleton endpoint must index one of them.
    const NUM_KEYPOINTS: usize = 17;

    #[test]
    fn test_skeleton_indices_in_range() {
        for pair in SKELETON {
            for &idx in &pair {
                assert!(idx < NUM_KEYPOINTS, "skeleton index {idx} out of range");
            }
            assert_ne!(
                pair[0], pair[1],
                "skeleton bone connects a keypoint to itself"
            );
        }
    }

    #[test]
    fn test_palette_lengths_match_spec() {
        // One color per bone, one color per keypoint.
        assert_eq!(SKELETON.len(), LIMB_COLOR_INDICES.len());
        assert_eq!(KPT_COLOR_INDICES.len(), NUM_KEYPOINTS);
    }

    #[test]
    fn test_color_indices_within_pose_palette() {
        // The pose palette has 20 entries (indices 0..=19).
        const POSE_PALETTE_LEN: usize = 20;
        for &c in &LIMB_COLOR_INDICES {
            assert!(c < POSE_PALETTE_LEN, "limb color index {c} out of palette");
        }
        for &c in &KPT_COLOR_INDICES {
            assert!(c < POSE_PALETTE_LEN, "kpt color index {c} out of palette");
        }
    }
}
