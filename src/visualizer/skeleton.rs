// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

/// COCO-Pose dataset skeleton structure (pairs of keypoint indices)
/// Defines which keypoints connect to form the pose skeleton
pub const SKELETON: [[usize; 2]; 19] = [
    [15, 13], // right ankle to left knee
    [13, 11], // left knee to left hip
    [16, 14], // right ankle (16) to right knee
    [14, 12], // right knee to right hip
    [11, 12], // left hip to right hip
    [5, 11],  // left shoulder to left hip
    [6, 12],  // right shoulder to right hip
    [5, 6],   // left shoulder to right shoulder
    [5, 7],   // left shoulder to left elbow
    [6, 8],   // right shoulder to right elbow
    [7, 9],   // left elbow to left wrist
    [8, 10],  // right elbow to right wrist
    [1, 2],   // left eye to right eye
    [0, 1],   // nose to left eye
    [0, 2],   // nose to right eye
    [1, 3],   // left eye to left ear
    [2, 4],   // right eye to right ear
    [3, 5],   // left ear to left shoulder
    [4, 6],   // right ear to right shoulder
];

/// Limb color indices mapping to `POSE_COLORS`
/// Defines which color from the pose palette to use for each limb
/// Mapping: arms=blue, legs=orange, face=green
pub const LIMB_COLOR_INDICES: [usize; 19] = [
    0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16,
];

/// Keypoint color indices mapping to `POSE_COLORS`
/// Defines which color from the pose palette to use for each keypoint
/// Mapping: arms=blue, legs=orange, face=green
pub const KPT_COLOR_INDICES: [usize; 17] = [16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0];
