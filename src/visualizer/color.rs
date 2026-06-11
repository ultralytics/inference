// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Color types and named constants used for drawing detections and overlays.

/// Color type for visualization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Color(pub u8, pub u8, pub u8);

impl Color {
    /// Red color.
    pub const RED: Self = Self(255, 0, 0);
    /// Green color.
    pub const GREEN: Self = Self(0, 255, 0);
    /// Blue color.
    pub const BLUE: Self = Self(0, 0, 255);
    /// White color.
    pub const WHITE: Self = Self(255, 255, 255);
    /// Black color.
    pub const BLACK: Self = Self(0, 0, 0);

    /// Create a new color from RGB values.
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self(r, g, b)
    }

    /// Get a color from the predefined palette by index (cycles through the
    /// standard Ultralytics color palette).
    #[must_use]
    pub const fn from_index(index: usize) -> Self {
        let color = COLORS[index % COLORS.len()];
        Self(color[0], color[1], color[2])
    }

    /// Get a color from the pose palette by index (cycles through the
    /// pose-specific palette).
    #[must_use]
    pub const fn from_pose_index(index: usize) -> Self {
        let color = POSE_COLORS[index % POSE_COLORS.len()];
        Self(color[0], color[1], color[2])
    }

    /// Format this color as a CSS hex string (`#rrggbb`).
    #[must_use]
    pub fn to_hex(self) -> String {
        format!("#{:02X}{:02X}{:02X}", self.0, self.1, self.2)
    }
}

/// Ultralytics Color Palette
pub const COLORS: [[u8; 3]; 20] = [
    [4, 42, 255],    // #042aff
    [11, 219, 235],  // #0bdbeb
    [243, 243, 243], // #f3f3f3
    [0, 223, 183],   // #00dfb7
    [17, 31, 104],   // #111f68
    [255, 111, 221], // #ff6fdd
    [255, 68, 79],   // #ff444f
    [204, 237, 0],   // #cced00
    [0, 243, 68],    // #00f344
    [189, 0, 255],   // #bd00ff
    [0, 180, 255],   // #00b4ff
    [221, 0, 186],   // #dd00ba
    [0, 255, 255],   // #00ffff
    [38, 192, 0],    // #26c000
    [1, 255, 179],   // #01ffb3
    [125, 36, 255],  // #7d24ff
    [123, 0, 104],   // #7b0068
    [255, 27, 108],  // #ff1b6c
    [252, 109, 47],  // #fc6d2f
    [162, 255, 11],  // #a2ff0b
];

/// Ultralytics Pose Color Palette
pub const POSE_COLORS: [[u8; 3]; 20] = [
    [255, 128, 0],   // #ff8000
    [255, 153, 51],  // #ff9933
    [255, 178, 102], // #ffb266
    [230, 230, 0],   // #e6e600
    [255, 153, 255], // #ff99ff
    [153, 204, 255], // #99ccff
    [255, 102, 255], // #ff66ff
    [255, 51, 255],  // #ff33ff
    [102, 178, 255], // #66b2ff
    [51, 153, 255],  // #3399ff
    [255, 153, 153], // #ff9999
    [255, 102, 102], // #ff6666
    [255, 51, 51],   // #ff3333
    [153, 255, 153], // #99ff99
    [102, 255, 102], // #66ff66
    [51, 255, 51],   // #33ff33
    [0, 255, 0],     // #00ff00
    [0, 0, 255],     // #0000ff
    [255, 0, 0],     // #ff0000
    [255, 255, 255], // #ffffff
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_constants() {
        assert_eq!(Color::RED, Color(255, 0, 0));
        assert_eq!(Color::BLUE, Color(0, 0, 255));
    }

    #[test]
    fn test_from_index() {
        assert_eq!(Color::from_index(0), Color(4, 42, 255));
        assert_eq!(Color::from_index(COLORS.len()), Color(4, 42, 255));
    }

    #[test]
    fn test_from_pose_index() {
        assert_eq!(Color::from_pose_index(0), Color(255, 128, 0));
    }

    #[test]
    fn test_to_hex() {
        assert_eq!(Color::from_index(0).to_hex(), "#042AFF");
        assert_eq!(Color::BLACK.to_hex(), "#000000");
    }
}
