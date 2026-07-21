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

/// Sample the INFERNO colormap at normalized position `t` (clamped to `[0, 1]`), returning RGB.
///
/// A 6th-order polynomial fit of `OpenCV`'s `COLORMAP_INFERNO` (the same colormap Ultralytics'
/// `colorize_depth` uses), accurate to within ~9/255 per channel, so it needs no lookup table.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn inferno(t: f32) -> [u8; 3] {
    // Per-channel [r, g, b] coefficients, order 0..=6, evaluated with Horner's method.
    const C: [[f32; 3]; 7] = [
        [0.000_218_940_4, 0.001_651_005, -0.019_480_9],
        [0.106_513_4, 0.563_956_4, 3.932_712],
        [11.602_49, -3.972_854, -15.942_39],
        [-41.704, 17.436_4, 44.354_15],
        [77.162_94, -33.402_36, -81.807_31],
        [-71.319_43, 32.626_06, 73.209_52],
        [25.131_13, -12.242_67, -23.070_32],
    ];
    let t = t.clamp(0.0, 1.0);
    let mut out = [0u8; 3];
    for (ch, o) in out.iter_mut().enumerate() {
        let mut v = C[6][ch];
        for row in C.iter().take(6).rev() {
            v = v.mul_add(t, row[ch]);
        }
        *o = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    }
    out
}

/// Sample the classic JET rainbow colormap at normalized position `t` (clamped to `[0, 1]`).
///
/// Returns RGB: dark blue (low) → cyan → green → yellow → dark red (high). The standard
/// piecewise-linear jet, so it needs no lookup table.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn jet(t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    let ch = |center: f32| (1.5 - t.mul_add(4.0, -center).abs()).clamp(0.0, 1.0);
    [
        (ch(3.0) * 255.0).round() as u8,
        (ch(2.0) * 255.0).round() as u8,
        (ch(1.0) * 255.0).round() as u8,
    ]
}

/// Sample the reversed-Spectral colormap at normalized position `t` (clamped to `[0, 1]`).
///
/// Diverging blue → cyan → green → yellow → red (matplotlib `Spectral_r`), linearly
/// interpolated between its 11 anchor colors.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn spectral(t: f32) -> [u8; 3] {
    const A: [[u8; 3]; 11] = [
        [94, 79, 162],
        [51, 135, 188],
        [102, 194, 165],
        [170, 220, 164],
        [230, 245, 152],
        [255, 254, 190],
        [254, 224, 139],
        [253, 173, 96],
        [244, 109, 67],
        [212, 61, 79],
        [158, 1, 66],
    ];
    let x = t.clamp(0.0, 1.0) * 10.0;
    let i = (x as usize).min(9);
    let f = x - i as f32;
    let (lo, hi) = (A[i], A[i + 1]);
    let lerp = |a: u8, b: u8| f.mul_add(f32::from(b) - f32::from(a), f32::from(a)).round() as u8;
    [lerp(lo[0], hi[0]), lerp(lo[1], hi[1]), lerp(lo[2], hi[2])]
}

/// Sample a grayscale ramp at normalized position `t` (clamped to `[0, 1]`): black (low) → white
/// (high). Raw normalized depth with no color.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn gray(t: f32) -> [u8; 3] {
    let g = (t.clamp(0.0, 1.0) * 255.0).round() as u8;
    [g, g, g]
}

/// A continuous colormap for depth visualization.
///
/// `Jet` (default) is the classic rainbow, matching Python's `colorize_depth` default and
/// the ramp the Ultralytics iOS app renders depth with; `Inferno` and the diverging
/// `Spectral` (`Spectral_r`) are Python's other two options; `Gray` is raw grayscale,
/// available here and in the wasm API only.
///
/// The CLI always renders the default (`Jet`), like Python's `yolo predict`. Pick another
/// through [`annotate_image_with`](crate::annotate::annotate_image_with).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Colormap {
    /// Perceptual black → purple → orange → yellow (matches Python depth plots).
    Inferno,
    /// Classic rainbow: blue → cyan → green → yellow → red.
    #[default]
    Jet,
    /// Diverging blue → green → yellow → red (matplotlib `Spectral_r`).
    Spectral,
    /// Raw grayscale black → white, no color.
    Gray,
}

impl Colormap {
    /// Sample this colormap at normalized position `t` (clamped to `[0, 1]`), returning RGB.
    #[must_use]
    pub fn sample(self, t: f32) -> [u8; 3] {
        match self {
            Self::Inferno => inferno(t),
            Self::Jet => jet(t),
            Self::Spectral => spectral(t),
            Self::Gray => gray(t),
        }
    }
}

impl std::str::FromStr for Colormap {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "inferno" => Ok(Self::Inferno),
            "jet" => Ok(Self::Jet),
            "spectral" | "spectral_r" => Ok(Self::Spectral),
            "gray" | "grey" | "grayscale" => Ok(Self::Gray),
            _ => Err(format!(
                "invalid colormap '{s}', expected one of: inferno, jet, spectral, gray"
            )),
        }
    }
}

/// Blend factor for the colorized depth overlay: `(1 - alpha) * image + alpha * depth`,
/// matching Python's `Annotator.depth_map` default.
///
/// Shared by the native annotator and the browser crate, which bakes it into the overlay's
/// alpha channel so the canvas composites to the same result.
pub const DEPTH_ALPHA: f32 = 0.6;

/// How depth values are normalized before colormapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DepthViz {
    /// Metric min/max over valid pixels — near = low color, far = high color. Matches
    /// Python's `colorize_depth`.
    Metric,
    /// Inverse depth (`1/d`, disparity) with a 2–98 percentile clip — near = high color
    /// (warm). The default: inverting depth spreads the color range over nearby detail
    /// instead of the distant background, and the percentile clip keeps a few stray
    /// pixels from washing it out.
    #[default]
    Disparity,
}

impl std::str::FromStr for DepthViz {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "metric" => Ok(Self::Metric),
            "disparity" | "depthanything" => Ok(Self::Disparity),
            _ => Err(format!(
                "invalid depth-viz '{s}', expected one of: metric, disparity"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jet_and_colormap() {
        // Jet endpoints: dark blue at the low end, dark red at the high end.
        assert_eq!(jet(0.0), [0, 0, 128]);
        assert_eq!(jet(1.0), [128, 0, 0]);
        // Colormap dispatches to the matching function and parses from strings.
        assert_eq!(Colormap::Inferno.sample(0.5), inferno(0.5));
        assert_eq!(Colormap::Jet.sample(0.5), jet(0.5));
        assert_eq!("jet".parse::<Colormap>().unwrap(), Colormap::Jet);
        assert_eq!("INFERNO".parse::<Colormap>().unwrap(), Colormap::Inferno);
        assert!("magma".parse::<Colormap>().is_err());
        // Gray is a plain black→white ramp: endpoints and an even midpoint, all channels equal.
        assert_eq!(gray(0.0), [0, 0, 0]);
        assert_eq!(gray(1.0), [255, 255, 255]);
        assert_eq!(Colormap::Gray.sample(0.5), gray(0.5));
        assert_eq!("gray".parse::<Colormap>().unwrap(), Colormap::Gray);
    }

    #[test]
    fn test_spectral_and_depth_viz() {
        // Spectral_r endpoints and midpoint from the matplotlib anchors.
        assert_eq!(spectral(0.0), [94, 79, 162]);
        assert_eq!(spectral(1.0), [158, 1, 66]);
        assert_eq!(spectral(0.5), [255, 254, 190]);
        assert_eq!(spectral(-1.0), [94, 79, 162]); // clamps
        assert_eq!(Colormap::Spectral.sample(1.0), [158, 1, 66]);
        assert_eq!(
            "spectral_r".parse::<Colormap>().unwrap(),
            Colormap::Spectral
        );
        // DepthViz parsing.
        assert_eq!("metric".parse::<DepthViz>().unwrap(), DepthViz::Metric);
        assert_eq!(
            "disparity".parse::<DepthViz>().unwrap(),
            DepthViz::Disparity
        );
        assert!("log".parse::<DepthViz>().is_err());
        // Depth defaults to the rainbow ramp with near = warm, matching the Ultralytics iOS app.
        assert_eq!(Colormap::default(), Colormap::Jet);
        assert_eq!(DepthViz::default(), DepthViz::Disparity);
    }

    #[test]
    fn test_inferno_range_and_clamp() {
        // Low end is near-black, high end is bright yellow (INFERNO's endpoints).
        let lo = inferno(0.0);
        let hi = inferno(1.0);
        assert!(lo.iter().all(|&c| c < 20), "low end should be dark: {lo:?}");
        assert!(
            hi[0] > 200 && hi[1] > 200,
            "high end should be bright: {hi:?}"
        );
        // Out-of-range values clamp to the endpoints.
        assert_eq!(inferno(-1.0), lo);
        assert_eq!(inferno(2.0), hi);
    }

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
