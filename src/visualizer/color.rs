// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Color types and named constants used for drawing detections and overlays.
//!
//! The palette and `Color` type now live in the wasm-safe [`crate::colors`]
//! module so the browser bindings can share them; this module re-exports them
//! for the native visualizer/annotator code paths.

pub use crate::colors::{COLORS, Color, POSE_COLORS};
