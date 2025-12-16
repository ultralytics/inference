// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Visualization tools for inference results.

/// Color definitions and palettes.
pub mod color;

#[cfg(feature = "visualize")]
pub mod viewer;

pub use color::Color;

#[cfg(feature = "visualize")]
pub use viewer::Viewer;
