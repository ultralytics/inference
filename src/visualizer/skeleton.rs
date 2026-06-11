// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Pose skeleton definition.
//!
//! The skeleton and pose color-index tables now live in the wasm-safe
//! [`crate::colors`] module so the browser bindings can share them; this module
//! re-exports them for the native visualizer/annotator code paths.

pub use crate::colors::{KPT_COLOR_INDICES, LIMB_COLOR_INDICES, SKELETON};
