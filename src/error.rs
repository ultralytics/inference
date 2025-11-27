// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Error types for the inference library

use std::fmt;

/// Result type alias for inference operations
pub type Result<T> = std::result::Result<T, InferenceError>;

/// Main error type for the inference library
#[derive(Debug)]
pub enum InferenceError {
    /// Error loading model
    ModelLoadError(String),
    /// Error during inference
    InferenceError(String),
    /// Image processing error
    ImageError(String),
    /// Invalid configuration
    ConfigError(String),
    /// IO error
    IoError(std::io::Error),
}

impl fmt::Display for InferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InferenceError::ModelLoadError(msg) => write!(f, "Model load error: {}", msg),
            InferenceError::InferenceError(msg) => write!(f, "Inference error: {}", msg),
            InferenceError::ImageError(msg) => write!(f, "Image error: {}", msg),
            InferenceError::ConfigError(msg) => write!(f, "Config error: {}", msg),
            InferenceError::IoError(err) => write!(f, "IO error: {}", err),
        }
    }
}

impl std::error::Error for InferenceError {}

impl From<std::io::Error> for InferenceError {
    fn from(err: std::io::Error) -> Self {
        InferenceError::IoError(err)
    }
}
