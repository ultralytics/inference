// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Error types for the inference library.

use std::fmt;

/// Result type alias for inference operations.
pub type Result<T> = std::result::Result<T, InferenceError>;

/// Main error type for the inference library.
#[derive(Debug)]
pub enum InferenceError {
    /// Error loading the ONNX model.
    ModelLoadError(String),
    /// Error during model inference.
    InferenceError(String),
    /// Error processing images.
    ImageError(String),
    /// Invalid configuration provided.
    ConfigError(String),
    /// IO error (file not found, permission denied, etc.).
    IoError(String),
    /// Wrapped `std::io::Error`
    Io(std::io::Error),
    /// Error parsing model metadata.
    MetadataError(String),
    /// Post-processing error.
    PostProcessingError(String),
    /// Visualizer error.
    VisualizerError(String),
    /// Video/stream processing error.
    VideoError(String),
    /// Feature not enabled.
    FeatureNotEnabled(String),
}

impl fmt::Display for InferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelLoadError(msg) => write!(f, "Model load error: {msg}"),
            Self::InferenceError(msg) => write!(f, "Inference error: {msg}"),
            Self::ImageError(msg) => write!(f, "Image error: {msg}"),
            Self::ConfigError(msg) => write!(f, "Config error: {msg}"),
            Self::IoError(msg) => write!(f, "IO error: {msg}"),
            Self::Io(err) => write!(f, "IO error: {err}"),
            Self::MetadataError(msg) => write!(f, "Metadata error: {msg}"),
            Self::PostProcessingError(msg) => write!(f, "Post-processing error: {msg}"),
            Self::VisualizerError(msg) => write!(f, "Visualizer error: {msg}"),
            Self::VideoError(msg) => write!(f, "Video error: {msg}"),
            Self::FeatureNotEnabled(msg) => write!(f, "Feature not enabled: {msg}"),
        }
    }
}

impl std::error::Error for InferenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for InferenceError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<image::ImageError> for InferenceError {
    fn from(err: image::ImageError) -> Self {
        Self::ImageError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = InferenceError::ModelLoadError("test".to_string());
        assert_eq!(err.to_string(), "Model load error: test");

        let err = InferenceError::InferenceError("test".to_string());
        assert_eq!(err.to_string(), "Inference error: test");
    }
}
