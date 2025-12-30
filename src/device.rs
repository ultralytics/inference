// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Hardware device support and abstraction.
use std::fmt;
use std::str::FromStr;

/// Hardware device for inference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Device {
    /// CPU (Central Processing Unit).
    Cpu,
    /// CUDA (Compute Unified Device Architecture) for NVIDIA GPUs.
    /// The argument specifies the device index (e.g., 0 for the first GPU).
    Cuda(usize),
    /// MPS (Metal Performance Shaders) for Apple Silicon (macOS).
    Mps,
    /// `CoreML` (Apple Core Machine Learning).
    CoreMl,
    /// `DirectML` (Direct Machine Learning) for Windows.
    /// The argument specifies the device index.
    DirectMl(usize),
    /// `OpenVINO` (Open Visual Inference and Neural Network Optimization) for Intel hardware.
    OpenVino,
    /// XNNPACK (optimized floating-point neural network inference operators) for CPU.
    Xnnpack,
    /// `TensorRT` (NVIDIA `TensorRT`) for high-performance deep learning inference.
    /// The argument specifies the device index.
    TensorRt(usize),
    /// `ROCm` (Radeon Open Compute) for AMD GPUs.
    /// The argument specifies the device index.
    Rocm(usize),
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Cuda(i) => write!(f, "cuda:{i}"),
            Self::Mps => write!(f, "mps"),
            Self::CoreMl => write!(f, "coreml"),
            Self::DirectMl(i) => write!(f, "directml:{i}"),
            Self::OpenVino => write!(f, "openvino"),
            Self::Xnnpack => write!(f, "xnnpack"),
            Self::TensorRt(i) => write!(f, "tensorrt:{i}"),
            Self::Rocm(i) => write!(f, "rocm:{i}"),
        }
    }
}

impl FromStr for Device {
    type Err = String;

    #[allow(clippy::option_if_let_else)]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.to_lowercase();
        match s.as_str() {
            "cpu" => Ok(Self::Cpu),
            "mps" => Ok(Self::Mps),
            "coreml" => Ok(Self::CoreMl),
            "openvino" => Ok(Self::OpenVino),
            "xnnpack" => Ok(Self::Xnnpack),
            _ => s.strip_prefix("cuda").map_or_else(
                || {
                    if let Some(rest) = s.strip_prefix("directml") {
                        let index = parse_device_index(rest).unwrap_or(0);
                        Ok(Self::DirectMl(index))
                    } else if let Some(rest) = s.strip_prefix("tensorrt") {
                        let index = parse_device_index(rest).unwrap_or(0);
                        Ok(Self::TensorRt(index))
                    } else if let Some(rest) = s.strip_prefix("rocm") {
                        let index = parse_device_index(rest).unwrap_or(0);
                        Ok(Self::Rocm(index))
                    } else {
                        Err(format!("Unknown device: {s}"))
                    }
                },
                |rest| {
                    let index = parse_device_index(rest).unwrap_or(0);
                    Ok(Self::Cuda(index))
                },
            ),
        }
    }
}

/// Helper to parse device index from string (e.g. ":0")
fn parse_device_index(s: &str) -> Option<usize> {
    if s.is_empty() {
        return None;
    }
    // Handle ":0", ":1" etc.
    s.strip_prefix(':')
        .and_then(|index_str| index_str.parse::<usize>().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_device() {
        assert_eq!(Device::from_str("cpu").unwrap(), Device::Cpu);
        assert_eq!(Device::from_str("cuda").unwrap(), Device::Cuda(0));
        assert_eq!(Device::from_str("cuda:0").unwrap(), Device::Cuda(0));
        assert_eq!(Device::from_str("cuda:1").unwrap(), Device::Cuda(1));
        assert_eq!(Device::from_str("mps").unwrap(), Device::Mps);
        assert_eq!(Device::from_str("coreml").unwrap(), Device::CoreMl);
        assert_eq!(Device::from_str("directml").unwrap(), Device::DirectMl(0));
        assert_eq!(Device::from_str("directml:1").unwrap(), Device::DirectMl(1));
    }
}
