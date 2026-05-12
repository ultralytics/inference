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
    /// CoreML execution provider for Apple Silicon / macOS.
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

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.to_lowercase();
        if let Some(rest) = s.strip_prefix("cuda") {
            return Ok(Self::Cuda(parse_device_index(rest)));
        }
        if let Some(rest) = s.strip_prefix("directml") {
            return Ok(Self::DirectMl(parse_device_index(rest)));
        }
        if let Some(rest) = s.strip_prefix("tensorrt") {
            return Ok(Self::TensorRt(parse_device_index(rest)));
        }
        if let Some(rest) = s.strip_prefix("rocm") {
            return Ok(Self::Rocm(parse_device_index(rest)));
        }
        match s.as_str() {
            "cpu" => Ok(Self::Cpu),
            "coreml" => Ok(Self::CoreMl),
            "openvino" => Ok(Self::OpenVino),
            "xnnpack" => Ok(Self::Xnnpack),
            _ => Err(format!("Unknown device: {s}")),
        }
    }
}

/// Parse a trailing device index like `":0"`, defaulting to `0` when absent.
fn parse_device_index(s: &str) -> usize {
    s.strip_prefix(':')
        .and_then(|i| i.parse().ok())
        .unwrap_or(0)
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
        assert_eq!(Device::from_str("coreml").unwrap(), Device::CoreMl);
        assert_eq!(Device::from_str("directml").unwrap(), Device::DirectMl(0));
        assert_eq!(Device::from_str("directml:1").unwrap(), Device::DirectMl(1));
    }
}
