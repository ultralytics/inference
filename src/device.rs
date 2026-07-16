// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Hardware device support and abstraction.
use std::fmt;
use std::str::FromStr;

/// Intel target hardware for the `OpenVINO` execution provider, selected with the
/// Ultralytics `intel:<type>` device name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intel {
    /// Intel CPU.
    Cpu,
    /// Intel GPU (integrated or discrete).
    Gpu,
    /// Intel NPU (Neural Processing Unit).
    Npu,
}

/// Hardware device for inference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Device {
    /// CPU (Central Processing Unit).
    Cpu,
    /// CUDA (Compute Unified Device Architecture) for NVIDIA GPUs.
    /// The argument specifies the device index (e.g., 0 for the first GPU).
    Cuda(usize),
    /// `CoreML` execution provider for Apple Silicon / macOS.
    CoreMl,
    /// `DirectML` (Direct Machine Learning) for Windows.
    /// The argument specifies the device index.
    DirectMl(usize),
    /// `OpenVINO` (Open Visual Inference and Neural Network Optimization) for Intel hardware,
    /// selected with `intel:cpu`, `intel:gpu`, or `intel:npu`.
    OpenVino(Intel),
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
            Self::OpenVino(Intel::Cpu) => write!(f, "intel:cpu"),
            Self::OpenVino(Intel::Gpu) => write!(f, "intel:gpu"),
            Self::OpenVino(Intel::Npu) => write!(f, "intel:npu"),
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
        // Ultralytics OpenVINO naming: `intel:cpu`, `intel:gpu`, `intel:npu`.
        if let Some(rest) = s.strip_prefix("intel") {
            return parse_intel(rest).map(Self::OpenVino);
        }
        match s.as_str() {
            "cpu" => Ok(Self::Cpu),
            "coreml" => Ok(Self::CoreMl),
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

/// Parse the Intel device from the suffix after `intel` (`":cpu"`, `":gpu"`, `":npu"`).
fn parse_intel(s: &str) -> Result<Intel, String> {
    match s {
        ":cpu" => Ok(Intel::Cpu),
        ":gpu" => Ok(Intel::Gpu),
        ":npu" => Ok(Intel::Npu),
        _ => Err(format!("Unknown OpenVINO device: intel{s}")),
    }
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
        // OpenVINO uses the Ultralytics `intel:<type>` naming.
        assert_eq!(
            Device::from_str("intel:cpu").unwrap(),
            Device::OpenVino(Intel::Cpu)
        );
        assert_eq!(
            Device::from_str("intel:gpu").unwrap(),
            Device::OpenVino(Intel::Gpu)
        );
        assert_eq!(
            Device::from_str("intel:npu").unwrap(),
            Device::OpenVino(Intel::Npu)
        );
        assert!(Device::from_str("intel").is_err());
        assert!(Device::from_str("intel:tpu").is_err());
        assert!(Device::from_str("openvino").is_err());
    }

    #[test]
    fn test_device_display() {
        assert_eq!(Device::Cpu.to_string(), "cpu");
        assert_eq!(Device::Cuda(0).to_string(), "cuda:0");
        assert_eq!(Device::Cuda(1).to_string(), "cuda:1");
        assert_eq!(Device::CoreMl.to_string(), "coreml");
        assert_eq!(Device::DirectMl(0).to_string(), "directml:0");
        assert_eq!(Device::OpenVino(Intel::Cpu).to_string(), "intel:cpu");
        assert_eq!(Device::OpenVino(Intel::Gpu).to_string(), "intel:gpu");
        assert_eq!(Device::OpenVino(Intel::Npu).to_string(), "intel:npu");
        assert_eq!(Device::Xnnpack.to_string(), "xnnpack");
        assert_eq!(Device::TensorRt(2).to_string(), "tensorrt:2");
        assert_eq!(Device::Rocm(3).to_string(), "rocm:3");
    }

    #[test]
    fn test_device_display_roundtrip() {
        for s in [
            "cpu",
            "cuda:0",
            "coreml",
            "directml:0",
            "intel:cpu",
            "intel:gpu",
            "intel:npu",
            "xnnpack",
        ] {
            assert_eq!(Device::from_str(s).unwrap().to_string(), s);
        }
    }
}
