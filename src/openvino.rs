// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Native `OpenVINO` IR (`.xml` + `.bin`) inference through the `openvino` crate.
//!
//! The runtime libraries are loaded when the first model opens, so building needs no
//! `OpenVINO` install. Exports from `yolo export format=openvino` keep f32 inputs and outputs
//! at every precision (FP32, FP16 weights, INT8), so only f32 tensors cross this boundary.

use std::path::{Path, PathBuf};

use ndarray::Array4;
use openvino::{CompiledModel, Core, DeviceType, ElementType, InferRequest, Shape, Tensor};

use crate::Device;
use crate::error::{InferenceError, Result};

/// A compiled `OpenVINO` model with one reusable inference request.
pub struct OvSession {
    // Declared in drop order: the request goes before the model, the model before the core.
    request: InferRequest,
    _compiled: CompiledModel,
    _core: Core,
    /// Input dimensions as `[N, C, H, W]`, with `-1` for a dynamic axis.
    input_dims: Vec<i64>,
    outputs: usize,
}

fn load_err(e: impl std::fmt::Display) -> InferenceError {
    InferenceError::ModelLoadError(format!("Failed to load OpenVINO model: {e}"))
}

fn run_err(e: impl std::fmt::Display) -> InferenceError {
    InferenceError::InferenceError(format!("OpenVINO inference failed: {e}"))
}

/// The `.xml` file of an IR model given as the file itself or its export folder.
pub fn resolve_xml(path: &Path) -> Result<PathBuf> {
    if path.is_dir() {
        let mut xmls = std::fs::read_dir(path)
            .map_err(load_err)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "xml"));
        return match (xmls.next(), xmls.next()) {
            (Some(xml), None) => Ok(xml),
            _ => Err(InferenceError::ModelLoadError(format!(
                "Expected exactly one .xml file in {}",
                path.display()
            ))),
        };
    }
    Ok(path.to_path_buf())
}

impl OvSession {
    /// Compile the IR at `xml` (weights next to it as `.bin`) for `device`.
    pub fn load(xml: &Path, device: Option<&Device>) -> Result<Self> {
        let device = match device {
            None | Some(Device::Cpu | Device::IntelCpu) => DeviceType::CPU,
            Some(Device::IntelGpu) => DeviceType::GPU,
            Some(Device::IntelNpu) => DeviceType::NPU,
            Some(other) => {
                return Err(InferenceError::ConfigError(format!(
                    "Device '{other}' cannot run an OpenVINO model; use cpu, intel:cpu, intel:gpu or intel:npu"
                )));
            }
        };
        let mut core = Core::new().map_err(load_err)?;
        let path = |p: &Path| p.to_string_lossy().into_owned();
        let model = core
            .read_model_from_file(&path(xml), &path(&xml.with_extension("bin")))
            .map_err(load_err)?;
        let input = model.get_input_by_index(0).map_err(load_err)?;
        if input.get_element_type().map_err(load_err)? != ElementType::F32 {
            return Err(InferenceError::ModelLoadError(
                "OpenVINO models must take f32 input".to_owned(),
            ));
        }
        let input_dims = input
            .get_partial_shape()
            .map_err(load_err)?
            .get_dimensions()
            .iter()
            .map(|d| if d.is_dynamic() { -1 } else { d.get_min() })
            .collect();
        let outputs = model.get_outputs_len().map_err(load_err)?;
        let mut compiled = core.compile_model(&model, device).map_err(load_err)?;
        let request = compiled.create_infer_request().map_err(load_err)?;
        Ok(Self {
            request,
            _compiled: compiled,
            _core: core,
            input_dims,
            outputs,
        })
    }

    /// Model input dimensions as `[N, C, H, W]`, `-1` where an axis is dynamic.
    pub fn input_dims(&self) -> &[i64] {
        &self.input_dims
    }

    /// Run one forward pass and return the output tensors.
    pub fn run(&mut self, input: &Array4<f32>) -> Result<Vec<Tensor>> {
        let input = input.as_standard_layout();
        #[allow(clippy::cast_possible_wrap)]
        let dims: Vec<i64> = input.shape().iter().map(|&d| d as i64).collect();
        let mut tensor =
            Tensor::new(ElementType::F32, &Shape::new(&dims).map_err(run_err)?).map_err(run_err)?;
        tensor
            .get_data_mut::<f32>()
            .map_err(run_err)?
            .copy_from_slice(input.as_slice().expect("standard layout is contiguous"));
        self.request.set_input_tensor(&tensor).map_err(run_err)?;
        self.request.infer().map_err(run_err)?;
        (0..self.outputs)
            .map(|i| self.request.get_output_tensor_by_index(i).map_err(run_err))
            .collect()
    }
}
