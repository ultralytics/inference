// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Native `OpenVINO` IR (`.xml` + `.bin`) inference through the `openvino` crate.
//!
//! The runtime libraries are loaded when the first model opens, so building needs no
//! `OpenVINO` install. Exports from `yolo export format=openvino` keep f32 inputs and outputs
//! at every precision (FP32, FP16 weights, INT8), so only f32 tensors cross this boundary.

use std::path::{Path, PathBuf};

use ndarray::Array4;
use openvino::{
    CompiledModel, Core, DeviceType, ElementType, InferRequest, PartialShape, RwPropertyKey, Shape,
    Tensor,
};

use crate::Device;
use crate::error::{InferenceError, Result};

/// A compiled `OpenVINO` model with one reusable inference request.
pub struct OvSession {
    // Declared in drop order: the request goes before the model, the model before the core.
    request: InferRequest,
    _compiled: CompiledModel,
    _core: Core,
    /// Input dimensions as `[N, C, H, W]`, always static once loaded.
    input_dims: Vec<i64>,
    outputs: usize,
    /// The only output is a `uint8` class map, as in semantic exports with `ArgMax` baked in.
    mask_output: bool,
    /// The input tensor, filled by every run.
    input: Tensor,
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

/// Value of `name="..."` inside an XML tag.
fn attr<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    let rest = &tag[tag.find(&format!(" {name}=\""))? + name.len() + 3..];
    rest.split('"').next()
}

/// The `<edge .../>` tag feeding port 0 of layer `id`.
fn edge_into<'a>(xml: &'a str, id: &str) -> Option<&'a str> {
    let at = xml.find(&format!("to-layer=\"{id}\" to-port=\"0\""))?;
    let start = xml[..at].rfind("<edge ")?;
    Some(&xml[start..at + xml[at..].find("/>")? + 2])
}

/// Rewire a depth IR so its output is the input of the final `Interpolate`, which the caller
/// resamples on the host straight to the image size. The graph's own bilinear
/// `align_corners` upsample is wrong on the NPU, and on every device this skips resampling
/// the map twice. `None` when the output is not fed by an `Interpolate`.
fn drop_final_upsample(xml: &str) -> Option<String> {
    let result = xml.rfind("type=\"Result\"")?;
    let layer = &xml[xml[..result].rfind("<layer ")?..result];
    let result_id = attr(layer, "id")?;
    let out_edge = edge_into(xml, result_id)?;
    let interp_id = attr(out_edge, "from-layer")?;
    let interp = xml.find(&format!("<layer id=\"{interp_id}\" "))?;
    if !xml[interp..xml[interp..].find('>')? + interp].contains("type=\"Interpolate\"") {
        return None;
    }
    let in_edge = edge_into(xml, interp_id)?;
    let rewired = format!(
        "<edge from-layer=\"{}\" from-port=\"{}\" to-layer=\"{result_id}\" to-port=\"0\" />",
        attr(in_edge, "from-layer")?,
        attr(in_edge, "from-port")?
    );
    Some(xml.replace(out_edge, &rewired))
}

impl OvSession {
    /// Compile the IR at `xml` (weights next to it as `.bin`) for `device`. The NPU compiles
    /// only bounded shapes, the GPU recompiles its kernels for every new shape (about a second
    /// each) and the CPU runs dynamic shapes about 1.5x slower, so a dynamic model is pinned to
    /// `static_input` (`[N, C, H, W]`) on every device. The NPU always
    /// runs at batch 1, for batched and dynamic exports alike: batching gains nothing there.
    #[allow(clippy::too_many_lines)]
    pub fn load(
        xml: &Path,
        device: Option<&Device>,
        static_input: [i64; 4],
        host_upsample: bool,
        threads: usize,
    ) -> Result<Self> {
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
        let bin = xml.with_extension("bin");
        if !bin.is_file() {
            return Err(InferenceError::ModelLoadError(format!(
                "OpenVINO model weights not found: {}",
                bin.display()
            )));
        }
        let mut core = Core::new().map_err(load_err)?;
        let path = |p: &Path| p.to_string_lossy().into_owned();
        let mut model = match host_upsample
            .then(|| {
                std::fs::read_to_string(xml)
                    .ok()
                    .and_then(|x| drop_final_upsample(&x))
            })
            .flatten()
        {
            Some(cut) => {
                let weights = std::fs::read(&bin).map_err(load_err)?;
                let shape = Shape::new(&[i64::try_from(weights.len()).map_err(load_err)?])
                    .map_err(load_err)?;
                let mut tensor = Tensor::new(ElementType::U8, &shape).map_err(load_err)?;
                tensor
                    .get_raw_data_mut()
                    .map_err(load_err)?
                    .copy_from_slice(&weights);
                core.read_model_from_buffer(cut.as_bytes(), Some(&tensor))
                    .map_err(load_err)?
            }
            None => core
                .read_model_from_file(&path(xml), &path(&bin))
                .map_err(load_err)?,
        };
        let shape = model
            .get_input_by_index(0)
            .and_then(|n| n.get_partial_shape())
            .map_err(load_err)?;
        let mut dims: Vec<i64> = shape
            .get_dimensions()
            .iter()
            .map(openvino::Dimension::get_min)
            .collect();
        if dims.len() != 4 {
            return Err(load_err("OpenVINO models must take an [N, C, H, W] input"));
        }
        let pinned = if shape.is_dynamic() {
            let batch = if device == DeviceType::NPU {
                1
            } else {
                static_input[0]
            };
            Some([batch, static_input[1], static_input[2], static_input[3]])
        } else if device == DeviceType::NPU && dims[0] > 1 {
            Some([1, dims[1], dims[2], dims[3]])
        } else {
            None
        };
        if let Some(pinned) = pinned {
            let shape = PartialShape::new_static(4, &pinned).map_err(load_err)?;
            model.reshape_single_input(&shape).map_err(load_err)?;
            dims = pinned.to_vec();
        }
        let input = model.get_input_by_index(0).map_err(load_err)?;
        if input.get_element_type().map_err(load_err)? != ElementType::F32 {
            return Err(InferenceError::ModelLoadError(
                "OpenVINO models must take f32 input".to_owned(),
            ));
        }
        let outputs = model.get_outputs_len().map_err(load_err)?;
        let mask_output = outputs == 1
            && model
                .get_output_by_index(0)
                .and_then(|n| n.get_element_type())
                .map_err(load_err)?
                == ElementType::U8;
        if threads > 0 && device == DeviceType::CPU {
            core.set_property(
                &device,
                &RwPropertyKey::InferenceNumThreads,
                &threads.to_string(),
            )
            .map_err(load_err)?;
        }
        let mut compiled = core.compile_model(&model, device).map_err(load_err)?;
        let mut request = compiled.create_infer_request().map_err(load_err)?;
        let input = Tensor::new(ElementType::F32, &Shape::new(&dims).map_err(load_err)?)
            .map_err(load_err)?;
        request.set_input_tensor(&input).map_err(load_err)?;
        Ok(Self {
            request,
            _compiled: compiled,
            _core: core,
            input_dims: dims,
            input,
            outputs,
            mask_output,
        })
    }

    /// Model input dimensions as `[N, C, H, W]`.
    pub fn input_dims(&self) -> &[i64] {
        &self.input_dims
    }

    /// Whether the only output is a `uint8` class map rather than f32 tensors.
    pub const fn mask_output(&self) -> bool {
        self.mask_output
    }

    /// Run one forward pass over `inputs`, which together fill the model's batch, and return
    /// the output tensors.
    pub fn run<'a>(
        &mut self,
        inputs: impl IntoIterator<Item = &'a Array4<f32>>,
    ) -> Result<Vec<Tensor>> {
        let data = self.input.get_data_mut::<f32>().map_err(run_err)?;
        let mut filled = 0;
        for input in inputs {
            let input = input.as_standard_layout();
            let src = input.as_slice().expect("standard layout is contiguous");
            data.get_mut(filled..filled + src.len())
                .ok_or_else(|| run_err("input is larger than the model input"))?
                .copy_from_slice(src);
            filled += src.len();
        }
        if filled != data.len() {
            return Err(run_err(format!(
                "input has {filled} values, the model takes {}",
                data.len()
            )));
        }
        self.request.infer().map_err(run_err)?;
        (0..self.outputs)
            .map(|i| self.request.get_output_tensor_by_index(i).map_err(run_err))
            .collect()
    }
}

/// Element types an output tensor is read as.
pub trait Element {
    /// The matching `OpenVINO` element type.
    const TYPE: ElementType;
}

impl Element for f32 {
    const TYPE: ElementType = ElementType::F32;
}

impl Element for u8 {
    const TYPE: ElementType = ElementType::U8;
}

/// An output tensor as its data and `[usize]` shape. `get_data` reinterprets the raw bytes,
/// so the element type is checked first.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn view<T: Element>(tensor: &Tensor) -> Result<(&[T], Vec<usize>)> {
    let dtype = tensor.get_element_type().map_err(run_err)?;
    if dtype != T::TYPE {
        return Err(run_err(format!(
            "expected a {:?} output, not {dtype:?}",
            T::TYPE
        )));
    }
    let shape = tensor.get_shape().map_err(run_err)?;
    let data = tensor.get_data::<T>().map_err(run_err)?;
    Ok((
        data,
        shape.get_dimensions().iter().map(|&d| d as usize).collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drop_final_upsample_rewires_the_output() {
        let xml = r#"<layers><layer id="5" name="up" type="Interpolate"></layer><layer id="6" name="r" type="Result"></layer></layers><edges><edge from-layer="4" from-port="2" to-layer="5" to-port="0" /><edge from-layer="5" from-port="3" to-layer="6" to-port="0" /></edges>"#;
        let cut = drop_final_upsample(xml).unwrap();
        assert!(cut.contains(r#"<edge from-layer="4" from-port="2" to-layer="6" to-port="0" />"#));
        assert!(!cut.contains(r#"from-layer="5" from-port="3""#));
        assert!(drop_final_upsample(&xml.replace("Interpolate", "Convolution")).is_none());
    }

    #[test]
    fn test_resolve_xml() {
        let dir = tempfile::tempdir().unwrap();
        assert!(resolve_xml(dir.path()).is_err(), "no .xml in the folder");
        std::fs::write(dir.path().join("m.xml"), "").unwrap();
        std::fs::write(dir.path().join("m.bin"), "").unwrap();
        assert_eq!(resolve_xml(dir.path()).unwrap(), dir.path().join("m.xml"));
        assert_eq!(
            resolve_xml(&dir.path().join("m.xml")).unwrap(),
            dir.path().join("m.xml")
        );
        std::fs::write(dir.path().join("n.xml"), "").unwrap();
        assert!(
            resolve_xml(dir.path()).is_err(),
            "two .xml files are ambiguous"
        );
    }

    #[test]
    fn test_load_rejects_missing_weights_and_foreign_devices() {
        let dir = tempfile::tempdir().unwrap();
        let xml = dir.path().join("m.xml");
        std::fs::write(&xml, "").unwrap();
        let err = OvSession::load(&xml, None, [1, 3, 4, 4], false, 0)
            .err()
            .unwrap();
        assert!(err.to_string().contains("weights not found"), "{err}");
        let err = OvSession::load(&xml, Some(&Device::Cuda(0)), [1, 3, 4, 4], false, 0)
            .err()
            .unwrap();
        assert!(
            err.to_string().contains("cannot run an OpenVINO model"),
            "{err}"
        );
    }
}
