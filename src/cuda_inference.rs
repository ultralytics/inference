// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Internal GPU preprocess for the `cuda-preprocess` feature.
//!
//! Implements a fused CUDA kernel that performs bilinear letterbox-resize +
//! `/255` normalize + (optional) BGR↔RGB swap + HWC→CHW packing in a single
//! launch, writing directly into the device-side input buffer that ONNX
//! Runtime's TRT/CUDA EP reads via [`ort::value::TensorRefMut::from_raw`].
//!
//! The whole module is `pub(crate)` — public API consumers reach this path
//! through [`crate::YOLOModel`] + [`crate::InferenceConfig`]'s
//! `with_cuda_preprocess` flag.

#![allow(unsafe_code)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    // `pub(crate)` here documents intent (these types are crate-internal API
    // for `model.rs`); the module just happens to be private too.
    clippy::redundant_pub_crate
)]

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx_with_opts;

use crate::error::{InferenceError, Result};

/// CUDA preprocess kernel source (NVRTC-compiled once per `CudaPreprocessor`).
///
/// Reads BGR/RGB HWC u8 input, writes RGB CHW f32 in `[0,1]` with letterbox padding.
/// `bgr_in=1` for OpenCV-style BGR inputs, `bgr_in=0` for RGB inputs (e.g. from the
/// `image` crate's `to_rgb8`).
const KERNEL_SRC: &str = r#"
extern "C" __global__ void preprocess(
    const unsigned char* __restrict__ src,
    int src_h, int src_w,
    float* __restrict__ dst,
    int dst_size,
    float scale,
    int pad_x, int pad_y,
    int bgr_in)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_size || y >= dst_size) return;

    int rx = x - pad_x;
    int ry = y - pad_y;
    int valid_w = (int)(src_w * scale + 0.5f);
    int valid_h = (int)(src_h * scale + 0.5f);

    float r, g, b;
    if (rx < 0 || ry < 0 || rx >= valid_w || ry >= valid_h) {
        r = g = b = 114.0f / 255.0f;
    } else {
        float fx = (float)rx / scale;
        float fy = (float)ry / scale;
        if (fx < 0.f) fx = 0.f;
        if (fy < 0.f) fy = 0.f;
        if (fx > (float)(src_w - 1)) fx = (float)(src_w - 1);
        if (fy > (float)(src_h - 1)) fy = (float)(src_h - 1);

        int x0 = (int)fx;
        int y0 = (int)fy;
        int x1 = x0 + 1; if (x1 > src_w - 1) x1 = src_w - 1;
        int y1 = y0 + 1; if (y1 > src_h - 1) y1 = src_h - 1;
        float dx = fx - (float)x0;
        float dy = fy - (float)y0;

        const unsigned char* p00 = src + (y0 * src_w + x0) * 3;
        const unsigned char* p01 = src + (y0 * src_w + x1) * 3;
        const unsigned char* p10 = src + (y1 * src_w + x0) * 3;
        const unsigned char* p11 = src + (y1 * src_w + x1) * 3;

        float w00 = (1.f - dx) * (1.f - dy);
        float w01 =        dx  * (1.f - dy);
        float w10 = (1.f - dx) *        dy ;
        float w11 =        dx  *        dy ;

        int ir = bgr_in ? 2 : 0;
        int ig = 1;
        int ib = bgr_in ? 0 : 2;

        float inv255 = 1.0f / 255.0f;
        r = (w00 * p00[ir] + w01 * p01[ir] + w10 * p10[ir] + w11 * p11[ir]) * inv255;
        g = (w00 * p00[ig] + w01 * p01[ig] + w10 * p10[ig] + w11 * p11[ig]) * inv255;
        b = (w00 * p00[ib] + w01 * p01[ib] + w10 * p10[ib] + w11 * p11[ib]) * inv255;
    }

    int hw = dst_size * dst_size;
    int idx = y * dst_size + x;
    dst[0 * hw + idx] = r;
    dst[1 * hw + idx] = g;
    dst[2 * hw + idx] = b;
}
"#;

/// Geometry produced by [`CudaPreprocessor::preprocess`].
///
/// `scale` is the uniform letterbox scale factor; `pad_x`/`pad_y` are the
/// pixel offsets applied along each axis. These values are needed by
/// post-processing to map model-space coordinates back to source pixels.
pub(crate) struct PreGeom {
    pub scale: f32,
    pub pad_x: i32,
    pub pad_y: i32,
}

/// Phase-1 of the fast-path init: a cudarc context + stream created before
/// the ORT session is built, so the raw stream pointer can be handed to the
/// TRT/CUDA EPs via `with_compute_stream`. Phase-2 ([`CudaPreprocessor::finalize`])
/// happens after the session is committed and the model's input size is known.
pub(crate) struct CudaStreamHandle {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

impl CudaStreamHandle {
    pub(crate) fn open(device_id: usize) -> Result<Self> {
        let ctx = CudaContext::new(device_id).map_err(|e| {
            InferenceError::ModelLoadError(format!("cudarc CudaContext::new({device_id}): {e:?}"))
        })?;
        let stream = ctx.default_stream();
        Ok(Self { ctx, stream })
    }

    /// Raw cudarc stream pointer, suitable for `ort`'s `with_compute_stream`.
    /// The handle (and therefore the stream) must outlive every session that
    /// binds to it.
    pub(crate) fn raw_stream_ptr(&self) -> *mut () {
        self.stream.cu_stream().cast()
    }
}

/// Fused GPU preprocess: holds the cudarc stream, compiled kernel, and
/// persistent device input/scratch buffers.
///
/// Built in two phases at model load time: [`CudaStreamHandle::open`] runs
/// *before* the ORT session is constructed (so the stream can be plumbed
/// into the TRT/CUDA EPs), then [`Self::finalize`] runs *after* the session
/// commit (so the model input edge length is known and the input buffer can
/// be sized correctly). Per-frame work happens in [`Self::preprocess`].
pub(crate) struct CudaPreprocessor {
    /// Kept alive so the stream below outlives every session bound to it.
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernel: CudaFunction,

    /// Persistent device buffer for the raw u8 source frame; grown on demand
    /// when frame size changes between calls.
    frame_dev: CudaSlice<u8>,
    frame_dev_capacity: usize,

    /// Persistent device buffer for the model input tensor (3*H*W f32).
    /// Pointer is stable for the buffer's lifetime — cached so callers can
    /// hand it to ORT via `TensorRefMut::from_raw` without re-querying.
    input_dev: CudaSlice<f32>,
    input_dev_ptr: u64,

    dst_size: usize,
}

impl CudaPreprocessor {
    /// Phase-2 init: NVRTC-compile the preprocess kernel and pre-allocate the
    /// device input buffer (`3 * dst_size^2` f32). Reuses the context+stream
    /// from [`CudaStreamHandle::open`] so ORT and this preprocessor share the
    /// same compute stream.
    pub(crate) fn finalize(handle: CudaStreamHandle, dst_size: usize) -> Result<Self> {
        let CudaStreamHandle { ctx, stream } = handle;

        let ptx = compile_ptx_with_opts(KERNEL_SRC, cudarc::nvrtc::CompileOptions::default())
            .map_err(|e| InferenceError::ModelLoadError(format!("NVRTC compile: {e:?}")))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| InferenceError::ModelLoadError(format!("load_module: {e:?}")))?;
        let kernel = module
            .load_function("preprocess")
            .map_err(|e| InferenceError::ModelLoadError(format!("load_function: {e:?}")))?;

        let input_elems = 3 * dst_size * dst_size;
        let input_dev = stream
            .alloc_zeros::<f32>(input_elems)
            .map_err(|e| InferenceError::ModelLoadError(format!("alloc input_dev: {e:?}")))?;
        let input_dev_ptr = {
            let (p, _g) = input_dev.device_ptr(&stream);
            p
        };

        let frame_dev = stream
            .alloc_zeros::<u8>(1)
            .map_err(|e| InferenceError::ModelLoadError(format!("alloc frame_dev: {e:?}")))?;

        Ok(Self {
            _ctx: ctx,
            stream,
            kernel,
            frame_dev,
            frame_dev_capacity: 1,
            input_dev,
            input_dev_ptr,
            dst_size,
        })
    }

    /// Device pointer of the model input buffer (`3 * dst_size^2` f32),
    /// stable for the lifetime of this preprocessor.
    pub(crate) const fn input_dev_ptr(&self) -> u64 {
        self.input_dev_ptr
    }

    /// Square model input edge (H == W).
    pub(crate) const fn dst_size(&self) -> usize {
        self.dst_size
    }

    /// H2D-copy the source frame, launch the fused preprocess kernel writing
    /// into the input buffer, and return the letterbox geometry needed by
    /// post-processing.
    ///
    /// The output of this call is enqueued on the same stream that the ORT
    /// session was bound to via [`CudaStreamHandle::raw_stream_ptr`], so the
    /// subsequent `session.run_binding(...)` sees the writes without an
    /// explicit synchronize.
    pub(crate) fn preprocess(
        &mut self,
        frame_hwc: &[u8],
        src_h: u32,
        src_w: u32,
        bgr_in: bool,
    ) -> Result<PreGeom> {
        let needed = (src_h as usize) * (src_w as usize) * 3;
        if frame_hwc.len() != needed {
            return Err(InferenceError::InferenceError(format!(
                "frame buffer length {} != src_h * src_w * 3 = {needed}",
                frame_hwc.len()
            )));
        }

        if self.frame_dev_capacity < needed {
            self.frame_dev = self
                .stream
                .alloc_zeros::<u8>(needed)
                .map_err(|e| InferenceError::InferenceError(format!("realloc frame_dev: {e:?}")))?;
            self.frame_dev_capacity = needed;
        }

        self.stream
            .memcpy_htod(frame_hwc, &mut self.frame_dev)
            .map_err(|e| InferenceError::InferenceError(format!("htod: {e:?}")))?;

        let dst = self.dst_size as f32;
        let scale = (dst / src_h as f32).min(dst / src_w as f32);
        let resized_w = (src_w as f32 * scale).round() as i32;
        let resized_h = (src_h as f32 * scale).round() as i32;
        let pad_x = (self.dst_size as i32 - resized_w) / 2;
        let pad_y = (self.dst_size as i32 - resized_h) / 2;
        let bgr_in_i = i32::from(bgr_in);

        let block_dim = (16u32, 16u32, 1u32);
        let grid_dim = (
            (self.dst_size as u32).div_ceil(block_dim.0),
            (self.dst_size as u32).div_ceil(block_dim.1),
            1u32,
        );
        let launch_cfg = LaunchConfig {
            grid_dim,
            block_dim,
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.kernel)
                .arg(&self.frame_dev)
                .arg(&(src_h as i32))
                .arg(&(src_w as i32))
                .arg(&mut self.input_dev)
                .arg(&(self.dst_size as i32))
                .arg(&scale)
                .arg(&pad_x)
                .arg(&pad_y)
                .arg(&bgr_in_i)
                .launch(launch_cfg)
                .map_err(|e| InferenceError::InferenceError(format!("kernel launch: {e:?}")))?;
        }

        Ok(PreGeom {
            scale,
            pad_x,
            pad_y,
        })
    }
}
