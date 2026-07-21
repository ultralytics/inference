// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Internal GPU preprocess for the `cuda-preprocess` feature.
//!
//! Implements a fused CUDA kernel that performs bilinear letterbox-resize +
//! `/255` normalize + (optional) BGR↔RGB swap + HWC→CHW packing in a single
//! launch, writing directly into the device-side input buffer that ONNX
//! Runtime's TRT/CUDA EP reads via [`ort::value::TensorRefMut::from_raw`].
//!
//! The whole module is `pub(crate)` - public API consumers reach this path
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
    int dst_h, int dst_w,
    float scale,
    int pad_x, int pad_y,
    int bgr_in)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    int rx = x - pad_x;
    int ry = y - pad_y;
    int valid_w = (int)(src_w * scale + 0.5f);
    int valid_h = (int)(src_h * scale + 0.5f);

    float r, g, b;
    if (rx < 0 || ry < 0 || rx >= valid_w || ry >= valid_h) {
        r = g = b = 114.0f / 255.0f;
    } else {
        // Half-pixel convention: src = (dst + 0.5) / scale - 0.5, matching the CPU
        // letterbox in preprocessing.rs and cv2/torch `align_corners=False`.
        float fx = ((float)rx + 0.5f) / scale - 0.5f;
        float fy = ((float)ry + 0.5f) / scale - 0.5f;
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

    int hw = dst_h * dst_w;
    int idx = y * dst_w + x;
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
    /// Target the frame was letterboxed into, `(height, width)`. Equals the caller's
    /// requested size, which is the model input for square inference and the
    /// stride-aligned rectangle for `rect`.
    pub dst_hw: (usize, usize),
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

    /// Persistent device buffer for the model input tensor (`3 * dst_h * dst_w` f32).
    /// Pointer is stable for the buffer's lifetime - cached so callers can
    /// hand it to ORT via `TensorRefMut::from_raw` without re-querying.
    input_dev: CudaSlice<f32>,
    input_dev_ptr: u64,

    /// Model input height/width (letterbox target). May be non-square.
    dst_h: usize,
    dst_w: usize,
}

impl CudaPreprocessor {
    /// Phase-2 init: NVRTC-compile the preprocess kernel and pre-allocate the
    /// device input buffer (`3 * dst_h * dst_w` f32). Reuses the context+stream
    /// from [`CudaStreamHandle::open`] so ORT and this preprocessor share the
    /// same compute stream.
    pub(crate) fn finalize(handle: CudaStreamHandle, dst_h: usize, dst_w: usize) -> Result<Self> {
        let CudaStreamHandle { ctx, stream } = handle;

        let ptx = compile_ptx_with_opts(KERNEL_SRC, cudarc::nvrtc::CompileOptions::default())
            .map_err(|e| InferenceError::ModelLoadError(format!("NVRTC compile: {e:?}")))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| InferenceError::ModelLoadError(format!("load_module: {e:?}")))?;
        let kernel = module
            .load_function("preprocess")
            .map_err(|e| InferenceError::ModelLoadError(format!("load_function: {e:?}")))?;

        let input_elems = 3 * dst_h * dst_w;
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
            dst_h,
            dst_w,
        })
    }

    /// Device pointer of the model input buffer (`3 * dst_h * dst_w` f32),
    /// stable for the lifetime of this preprocessor.
    pub(crate) const fn input_dev_ptr(&self) -> u64 {
        self.input_dev_ptr
    }

    /// Model input dimensions `(height, width)` - may be non-square.
    pub(crate) const fn dst_hw(&self) -> (usize, usize) {
        (self.dst_h, self.dst_w)
    }

    /// H2D-copy the source frame, launch the fused preprocess kernel writing
    /// into the input buffer, and return the letterbox geometry needed by
    /// post-processing.
    ///
    /// `dst` is the letterbox target `(height, width)`: the model input for square
    /// inference, or the smaller stride-aligned rectangle for `rect`. It must fit the
    /// buffer sized at [`Self::finalize`], which every rect target does since
    /// `calculate_rect_size` only shrinks axes.
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
        dst: (usize, usize),
    ) -> Result<PreGeom> {
        let (dst_h, dst_w) = dst;
        if dst_h * dst_w > self.dst_h * self.dst_w {
            return Err(InferenceError::InferenceError(format!(
                "letterbox target {dst_h}x{dst_w} exceeds the input buffer sized for {}x{}",
                self.dst_h, self.dst_w
            )));
        }
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

        // Letterbox into the (possibly non-square) dst_h × dst_w target:
        // single uniform scale = min over both axes, centered padding.
        let scale = (dst_h as f32 / src_h as f32).min(dst_w as f32 / src_w as f32);
        let resized_w = (src_w as f32 * scale).round() as i32;
        let resized_h = (src_h as f32 * scale).round() as i32;
        let pad_x = (dst_w as i32 - resized_w) / 2;
        let pad_y = (dst_h as i32 - resized_h) / 2;
        let bgr_in_i = i32::from(bgr_in);

        let block_dim = (16u32, 16u32, 1u32);
        let grid_dim = (
            (dst_w as u32).div_ceil(block_dim.0),
            (dst_h as u32).div_ceil(block_dim.1),
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
                .arg(&(dst_h as i32))
                .arg(&(dst_w as i32))
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
            dst_hw: (dst_h, dst_w),
        })
    }
}

#[cfg(test)]
mod tests {
    // Runs only on an NVIDIA GPU: the whole module is `cuda-preprocess`-gated,
    // so these compile and execute exclusively on the GPU CI runner. Inline
    // tests reach the preprocessor's private `stream`/`input_dev` fields to copy
    // the kernel output back to the host and assert on it.
    use super::*;

    /// Padded background emitted by the kernel (`114/255`).
    const PAD: f32 = 114.0 / 255.0;
    /// Tolerance for GPU-computed float comparisons.
    const EPS: f32 = 1e-4;

    /// Open device 0 and build a preprocessor targeting `dst_h` x `dst_w`.
    fn preprocessor(dst_h: usize, dst_w: usize) -> CudaPreprocessor {
        let handle = CudaStreamHandle::open(0).expect("open CUDA device 0");
        CudaPreprocessor::finalize(handle, dst_h, dst_w).expect("finalize preprocessor")
    }

    /// Build an HWC RGB buffer filled with a single solid color.
    fn solid(height: usize, width: usize, red: u8, green: u8, blue: u8) -> Vec<u8> {
        let mut buf = Vec::with_capacity(height * width * 3);
        for _ in 0..height * width {
            buf.extend_from_slice(&[red, green, blue]);
        }
        buf
    }

    /// Copy the device input buffer back to the host as flat CHW f32 planes.
    fn readback(pre: &CudaPreprocessor) -> Vec<f32> {
        pre.stream
            .clone_dtoh(&pre.input_dev)
            .expect("device to host copy")
    }

    #[test]
    fn stream_handle_open() {
        let handle = CudaStreamHandle::open(0).expect("open CUDA device 0");
        // cudarc hands back CUDA's default stream, whose raw pointer is null by
        // design, so only assert that the accessor is callable (it feeds ort's
        // with_compute_stream) rather than that it is non-null.
        let _ = handle.raw_stream_ptr();
    }

    #[test]
    fn preprocessor_finalize_allocates() {
        let pre = preprocessor(640, 640);
        assert_eq!(pre.dst_hw(), (640, 640));
        assert_ne!(
            pre.input_dev_ptr(),
            0,
            "device input buffer must be allocated"
        );
    }

    #[test]
    fn preprocess_square_geometry() {
        let mut pre = preprocessor(640, 640);
        let (src_h, src_w) = (480u32, 640u32);
        let frame = solid(src_h as usize, src_w as usize, 200, 100, 50);

        let geom = pre
            .preprocess(&frame, src_h, src_w, false, pre.dst_hw())
            .expect("preprocess");

        // 640/640 = 1.0 is the binding axis; the 480-tall image is centered.
        assert!((geom.scale - 1.0).abs() < EPS, "scale = {}", geom.scale);
        assert_eq!(geom.pad_x, 0);
        assert_eq!(geom.pad_y, 80);

        let host = readback(&pre);
        assert!(
            host.iter().all(|&v| (0.0..=1.0).contains(&v)),
            "every output value must be normalized into [0, 1]"
        );
    }

    #[test]
    fn preprocess_non_square_target() {
        let mut pre = preprocessor(384, 640);
        assert_eq!(pre.dst_hw(), (384, 640));

        // 640x640 into 384x640: scale 0.6 fills the width, pads the height to 0.
        let frame = solid(640, 640, 10, 20, 30);
        let geom = pre
            .preprocess(&frame, 640, 640, false, pre.dst_hw())
            .expect("preprocess");

        assert_eq!(geom.pad_y, 0);
        assert!(
            geom.pad_x > 0,
            "expected horizontal padding, got {}",
            geom.pad_x
        );
    }

    #[test]
    fn preprocess_padding_value() {
        let (dst, src_h, src_w) = (128usize, 64u32, 32u32);
        let mut pre = preprocessor(dst, dst);
        let frame = solid(src_h as usize, src_w as usize, 255, 255, 255);

        let geom = pre
            .preprocess(&frame, src_h, src_w, false, pre.dst_hw())
            .expect("preprocess");
        assert!(
            geom.pad_x > 0,
            "expected horizontal padding, got {}",
            geom.pad_x
        );

        // The top-left pixel is outside the resized image, so it is background.
        let host = readback(&pre);
        let hw = dst * dst;
        for plane in 0..3 {
            let v = host[plane * hw];
            assert!(
                (v - PAD).abs() < EPS,
                "padding plane {plane} = {v}, want {PAD}"
            );
        }
    }

    #[test]
    fn preprocess_center_normalize() {
        let dst = 64usize;
        let mut pre = preprocessor(dst, dst);
        // Square input, no letterbox: every pixel is R=255, G=0, B=0.
        let frame = solid(dst, dst, 255, 0, 0);
        pre.preprocess(&frame, dst as u32, dst as u32, false, pre.dst_hw())
            .expect("preprocess");

        let host = readback(&pre);
        let hw = dst * dst;
        let idx = (dst / 2) * dst + dst / 2;
        assert!((host[idx] - 1.0).abs() < EPS, "R plane = {}", host[idx]);
        assert!(host[hw + idx].abs() < EPS, "G plane = {}", host[hw + idx]);
        assert!(
            host[2 * hw + idx].abs() < EPS,
            "B plane = {}",
            host[2 * hw + idx]
        );
    }

    #[test]
    fn preprocess_bgr_swap() {
        let dst = 64usize;
        let frame = solid(dst, dst, 255, 0, 0); // byte order [255, 0, 0]
        let hw = dst * dst;
        let idx = (dst / 2) * dst + dst / 2;

        let mut rgb = preprocessor(dst, dst);
        rgb.preprocess(&frame, dst as u32, dst as u32, false, rgb.dst_hw())
            .expect("rgb preprocess");
        let rgb_host = readback(&rgb);

        let mut bgr = preprocessor(dst, dst);
        bgr.preprocess(&frame, dst as u32, dst as u32, true, bgr.dst_hw())
            .expect("bgr preprocess");
        let bgr_host = readback(&bgr);

        // RGB reads byte[0]=255 into R; BGR reads the same byte into B.
        assert!(
            (rgb_host[idx] - 1.0).abs() < EPS,
            "rgb R = {}",
            rgb_host[idx]
        );
        assert!(bgr_host[idx].abs() < EPS, "bgr R = {}", bgr_host[idx]);
        assert!(
            (bgr_host[2 * hw + idx] - 1.0).abs() < EPS,
            "bgr B = {}",
            bgr_host[2 * hw + idx]
        );
    }

    #[test]
    fn preprocess_rejects_wrong_len() {
        let mut pre = preprocessor(64, 64);
        let bad = vec![0u8; 10]; // != 64 * 64 * 3
        let result = pre.preprocess(&bad, 64, 64, false, pre.dst_hw());
        assert!(
            matches!(result, Err(InferenceError::InferenceError(_))),
            "wrong-length frame must be rejected"
        );
    }

    #[test]
    fn preprocess_frame_buffer_growth() {
        let mut pre = preprocessor(128, 128);
        // First a small frame, then a larger one to force `frame_dev` to grow.
        let small = solid(32, 32, 1, 2, 3);
        pre.preprocess(&small, 32, 32, false, pre.dst_hw())
            .expect("small frame");
        let large = solid(256, 256, 4, 5, 6);
        pre.preprocess(&large, 256, 256, false, pre.dst_hw())
            .expect("large frame after buffer growth");
    }

    #[test]
    fn preprocess_matches_cpu_letterbox() {
        // The GPU kernel and the CPU letterbox must produce the same tensor, otherwise
        // `--device cuda` silently infers on different pixels than `--device cpu`. A
        // gradient (not a solid color) is required: it is the only input that exposes a
        // half-pixel offset in the bilinear sampling.
        let (src_h, src_w) = (270usize, 480usize);
        let mut frame = Vec::with_capacity(src_h * src_w * 3);
        for y in 0..src_h {
            for x in 0..src_w {
                frame.extend_from_slice(&[(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8]);
            }
        }

        // A rect target, so this also covers the non-square letterbox the `rect` path uses.
        let dst = (192usize, 320usize);
        let mut pre = preprocessor(dst.0, dst.1);
        pre.preprocess(&frame, src_h as u32, src_w as u32, false, dst)
            .expect("gpu preprocess");
        let gpu = readback(&pre);

        let img = image::DynamicImage::ImageRgb8(
            image::RgbImage::from_raw(src_w as u32, src_h as u32, frame).expect("rgb image"),
        );
        let cpu = crate::preprocessing::preprocess_image_with_precision(&img, dst, 32, false);
        let cpu = cpu.tensor.as_slice().expect("contiguous cpu tensor");

        let n = 3 * dst.0 * dst.1;
        let max_diff = (0..n).fold(0.0f32, |m, i| m.max((gpu[i] - cpu[i]).abs()));
        // 1/255 == 0.0039: anything above one input quantization step is a real
        // resampling mismatch, not float noise.
        assert!(
            max_diff < 1.0 / 255.0,
            "gpu letterbox deviates from cpu by {max_diff}"
        );
    }
}
