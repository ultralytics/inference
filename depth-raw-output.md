# Raw depth map: what it is and how to use it downstream

## What the raw output is

The depth head does **not** produce an image. Its raw output is a single-channel
`float32` array of shape `(H, W)` where every pixel holds a **depth value in
metric-looking units**. There is no color and no RGB — colorization (grayscale,
INFERNO, Spectral, ...) only happens at render time. Invalid / background pixels are `0`.

> **Relative vs metric.** The head actually predicts _relative_ (affine-invariant)
> scene structure in log space, like DepthAnything — it has no inherent absolute scale.
> The "meters" come from a global 2-parameter log-affine calibration
> `d' = exp(a·log d + b)` baked into the model (`cal_a`/`cal_b`; the default weights
> use `a=1.0, b≈-0.25`, i.e. a single global scale of ~0.78). So **relative structure /
> ordering is reliable, but absolute values are only approximate** unless you
> re-calibrate to your camera/domain with `model.calibrate()` against ground-truth
> depth. Treat the numbers as calibrated relative depth, not measured metric depth.

Example (`bus.jpg`, imgsz 768, `yolo26l-depth`):

- `dtype=float32`, `shape=(1080, 810)`, `ndim=2`
- meters: `min 1.91` (nearest) → `max 19.26` (farthest), `mean ~6.0`
- a raw 4x4 sample block: `[[9.75, 9.75, 9.75, 9.74], ...]` — literal meters

Pipeline is always: **raw float meters → normalize → colormap → displayable image**.
The library keeps the raw meters; the picture is a downstream view of them.

- **grayscale** = 1 channel, min-max meters (near dark, far light)
- **INFERNO / JET / Spectral** = same array through a 3-channel color LUT
- **metric vs disparity** = only changes how meters are normalized (raw `d` vs `1/d`
  clipped to the 2nd-98th percentile) before the LUT

## Getting the raw array

### Python

```python
r = model.predict("bus.jpg", imgsz=768)[0]
depth = r.depth.data  # torch.Tensor or np.ndarray, (H, W), meters
depth = depth.cpu().numpy() if hasattr(depth, "cpu") else depth
valid = depth > 0  # background / invalid pixels are 0
```

### Rust

```rust
let results = model.predict(&source)?;
if let Some(depth) = &results[0].depth {
    let map = &depth.data;                 // ndarray Array2<f32>, meters
    let (lo, hi) = (depth.min_depth(), depth.max_depth()); // Option<f32> over valid (>0) px
}
```

## Using it in a postprocessing pipeline

Because it is plain metric data, you can feed it straight into downstream steps:

- **Distance thresholding / zone masks** — keep only pixels within a range:
  `near = (depth > 0) & (depth < 2.0)`. Combine with detection boxes to flag
  "objects closer than X meters".
- **Per-object depth** — index the map with a detection box or segment mask and take
  the median to get that instance's distance (median is robust to edge pixels).
- **3D back-projection / point cloud** — with camera intrinsics `fx, fy, cx, cy`:
  `Z = depth`, `X = (u - cx) * Z / fx`, `Y = (v - cy) * Z / fy`. Stack `(X, Y, Z)`
  over valid pixels for an `(N, 3)` point cloud.
- **Disparity / relative depth** — `disp = 1.0 / depth` on valid pixels for a
  scale-invariant map (what the DepthAnything-style visualization uses).
- **Foreground extraction / bokeh** — blur or composite by depth bands.
- **Feeding another model** — normalize (metric or percentile) and pass the map as an
  extra input channel or as a mask to a downstream network.

## Exporting the raw map (lossless)

The colorized JPEG throws away the meters. To keep them for offline processing, save
the raw array instead of (or alongside) the picture:

- **`.npy`** — exact float meters, easiest to reload in Python/NumPy.
- **16-bit PNG in millimeters** — `(depth * 1000).clip(0, 65535).astype(uint16)`,
  written with `cv2.imwrite("depth.png", mm16)`; portable and lossless to ~1 mm,
  reload as `png / 1000.0` to get meters back.
- **EXR (32-bit float)** — if you need exact meters in an image container.

> Note: the CLI currently only saves the colorized side-by-side PNG. A `--save-depth`
> option that writes the raw `.npy` / 16-bit-mm PNG is a planned follow-up; the data is
> already available in `results.depth.data` for programmatic use today.
