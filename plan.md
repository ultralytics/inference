# TASK: Optimize WebAssembly YOLO Inference for `ultralytics/inference`

## 🎯 Objective
Refactor the current Rust/WASM implementation in the `ultralytics/inference` repository to maximize YOLO inference speed in the browser using the `ort` crate. The focus is on enabling WebGPU, configuring WASM SIMD/Atomics, and implementing zero-copy memory transfers between JS and Rust.

---

## 🛠️ Step 1: Configure Cargo for SIMD and Atomics
**Target File:** `.cargo/config.toml` (Create if it does not exist in the repo root)
**Action:** Add the following compiler flags. This ensures that if the user's browser lacks WebGPU, the fallback WASM CPU execution uses vector instructions and multithreading.

```toml
[build]
target = "wasm32-unknown-unknown"

[target.wasm32-unknown-unknown]
rustflags = [
    "-C", "target-feature=+simd128,+atomics,+bulk-memory",
    "-C", "link-arg=--shared-memory"
]
```

---

## 🛠️ Step 2: Enable WebGPU Execution Provider
**Target File:** The Rust source file handling `ort::Session` initialization (likely `src/model.rs`, `src/lib.rs`, or similar).
**Action:** Modify the `Session::builder()` pipeline to explicitly request the `WebGPUExecutionProvider`.

```rust
// Replace existing session initialization with this optimized builder
let session = ort::Session::builder()?
    .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
    .with_execution_providers([
        ort::execution_providers::WebGPUExecutionProvider::default().build()
    ])?
    .commit_from_file("model.ort")?; // Note: Ensure logic points to .ort extensions, not .onnx
```

---

## 🛠️ Step 3: Implement Zero-Copy Memory for Image Tensors
**Target File:** The WASM binding file where JavaScript calls Rust (e.g., `src/lib.rs`).
**Action:** Remove any logic that serializes image data (like JSON or heavy array cloning). Use `js-sys` to access shared linear memory, allowing JS to write directly to Rust's memory space or allowing Rust to read JS arrays without cloning.

```rust
use wasm_bindgen::prelude::*;
use js_sys::Float32Array;

#[wasm_bindgen]
pub struct YoloEngine {
    // Session state goes here
}

#[wasm_bindgen]
impl YoloEngine {
    /// Accepts a direct reference to the Float32Array allocated in JS
    pub fn run_inference(&self, input_array: &Float32Array) -> Result<JsValue, JsValue> {
        // Extract data with minimal copying
        let mut buffer = vec![0.0; input_array.length() as usize];
        input_array.copy_to(&mut buffer); 
        
        // Advanced: For true zero-copy, allocate a buffer in Rust, 
        // return its pointer to JS, let JS write to it, and read it directly here.

        // ... Feed buffer to ORT ...
        
        Ok(JsValue::NULL) // Return Bounding Box struct
    }
}
```

---

## 🛠️ Step 4: Asset Pipeline & Type Constraints
**Target File:** `README.md`, Python export scripts, or deployment docs.
**Action:** Add documentation stating two strict rules for WASM deployment:
1. **Use `.ort` files:** `.onnx` files are too bloated for web. Provide this command to developers:
   ```bash
   python -m onnxruntime.tools.convert_onnx_models_to_ort model.onnx
   ```
2. **Avoid FP16 for WASM:** Instruct users to use INT8 Quantization or standard FP32. WebAssembly CPU fallback does not natively support FP16 math and will result in extreme slow-downs due to software emulation.

---

## 🛠️ Step 5: Dev Server & Production Headers
**Target File:** The web framework config (e.g., `vite.config.ts`, `next.config.js`, or `vercel.json`).
**Action:** Multithreading (Atomics) requires `SharedArrayBuffer` support in the browser. Add these headers to the dev server and production responses:
- `Cross-Origin-Opener-Policy: same-origin`
- `Cross-Origin-Embedder-Policy: require-corp`

---
---

# 📓 OPTIMIZATION LOG (branch `speed-up-web-inference`)

Goal: match LiteRT.js (~60 fps) on the **same model + same 640×640 resolution**.
Test machine context: WebGPU shows ~30 fps where LiteRT.js shows ~60. Lowering
resolution is explicitly **out of scope** (LiteRT doesn't, so neither do we).

## Measured per-stage breakdown (yolo11n detect, 640, webcam, WebGPU)

| Stage       | Reading                | Conclusion |
|-------------|------------------------|------------|
| `pre`       | **2.5 ms, always**     | Our Rust preprocess (now SIMD) is NOT the bottleneck. Done. |
| `infer` CPU | ~100 ms, flat          | Pure wasm/asyncify CPU path. |
| `infer` GPU | **13 ms → 30 ms**      | Starts fast, settles to ~2× after a few seconds, **recovers instantly on page reload**. |
| `post`      | small, stable          | SIMD NMS fine. |

Key deductions:
- WebGPU is genuinely engaged (100 ms → 30 ms is a real ~3× GPU win; `device:
  webgpu` confirmed; `chrome://gpu` shows WebGPU hardware-accelerated).
- The runtime in use is **`ort-wasm-simd-threaded.asyncify.wasm`** (27 MB), NOT
  the `jsep.wasm` (which is preloaded-but-unused). pyke `ort-web` 0.2.1 bridges
  Rust `run_async` via **Asyncify**, so the asyncify build loads even on WebGPU
  with JSPI available. Asyncify ~doubles CPU-side per-op dispatch cost → the most
  likely structural reason we trail LiteRT at 640. **Not fixable without an
  ort-web change.**
- The 13→30 ms degradation **recovers on reload** ⇒ it is NOT thermal (heat
  persists across a reload). It is per-page state: GPU clock settling and/or
  ORT-web WebGPU resource accumulation.

## Changes landed so far

| # | Change | File(s) | Status / effect |
|---|--------|---------|-----------------|
| 1 | **wasm `+simd128`** via scoped cargo config (un-ignored in `.gitignore`) | `crates/web/.cargo/config.toml`, `.gitignore` | Activates `wide` f32x8 NMS + vectorized preprocess that were silently scalar. `pre`/`post` dropped. ✅ |
| 2 | **Drop per-frame input clone** (`mem::take` tensor into ORT) | `crates/web/src/lib.rs` | −1 copy (~4.9 MB) per frame. ✅ |
| 3 | **COOP/COEP serve script** (`npm run serve`) | `web/serve.mjs`, `web/package.json`, `web/README.md` | Enables SharedArrayBuffer so ORT threaded wasm can use workers. ✅ |
| 4 | **Decouple loop from rAF** (single-slot) | `web/example/index.html` | Removed the rAF/vsync quantization that capped FPS. Faster start (45–50 fps). ✅ |
| 5 | **Zero-gap scheduler** (MessageChannel, no `setTimeout` ≥4 ms clamp) | `web/example/index.html` | Keeps GPU fed; reclaims ~4 ms/frame. ⚠️ Did **not** stop the 10–15 s degradation → rules out "GPU idle between frames" as the cause. |
| 6 | **Reuse readback canvas** (was a new `OffscreenCanvas` + ctx every frame) | `web/src/index.ts` | Less GC churn on `pre`. ✅ |
| 7 | **Reuse `RunOptions`** across frames (struct field) | `crates/web/src/lib.rs` | −1 ORT handle create/drop per frame. ✅ |

## Tried and rejected

- **WebNN EP** (`ort::ep::WebNN`, `DeviceType::GPU`): wired in fully, but stock
  YOLO exports throw `WebNN backend does not support data type: int64` at predict
  time (WebNN spec has no int64; the graphs use it). Device-type-agnostic. Fully
  **reverted**. (Would only work on int64-free / re-exported models.)
- **Lower input resolution** (320/416/480): would work but is out of scope —
  LiteRT hits 60 at full res, so we must too.
- **`.ort` model format** (original plan Step 4): rejected — breaks the embedded
  `.onnx` metadata parser (`onnx_meta.rs`); only affects download size.

## Open problem: the 10–15 s "drops to half" degradation

Persists even with the GPU continuously fed (change #5). Two remaining causes:
1. **GPU clock settling** (driver drops boost clock under sustained load) — mostly
   outside our control from the web layer.
2. **ORT-web WebGPU resource accumulation** (per-run GPU buffers/staging not
   reused; pool grows until allocation/validation slows) — fixable via buffer
   cache modes or io-binding.

**Distinguishing experiment (next):** cap the loop to ~10 fps. If the slowdown
still hits at ~10–15 s wall-clock → thermal/clock. If it now takes much longer (it
tracks *frame count*, not time) → resource accumulation, and we chase the ORT-web
WebGPU buffer cache.

## Investigated and BLOCKED by the ort-web 0.2.1 stack

- **WebGPU graph capture** — NOT VIABLE. Traced through source:
  - `ort`'s `WebGPU::with_enable_graph_capture(true)` only sets the EP option
    string `ep.webgpuexecutionprovider.enableGraphCapture`.
  - ort-web's `SessionOptionsAppendExecutionProvider` reads **only**
    `preferredLayout` for WebGPU and ignores everything else; it implements **no**
    session-option setters at all (no `enable_graph_capture`/`enable_mem_pattern`/
    `graph_optimization_level` writer), so the flag never reaches onnxruntime-web.
  - Even if forwarded, graph capture needs **GPU-resident, fixed-address inputs**
    (io-binding). ort-web's OrtApi shim only exposes `CreateTensorWithDataAsOrtValue`
    (CPU memory) — no io-binding, no GPU-buffer input path for the `ort` crate. The
    binding layer has `Tensor.fromGpuBuffer`, but `ort` can't reach it.
  - Verdict: requires reimplementing the GPU input path across `ort` + `ort-web`
    (upstream-scale), and even then onnxruntime-web may refuse with CPU inputs.
- **WebGPU buffer-cache modes / mem-pattern / graph-opt overrides** — same wall:
  all are WebGPU EP / session options that ort-web silently drops. Unreachable.
- `cargo search ort-web` → newest is `0.2.1+1.24` (what we pin); no upgrade
  forwards these options.

## Next levers that are actually reachable

1. **Diagnose the degradation from OUR side** — since the ORT buffer-cache knobs
   are unreachable, the per-run resource growth (if that's the cause) can only be
   addressed in how we create/drop the input `Tensor` and `SessionOutputs` each
   frame, or confirmed as GPU clock settling via the 10 fps cap experiment.
2. **wasm threads** for Rust pre/postprocess (nightly `-Zbuild-std` +
   wasm-bindgen-rayon) — deferred; only helps `pre`/`post`, already small.
3. **Upstream**: contribute session-option + io-binding forwarding to ort-web (and
   a GPU input path) — the only way to unlock graph capture / EP tuning long-term.

## Structural ceiling (record)

The asyncify dispatch tax is inherent to ort-web 0.2.1's WebGPU path. Matching
LiteRT exactly at 640 may require an upstream ort-web change (JSPI-based suspension
instead of Asyncify) or a different runtime. Everything above narrows the gap; this
note explains why some gap may remain.