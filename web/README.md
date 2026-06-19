<!-- Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license -->

<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Ultralytics YOLO npm Inference

[![npm version](https://img.shields.io/npm/v/@ultralytics/yolo?logo=npm&logoColor=white&label=npm&color=CB3837)](https://www.npmjs.com/package/@ultralytics/yolo)
[![npm downloads](https://img.shields.io/npm/dm/@ultralytics/yolo?logo=npm&logoColor=white&label=downloads&color=CB3837)](https://www.npmjs.com/package/@ultralytics/yolo)
[![CI](https://github.com/ultralytics/inference/actions/workflows/ci.yml/badge.svg)](https://github.com/ultralytics/inference/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](https://github.com/ultralytics/inference/blob/main/LICENSE)

[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://www.reddit.com/r/Ultralytics/)

Run [Ultralytics](https://www.ultralytics.com/) YOLO models directly in the browser,
with no server and no Python. It runs on **WebGPU** (with an automatic CPU/wasm
fallback) and covers detection, segmentation, pose, classification, OBB, and
semantic segmentation, behind a small TypeScript API with a built-in
`annotate()` that draws results straight to a canvas.

```ts
import { YOLO, annotate } from "@ultralytics/yolo";

const model = await YOLO.load("yolo26n.onnx");
const results = await model.predict("bus.jpg");
await annotate(document.querySelector("canvas"), "bus.jpg", results);
```

It is a **library only** (no CLI; that is the native Rust crate). Under the hood
the engine is the `ultralytics-inference` Rust crate compiled to WebAssembly.
Inference runs on [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)
via [`ort-web`](https://ort.pyke.io/backends/web), and all pre/postprocessing,
colors, and the pose skeleton come from that shared Rust code, so results and
visuals match the native and Python paths.

## 📦 Install

```bash
npm install @ultralytics/yolo
# or
pnpm add @ultralytics/yolo
yarn add @ultralytics/yolo
bun add @ultralytics/yolo
```

It ships as an ES module with TypeScript types and works in any modern bundler
(Vite, webpack, esbuild, Bun) or directly via a CDN such as
[esm.sh](https://esm.sh/@ultralytics/yolo).

## 🚀 Quick Start

```ts
import { YOLO, annotate } from "@ultralytics/yolo";

// Loads the model and initializes WebGPU + ONNX Runtime Web on first use.
const model = await YOLO.load("yolo26n.onnx");

const results = await model.predict("bus.jpg");
for (const box of results.boxes) {
  console.log(box.name, box.conf.toFixed(2), [box.x1, box.y1, box.x2, box.y2]);
}

// Draw boxes, OBB, pose, and labels onto a canvas in one call (no canvas code).
await annotate(document.querySelector("canvas"), "bus.jpg", results);
```

`predict()` accepts a URL/path, a `Blob`/`File`, raw encoded image bytes
(`Uint8Array`/`ArrayBuffer`), `ImageData`, an `HTMLImageElement`,
`HTMLCanvasElement`, `HTMLVideoElement`, or an `ImageBitmap`.

```ts
const results = await model.predict(canvas, { conf: 0.25, iou: 0.7 });
console.log(model.device); // "webgpu" or "cpu"
```

### Webcam / Video

Drawable sources (`<video>`, canvas, `ImageBitmap`, `ImageData`) take a
raw-pixel fast path with no re-encoding, so a render loop is smooth:

```ts
const model = await YOLO.load("yolo26n.onnx");
async function frame() {
  const results = await model.predict(video); // <video> element
  await annotate(canvas, video, results);
  requestAnimationFrame(frame);
}
```

## ✨ Models

<a href="https://docs.ultralytics.com/tasks" target="_blank">
    <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/docs/ultralytics-yolov8-tasks-banner.avif" alt="Ultralytics YOLO supported tasks">
</a>
<br>
<br>

Runs [YOLOv8](https://docs.ultralytics.com/models/yolov8),
[YOLO11](https://docs.ultralytics.com/models/yolo11), and
[YOLO26](https://docs.ultralytics.com/models/yolo26) ONNX exports for detection,
segmentation, pose, OBB, classification, and semantic segmentation.

Pass a bare ONNX name and it is **auto-downloaded** from the
[Ultralytics assets release](https://github.com/ultralytics/assets/releases) (the
same weights the native crate and Python use):

```ts
await YOLO.load("yolo26n.onnx"); // auto-downloads from the release: .../download/v8.4.0/yolo26n.onnx
```

Auto-download covers **yolo26**, **yolo11**, and **yolov8** in sizes `n/s/m/l/x`
with task suffixes `-seg`, `-pose`, `-cls`, `-obb`, and `-sem` (semantic, yolo26
only). A value containing a `/` or a scheme is used as a URL/path as-is.

> **CORS note:** GitHub release assets do not send `Access-Control-Allow-Origin`,
> so a browser cannot fetch them cross-origin. Host the `.onnx` **same-origin**
> (e.g. `YOLO.load("/models/yolo26n.onnx")`) or behind a CORS-enabled origin /
> proxy. The bare-name shortcut is convenient when you mirror the assets on such
> a host.

## 📐 Results Shape

`predict()` resolves to a `Results` object shaped like the Ultralytics
`Results`:

Field names match the Rust/Ultralytics `Results` API 1-1:

| Field              | Type                                                      | Tasks                 |
| ------------------ | --------------------------------------------------------- | --------------------- |
| `task`             | `string`                                                  | all                   |
| `width` / `height` | `number`                                                  | all                   |
| `boxes`            | `{ x1, y1, x2, y2, conf, cls, name, color }[]`            | detect, segment, pose |
| `obb`              | `{ x, y, w, h, angle, conf, cls, name, color }[]`         | obb                   |
| `keypoints`        | `{ points: [x, y, conf][], color }[]`                     | pose                  |
| `probs`            | `{ top1, top5, top1conf, top5conf, name, color } \| null` | classify              |
| `masks`            | `Uint8Array` (RGBA overlay, `width*height*4`)             | segment, semantic     |
| `semantic`         | `Uint16Array` (class id per pixel, `width*height`)        | semantic              |
| `speed`            | `{ preprocess, inference, postprocess }` ms               | all                   |

`model.names` is the class id to name map (like `model.names` in Python). Every
detection carries its Ultralytics palette `color`, and `annotate()` draws the
`masks` overlay and the pose skeleton with the same per-limb/keypoint colors as
the native renderer. None of this is duplicated in JS.

## ⚙️ Requirements & Notes

- **WebGPU** (Chrome/Edge, or Firefox with WebGPU enabled) from a **secure
  context** (`https://` or `http://localhost`) gives the fast path. Without
  WebGPU (older browsers, some phones), `YOLO.load` automatically falls back to a
  portable **CPU/wasm** build that runs everywhere. Pick the device with
  `YOLO.load("yolo26n.onnx", { device: "webgpu" | "cpu" })` (default `"auto"`). If
  WebGPU cannot engage, the load falls back to CPU; `model.device` reports what
  actually ran.
- **Model format**: export your model to ONNX with Ultralytics so the metadata
  (task, class names, `imgsz`) is embedded:

  ```python
  from ultralytics import YOLO

  YOLO("yolo26n.pt").export(format="onnx")
  ```

- **Runtime assets**: on first load, `ort-web` fetches the ONNX Runtime Web wasm
  bundle (~25 MB, browser-cached afterward) from `cdn.pyke.io`. If you set a
  Content-Security-Policy, allow that origin in `script-src`/`connect-src`. To
  avoid the CDN entirely, self-host the runtime and point to it:
  ```ts
  const model = await YOLO.load("yolo26n.onnx", { ortBaseUrl: "/ort/" });
  ```
  The folder must contain the ONNX Runtime Web entry scripts (`ort.webgpu.min.js`
  and `ort.wasm.min.js` for the CPU fallback) plus the
  `ort-wasm-simd-threaded.{jsep,asyncify,}.{mjs,wasm}` binaries.
- **Telemetry**: `ort-web` reports the page domain to pyke on first session
  creation. See the [ort-web docs](https://ort.pyke.io/backends/web) to review or
  disable it.

## 🔨 Building From Source

This package builds the wasm from the Rust crate with
[`wasm-pack`](https://github.com/wasm-bindgen/wasm-pack):

```bash
npm run build # wasm-pack build + tsc
```

Serve the built `dist/` + `pkg/` over `localhost` (a secure context) and open it
in a WebGPU browser.

## 💡 Contributing

Ultralytics thrives on community collaboration, and we deeply value your contributions! Whether it's reporting bugs,
suggesting features, or submitting code changes, your involvement is crucial.

- **Report Issues**: Found a bug? [Open an issue](https://github.com/ultralytics/inference/issues)
- **Feature Requests**: Have an idea? [Share it](https://github.com/ultralytics/inference/issues)
- **Pull Requests**: Read our [Contributing Guide](https://docs.ultralytics.com/help/contributing) first
- **Feedback**: Take our [Survey](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey)

A heartfelt thank you 🙏 goes out to all our contributors! Your efforts help make Ultralytics tools better for everyone.

[![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/ultralytics/graphs/contributors)

## 📜 License

Ultralytics offers two licensing options to suit different needs:

- **AGPL-3.0 License**: This [OSI-approved](https://opensource.org/license/agpl-3.0) open-source license is perfect for students, researchers, and enthusiasts. It encourages open collaboration and knowledge sharing. See the [LICENSE](https://github.com/ultralytics/inference/blob/main/LICENSE) file for full details.
- **Ultralytics Enterprise License**: Designed for commercial use, this license allows for the seamless integration of Ultralytics software and AI models into commercial products and services, bypassing the open-source requirements of AGPL-3.0. If your use case involves commercial deployment, please contact us via [Ultralytics Licensing](https://www.ultralytics.com/license).

## 📮 Contact

- **GitHub Issues**: [Bug reports and feature requests](https://github.com/ultralytics/inference/issues)
- **Discord**: [Join our community](https://discord.com/invite/ultralytics)
- **Documentation**: [docs.ultralytics.com](https://docs.ultralytics.com)

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://x.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
