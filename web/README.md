<!-- Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license -->

# Ultralytics Inference (Browser / WebGPU)

[![npm version](https://img.shields.io/npm/v/@ultralytics/yolo?logo=npm&logoColor=white&label=npm&color=CB3837)](https://www.npmjs.com/package/@ultralytics/yolo)
[![npm downloads](https://img.shields.io/npm/dm/@ultralytics/yolo?logo=npm&logoColor=white&label=downloads&color=CB3837)](https://www.npmjs.com/package/@ultralytics/yolo)
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](https://github.com/ultralytics/inference/blob/main/LICENSE)

[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://reddit.com/r/ultralytics)

Run [Ultralytics](https://ultralytics.com) YOLO models directly in the browser on
**WebGPU** — no server, no Python. Detection, segmentation, pose, classification,
OBB, and semantic segmentation, with a small Python-like API and a built-in
`annotate()` for drawing.

```ts
import { YOLO, annotate } from "@ultralytics/yolo";

const model = await YOLO.load("yolo26n.onnx");
const results = await model.predict("bus.jpg");
await annotate(document.querySelector("canvas"), "bus.jpg", results);
```

It is a **library only** (no CLI — that's the native Rust crate). Under the hood
the engine is the `ultralytics-inference` Rust crate compiled to WebAssembly;
inference runs on [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)
via [`ort-web`](https://ort.pyke.io/backends/web), and all pre/postprocessing,
colors, and the pose skeleton come from that shared Rust code — so results and
visuals match the native and Python paths.

## Install

```bash
npm install @ultralytics/yolo
```

## Quick start

```ts
import { YOLO, annotate } from "@ultralytics/yolo";

// Loads the model and initializes WebGPU + ONNX Runtime Web on first use.
const model = await YOLO.load("yolo26n.onnx");

const results = await model.predict("bus.jpg");
for (const box of results.boxes) {
  console.log(box.name, box.conf.toFixed(2), [box.x1, box.y1, box.x2, box.y2]);
}

// Draw boxes / OBB / pose skeletons / labels onto a canvas in one call —
// no manual canvas code required.
await annotate(document.querySelector("canvas"), "bus.jpg", results);
```

`predict()` accepts a URL/path, a `Blob`/`File`, raw encoded image bytes
(`Uint8Array`/`ArrayBuffer`), `ImageData`, an `HTMLImageElement`,
`HTMLCanvasElement`, `HTMLVideoElement`, or an `ImageBitmap`.

```ts
const results = await model.predict(canvas, { conf: 0.25, iou: 0.7 });
console.log(model.backend); // "WebGPU" or "CPU (wasm)"
```

### Webcam / video

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

## Models

A bare ONNX name resolves to the
[Ultralytics assets release](https://github.com/ultralytics/assets/releases):

```ts
await YOLO.load("yolo26n.onnx"); // -> .../releases/download/v8.4.0/yolo26n.onnx
```

Supports **yolo26**, **yolo11**, and **yolov8** in sizes `n/s/m/l/x` with task
suffixes `-seg`, `-pose`, `-cls`, `-obb`, and `-sem` (semantic, yolo26 only). A
value containing a `/` or a scheme is used as a URL/path as-is.

> **CORS note:** GitHub release assets do not send `Access-Control-Allow-Origin`,
> so a browser cannot fetch them cross-origin. Host the `.onnx` **same-origin**
> (e.g. `YOLO.load("/models/yolo26n.onnx")`) or behind a CORS-enabled origin /
> proxy. The bare-name shortcut is convenient when you mirror the assets on such
> a host.

## Results shape

`predict()` resolves to a `Results` object shaped like the Ultralytics Python
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
| `speed`            | `{ preprocess, inference, postprocess }` ms               | all                   |

`model.names` is the class id -> name map (like `model.names` in Python). Every
detection carries its Ultralytics palette `color`, and `annotate()` draws the
`masks` overlay and the pose skeleton with the same per-limb/keypoint colors as
the native renderer — none of which is duplicated in JS.

## Requirements & notes

- **WebGPU** (Chrome/Edge, or Firefox with WebGPU enabled) from a **secure
  context** (`https://` or `http://localhost`) gives the fast path. Without
  WebGPU (older browsers, some phones), `YOLO.load` automatically falls back to a
  portable **CPU/wasm** build — slower, but it runs everywhere. Force a backend
  with `YOLO.load(model, { backend: "webgpu" | "cpu" })`.
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

## Building from source

This package builds the wasm from the Rust crate with
[`wasm-pack`](https://github.com/rustwasm/wasm-pack):

```bash
npm run build # wasm-pack build + tsc
```

Serve the built `dist/` + `pkg/` over `localhost` (a secure context) and open it
in a WebGPU browser.

## License

AGPL-3.0. See the repository [LICENSE](https://github.com/ultralytics/inference/blob/main/LICENSE).
