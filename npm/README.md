<!-- Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license -->

# Ultralytics Inference (Browser / WebGPU)

Run [Ultralytics](https://ultralytics.com) YOLO models in the browser, entirely
client-side, accelerated by **WebGPU**. The inference engine is the
`ultralytics-inference` Rust crate compiled to WebAssembly; the forward pass runs
on the official [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)
build, bridged through [`ort-web`](https://ort.pyke.io/backends/web). All
preprocessing and postprocessing is shared with the native Rust and Python
implementations, so results match.

The API mirrors the Ultralytics Python package as closely as the browser allows.

## Install

```bash
npm install ultralytics-inference
```

## Quick start

```ts
import { YOLO } from "ultralytics-inference";

// Loads the model and initializes WebGPU + ONNX Runtime Web on first use.
const model = await YOLO.load("yolo26n.onnx");

const results = await model.predict("bus.jpg");
for (const box of results.boxes) {
  console.log(box.class_name, box.confidence.toFixed(2), [box.x1, box.y1, box.x2, box.y2]);
}
```

`predict()` accepts a URL/path, a `Blob`/`File`, raw encoded image bytes
(`Uint8Array`/`ArrayBuffer`), `ImageData`, an `HTMLImageElement`, an
`HTMLCanvasElement`, or an `ImageBitmap`.

```ts
const results = await model.predict(canvas, { conf: 0.25, iou: 0.7 });
```

## Results shape

`predict()` resolves to a `Results` object shaped like the Ultralytics Python
`Results`:

| Field | Type | Tasks |
| --- | --- | --- |
| `task` | `string` | all |
| `boxes` | `{ x1, y1, x2, y2, confidence, class_id, class_name }[]` | detect, segment, pose |
| `obb` | `{ x, y, width, height, angle, confidence, class_id, class_name }[]` | obb |
| `keypoints` | `{ points: [x, y, conf][] }[]` | pose |
| `probs` | `{ top1, top1_name, top1_conf, top5, top5_conf } \| null` | classify |
| `mask_count` | `number` | segment |
| `orig_width` / `orig_height` | `number` | all |

Detection, segmentation, pose, classification, and OBB are supported. (Segment
returns boxes plus a mask count today; raw mask pixels are a planned addition.)

## Requirements & notes

- **WebGPU** browser (Chrome/Edge, or Firefox with WebGPU enabled) served from a
  **secure context** (`https://` or `http://localhost`).
- **Model format**: export your model to ONNX with Ultralytics so the metadata
  (task, class names, `imgsz`) is embedded:
  ```python
  from ultralytics import YOLO
  YOLO("yolo26n.pt").export(format="onnx")
  ```
- **Runtime assets**: on first load, `ort-web` fetches the ONNX Runtime Web wasm
  bundle from `cdn.pyke.io`. If you set a Content-Security-Policy, allow that
  origin in `script-src`/`connect-src` (or self-host the runtime).
- **Telemetry**: `ort-web` reports the page domain to pyke on first session
  creation. See the [ort-web docs](https://ort.pyke.io/backends/web) to review or
  disable it.

## Building from source

This package builds the wasm from the Rust crate with
[`wasm-pack`](https://rustwasm.github.io/wasm-pack/):

```bash
npm run build      # wasm-pack build + tsc
```

See `example/index.html` for a complete, runnable demo (serve the `npm/` folder
over `localhost` and open it in a WebGPU browser).

## License

AGPL-3.0. See the repository [LICENSE](https://github.com/ultralytics/inference/blob/main/LICENSE).
