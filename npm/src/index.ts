// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

/**
 * Browser bindings for Ultralytics YOLO inference on WebGPU.
 *
 * Thin, Python-like wrapper around the WebAssembly module compiled from the
 * `ultralytics-inference` Rust crate. Inference runs on the GPU through ONNX
 * Runtime Web (bridged by `ort-web`); this module hides the wasm init, GPU
 * setup, and image encoding behind a small `YOLO` class:
 *
 * ```ts
 * import { YOLO } from "ultralytics-inference";
 *
 * const model = await YOLO.load("yolo26n.onnx");
 * const results = await model.predict("bus.jpg");
 * for (const box of results.boxes) {
 *   console.log(box.class_name, box.confidence, box.x1, box.y1, box.x2, box.y2);
 * }
 * ```
 */

import init, { YoloModel, start } from "../pkg/ultralytics_inference_web.js";

/** A single detected box in original-image pixel coordinates (`xyxy`). */
export interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
  class_id: number;
  class_name: string;
}

/** A single oriented bounding box (`xywhr`, angle in radians). */
export interface OrientedBox {
  x: number;
  y: number;
  width: number;
  height: number;
  angle: number;
  confidence: number;
  class_id: number;
  class_name: string;
}

/** Keypoints for one detection: `[x, y, confidence]` per keypoint. */
export interface Keypoints {
  points: Array<[number, number, number]>;
}

/** Classification probabilities (top-5). */
export interface Probs {
  top1: number;
  top1_name: string;
  top1_conf: number;
  top5: number[];
  top5_conf: number[];
}

/** Inference results, shaped to mirror the Ultralytics Python `Results` API. */
export interface Results {
  /** Task name: `detect`, `segment`, `pose`, `classify`, `obb`, or `semantic`. */
  task: string;
  orig_width: number;
  orig_height: number;
  /** Axis-aligned detections (detect/segment/pose tasks). */
  boxes: Box[];
  /** Oriented detections (obb task). */
  obb: OrientedBox[];
  /** Per-detection keypoints (pose task). */
  keypoints: Keypoints[];
  /** Classification probabilities (classify task), else `null`. */
  probs: Probs | null;
  /** Number of instance masks produced (segment task). */
  mask_count: number;
}

/** Options for {@link YOLO.load}. */
export interface LoadOptions {
  /**
   * Optional URL/path to the wasm binary (`ultralytics_inference_web_bg.wasm`).
   * Only needed if your bundler does not resolve it automatically.
   */
  wasmUrl?: string | URL;
}

/** Options for {@link YOLO.predict}. */
export interface PredictOptions {
  /** Confidence threshold. Default `0.25` (matches Ultralytics). */
  conf?: number;
  /** NMS IoU threshold. Default `0.7` (matches Ultralytics). */
  iou?: number;
}

/** Image-like inputs accepted by {@link YOLO.predict}. */
export type ImageInput =
  | string
  | URL
  | Blob
  | ArrayBuffer
  | Uint8Array
  | ImageData
  | HTMLImageElement
  | HTMLCanvasElement
  | ImageBitmap;

let initialized = false;

/** Initialize the wasm module exactly once. */
async function ensureInit(wasmUrl?: string | URL): Promise<void> {
  if (initialized) return;
  await init(wasmUrl ? { module_or_path: wasmUrl } : undefined);
  start();
  initialized = true;
}

/** Encode an arbitrary image input to PNG/JPEG bytes the wasm side can decode. */
async function toImageBytes(input: ImageInput): Promise<Uint8Array> {
  if (input instanceof Uint8Array) return input;
  if (input instanceof ArrayBuffer) return new Uint8Array(input);
  if (typeof input === "string" || input instanceof URL) {
    const resp = await fetch(input);
    if (!resp.ok) throw new Error(`failed to fetch image: ${input} (${resp.status})`);
    return new Uint8Array(await resp.arrayBuffer());
  }
  if (input instanceof Blob) return new Uint8Array(await input.arrayBuffer());

  // Drawable sources: render to a canvas and encode to PNG.
  const { width, height } = drawableSize(input);
  const canvas = makeCanvas(width, height);
  const ctx = canvas.getContext("2d") as
    | CanvasRenderingContext2D
    | OffscreenCanvasRenderingContext2D
    | null;
  if (!ctx) throw new Error("could not acquire a 2D canvas context");
  if (input instanceof ImageData) {
    ctx.putImageData(input, 0, 0);
  } else {
    ctx.drawImage(input as CanvasImageSource, 0, 0);
  }
  const blob = await canvasToBlob(canvas);
  return new Uint8Array(await blob.arrayBuffer());
}

function drawableSize(input: ImageData | HTMLImageElement | HTMLCanvasElement | ImageBitmap): {
  width: number;
  height: number;
} {
  if (input instanceof HTMLImageElement) {
    return { width: input.naturalWidth, height: input.naturalHeight };
  }
  return { width: (input as { width: number }).width, height: (input as { height: number }).height };
}

function makeCanvas(width: number, height: number): HTMLCanvasElement | OffscreenCanvas {
  if (typeof OffscreenCanvas !== "undefined") return new OffscreenCanvas(width, height);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

async function canvasToBlob(canvas: HTMLCanvasElement | OffscreenCanvas): Promise<Blob> {
  if (canvas instanceof OffscreenCanvas) return canvas.convertToBlob({ type: "image/png" });
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("canvas.toBlob failed"))), "image/png");
  });
}

/**
 * A loaded YOLO model. Mirrors the Ultralytics Python `YOLO` class as closely as
 * the browser allows (loading is asynchronous, so use {@link YOLO.load} instead
 * of a constructor).
 */
export class YOLO {
  private constructor(private readonly model: YoloModel) {}

  /**
   * Load a model and initialize the WebGPU backend.
   *
   * @param source ONNX model URL/path, or its raw bytes.
   * @param options Optional loader options (e.g. an explicit wasm URL).
   */
  static async load(source: string | URL | ArrayBuffer | Uint8Array, options?: LoadOptions): Promise<YOLO> {
    await ensureInit(options?.wasmUrl);
    let model: YoloModel;
    if (typeof source === "string" || source instanceof URL) {
      model = await YoloModel.load_url(source.toString());
    } else {
      model = await YoloModel.load_bytes(source instanceof Uint8Array ? source : new Uint8Array(source));
    }
    return new YOLO(model);
  }

  /** The model's task (`detect`, `segment`, `pose`, `classify`, `obb`, `semantic`). */
  get task(): string {
    return this.model.task;
  }

  /**
   * Run inference on an image.
   *
   * @param image A URL/path, `Blob`/`File`, raw encoded bytes, `ImageData`,
   *   `HTMLImageElement`, `HTMLCanvasElement`, or `ImageBitmap`.
   * @param options Confidence/IoU thresholds.
   */
  async predict(image: ImageInput, options?: PredictOptions): Promise<Results> {
    const bytes = await toImageBytes(image);
    const conf = options?.conf ?? 0.25;
    const iou = options?.iou ?? 0.7;
    return (await this.model.predict(bytes, conf, iou)) as Results;
  }

  /** Release the underlying wasm model. Call when you are done with it. */
  free(): void {
    this.model.free();
  }
}

export default YOLO;
