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
  /** Ultralytics palette color for this class (`#rrggbb`), supplied by the engine. */
  color: string;
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
  /** Ultralytics palette color for this class (`#rrggbb`), supplied by the engine. */
  color: string;
}

/** Keypoints for one detection: `[x, y, confidence]` per keypoint. */
export interface Keypoints {
  points: Array<[number, number, number]>;
  /** Palette color for this instance (`#rrggbb`), supplied by the engine. */
  color: string;
}

/** Classification probabilities (top-5). */
export interface Probs {
  top1: number;
  top1_name: string;
  top1_conf: number;
  top5: number[];
  top5_conf: number[];
  /** Palette color for the top-1 class (`#rrggbb`), supplied by the engine. */
  color: string;
}

/** Per-stage timing in milliseconds (mirrors Ultralytics `results.speed`). */
export interface Speed {
  /** Image decode + letterbox/normalize. */
  preprocess: number;
  /** GPU inference (`run_async`) + cross-context output sync. */
  inference: number;
  /** NMS, decoding, and coordinate scaling. */
  postprocess: number;
}

/** Inference results, shaped to mirror the Ultralytics Python `Results` API. */
export interface Results {
  /** Task name: `detect`, `segment`, `pose`, `classify`, `obb`, or `semantic`. */
  task: string;
  orig_width: number;
  orig_height: number;
  /** Per-stage timing in milliseconds. */
  speed: Speed;
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

/** Options for {@link annotate}. */
export interface AnnotateOptions {
  /** Box/line width in pixels. Defaults to a value scaled to the image size. */
  lineWidth?: number;
  /** Font CSS string. Defaults to a size scaled to the image. */
  font?: string;
  /** Draw `class_name confidence%` labels. Default `true`. */
  labels?: boolean;
  /** Draw pose keypoints and skeleton. Default `true`. */
  keypoints?: boolean;
  /** Keypoint confidence threshold for drawing. Default `0.5`. */
  keypointThreshold?: number;
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
    let bytes: Uint8Array;
    if (typeof source === "string" || source instanceof URL) {
      const resp = await fetch(source);
      if (!resp.ok) throw new Error(`failed to fetch model: ${source} (${resp.status})`);
      bytes = new Uint8Array(await resp.arrayBuffer());
    } else {
      bytes = source instanceof Uint8Array ? source : new Uint8Array(source);
    }
    return new YOLO(await YoloModel.load_bytes(bytes));
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

// Colors come from the Rust engine (the Ultralytics palette) on each detection,
// so there is no palette duplicated here.

/** Pick black or white text for readability over a given hex background. */
function textColorFor(hex: string): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  // Perceived luminance (same threshold Ultralytics uses).
  return r * 0.299 + g * 0.587 + b * 0.114 > 140 ? "#000000" : "#FFFFFF";
}

// COCO 17-keypoint skeleton (0-indexed pairs) for pose rendering.
const COCO_SKELETON: Array<[number, number]> = [
  [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6],
  [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
  [3, 5], [4, 6],
];

/**
 * Draw inference results onto a canvas, on top of the source image.
 *
 * Handles every task: axis-aligned boxes, oriented boxes (OBB), pose keypoints
 * with the COCO skeleton, and a top-1 banner for classification. The canvas is
 * resized to the original image dimensions so detection coordinates line up.
 *
 * ```ts
 * const results = await model.predict("bus.jpg");
 * await annotate(document.querySelector("canvas"), "bus.jpg", results);
 * ```
 *
 * @param canvas Target `<canvas>` (or `OffscreenCanvas`).
 * @param image The same image passed to `predict` (URL, Blob, bytes, element, ...).
 * @param results The `Results` returned by {@link YOLO.predict}.
 * @param options Styling options.
 */
export async function annotate(
  canvas: HTMLCanvasElement | OffscreenCanvas,
  image: ImageInput,
  results: Results,
  options?: AnnotateOptions,
): Promise<void> {
  const bitmap = await resolveBitmap(image);
  const width = results.orig_width || (bitmap as { width: number }).width;
  const height = results.orig_height || (bitmap as { height: number }).height;
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext("2d") as
    | CanvasRenderingContext2D
    | OffscreenCanvasRenderingContext2D
    | null;
  if (!ctx) throw new Error("could not acquire a 2D canvas context");

  ctx.drawImage(bitmap as CanvasImageSource, 0, 0, width, height);

  const lineWidth = options?.lineWidth ?? Math.max(2, Math.round(width / 320));
  const fontSize = Math.max(12, Math.round(width / 40));
  const font = options?.font ?? `${fontSize}px sans-serif`;
  const showLabels = options?.labels ?? true;
  const showKpts = options?.keypoints ?? true;
  const kptThresh = options?.keypointThreshold ?? 0.5;
  ctx.lineWidth = lineWidth;
  ctx.font = font;
  ctx.textBaseline = "top";

  const label = (text: string, x: number, y: number, color: string) => {
    if (!showLabels) return;
    const pad = Math.round(fontSize * 0.2);
    const w = ctx.measureText(text).width + pad * 2;
    const h = fontSize + pad * 2;
    const ly = Math.max(0, y - h);
    ctx.fillStyle = color;
    ctx.fillRect(x, ly, w, h);
    ctx.fillStyle = textColorFor(color);
    ctx.fillText(text, x + pad, ly + pad);
  };

  // Axis-aligned boxes (detect / segment / pose).
  for (const b of results.boxes) {
    ctx.strokeStyle = b.color;
    ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    label(`${b.class_name} ${(b.confidence * 100).toFixed(0)}%`, b.x1, b.y1, b.color);
  }

  // Oriented boxes (obb): draw the rotated rectangle.
  for (const o of results.obb) {
    const color = o.color;
    ctx.strokeStyle = color;
    const cos = Math.cos(o.angle);
    const sin = Math.sin(o.angle);
    const hw = o.width / 2;
    const hh = o.height / 2;
    const corners: Array<[number, number]> = [
      [-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh],
    ].map(([dx, dy]) => [o.x + dx * cos - dy * sin, o.y + dx * sin + dy * cos]);
    ctx.beginPath();
    ctx.moveTo(corners[0][0], corners[0][1]);
    for (let i = 1; i < corners.length; i++) ctx.lineTo(corners[i][0], corners[i][1]);
    ctx.closePath();
    ctx.stroke();
    label(`${o.class_name} ${(o.confidence * 100).toFixed(0)}%`, corners[0][0], corners[0][1], color);
  }

  // Pose keypoints + skeleton.
  if (showKpts) {
    const radius = Math.max(2, Math.round(width / 300));
    for (const kp of results.keypoints) {
      ctx.strokeStyle = kp.color;
      ctx.lineWidth = Math.max(1, Math.round(lineWidth / 2));
      if (kp.points.length === 17) {
        for (const [a, b] of COCO_SKELETON) {
          const pa = kp.points[a];
          const pb = kp.points[b];
          if (pa[2] < kptThresh || pb[2] < kptThresh) continue;
          ctx.beginPath();
          ctx.moveTo(pa[0], pa[1]);
          ctx.lineTo(pb[0], pb[1]);
          ctx.stroke();
        }
      }
      ctx.fillStyle = kp.color;
      for (const [x, y, c] of kp.points) {
        if (c < kptThresh) continue;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.lineWidth = lineWidth;
  }

  // Classification: top-1 banner.
  if (results.probs) {
    label(
      `${results.probs.top1_name} ${(results.probs.top1_conf * 100).toFixed(1)}%`,
      0,
      fontSize + 8,
      results.probs.color,
    );
  }
}

/** Resolve any {@link ImageInput} to something drawable on a canvas. */
async function resolveBitmap(input: ImageInput): Promise<ImageBitmap | HTMLImageElement> {
  if (typeof input === "string" || input instanceof URL) {
    const resp = await fetch(input);
    if (!resp.ok) throw new Error(`failed to fetch image: ${input} (${resp.status})`);
    return createImageBitmap(await resp.blob());
  }
  if (input instanceof Uint8Array) return createImageBitmap(new Blob([input as BlobPart]));
  if (input instanceof ArrayBuffer) return createImageBitmap(new Blob([input]));
  // Blob, ImageData, HTMLImageElement, HTMLCanvasElement, ImageBitmap are all
  // accepted directly by createImageBitmap.
  return createImageBitmap(input as ImageBitmapSource);
}

export default YOLO;
