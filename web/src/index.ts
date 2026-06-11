// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

/**
 * Browser bindings for Ultralytics YOLO inference on WebGPU.
 *
 * Thin TypeScript wrapper around the WebAssembly module compiled from the
 * `ultralytics-inference` Rust crate. Inference runs on the GPU through ONNX
 * Runtime Web (bridged by `ort-web`); this module hides the wasm init, GPU
 * setup, and image encoding behind a small `YOLO` class:
 *
 * ```ts
 * import { YOLO } from "@ultralytics/yolo";
 *
 * const model = await YOLO.load("yolo26n.onnx");
 * const results = await model.predict("bus.jpg");
 * for (const box of results.boxes) {
 *   console.log(box.name, box.conf, box.x1, box.y1, box.x2, box.y2);
 * }
 * ```
 */

import init, { YoloModel, start, pose_palette } from "../pkg/ultralytics_inference_web.js";

/** Pose drawing scheme supplied by the engine (skeleton + palette colors). */
interface PoseScheme {
  skeleton: Array<[number, number]>;
  keypoint_colors: string[];
  limb_colors: string[];
}

let poseSchemeCache: PoseScheme | null = null;
/** The Ultralytics pose skeleton + per-keypoint/limb colors (constant). */
function poseScheme(): PoseScheme {
  poseSchemeCache ??= pose_palette() as PoseScheme;
  return poseSchemeCache;
}

// Field names mirror the Rust/Ultralytics `Results` API 1-1. Every detection
// carries its palette `color` (`#rrggbb`), supplied by the engine.

/** Axis-aligned box, pixel `xyxy`. */
export interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  conf: number;
  cls: number;
  name: string;
  color: string;
}

/** Oriented box, `xywhr` (angle in radians). */
export interface Obb {
  x: number;
  y: number;
  w: number;
  h: number;
  angle: number;
  conf: number;
  cls: number;
  name: string;
  color: string;
}

/** Keypoints for one detection: `[x, y, conf]` per point. */
export interface Keypoints {
  points: Array<[number, number, number]>;
  color: string;
}

/** Classification probabilities. */
export interface Probs {
  top1: number;
  top5: number[];
  top1conf: number;
  top5conf: number[];
  name: string;
  color: string;
}

/** Per-stage timing in ms. */
export interface Speed {
  preprocess: number;
  inference: number;
  postprocess: number;
}

/** Inference results, mirroring the Ultralytics `Results` API. */
export interface Results {
  task: string;
  width: number;
  height: number;
  boxes: Box[];
  obb: Obb[];
  keypoints: Keypoints[];
  probs: Probs | null;
  /** Segment/semantic masks as a translucent RGBA overlay (`width*height*4`), else empty. */
  masks: Uint8Array;
  /**
   * Semantic segmentation class id per pixel, row-major (`width*height`),
   * `undefined` for other tasks. The sentinel `65535` marks background or
   * class-filtered pixels. The `masks` overlay is its renderable form.
   */
  semantic?: Uint16Array;
  speed: Speed;
}

/** Options for {@link YOLO.load}. */
export interface LoadOptions {
  /**
   * Optional URL/path to the wasm binary (`ultralytics_inference_web_bg.wasm`).
   * Only needed if your bundler does not resolve it automatically.
   */
  wasmUrl?: string | URL;
  /**
   * Optional base URL to self-host the ONNX Runtime Web build instead of
   * fetching it from `cdn.pyke.io`. The directory must contain
   * `ort.webgpu.min.js`, `ort-wasm-simd-threaded.jsep.wasm`, and
   * `ort-wasm-simd-threaded.jsep.mjs`. Use an absolute URL ending in `/`.
   */
  ortBaseUrl?: string | URL;
  /**
   * Which device to run on, mirroring the native `device` option. `"auto"`
   * (default) picks WebGPU when the browser has a working adapter, otherwise the
   * portable CPU/wasm build. `"webgpu"` or `"cpu"` force one; the GPU adapter is
   * chosen automatically by the browser. If WebGPU fails to engage the model
   * falls back to CPU; read {@link YOLO.device} to see what actually ran.
   */
  device?: "auto" | "webgpu" | "cpu";
}

/**
 * Resolve the device string passed to wasm. An explicit choice is honored;
 * `"auto"` feature-detects a WebGPU adapter and otherwise falls back to CPU.
 */
async function resolveDevice(pref?: "auto" | "webgpu" | "cpu"): Promise<string> {
  if (pref === "cpu" || pref === "webgpu") return pref;
  const gpu = (navigator as { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  if (!gpu) return "cpu";
  try {
    return (await gpu.requestAdapter()) ? "webgpu" : "cpu";
  } catch {
    return "cpu";
  }
}

/** Ultralytics model assets release (ONNX exports of yolo26 / yolo11 / yolov8). */
const ASSETS = "https://github.com/ultralytics/assets/releases/download/v8.4.0/";

/**
 * Resolve a model reference to a URL. A bare ONNX name (e.g. `"yolo26n.onnx"`,
 * `"yolo11s-seg.onnx"`, `"yolov8n.onnx"`) auto-downloads from the Ultralytics
 * assets release; anything with a slash or scheme is used as-is.
 */
function resolveModel(src: string): string {
  return /^[\w.-]+\.onnx$/i.test(src) ? ASSETS + src : src;
}

/**
 * Reinterpret the wasm payload's little-endian `semantic` bytes as a
 * `Uint16Array` of class ids (a zero-copy view), leaving every other field as-is.
 */
function decodeResults(raw: unknown): Results {
  const r = raw as Results & { semantic?: Uint8Array | Uint16Array };
  const sem = r.semantic as Uint8Array | undefined;
  r.semantic = sem && sem.length ? new Uint16Array(sem.buffer, sem.byteOffset, sem.length / 2) : undefined;
  return r as Results;
}

/** Options for {@link YOLO.predict}. */
export interface PredictOptions {
  /** Confidence threshold. Default `0.25` (matches Ultralytics). */
  conf?: number;
  /** NMS IoU threshold. Default `0.7` (matches Ultralytics). */
  iou?: number;
  /**
   * Keep only these class ids. Omit to keep all. Filters detections for
   * detect/segment/pose/obb, and for semantic marks other pixels as background.
   */
  classes?: number[];
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
  /** Keypoint confidence threshold for drawing. Default `0.25` (matches Ultralytics). */
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
  | HTMLVideoElement
  | ImageBitmap;

let initialized = false;

/** Initialize the wasm module exactly once. */
async function ensureInit(wasmUrl?: string | URL): Promise<void> {
  if (initialized) return;
  await init(wasmUrl ? { module_or_path: wasmUrl } : undefined);
  start();
  initialized = true;
}

/** Encoded-image inputs (decoded by the wasm side via the `image` crate). */
type EncodedInput = string | URL | Blob | ArrayBuffer | Uint8Array;
/** Drawable inputs that can be read as raw pixels without re-encoding. */
type DrawableInput = ImageData | HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | ImageBitmap;

function isDrawable(input: ImageInput): input is DrawableInput {
  return (
    input instanceof ImageData ||
    input instanceof HTMLImageElement ||
    input instanceof HTMLCanvasElement ||
    input instanceof HTMLVideoElement ||
    input instanceof ImageBitmap
  );
}

/** Fetch/normalize an encoded image input to bytes the wasm side can decode. */
async function toEncodedBytes(input: EncodedInput): Promise<Uint8Array> {
  if (input instanceof Uint8Array) return input;
  if (input instanceof ArrayBuffer) return new Uint8Array(input);
  if (input instanceof Blob) return new Uint8Array(await input.arrayBuffer());
  const resp = await fetch(input);
  if (!resp.ok) throw new Error(`failed to fetch image: ${input} (${resp.status})`);
  return new Uint8Array(await resp.arrayBuffer());
}

/** Read a drawable source as raw RGBA pixels (the fast path, no re-encoding). */
function toImageData(input: DrawableInput): { data: Uint8Array; width: number; height: number } {
  if (input instanceof ImageData) {
    return { data: new Uint8Array(input.data.buffer), width: input.width, height: input.height };
  }
  let width: number;
  let height: number;
  if (input instanceof HTMLImageElement) {
    [width, height] = [input.naturalWidth, input.naturalHeight];
  } else if (input instanceof HTMLVideoElement) {
    [width, height] = [input.videoWidth, input.videoHeight];
  } else {
    [width, height] = [input.width, input.height];
  }
  const canvas = makeCanvas(width, height);
  const ctx = get2d(canvas, { willReadFrequently: true });
  ctx.drawImage(input as CanvasImageSource, 0, 0, width, height);
  const img = ctx.getImageData(0, 0, width, height);
  return { data: new Uint8Array(img.data.buffer), width, height };
}

function makeCanvas(width: number, height: number): HTMLCanvasElement | OffscreenCanvas {
  if (typeof OffscreenCanvas !== "undefined") return new OffscreenCanvas(width, height);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

type Ctx2D = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;

/** Get a 2D context or throw a clear error. */
function get2d(canvas: HTMLCanvasElement | OffscreenCanvas, options?: CanvasRenderingContext2DSettings): Ctx2D {
  const ctx = canvas.getContext("2d", options) as Ctx2D | null;
  if (!ctx) throw new Error("could not acquire a 2D canvas context");
  return ctx;
}

/**
 * A loaded YOLO model. Loading is asynchronous, so use {@link YOLO.load} instead
 * of a constructor.
 */
export class YOLO {
  private constructor(private readonly model: YoloModel) {}

  /**
   * Load a model and initialize the ONNX Runtime backend.
   *
   * @param source ONNX model URL/path, or its raw bytes.
   * @param options Optional loader options (e.g. an explicit wasm URL).
   */
  static async load(source: string | URL | ArrayBuffer | Uint8Array, options?: LoadOptions): Promise<YOLO> {
    await ensureInit(options?.wasmUrl);
    let bytes: Uint8Array;
    if (typeof source === "string" || source instanceof URL) {
      const url = resolveModel(source.toString());
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`failed to fetch model: ${url} (${resp.status})`);
      bytes = new Uint8Array(await resp.arrayBuffer());
    } else {
      bytes = source instanceof Uint8Array ? source : new Uint8Array(source);
    }
    const ortBaseUrl = options?.ortBaseUrl ? options.ortBaseUrl.toString() : undefined;
    const device = await resolveDevice(options?.device);
    return new YOLO(await YoloModel.load_bytes(bytes, ortBaseUrl, device));
  }

  /** The model's task (`detect`, `segment`, `pose`, `classify`, `obb`, `semantic`). */
  get task(): string {
    return this.model.task;
  }

  /** The active device that ran inference (`"webgpu"` or `"cpu"`). */
  get device(): string {
    return this.model.device;
  }

  /** Class id -> name map (like Ultralytics `model.names`). */
  get names(): Record<number, string> {
    return this.model.names as Record<number, string>;
  }

  /**
   * Run inference on an image.
   *
   * Drawable sources (`ImageData`, `HTMLImageElement`, `HTMLCanvasElement`,
   * `HTMLVideoElement`, `ImageBitmap`) take a fast path that reads raw pixels
   * with no re-encoding, ideal for webcam/video. URLs, `Blob`s, and raw bytes
   * are decoded inside wasm.
   *
   * @param image The image to run on.
   * @param options Confidence/IoU thresholds.
   */
  async predict(image: ImageInput, options?: PredictOptions): Promise<Results> {
    const conf = options?.conf ?? 0.25;
    const iou = options?.iou ?? 0.7;
    const classes = options?.classes ? new Uint32Array(options.classes) : undefined;
    if (isDrawable(image)) {
      const { data, width, height } = toImageData(image);
      return decodeResults(await this.model.predict_rgba(data, width, height, conf, iou, classes));
    }
    const bytes = await toEncodedBytes(image);
    return decodeResults(await this.model.predict(bytes, conf, iou, classes));
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
  const width = results.width || (bitmap as { width: number }).width;
  const height = results.height || (bitmap as { height: number }).height;
  canvas.width = width;
  canvas.height = height;

  const ctx = get2d(canvas);

  ctx.drawImage(bitmap as CanvasImageSource, 0, 0, width, height);

  // Segment: composite the translucent colored mask overlay from the engine.
  const overlay = results.masks;
  if (overlay && overlay.length === width * height * 4) {
    const tmp = makeCanvas(width, height);
    get2d(tmp).putImageData(new ImageData(new Uint8ClampedArray(overlay), width, height), 0, 0);
    ctx.drawImage(tmp as CanvasImageSource, 0, 0, width, height);
  }

  const lineWidth = options?.lineWidth ?? Math.max(2, Math.round(width / 320));
  const fontSize = Math.max(12, Math.round(width / 40));
  const font = options?.font ?? `${fontSize}px sans-serif`;
  const showLabels = options?.labels ?? true;
  const showKpts = options?.keypoints ?? true;
  const kptThresh = options?.keypointThreshold ?? 0.25;
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
    label(`${b.name} ${(b.conf * 100).toFixed(0)}%`, b.x1, b.y1, b.color);
  }

  // Oriented boxes (obb): draw the rotated rectangle.
  for (const o of results.obb) {
    const color = o.color;
    ctx.strokeStyle = color;
    const cos = Math.cos(o.angle);
    const sin = Math.sin(o.angle);
    const hw = o.w / 2;
    const hh = o.h / 2;
    const corners: Array<[number, number]> = [
      [-hw, -hh],
      [hw, -hh],
      [hw, hh],
      [-hw, hh],
    ].map(([dx, dy]) => [o.x + dx * cos - dy * sin, o.y + dx * sin + dy * cos]);
    ctx.beginPath();
    ctx.moveTo(corners[0][0], corners[0][1]);
    for (let i = 1; i < corners.length; i++) ctx.lineTo(corners[i][0], corners[i][1]);
    ctx.closePath();
    ctx.stroke();
    label(`${o.name} ${(o.conf * 100).toFixed(0)}%`, corners[0][0], corners[0][1], color);
  }

  // Pose keypoints + skeleton, colored with the engine's pose palette (per-limb
  // and per-keypoint), matching the native/Python renderer.
  if (showKpts && results.keypoints.length) {
    const { skeleton, keypoint_colors, limb_colors } = poseScheme();
    const radius = Math.max(2, Math.round(width / 300));
    ctx.lineWidth = Math.max(1, Math.round(lineWidth / 1.5));
    for (const kp of results.keypoints) {
      skeleton.forEach(([a, b], li) => {
        const pa = kp.points[a];
        const pb = kp.points[b];
        if (!pa || !pb || pa[2] < kptThresh || pb[2] < kptThresh) return;
        ctx.strokeStyle = limb_colors[li] ?? kp.color;
        ctx.beginPath();
        ctx.moveTo(pa[0], pa[1]);
        ctx.lineTo(pb[0], pb[1]);
        ctx.stroke();
      });
      kp.points.forEach(([x, y, c], ki) => {
        if (c < kptThresh) return;
        ctx.fillStyle = keypoint_colors[ki] ?? kp.color;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
      });
    }
    ctx.lineWidth = lineWidth;
  }

  // Classification: top-1 banner.
  if (results.probs) {
    label(`${results.probs.name} ${(results.probs.top1conf * 100).toFixed(1)}%`, 0, fontSize + 8, results.probs.color);
  }
}

/** Resolve any {@link ImageInput} to something `drawImage` accepts. */
async function resolveBitmap(input: ImageInput): Promise<CanvasImageSource> {
  if (typeof input === "string" || input instanceof URL) {
    const resp = await fetch(input);
    if (!resp.ok) throw new Error(`failed to fetch image: ${input} (${resp.status})`);
    return createImageBitmap(await resp.blob());
  }
  if (input instanceof Blob) return createImageBitmap(input);
  if (input instanceof Uint8Array) return createImageBitmap(new Blob([input as BlobPart]));
  if (input instanceof ArrayBuffer) return createImageBitmap(new Blob([input]));
  if (input instanceof ImageData) return createImageBitmap(input);
  // <img>, <canvas>, <video>, ImageBitmap are valid drawImage sources as-is;
  // createImageBitmap on a live <video> can throw InvalidStateError.
  return input;
}

export default YOLO;
