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

import init, { YoloModel, YoloPipeline, start, pose_palette } from "../pkg/ultralytics_inference_web.js";

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
  /** Display names for each {@link top5} class. */
  top5names: string[];
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
  semantic_mask?: Uint16Array;
  /**
   * Depth map as an opaque colorized RGBA image (`width*height*4`), using the
   * colormap from {@link PredictOptions.colormap}; empty for other tasks.
   * Drawable straight onto a canvas via `ImageData`.
   */
  depth: Uint8Array;
  /** Depth range `[min, max]` in meters over valid pixels, for depth models. */
  depth_range?: [number, number];
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
  /**
   * Base URL of the `@litertjs/core` wasm assets folder, used when the model is a
   * `.tflite` (run through LiteRT.js). Defaults to the jsDelivr CDN. Self-host by
   * copying `node_modules/@litertjs/core/wasm/` and pointing here (absolute URL
   * ending in `/`). Requires the optional `@litertjs/core` peer dependency.
   */
  litertWasmUrl?: string | URL;
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

/** Ultralytics default thresholds (mirrored in the option JSDoc above). */
const DEFAULT_CONF = 0.25;
const DEFAULT_IOU = 0.7;
const DEFAULT_KEYPOINT_THRESHOLD = 0.25;

/**
 * Resolve a model reference to a URL. A bare ONNX name (e.g. `"yolo26n.onnx"`,
 * `"yolo11s-seg.onnx"`, `"yolov8n.onnx"`) auto-downloads from the Ultralytics
 * assets release; anything with a slash or scheme is used as-is.
 */
function resolveModel(src: string): string {
  return /^[\w.-]+\.onnx$/i.test(src) ? ASSETS + src : src;
}

/** A model source: a URL/path, raw bytes, or a `Blob`/`File` (e.g. a dropped file). */
export type ModelSource = string | URL | ArrayBuffer | Uint8Array | Blob;

/** Whether `bytes` begin with a TFLite flatbuffer (the `TFL3` identifier at offset 4). */
function isTflite(bytes: Uint8Array): boolean {
  return bytes.length >= 8 && bytes[4] === 0x54 && bytes[5] === 0x46 && bytes[6] === 0x4c && bytes[7] === 0x33;
}

/**
 * Pick the inference backend from the model: `.tflite` -> `litert`, `.onnx` ->
 * `ort`. When the source has no known extension (raw bytes, a `Blob`/`File`, or an
 * extensionless/signed URL), the already-fetched `bytes` are sniffed for `TFL3`.
 */
function inferBackend(source: ModelSource, bytes: Uint8Array): "ort" | "litert" {
  if (typeof source === "string" || source instanceof URL) {
    // Match on the pathname only so `model.tflite?v=1` or a signed URL still works.
    const path = (source instanceof URL ? source.pathname : source).split(/[?#]/, 1)[0];
    if (/\.tflite$/i.test(path)) return "litert";
    if (/\.onnx$/i.test(path)) return "ort";
  }
  return isTflite(bytes) ? "litert" : "ort";
}

/**
 * Reinterpret the wasm payload's little-endian `semantic_mask` bytes as a
 * `Uint16Array` of class ids (a zero-copy view), leaving every other field as-is.
 */
function decodeResults(raw: unknown): Results {
  const r = raw as Results & { semantic_mask?: Uint8Array | Uint16Array };
  const sem = r.semantic_mask as Uint8Array | undefined;
  r.semantic_mask = sem && sem.length ? new Uint16Array(sem.buffer, sem.byteOffset, sem.length / 2) : undefined;
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
  /**
   * Depth colormap for the `depth` overlay: `"inferno"` (default), `"jet"`,
   * `"spectral"` (`Spectral_r`, DepthAnything-style), or `"gray"` (raw grayscale).
   * Ignored by every other task.
   */
  colormap?: "inferno" | "jet" | "spectral" | "gray";
  /**
   * Depth normalization: `"metric"` (default, min/max) or `"disparity"` (inverse depth +
   * percentile clip, the DepthAnything look). Ignored by every other task.
   */
  depthViz?: "metric" | "disparity";
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

/** Fetch a URL and throw a clear error naming `what` if the response is not OK. */
async function fetchOk(url: string | URL, what: string): Promise<Response> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`failed to fetch ${what}: ${url} (${resp.status})`);
  return resp;
}

/** Fetch/normalize an encoded image input to bytes the wasm side can decode. */
async function toEncodedBytes(input: EncodedInput): Promise<Uint8Array> {
  if (input instanceof Uint8Array) return input;
  if (input instanceof ArrayBuffer) return new Uint8Array(input);
  if (input instanceof Blob) return new Uint8Array(await input.arrayBuffer());
  const resp = await fetchOk(input, "image");
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

// ── Inference engines ───────────────────────────────────────────────────────
// `YOLO` is backend-agnostic: it holds an `Engine` that runs inference and
// returns the wasm `Results` payload. `ort` runs the whole pipeline in wasm;
// `litert` keeps pre/postprocessing in wasm and runs inference in LiteRT.js.

/** Minimal subset of the `@litertjs/core` API this wrapper uses, declared locally
 * so the package stays an *optional* peer dependency (no build-time coupling). */
interface LiteRtTensor {
  // Shape lives at type.layout.dimensions in @litertjs/core (not a top-level `shape`).
  readonly type: { layout: { dimensions: Int32Array | number[] } };
  moveTo(backend: string): Promise<LiteRtTensor>;
  toTypedArray(): Float32Array | Int32Array;
  delete?(): void;
}
interface LiteRtCompiledModel {
  run(input: LiteRtTensor | LiteRtTensor[]): Promise<LiteRtTensor[]>;
  delete?(): void;
}
interface LiteRtModule {
  loadLiteRt(wasmPath: string, opts?: { threads?: boolean }): Promise<void>;
  loadAndCompile(model: Uint8Array | string, opts: { accelerator: string | string[] }): Promise<LiteRtCompiledModel>;
  Tensor: new (data: Float32Array, shape: number[]) => LiteRtTensor;
}

/** Default CDN for the LiteRT.js wasm assets (override via `litertWasmUrl`). */
const DEFAULT_LITERT_WASM = "https://cdn.jsdelivr.net/npm/@litertjs/core/wasm/";

/** Import the optional `@litertjs/core` peer dep. An indirect specifier keeps it
 * out of the build graph, so the package never hard-depends on it. */
async function importLiteRt(): Promise<LiteRtModule> {
  const pkg = "@litertjs/core";
  try {
    return (await import(/* @vite-ignore */ pkg)) as unknown as LiteRtModule;
  } catch (e) {
    // Keep the original error visible so a real load failure is not mistaken for
    // a missing install.
    throw new Error(
      `failed to load the optional peer dependency "@litertjs/core" for the .tflite backend (run \`npm install @litertjs/core\`): ${e}`,
    );
  }
}

// LiteRT.js initializes its wasm runtime once per page; calling `loadLiteRt`
// again throws "LiteRT is already loading / loaded". Cache the first init so
// loading a second model (e.g. switching models in a demo) reuses it. The init
// is process-global, so a later load asking for a different `wasmUrl` is rejected
// (reload to switch) rather than silently reusing the original assets.
let litertInit: Promise<void> | null = null;
let litertInitUrl: string | null = null;
function ensureLiteRtRuntime(litert: LiteRtModule, wasmUrl: string): Promise<void> {
  if (litertInit) {
    if (litertInitUrl !== wasmUrl) {
      return Promise.reject(
        new Error(
          `LiteRT.js wasm is initialized once per page and shared by every model: already loaded from "${litertInitUrl}", cannot switch to "${wasmUrl}". Use the same litertWasmUrl, or reload the page.`,
        ),
      );
    }
    return litertInit;
  }
  litertInitUrl = wasmUrl;
  // Enable WASM multithreading when the page is cross-origin isolated (COOP/COEP
  // set, so SharedArrayBuffer is available); otherwise LiteRT.js runs single-
  // threaded. SIMD is detected by LiteRT itself. Isolation is a page-global
  // property, so this needs no per-load option.
  const threads = typeof crossOriginIsolated !== "undefined" && crossOriginIsolated;
  litertInit = litert.loadLiteRt(wasmUrl, { threads }).catch((e) => {
    litertInit = null; // let a later load retry if the first init failed
    litertInitUrl = null;
    throw e;
  });
  return litertInit;
}

/** Runs a `.tflite` model with LiteRT.js; inference only, pre/post stay in Rust. */
class LiteRtBackend {
  private constructor(
    private readonly litert: LiteRtModule,
    private readonly model: LiteRtCompiledModel,
    readonly device: string,
  ) {}

  /** Import LiteRT.js, init its wasm, and compile the model, falling back from
   * WebGPU to wasm (CPU) if the GPU accelerator cannot compile. */
  static async load(tflite: Uint8Array, wasmUrl: string, accelerator: "webgpu" | "wasm"): Promise<LiteRtBackend> {
    const litert = await importLiteRt();
    await ensureLiteRtRuntime(litert, wasmUrl);
    try {
      const model = await litert.loadAndCompile(tflite, { accelerator });
      return new LiteRtBackend(litert, model, accelerator);
    } catch (e) {
      if (accelerator === "wasm") throw e;
      const model = await litert.loadAndCompile(tflite, { accelerator: "wasm" });
      return new LiteRtBackend(litert, model, "wasm");
    }
  }

  /** Upload the input tensor, run, and read every output back to the CPU. */
  async run(
    input: Float32Array,
    shape: number[],
  ): Promise<{ outputs: Float32Array[]; shapes: number[][]; inferenceMs: number }> {
    const t0 = performance.now();
    const cpuIn = new this.litert.Tensor(input, shape);
    let gpuIn: LiteRtTensor | undefined;
    try {
      gpuIn = this.device === "wasm" ? cpuIn : await cpuIn.moveTo("webgpu");
      const tensors = await this.model.run(gpuIn);
      const outputs: Float32Array[] = [];
      const shapes: number[][] = [];
      for (const t of tensors) {
        const host = await t.moveTo("wasm");
        // Copy out before deleting; toTypedArray() may view the freed buffer.
        outputs.push(new Float32Array(host.toTypedArray() as Float32Array));
        shapes.push(Array.from(host.type.layout.dimensions));
        host.delete?.();
        // moveTo("wasm") returns the same tensor when it is already on wasm; only
        // delete the original separately when it is a distinct GPU tensor.
        if (t !== host) t.delete?.();
      }
      return { outputs, shapes, inferenceMs: performance.now() - t0 };
    } finally {
      // Free the input tensors on both success and failure (per-frame loops leak fast).
      if (gpuIn && gpuIn !== cpuIn) gpuIn.delete?.();
      cpuIn.delete?.();
    }
  }

  free(): void {
    this.model.delete?.();
  }
}

/** Backend behind {@link YOLO}: returns the wasm `Results` payload either way. */
interface Engine {
  readonly task: string;
  readonly device: string;
  readonly names: Record<number, string>;
  predictDrawable(
    data: Uint8Array,
    width: number,
    height: number,
    conf: number,
    iou: number,
    classes: Uint32Array | undefined,
    colormap: string,
    depthViz: string,
  ): Promise<unknown>;
  predictEncoded(
    bytes: Uint8Array,
    conf: number,
    iou: number,
    classes: Uint32Array | undefined,
    colormap: string,
    depthViz: string,
  ): Promise<unknown>;
  free(): void;
}

/** ONNX Runtime Web engine: the full pipeline runs inside wasm. */
class OrtEngine implements Engine {
  constructor(private readonly model: YoloModel) {}
  get task(): string {
    return this.model.task;
  }
  get device(): string {
    return this.model.device;
  }
  get names(): Record<number, string> {
    return this.model.names as Record<number, string>;
  }
  predictDrawable(
    data: Uint8Array,
    width: number,
    height: number,
    conf: number,
    iou: number,
    classes: Uint32Array | undefined,
    colormap: string,
    depthViz: string,
  ): Promise<unknown> {
    return this.model.predict_rgba(data, width, height, conf, iou, classes, colormap, depthViz);
  }
  predictEncoded(
    bytes: Uint8Array,
    conf: number,
    iou: number,
    classes: Uint32Array | undefined,
    colormap: string,
    depthViz: string,
  ): Promise<unknown> {
    return this.model.predict(bytes, conf, iou, classes, colormap, depthViz);
  }
  free(): void {
    this.model.free();
  }
}

/** LiteRT.js engine: Rust preprocess → LiteRT inference → Rust postprocess. */
class LiteRtEngine implements Engine {
  // `YoloPipeline` keeps exactly one pending frame in wasm state between
  // `preprocess_rgba` and `postprocess`, so predictions must not interleave.
  // Serialize them: a new call waits for the in-flight one to finish.
  private queue: Promise<unknown> = Promise.resolve();

  constructor(
    private readonly pipeline: YoloPipeline,
    private readonly backend: LiteRtBackend,
  ) {}
  get task(): string {
    return this.pipeline.task;
  }
  get device(): string {
    // Report the CPU path as "cpu" (the package-wide label); the LiteRT backend
    // tracks it internally as "wasm".
    return this.backend.device === "wasm" ? "cpu" : this.backend.device;
  }
  get names(): Record<number, string> {
    return this.pipeline.names as Record<number, string>;
  }

  predictDrawable(
    data: Uint8Array,
    width: number,
    height: number,
    conf: number,
    iou: number,
    classes: Uint32Array | undefined,
    colormap: string,
    depthViz: string,
  ): Promise<unknown> {
    return this.serialize(() => this.runDrawable(data, width, height, conf, iou, classes, colormap, depthViz));
  }

  /** LiteRT takes raw pixels, so decode encoded inputs to RGBA in JS first. */
  predictEncoded(
    bytes: Uint8Array,
    conf: number,
    iou: number,
    classes: Uint32Array | undefined,
    colormap: string,
    depthViz: string,
  ): Promise<unknown> {
    return this.serialize(async () => {
      const bitmap = await createImageBitmap(new Blob([bytes as BlobPart]));
      try {
        const { data, width, height } = toImageData(bitmap);
        return await this.runDrawable(data, width, height, conf, iou, classes, colormap, depthViz);
      } finally {
        bitmap.close();
      }
    });
  }

  private async runDrawable(
    data: Uint8Array,
    width: number,
    height: number,
    conf: number,
    iou: number,
    classes: Uint32Array | undefined,
    colormap: string,
    depthViz: string,
  ): Promise<unknown> {
    const input = this.pipeline.preprocess_rgba(data, width, height);
    const { outputs, shapes, inferenceMs } = await this.backend.run(input, Array.from(this.pipeline.inputShape));
    // Flatten shapes as [rank, dims..., rank, dims...] for the wasm boundary.
    const flatShapes: number[] = [];
    for (const s of shapes) flatShapes.push(s.length, ...s);
    return this.pipeline.postprocess(outputs, new Uint32Array(flatShapes), inferenceMs, conf, iou, classes, colormap, depthViz);
  }

  /** Chain `fn` after any in-flight prediction so wasm frame state is never shared. */
  private serialize<T>(fn: () => Promise<T>): Promise<T> {
    const result = this.queue.then(fn, fn);
    this.queue = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }

  free(): void {
    this.pipeline.free();
    this.backend.free();
  }
}

/** Fetch model bytes from a URL/path, or pass provided bytes through. */
async function fetchModelBytes(source: ModelSource): Promise<Uint8Array> {
  if (typeof source === "string" || source instanceof URL) {
    const resp = await fetchOk(source.toString(), "model");
    return new Uint8Array(await resp.arrayBuffer());
  }
  if (source instanceof Blob) return new Uint8Array(await source.arrayBuffer());
  return source instanceof Uint8Array ? source : new Uint8Array(source);
}

/**
 * A loaded YOLO model. Loading is asynchronous, so use {@link YOLO.load} instead
 * of a constructor.
 */
export class YOLO {
  private constructor(private readonly engine: Engine) {}

  /**
   * Load a model and initialize the inference backend.
   *
   * The engine is chosen from the model: a `.tflite` runs on LiteRT.js, an
   * `.onnx` on ONNX Runtime Web.
   *
   * @param source Model URL/path, raw bytes, or a `Blob`/`File` (e.g. from a file
   *   input or drag-and-drop). `.onnx` or `.tflite`.
   * @param options Loader options.
   */
  static async load(source: ModelSource, options?: LoadOptions): Promise<YOLO> {
    await ensureInit(options?.wasmUrl);
    // Fetch once (a bare `.onnx` name resolves to the Ultralytics assets release),
    // then pick the backend from the extension, or the bytes when it is unknown.
    const bytes = await fetchModelBytes(typeof source === "string" ? resolveModel(source) : source);
    if (inferBackend(source, bytes) === "litert") {
      return YOLO.loadLiteRt(bytes, options);
    }
    const ortBaseUrl = options?.ortBaseUrl ? options.ortBaseUrl.toString() : undefined;
    const device = await resolveDevice(options?.device);
    return new YOLO(new OrtEngine(await YoloModel.load_bytes(bytes, ortBaseUrl, device)));
  }

  /** Load the LiteRT (`.tflite`) backend. Metadata (task, class names, `imgsz`)
   * is read from the single `.tflite` file, the same as the `.onnx` path. */
  private static async loadLiteRt(tflite: Uint8Array, options?: LoadOptions): Promise<YOLO> {
    const pipeline = new YoloPipeline(tflite);
    const wasmUrl = options?.litertWasmUrl ? options.litertWasmUrl.toString() : DEFAULT_LITERT_WASM;
    // Honor `device` including `"auto"`, which feature-detects WebGPU (so a
    // browser without it goes straight to wasm instead of a failed compile).
    let accelerator: "webgpu" | "wasm" = (await resolveDevice(options?.device)) === "webgpu" ? "webgpu" : "wasm";
    // End-to-end exports (YOLO26) do NMS/top-k with int64 + gather_nd ops the
    // LiteRT WebGPU delegate cannot run: it fails to invoke and returns zeros.
    // Force CPU/wasm so the model works. For WebGPU speed, re-export the model
    // with `end2end=False` so the standard head (with NMS in Rust) is used.
    if (accelerator === "webgpu" && pipeline.end2end) {
      accelerator = "wasm";
      console.warn(
        "LiteRT: this is an end2end (NMS-free) model; its WebGPU delegate can't run the int64 ops, so it runs on CPU/wasm. Re-export with `end2end=False` for WebGPU.",
      );
    }
    const backend = await LiteRtBackend.load(tflite, wasmUrl, accelerator);
    return new YOLO(new LiteRtEngine(pipeline, backend));
  }

  /** The model's task (`detect`, `segment`, `pose`, `classify`, `obb`, `semantic`). */
  get task(): string {
    return this.engine.task;
  }

  /** The active device that ran inference (`"webgpu"`, `"wasm"`, or `"cpu"`). */
  get device(): string {
    return this.engine.device;
  }

  /** Class id -> name map (like Ultralytics `model.names`). */
  get names(): Record<number, string> {
    return this.engine.names;
  }

  /**
   * Run inference on an image.
   *
   * Drawable sources (`ImageData`, `HTMLImageElement`, `HTMLCanvasElement`,
   * `HTMLVideoElement`, `ImageBitmap`) take a fast path that reads raw pixels
   * with no re-encoding, ideal for webcam/video. URLs, `Blob`s, and raw bytes
   * are decoded (in wasm for `ort`, via a canvas for `litert`).
   *
   * @param image The image to run on.
   * @param options Confidence/IoU thresholds.
   */
  async predict(image: ImageInput, options?: PredictOptions): Promise<Results> {
    const conf = options?.conf ?? DEFAULT_CONF;
    const iou = options?.iou ?? DEFAULT_IOU;
    const classes = options?.classes ? new Uint32Array(options.classes) : undefined;
    const colormap = options?.colormap ?? "inferno";
    const depthViz = options?.depthViz ?? "metric";
    if (isDrawable(image)) {
      const { data, width, height } = toImageData(image);
      return decodeResults(await this.engine.predictDrawable(data, width, height, conf, iou, classes, colormap, depthViz));
    }
    const bytes = await toEncodedBytes(image);
    return decodeResults(await this.engine.predictEncoded(bytes, conf, iou, classes, colormap, depthViz));
  }

  /** Release the underlying wasm model/engine. Call when you are done with it. */
  free(): void {
    this.engine.free();
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
 * with the COCO skeleton, and a top-5 list for classification. The canvas is
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

  // Depth: the engine returns an opaque colorized depth image; draw it over the
  // frame so the canvas shows the depth map.
  const depth = results.depth;
  if (depth && depth.length === width * height * 4) {
    const tmp = makeCanvas(width, height);
    get2d(tmp).putImageData(new ImageData(new Uint8ClampedArray(depth), width, height), 0, 0);
    ctx.drawImage(tmp as CanvasImageSource, 0, 0, width, height);
  }

  const lineWidth = options?.lineWidth ?? Math.max(2, Math.round(width / 320));
  const fontSize = Math.max(12, Math.round(width / 40));
  const font = options?.font ?? `${fontSize}px sans-serif`;
  const showLabels = options?.labels ?? true;
  const showKpts = options?.keypoints ?? true;
  const kptThresh = options?.keypointThreshold ?? DEFAULT_KEYPOINT_THRESHOLD;
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

  // Classification: top-5 list on a translucent black panel (matches the native
  // and Python renderers, which list the top classes rather than a single banner).
  if (results.probs && showLabels) {
    const { top5, top5conf, top5names } = results.probs;
    const lines = top5.map((_, i) => `${top5names[i] ?? top5[i]} ${top5conf[i].toFixed(2)}`);
    const pad = Math.round(fontSize * 0.3);
    const lineH = Math.round(fontSize * 1.25);
    const boxW = Math.max(...lines.map((t) => ctx.measureText(t).width)) + pad * 2;
    const boxH = lines.length * lineH + pad * 2;
    ctx.fillStyle = "rgba(0, 0, 0, 0.4)";
    ctx.fillRect(pad, pad, boxW, boxH);
    ctx.fillStyle = "#ffffff";
    lines.forEach((t, i) => ctx.fillText(t, pad * 2, pad * 2 + i * lineH));
  }
}

/** Resolve any {@link ImageInput} to something `drawImage` accepts. */
async function resolveBitmap(input: ImageInput): Promise<CanvasImageSource> {
  if (typeof input === "string" || input instanceof URL) {
    const resp = await fetchOk(input, "image");
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
