// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

/**
 * Browser bindings for Ultralytics YOLO inference on WebGPU or CPU.
 *
 * Thin TypeScript wrapper around the WebAssembly module compiled from the
 * `ultralytics-inference` Rust crate. `.onnx` models run through ONNX Runtime Web
 * (bridged by `ort-web`) and `.tflite` models through LiteRT.js; this module hides
 * the wasm init, device setup, and image encoding behind a small `YOLO` class:
 *
 * ```ts
 * import { YOLO } from "@ultralytics/yolo";
 *
 * const model = await YOLO.load("/models/yolo26n.onnx");
 * const results = await model.predict("bus.jpg");
 * for (const box of results.boxes) {
 *   console.log(box.name, box.conf, box.x1, box.y1, box.x2, box.y2);
 * }
 *
 * // Tracks carry over between calls, like `model.track(persist=True)` in Ultralytics.
 * const tracked = await model.track(videoElement, { tracker: "bytetrack.yaml" });
 * console.log(tracked.boxes.map((box) => box.id));
 * ```
 */

import init, { YoloModel, YoloPipeline, start, pose_palette, bidi_runs } from "../pkg/ultralytics_inference_web.js";

/** Pose drawing scheme supplied by the engine (skeleton + palette colors). */
interface PoseScheme {
  skeleton: Array<[number, number]>;
  keypoint_colors: string[];
  limb_colors: string[];
  palette: string[];
}

let poseSchemeCache: PoseScheme | null = null;
/** The Ultralytics pose skeleton + per-keypoint/limb colors (constant). */
function poseScheme(): PoseScheme {
  poseSchemeCache ??= pose_palette() as PoseScheme;
  return poseSchemeCache;
}

// Field names mirror the Rust/Ultralytics `Results` API 1-1. Every detection
// carries its class `color` (`#rrggbb`), supplied by the engine.

/** Axis-aligned box, pixel `xyxy`. */
export interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  /** Track id, set by {@link YOLO.track}. */
  id?: number;
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
  /** Track id, set by {@link YOLO.track}. */
  id?: number;
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
   * Depth map as an opaque colorized RGBA image (`width*height*4`), using the colormap
   * from {@link PredictOptions.colormap}; empty for other tasks. {@link annotate} blends
   * it over the frame at {@link AnnotateOptions.depthAlpha} (default 0.6); draw it directly
   * for the raw map.
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
   * `ort-wasm-simd-threaded.jsep.mjs` for WebGPU, plus `ort.wasm.min.js`,
   * `ort-wasm-simd-threaded.wasm`, and `ort-wasm-simd-threaded.mjs` for the CPU build, which is what a `"cpu"`
   * device and the WebGPU fallback below load. Use an absolute URL ending in `/`.
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
   *
   * Required on a cross-origin-isolated page: `COEP: require-corp` blocks the worker
   * LiteRT starts from the default CDN, so the wasm has to come from your own origin.
   */
  litertWasmUrl?: string | URL;
}

/**
 * Resolve the device string passed to wasm. `"cpu"` is honored as-is; anything else
 * needs a real adapter, since `navigator.gpu` can exist while `requestAdapter()`
 * still returns `null` — so `YOLO.device` reports what actually ran.
 */
async function resolveDevice(pref?: "auto" | "webgpu" | "cpu"): Promise<string> {
  if (pref === "cpu") return "cpu";
  const gpu = (navigator as { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  if (await gpu?.requestAdapter().catch(() => null)) return "webgpu";
  if (pref === "webgpu") console.warn("[@ultralytics/yolo] WebGPU requested but no adapter found; running on CPU.");
  return "cpu";
}

/** Ultralytics default keypoint confidence threshold for drawing. */
const DEFAULT_KEYPOINT_THRESHOLD = 0.25;

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
  /** NMS IoU threshold. Default `0.5` (matches Ultralytics predict). */
  iou?: number;
  /**
   * Keep only these class ids. Omit to keep all. Filters detections for
   * detect/segment/pose/obb, and for semantic marks other pixels as background.
   */
  classes?: number[];
  /**
   * Depth colormap for the `depth` overlay: `"jet"` (default, classic rainbow),
   * `"inferno"`, `"spectral"` (`Spectral_r`), or `"gray"` (raw grayscale). Ignored by
   * every other task.
   */
  colormap?: "inferno" | "jet" | "spectral" | "gray";
  /**
   * Depth normalization: `"disparity"` (default, inverse depth + percentile clip) or
   * `"metric"` (min/max). Ignored by every other task.
   */
  depthViz?: "metric" | "disparity";
}

/** Options for {@link YOLO.track}. */
export interface TrackOptions extends PredictOptions {
  /**
   * Tracker: `"botsort.yaml"`, `"bytetrack.yaml"`, `"ocsort.yaml"`, `"deepocsort.yaml"`, `"fasttrack.yaml"` or
   * `"tracktrack.yaml"` (default, like Ultralytics). Changing it starts new tracks.
   */
  tracker?: string;
}

/** Options for {@link annotate}. */
export interface AnnotateOptions {
  /** Box/line width in pixels. Defaults to a value scaled to the image size. */
  lineWidth?: number;
  /** Font CSS string. Defaults to a size scaled to the image. */
  font?: string;
  /** Draw `id:track_id class_name confidence%` labels. Default `true`. */
  labels?: boolean;
  /** Draw pose keypoints and skeleton. Default `true`. */
  keypoints?: boolean;
  /** Keypoint confidence threshold for drawing. Default `0.25` (matches Ultralytics). */
  keypointThreshold?: number;
  /** Keypoint radius in pixels (`kpt_radius`). Default `5`, like Ultralytics. */
  keypointRadius?: number;
  /** Draw the lines between keypoints (`kpt_line`). Default `true`. */
  keypointLines?: boolean;
  /** One CSS color for every keypoint and line (`kpt_color`). Defaults to the palette. */
  keypointColor?: string;
  /**
   * Pairs of keypoint indices to connect. Defaults to the human skeleton for 17-keypoint
   * (COCO-Pose) models and no lines for other layouts, like Ultralytics.
   */
  skeleton?: Array<[number, number]>;
  /**
   * Blend opacity for the depth overlay, `0`..`1`. Default `0.6` (matches the native
   * annotator); `1` shows the raw colorized depth map, `0` shows only the source.
   */
  depthAlpha?: number;
  /**
   * Optional URL/path to the wasm binary, as in {@link LoadOptions.wasmUrl}. Labels with
   * right-to-left text need the engine, so pass it when no model has loaded it in this thread.
   */
  wasmUrl?: string | URL;
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
  | OffscreenCanvas
  | ImageBitmap;

let initialized: Promise<void> | undefined;

/** Initialize the wasm module exactly once. */
function ensureInit(wasmUrl?: string | URL): Promise<void> {
  initialized ??= init(wasmUrl ? { module_or_path: wasmUrl } : undefined)
    .then(() => start())
    .catch((error) => {
      initialized = undefined;
      throw error;
    });
  return initialized;
}

/** Encoded-image inputs (decoded by the wasm side via the `image` crate). */
type EncodedInput = string | URL | Blob | ArrayBuffer | Uint8Array;
/** Drawable inputs that can be read as raw pixels without re-encoding. */
type DrawableInput =
  ImageData | HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | OffscreenCanvas | ImageBitmap;

function isDrawable(input: ImageInput): input is DrawableInput {
  return (
    (typeof ImageData !== "undefined" && input instanceof ImageData) ||
    (typeof HTMLImageElement !== "undefined" && input instanceof HTMLImageElement) ||
    (typeof HTMLCanvasElement !== "undefined" && input instanceof HTMLCanvasElement) ||
    (typeof HTMLVideoElement !== "undefined" && input instanceof HTMLVideoElement) ||
    (typeof OffscreenCanvas !== "undefined" && input instanceof OffscreenCanvas) ||
    (typeof ImageBitmap !== "undefined" && input instanceof ImageBitmap)
  );
}

/** Fetch a URL and throw a clear error naming `what` if the response is not OK. */
async function fetchOk(url: string | URL, what: string): Promise<Response> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`failed to fetch ${what}: ${url} (${resp.status})`);
  return resp;
}

/** Fetch a URL/path, or normalize provided bytes, into bytes the wasm side can read. */
async function toBytes(input: EncodedInput, what: string): Promise<Uint8Array> {
  if (input instanceof Uint8Array) return input;
  if (input instanceof ArrayBuffer) return new Uint8Array(input);
  if (input instanceof Blob) return new Uint8Array(await input.arrayBuffer());
  const resp = await fetchOk(input, what);
  return new Uint8Array(await resp.arrayBuffer());
}

/** Read a drawable source as raw RGBA pixels (the fast path, no re-encoding). */
function toImageData(input: DrawableInput): { data: Uint8Array; width: number; height: number } {
  if (typeof ImageData !== "undefined" && input instanceof ImageData) {
    return {
      // Snapshot caller-owned pixels before a queued prediction can yield.
      data: new Uint8Array(input.data),
      width: input.width,
      height: input.height,
    };
  }
  let width: number;
  let height: number;
  if (typeof HTMLImageElement !== "undefined" && input instanceof HTMLImageElement) {
    [width, height] = [input.naturalWidth, input.naturalHeight];
  } else if (typeof HTMLVideoElement !== "undefined" && input instanceof HTMLVideoElement) {
    [width, height] = [input.videoWidth, input.videoHeight];
  } else {
    [width, height] = [input.width, input.height];
  }
  const canvas = makeCanvas(width, height);
  const ctx = get2d(canvas, { willReadFrequently: true });
  ctx.drawImage(input as CanvasImageSource, 0, 0, width, height);
  const img = ctx.getImageData(0, 0, width, height);
  return { data: new Uint8Array(img.data.buffer, img.data.byteOffset, img.data.byteLength), width, height };
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
  readonly options: { accelerator: "webgpu" | "wasm" };
  readonly isFullyAccelerated: boolean;
  run(input: LiteRtTensor | LiteRtTensor[]): Promise<LiteRtTensor[]>;
  // The compiled model knows its real input shape; the metadata only claims one.
  getInputDetails?(): readonly { readonly shape: Int32Array | number[] }[];
  delete?(): void;
}
interface LiteRtModule {
  loadLiteRt(wasmPath: string, opts?: { threads?: boolean }): Promise<void>;
  supportsFeature(feature: "relaxedSimd"): Promise<boolean>;
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
/** Reject a cross-origin wasm URL on an isolated page before LiteRT fails on it.
 *
 * Threads need `crossOriginIsolated`, which needs `COEP: require-corp`, and that blocks
 * the worker LiteRT starts from another origin. The default URL is a CDN, so the two
 * settings are mutually exclusive and the failure otherwise arrives as a bare
 * `SecurityError: Failed to construct 'Worker'`.
 */
function checkWasmUrlReachable(wasmUrl: string): void {
  if (typeof crossOriginIsolated === "undefined" || !crossOriginIsolated) return;
  if (typeof location === "undefined") return;
  const origin = new URL(wasmUrl, location.href).origin;
  if (origin === location.origin) return;
  throw new Error(
    `LiteRT.js wasm must be served from this origin on a cross-origin-isolated page: "${wasmUrl}" is on ${origin}, and COEP blocks the worker it starts. Copy node_modules/@litertjs/core/wasm/ next to your app and pass it as litertWasmUrl, or drop COOP/COEP to give up WASM threads.`,
  );
}

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
  checkWasmUrlReachable(wasmUrl);
  litertInitUrl = wasmUrl;
  // Enable WASM multithreading when the page is cross-origin isolated (COOP/COEP
  // set, so SharedArrayBuffer is available); otherwise LiteRT.js runs single-
  // threaded. LiteRT's threaded build also needs relaxed SIMD, which Safari lacks.
  // Isolation is a page-global property, so this needs no per-load option.
  const isolated = typeof crossOriginIsolated !== "undefined" && crossOriginIsolated;
  litertInit = (async () => {
    const threads = isolated && (await litert.supportsFeature("relaxedSimd"));
    await litert.loadLiteRt(wasmUrl, { threads });
  })().catch((e) => {
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

  /** Input shape the compiled model actually expects, when LiteRT reports it. */
  get inputShape(): number[] | undefined {
    const shape = this.model.getInputDetails?.()[0]?.shape;
    return shape && Array.from(shape);
  }

  /** Import LiteRT.js, init its wasm, and compile the model, falling back from
   * WebGPU to wasm (CPU) if the GPU accelerator cannot compile. */
  static async load(tflite: Uint8Array, wasmUrl: string, accelerator: "webgpu" | "wasm"): Promise<LiteRtBackend> {
    const litert = await importLiteRt();
    await ensureLiteRtRuntime(litert, wasmUrl);
    let model: LiteRtCompiledModel;
    try {
      model = await litert.loadAndCompile(tflite, { accelerator });
    } catch (e) {
      if (accelerator === "wasm") throw e;
      model = await litert.loadAndCompile(tflite, { accelerator: "wasm" });
    }
    // This runtime is loaded without JSPI. A partially delegated model needs it to
    // switch between GPU and CPU, even when the browser itself supports JSPI.
    if (model.options.accelerator === "webgpu" && !model.isFullyAccelerated) {
      model.delete?.();
      model = await litert.loadAndCompile(tflite, { accelerator: "wasm" });
    }
    return new LiteRtBackend(litert, model, model.options.accelerator);
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
      for (const tensor of tensors) {
        const host = await tensor.moveTo("wasm");
        // LiteRT's toTypedArray() already copies the host data.
        const data = host.toTypedArray();
        outputs.push(data instanceof Float32Array ? data : new Float32Array(data));
        shapes.push(Array.from(host.type.layout.dimensions));
        host.delete?.();
        if (tensor !== host) tensor.delete?.();
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

/** Prediction knobs passed straight through to wasm, shared by every engine call. */
interface PredictParams {
  conf: number | undefined;
  iou: number | undefined;
  classes: Uint32Array | undefined;
  colormap: string;
  depthViz: string;
  tracker: string | undefined;
}

/** Backend behind {@link YOLO}: returns the wasm `Results` payload either way. */
interface Engine {
  readonly task: string;
  readonly device: string;
  readonly names: Record<number, string>;
  predictDrawable(data: Uint8Array, width: number, height: number, p: PredictParams): Promise<unknown>;
  predictEncoded(bytes: Uint8Array, p: PredictParams): Promise<unknown>;
  resetTracker(): void;
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
  predictDrawable(data: Uint8Array, width: number, height: number, p: PredictParams): Promise<unknown> {
    return this.model.predict_rgba(data, width, height, p.conf, p.iou, p.classes, p.colormap, p.depthViz, p.tracker);
  }
  predictEncoded(bytes: Uint8Array, p: PredictParams): Promise<unknown> {
    return this.model.predict(bytes, p.conf, p.iou, p.classes, p.colormap, p.depthViz, p.tracker);
  }
  resetTracker(): void {
    this.model.reset_tracker();
  }
  free(): void {
    this.model.free();
  }
}

/** LiteRT.js engine: Rust preprocess → LiteRT inference → Rust postprocess. */
class LiteRtEngine implements Engine {
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

  /** LiteRT takes raw pixels, so decode encoded inputs to RGBA in JS first. */
  async predictEncoded(bytes: Uint8Array, p: PredictParams): Promise<unknown> {
    const bitmap = await createImageBitmap(new Blob([bytes as BlobPart]));
    try {
      const { data, width, height } = toImageData(bitmap);
      return await this.predictDrawable(data, width, height, p);
    } finally {
      bitmap.close();
    }
  }

  async predictDrawable(data: Uint8Array, width: number, height: number, p: PredictParams): Promise<unknown> {
    const input = this.pipeline.preprocess_rgba(data, width, height);
    const { outputs, shapes, inferenceMs } = await this.backend.run(input, Array.from(this.pipeline.inputShape));
    return this.pipeline.postprocess(
      outputs,
      shapes.map((s) => new Uint32Array(s)),
      inferenceMs,
      p.conf,
      p.iou,
      p.classes,
      p.colormap,
      p.depthViz,
      p.tracker,
    );
  }

  resetTracker(): void {
    this.pipeline.reset_tracker();
  }

  free(): void {
    this.pipeline.free();
    this.backend.free();
  }
}

/**
 * A loaded YOLO model. Loading is asynchronous, so use {@link YOLO.load} instead
 * of a constructor.
 */
export class YOLO {
  // The wasm models keep mutable per-frame state; run one prediction per model at a time.
  private queue: Promise<unknown> = Promise.resolve();

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
    // Fetch once, then pick the backend from the extension, or the bytes when it is unknown.
    const bytes = await toBytes(source, "model");
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
    const accelerator: "webgpu" | "wasm" = (await resolveDevice(options?.device)) === "webgpu" ? "webgpu" : "wasm";
    const backend = await LiteRtBackend.load(tflite, wasmUrl, accelerator);
    // The compiled model outranks the metadata: `pipeline.inputShape` is what sizes the
    // input tensor below, so a stale `imgsz` would build one the model rejects.
    // Signed on purpose: the engine reports a dynamic axis as a negative number, and wasm
    // rejects those rather than reading one as a huge unsigned size.
    const shape = backend.inputShape;
    if (shape) pipeline.setInputShape(new Int32Array(shape));
    return new YOLO(new LiteRtEngine(pipeline, backend));
  }

  /** The model's task (`detect`, `segment`, `semantic`, `depth`, `classify`, `pose`, `obb`). */
  get task(): string {
    return this.engine.task;
  }

  /** The active device that ran inference (`"webgpu"` or `"cpu"`). */
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
   * `HTMLVideoElement`, `OffscreenCanvas`, `ImageBitmap`) take a fast path that reads raw pixels
   * with no re-encoding, ideal for webcam/video. URLs, `Blob`s, and raw bytes
   * are decoded (in wasm for `ort`, via a canvas for `litert`).
   *
   * @param image The image to run on.
   * @param options Confidence/IoU thresholds, class filter, and depth colorizing.
   */
  async predict(image: ImageInput, options?: PredictOptions): Promise<Results> {
    return this.run(image, options);
  }

  /**
   * Run inference on the next frame of a video and track its objects, like Ultralytics
   * `model.track(persist=True)`: tracks carry over between calls and every kept box gets an
   * `id`. Confidence defaults to `0.1` here, as in Ultralytics. Call {@link resetTracker}
   * before a new video.
   *
   * @param image The frame to run on, as in {@link predict}.
   * @param options The {@link predict} options and the tracker.
   */
  async track(image: ImageInput, options?: TrackOptions): Promise<Results> {
    return this.run(image, options, options?.tracker ?? "tracktrack.yaml");
  }

  /** Forget the tracks of {@link track}, so the next frame starts a new video. */
  resetTracker(): void {
    this.queue = this.queue.then(() => this.engine.resetTracker());
  }

  /** Run one prediction in turn, tracked by `tracker` when given. */
  private async run(image: ImageInput, options: PredictOptions | undefined, tracker?: string): Promise<Results> {
    const p: PredictParams = {
      // Unset thresholds fall back to the Rust `InferenceConfig` defaults.
      conf: options?.conf,
      iou: options?.iou,
      classes: options?.classes ? new Uint32Array(options.classes) : undefined,
      // Empty means "unset": the wasm side falls back to the Rust `Colormap`/`DepthViz`
      // defaults, so the default lives in exactly one place.
      colormap: options?.colormap ?? "",
      depthViz: options?.depthViz ?? "",
      tracker,
    };
    let predict: () => Promise<unknown>;
    if (isDrawable(image)) {
      const { data, width, height } = toImageData(image);
      predict = () => this.engine.predictDrawable(data, width, height, p);
    } else {
      const bytes =
        image instanceof Uint8Array
          ? image.slice()
          : image instanceof ArrayBuffer
            ? new Uint8Array(image.slice(0))
            : await toBytes(image, "image");
      predict = () => this.engine.predictEncoded(bytes, p);
    }
    const result = this.queue.then(async () => decodeResults(await predict()));
    this.queue = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }

  /** Release the underlying wasm model/engine. Call when you are done with it. */
  free(): void {
    this.engine.free();
  }
}

/** Label text color over a `#rrggbb` background, the rule of `Color::text_color` in the engine. */
function textColorFor(hex: string): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return r * 0.299 + g * 0.587 + b * 0.114 > 150 ? "#111F68" : "#FFFFFF";
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

  // Depth: the engine returns the opaque colorized depth map, blended over the frame here
  // at `depthAlpha` (default 0.6; pass 1 for the raw map, 0 for none). Like the segment
  // overlay above, this assumes an opaque source (photo/camera/video frame): a source with
  // transparent pixels keeps that transparency through the blend.
  const depth = results.depth;
  if (depth && depth.length === width * height * 4) {
    const tmp = makeCanvas(width, height);
    get2d(tmp).putImageData(new ImageData(new Uint8ClampedArray(depth), width, height), 0, 0);
    ctx.globalAlpha = options?.depthAlpha ?? 0.6;
    ctx.drawImage(tmp as CanvasImageSource, 0, 0, width, height);
    ctx.globalAlpha = 1;
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
  ctx.textAlign = "left";

  // Labels that may hold right-to-left text (Unicode's right-to-left blocks and controls) are
  // drawn as the engine's bidi runs, each in its resolved direction, since Safari's canvas
  // does not pair brackets in right-to-left text. Other labels read left to right as they are.
  const rtlText =
    /[\u0590-\u08FF\u200F\u202B\u202E\u2067\u2068\uFB1D-\uFDFF\uFE70-\uFEFF\u{10800}-\u{10FFF}\u{1E800}-\u{1EFFF}]/u;
  const names = [...results.boxes, ...results.obb].map((b) => b.name);
  if (showLabels && names.concat(results.probs?.top5names ?? []).some((n) => rtlText.test(n))) {
    await ensureInit(options?.wasmUrl);
  }
  // Class names repeat across labels, so each run's width is measured once per call.
  const widths = new Map<string, number>();
  type Run = { text: string; rtl: boolean; width: number };
  const layout = (text: string): Run[] =>
    (rtlText.test(text) ? bidi_runs(text) : [text]).map((run) => {
      const rtl = run.startsWith("\u202E");
      let width = widths.get(run);
      if (width === undefined) {
        ctx.direction = rtl ? "rtl" : "ltr";
        width = ctx.measureText(run).width;
        widths.set(run, width);
      }
      return { text: run, rtl, width };
    });
  const runsWidth = (runs: Run[]) => runs.reduce((sum, run) => sum + run.width, 0);
  const fillText = (runs: Run[], x: number, y: number) => {
    for (const run of runs) {
      ctx.direction = run.rtl ? "rtl" : "ltr";
      ctx.fillText(run.text, x, y);
      x += run.width;
    }
  };

  const caption = (d: Box | Obb) => `${d.id === undefined ? "" : `id:${d.id} `}${d.name} ${(d.conf * 100).toFixed(0)}%`;
  const label = (text: string, x: number, y: number, color: string) => {
    if (!showLabels) return;
    const pad = Math.round(fontSize * 0.2);
    const runs = layout(text);
    const w = runsWidth(runs) + pad * 2;
    const h = fontSize + pad * 2;
    const ly = Math.max(0, y - h);
    ctx.fillStyle = color;
    ctx.fillRect(x, ly, w, h);
    ctx.fillStyle = textColorFor(color);
    fillText(runs, x + pad, ly + pad);
  };

  // Axis-aligned boxes (detect / segment / pose).
  for (const b of results.boxes) {
    ctx.strokeStyle = b.color;
    ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    label(caption(b), b.x1, b.y1, b.color);
  }

  // Oriented boxes (obb): draw the rotated rectangle.
  for (const o of results.obb) {
    const color = o.color;
    ctx.strokeStyle = color;
    const cos = Math.cos(o.angle);
    const sin = Math.sin(o.angle);
    const hw = o.w / 2;
    const hh = o.h / 2;
    // Same corner order as the native `Obb::xyxyxyxy` and Ultralytics `xywhr2xyxyxyxy`.
    const corners: Array<[number, number]> = [
      [hw, hh],
      [hw, -hh],
      [-hw, -hh],
      [-hw, hh],
    ].map(([dx, dy]) => [o.x + dx * cos - dy * sin, o.y + dx * sin + dy * cos]);
    ctx.beginPath();
    ctx.moveTo(corners[0][0], corners[0][1]);
    for (let i = 1; i < corners.length; i++) ctx.lineTo(corners[i][0], corners[i][1]);
    ctx.closePath();
    ctx.stroke();
    label(caption(o), corners[0][0], corners[0][1], color);
  }

  // Pose keypoints + skeleton, colored with the engine's pose palette (per-limb
  // and per-keypoint), matching the native/Python renderer.
  if (showKpts && results.keypoints.length) {
    const scheme = poseScheme();
    const radius = options?.keypointRadius ?? 5;
    const fixed = options?.keypointColor;
    ctx.lineWidth = Math.max(1, Math.round(lineWidth / 1.5));
    for (const kp of results.keypoints) {
      // Like Ultralytics, only the 17-keypoint COCO-Pose layout gets the pose palette and the
      // human skeleton; other layouts take one class palette color per keypoint and no lines,
      // and a custom skeleton colors each line like its start keypoint.
      const coco = kp.points.length === scheme.keypoint_colors.length;
      const skeleton = options?.skeleton ?? (coco ? scheme.skeleton : []);
      const paletteAt = (i: number) => scheme.palette[i % scheme.palette.length];
      const kptColor = (i: number) => fixed ?? (coco ? scheme.keypoint_colors[i] : paletteAt(i));
      const limbColor = (i: number, a: number) =>
        coco && !options?.skeleton ? (fixed ?? scheme.limb_colors[i]) : kptColor(a);
      if (options?.keypointLines ?? true) {
        skeleton.forEach(([a, b], li) => {
          const pa = kp.points[a];
          const pb = kp.points[b];
          if (!pa || !pb || pa[2] < kptThresh || pb[2] < kptThresh) return;
          ctx.strokeStyle = limbColor(li, a);
          ctx.beginPath();
          ctx.moveTo(pa[0], pa[1]);
          ctx.lineTo(pb[0], pb[1]);
          ctx.stroke();
        });
      }
      kp.points.forEach(([x, y, c], ki) => {
        if (c < kptThresh) return;
        ctx.fillStyle = kptColor(ki);
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
    const lines = top5.map((_, i) => layout(`${top5names[i] ?? top5[i]} ${top5conf[i].toFixed(2)}`));
    const pad = Math.round(fontSize * 0.3);
    const lineH = Math.round(fontSize * 1.25);
    const boxW = Math.max(...lines.map(runsWidth)) + pad * 2;
    const boxH = lines.length * lineH + pad * 2;
    ctx.fillStyle = "rgba(0, 0, 0, 0.4)";
    ctx.fillRect(pad, pad, boxW, boxH);
    ctx.fillStyle = "#ffffff";
    lines.forEach((t, i) => fillText(t, pad * 2, pad * 2 + i * lineH));
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
