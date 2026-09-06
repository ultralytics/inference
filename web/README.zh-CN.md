<!-- Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license -->

<div align="center">
  <p>
    <a href="https://www.ultralytics.com/events/yolovision?utm_source=github&utm_medium=social&utm_campaign=yolovision26&utm_content=banner" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="Ultralytics YOLO 横幅"></a>
  </p>

[中文](https://docs.ultralytics.com/zh) | [한국어](https://docs.ultralytics.com/ko) | [日本語](https://docs.ultralytics.com/ja) | [Русский](https://docs.ultralytics.com/ru) | [Deutsch](https://docs.ultralytics.com/de) | [Français](https://docs.ultralytics.com/fr) | [Español](https://docs.ultralytics.com/es) | [Português](https://docs.ultralytics.com/pt) | [Türkçe](https://docs.ultralytics.com/tr) | [Tiếng Việt](https://docs.ultralytics.com/vi) | [العربية](https://docs.ultralytics.com/ar) <br>

</div>

# Ultralytics YOLO npm Inference

<div align="center">

[English](README.md) | [简体中文](README.zh-CN.md)

</div>

<div align="center">

[![npm version](https://img.shields.io/npm/v/@ultralytics/yolo?logo=npm&logoColor=white&label=npm&color=CB3837)](https://www.npmjs.com/package/@ultralytics/yolo)
[![npm downloads](https://img.shields.io/npm/dm/@ultralytics/yolo?logo=npm&logoColor=white&label=downloads&color=CB3837)](https://www.npmjs.com/package/@ultralytics/yolo)
[![CI](https://github.com/ultralytics/inference/actions/workflows/ci.yml/badge.svg)](https://github.com/ultralytics/inference/actions/workflows/ci.yml)
[![License](https://img.shields.io/npm/l/@ultralytics/yolo?label=license&color=blue)](https://github.com/ultralytics/inference/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2606.03748-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.03748)

[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://www.reddit.com/r/Ultralytics/)

</div>

直接在浏览器中运行 [Ultralytics](https://www.ultralytics.com) YOLO 模型，无需服务器，也无需
Python。本库基于 **WebGPU**（并自动回退到 CPU/wasm），支持检测、分割、姿态、分类、OBB、语义分割和
深度估计，接口是一个小巧的 TypeScript API，内置的 `annotate()` 可直接把结果绘制到 canvas 上。

```ts
import { YOLO, annotate } from "@ultralytics/yolo";

const model = await YOLO.load("/models/yolo26n.onnx");
const results = await model.predict("bus.jpg");
await annotate(document.querySelector("canvas"), "bus.jpg", results);
```

本包**仅为库**（不含 CLI，CLI 属于原生 Rust crate）。底层引擎是编译为 WebAssembly 的
`ultralytics-inference` Rust crate。推理通过
[`ort-web`](https://ort.pyke.io/backends/web) 运行在
[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) 上，所有前处理/后处理、
配色和姿态骨架都来自同一份共享 Rust 代码，因此结果和视觉效果与原生及 Python 路径保持一致。

## 📦 安装

```bash
npm install @ultralytics/yolo
# 或
pnpm add @ultralytics/yolo
yarn add @ultralytics/yolo
bun add @ultralytics/yolo
```

本包以 ES module 形式发布并自带 TypeScript 类型，可用于任意现代打包工具（Vite、webpack、
esbuild、Bun），也可直接通过 [esm.sh](https://esm.sh/@ultralytics/yolo) 等 CDN 使用。

## 🚀 快速开始

```ts
import { YOLO, annotate } from "@ultralytics/yolo";

// 首次使用时加载模型并初始化 WebGPU + ONNX Runtime Web。
const model = await YOLO.load("/models/yolo26n.onnx");

const results = await model.predict("bus.jpg");
for (const box of results.boxes) {
  console.log(box.name, box.conf.toFixed(2), [box.x1, box.y1, box.x2, box.y2]);
}

// 一次调用即可把框、OBB、姿态和标签绘制到 canvas（无需自己写 canvas 代码）。
await annotate(document.querySelector("canvas"), "bus.jpg", results);
```

`predict()` 接受 URL/路径、`Blob`/`File`、原始编码图片字节（`Uint8Array`/`ArrayBuffer`）、
`ImageData`、`HTMLImageElement`、`HTMLCanvasElement`、`HTMLVideoElement` 或 `ImageBitmap`。

```ts
const results = await model.predict(canvas, { conf: 0.25, iou: 0.7 });
console.log(model.device); // "webgpu" 或 "cpu"
```

`YOLO.load` 同样接受 `Blob`/`File`，因此可以加载用户拖入或选择的模型。后端根据字节内容自动检测，
所以同一个调用可同时处理 `.onnx` 和 `.tflite`：

```ts
const model = await YOLO.load(fileInput.files[0]); // 拖入/选择的 .onnx 或 .tflite
```

### 摄像头 / 视频

可绘制的输入源（`<video>`、canvas、`ImageBitmap`、`ImageData`）走原始像素快速路径，无需重新编码，
因此渲染循环很流畅：

```ts
const model = await YOLO.load("/models/yolo26n.onnx");
async function frame() {
  const results = await model.predict(video); // <video> 元素
  await annotate(canvas, video, results);
  requestAnimationFrame(frame);
}
```

## ✨ 模型

<a href="https://docs.ultralytics.com/tasks" target="_blank">
    <img width="100%" src="https://cdn.ul.run/i/c99d914c3958d0755b5a3d7204b6f24a.avif" alt="Ultralytics YOLO 支持的任务">
</a>
<br>
<br>

可运行 [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8)、
[Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11) 和
[Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26) 的 ONNX 导出模型，覆盖
[检测](https://docs.ultralytics.com/tasks/detect)、
[分割](https://docs.ultralytics.com/tasks/segment)、
[姿态](https://docs.ultralytics.com/tasks/pose)、
[OBB](https://docs.ultralytics.com/tasks/obb)、
[分类](https://docs.ultralytics.com/tasks/classify)、
[语义分割](https://docs.ultralytics.com/tasks/semantic)和
[深度估计](https://docs.ultralytics.com/tasks/depth)。

`YOLO.load` 接受 URL 或路径，浏览器会像加载其他静态资源一样获取它。请从 [Ultralytics assets release](https://github.com/ultralytics/assets/releases) 下载所需权重（与原生 crate 和 Python 使用的是同一份文件），并部署在**同源**位置，或放在启用了 CORS 的源之后：

```ts
await YOLO.load("/models/yolo26n.onnx");
```

GitHub release 资源不会返回 `Access-Control-Allow-Origin`，因此浏览器无法直接从 release URL 获取它们。该 URL 适用于不受 CORS 限制的原生 crate 和 Python，但在这里不适用。

## 📐 结果结构

`predict()` 返回的 `Results` 对象，其字段名与 Rust/Ultralytics 的 `Results` API 一一对应：

| 字段               | 类型                                                                 | 任务                  |
| ------------------ | -------------------------------------------------------------------- | --------------------- |
| `task`             | `string`                                                             | 全部                  |
| `width` / `height` | `number`                                                             | 全部                  |
| `boxes`            | `{ x1, y1, x2, y2, conf, cls, name, color }[]`                       | detect、segment、pose |
| `obb`              | `{ x, y, w, h, angle, conf, cls, name, color }[]`                    | obb                   |
| `keypoints`        | `{ points: [x, y, conf][], color }[]`                                | pose                  |
| `probs`            | `{ top1, top5, top1conf, top5conf, name, top5names, color } \| null` | classify              |
| `masks`            | `Uint8Array`（RGBA 叠加层，`width*height*4`）                        | segment、semantic     |
| `semantic_mask`    | `Uint16Array`（每像素的类别 id，`width*height`）                     | semantic              |
| `depth`            | `Uint8Array`（不透明的彩色深度图，`width*height*4`）                 | depth                 |
| `depth_range`      | `[min, max]`，单位为米                                               | depth                 |
| `speed`            | `{ preprocess, inference, postprocess }`，单位 ms                    | 全部                  |

`model.names` 是类别 id 到名称的映射（相当于 Python 中的 `model.names`）。每个检测结果都带有
Ultralytics 调色板中的 `color`，`annotate()` 绘制 `masks` 叠加层和姿态骨架时，使用的每条肢体/
关键点配色与原生渲染器完全一致。这些逻辑都没有在 JS 中重复实现。

对于 `depth` 任务，`predict(img, { colormap, depthViz })` 用于选择配色方案（默认 `"jet"`，
另有 `"inferno"`、`"spectral"`、`"gray"`）和归一化方式（默认 `"disparity"`，另有 `"metric"`）；
`annotate()` 会以 `depthAlpha`（默认 `0.6`，设为 `1` 则显示原始深度图）把返回的深度图叠加到画面上：

```ts
const results = await model.predict(img, { colormap: "spectral", depthViz: "metric" });
await annotate(canvas, img, results, { depthAlpha: 0.6 });
```

## ⚙️ 环境要求与注意事项

- **WebGPU**（Chrome/Edge，或启用了 WebGPU 的 Firefox）配合**安全上下文**（`https://` 或
  `http://localhost`）可获得快速路径。在没有 WebGPU 的环境（较旧的浏览器、部分手机）中，
  `YOLO.load` 会自动回退到通用的 **CPU/wasm** 构建，随处可用。可通过
  `YOLO.load("/models/yolo26n.onnx", { device: "webgpu" | "cpu" })` 指定设备（默认 `"auto"`）。若 WebGPU
  无法启用，加载会回退到 CPU；`model.device` 会报告实际使用的设备。
- **模型格式**：请使用 Ultralytics `>=8.4.142` 导出为 ONNX，以便元数据（任务、类别名称、`imgsz`）被嵌入模型：

  ```python
  from ultralytics import YOLO

  YOLO("yolo26n.pt").export(format="onnx")  # FP32 (默认)
  YOLO("yolo26n.pt").export(format="onnx", quantize=16)  # FP16 (体积约小 50%)
  ```

  对于 detect、segment、pose 和 OBB，`nms=None`（默认值）导出原始输出，由 Rust 执行 NMS；
  `nms=False` 在模型支持时选择无 NMS 检测头（例如 YOLO26）；`nms=True` 将 NMS 嵌入 ONNX 图中。
  semantic、depth 和 classify 使用 `nms=None`。现有的兼容 ONNX 文件无需重新导出。

  > Ultralytics ≥8.4 使用 `quantize` 参数，取代已弃用的 `half=True` / `int8=True` 标志。
  > 对于 ONNX，支持的取值为 `32`/`fp32`（默认）、`16`/`fp16` 和 `8`/`int8`；旧标志
  > 仍可使用，但会触发弃用警告。

- **运行时资源**：首次加载时，`ort-web` 会从 `cdn.pyke.io` 获取 ONNX Runtime Web 的 wasm 包
  （约 25 MB，之后由浏览器缓存）。如果你设置了 Content-Security-Policy，请在
  `script-src`/`connect-src` 中放行该源。若想完全避开 CDN，可自行托管运行时并指向它：
  ```ts
  const model = await YOLO.load("/models/yolo26n.onnx", { ortBaseUrl: "/ort/" });
  ```
  该目录需包含 ONNX Runtime Web 的入口脚本（`ort.webgpu.min.js`，以及 CPU 回退所需的
  `ort.wasm.min.js`）和 `ort-wasm-simd-threaded.{jsep,asyncify,}.{mjs,wasm}` 二进制文件。
- **遥测**：`ort-web` 会在首次创建会话时向 pyke 上报页面域名。查看或关闭的方法见
  [ort-web 文档](https://ort.pyke.io/backends/web)。

## ⚡ LiteRT.js 后端

这是一个可选的推理引擎，通过 [**LiteRT.js**](https://developers.google.com/edge/litert/web)
（Google 面向 Web 的 LiteRT）运行 Ultralytics 导出的 **`.tflite`** 模型，在 WebGPU 上
**通常比 ONNX Runtime Web 快约 2 倍**。只有推理引擎发生变化，前处理、后处理、绘制和 `Results`
结构仍是同一份共享 Rust 代码，因此输出与 `ort` 路径一致。

后端根据文件扩展名选择：`.tflite` 使用 LiteRT.js，`.onnx` 使用 ONNX Runtime Web。LiteRT.js 的
wasm 默认从 CDN 加载，因此唯一需要做的就是让 `@litertjs/core` 能被解析（连同它的
`@litertjs/wasm-utils` 依赖，npm 会自动安装，下面的 import map 中也显式列出）。

**使用 npm（配合打包工具）：**

```bash
npm install @ultralytics/yolo @litertjs/core
```

```ts
import { YOLO, annotate } from "@ultralytics/yolo";

const model = await YOLO.load("/models/yolo26n.tflite"); // .tflite -> LiteRT.js
const results = await model.predict("bus.jpg");
await annotate(document.querySelector("canvas"), "bus.jpg", results);
```

**无需构建步骤（CDN）：** 把模块映射到 CDN，然后使用与上面完全相同的代码：

```html
<script type="importmap">
  {
    "imports": {
      "@ultralytics/yolo": "https://esm.sh/@ultralytics/yolo",
      "@litertjs/core": "https://esm.sh/@litertjs/core",
      "@litertjs/wasm-utils": "https://esm.sh/@litertjs/wasm-utils"
    }
  }
</script>
```

对于摄像头或视频，每帧传入 `<video>` 元素即可：

```ts
const results = await model.predict(video);
await annotate(canvas, video, results);
```

wasm 默认从 jsDelivr CDN 加载；向 `YOLO.load` 传入 `litertWasmUrl: "/litert/"` 可自行托管
（复制 `node_modules/@litertjs/core/wasm/` 即可）。

注意事项：

- **模型**：使用 Ultralytics 导出为 `.tflite`（WebGPU 需要 float32）。模型从单个文件加载，
  元数据（任务、类别名称、`imgsz`、stride）直接从 `.tflite` 中读取，与 `.onnx` 路径相同，
  无需额外的附属文件。
- **现有模型兼容性**：带内嵌元数据的单文件 LiteRT 导出自
  [v8.4.83](https://github.com/ultralytics/ultralytics/releases/tag/v8.4.83) 起提供。更早的版本
  会导出旧版 TFLite 格式，无法在这里加载。
- **为 WebGPU 保留一对多检测头**（`nms=None`）：以 `nms=False` 导出的 YOLO26 无 NMS 检测头包含
  `int64` / `gather_nd` 算子，无法在 LiteRT 的 **WebGPU** delegate 上运行，因此这类导出会
  回退到 CPU/wasm。以下命令需要 Ultralytics `>=8.4.142`，默认导出一对多检测头，NMS 由本包的 Rust 代码执行，
  推理得以保持在 WebGPU 上：

  ```bash
  yolo export model=yolo26n.pt format=litert nms=None
  ```

  如果加载了无 NMS 的 `.tflite`，后端会自动切换到 wasm（较慢）并打印警告，而不是返回空结果。

- **支持的任务**：detect、segment、pose、obb、classify、semantic 和 depth 均已支持。
- **跨源隔离**：LiteRT 的多线程 wasm 需要 `SharedArrayBuffer`，因此请以
  `Cross-Origin-Opener-Policy: same-origin` 和 `Cross-Origin-Embedder-Policy: require-corp`
  提供服务。

## 🔨 从源码构建

本包使用 [`wasm-pack`](https://github.com/wasm-bindgen/wasm-pack) 从 Rust crate 构建 wasm：

```bash
npm run build # wasm-pack build + tsc
```

构建完成后，在 `localhost`（安全上下文）上以上述两个跨源隔离响应头提供服务，然后用支持 WebGPU 的
浏览器打开。

## 💡 贡献

Ultralytics 依靠社区协作持续发展，我们重视每一份贡献。无论是报告 bug、提出功能建议，还是提交代码改动，都欢迎参与。

- **报告问题**：[打开 issue](https://github.com/ultralytics/inference/issues)。
- **功能请求**：[提交想法](https://github.com/ultralytics/inference/issues)。
- **Pull Request**：请先阅读[贡献指南](https://docs.ultralytics.com/help/contributing)。
- **反馈**：填写 [Ultralytics 调查问卷](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey)。

感谢所有贡献者！你们的努力让 Ultralytics 工具持续变得更好。

[![Ultralytics 开源贡献者](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/ultralytics/graphs/contributors)

## 📄 许可证

Ultralytics 提供两种许可方式：

- **AGPL-3.0 许可证**：经 [OSI 批准](https://opensource.org/license/agpl-3.0)的开源许可证，适合学生、研究者和爱好者，鼓励开放协作和知识共享。完整详情请参阅 [LICENSE](https://github.com/ultralytics/inference/blob/main/LICENSE) 文件。
- **Ultralytics 企业许可证**：面向商业使用，允许将 Ultralytics 软件和 AI 模型集成到商业产品与服务中，而无需遵循 AGPL-3.0 的开源要求。如需商业部署，请通过 [Ultralytics Licensing](https://www.ultralytics.com/license) 联系我们。

## 📮 联系方式

- **GitHub Issues**：[bug 报告和功能请求](https://github.com/ultralytics/inference/issues)。
- **Discord**：[加入社区](https://discord.com/invite/ultralytics)。
- **文档**：[docs.ultralytics.com](https://docs.ultralytics.com)。

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
