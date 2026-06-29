// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
// Minimal static dev server for the browser example. Its only job beyond serving
// files is to send the two cross-origin isolation headers:
//
//   Cross-Origin-Opener-Policy: same-origin
//   Cross-Origin-Embedder-Policy: require-corp
//
// Those make `crossOriginIsolated` (and therefore SharedArrayBuffer) available,
// which is what lets ONNX Runtime's `*-simd-threaded` wasm spawn worker threads.
// Any production host serving this example must send the same two headers.
//
// Usage: `npm run serve` then open http://localhost:8080/example/

import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, join, normalize, sep } from "node:path";
import { fileURLToPath } from "node:url";

// the web/ directory, without a trailing separator
const ROOT = fileURLToPath(new URL(".", import.meta.url)).replace(/[\\/]+$/, "");
const PORT = Number(process.env.PORT) || 8080;

// Content types for the assets the example actually serves.
const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".wasm": "application/wasm",
  ".onnx": "application/octet-stream",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".map": "application/json; charset=utf-8",
};

const server = createServer(async (req, res) => {
  // Cross-origin isolation headers on every response (needed for threads).
  res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  res.setHeader("Cache-Control", "no-cache");

  let pathname = decodeURIComponent(new URL(req.url, `http://${req.headers.host}`).pathname);

  // The package has no top-level index; send root straight to the example.
  if (pathname === "/") {
    res.writeHead(302, { Location: "/example/" }).end();
    return;
  }

  // Resolve within ROOT and reject any traversal outside it.
  const reqPath = normalize(join(ROOT, pathname));
  if (reqPath !== ROOT && !reqPath.startsWith(ROOT + sep)) {
    res.writeHead(403).end("Forbidden");
    return;
  }

  // For a directory request, append index.html (and add the trailing slash via a
  // redirect first so the page's relative imports resolve against the dir).
  let filePath = reqPath;
  if (pathname.endsWith("/")) {
    filePath = join(reqPath, "index.html");
  } else {
    try {
      if ((await stat(reqPath)).isDirectory()) {
        res.writeHead(302, { Location: pathname + "/" }).end();
        return;
      }
    } catch {
      // Not a directory (or missing); fall through to the file read below.
    }
  }

  try {
    const body = await readFile(filePath);
    res.writeHead(200, { "Content-Type": MIME[extname(filePath)] || "application/octet-stream" });
    res.end(body);
  } catch {
    res.writeHead(404).end("Not found");
  }
});

server.listen(PORT, () => {
  console.log(`Serving ${ROOT} at http://localhost:${PORT}/example/ (COOP/COEP enabled)`);
});
