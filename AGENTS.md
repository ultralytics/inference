# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in this repository. CLAUDE.md is a symlink to this file.

`ultralytics-inference` (crates.io, AGPL-3.0) is the official Rust package for YOLO-family vision model inference — detection, instance and semantic segmentation, classification, pose, oriented boxes, and depth estimation — over ONNX Runtime, with image/video/webcam sources, annotation and visualization, the `ultralytics-inference` CLI, and a WebGPU/wasm build published as `@ultralytics/yolo` on npm. The supported floor is Rust 1.89 (edition 2024).

## Core Principles (CRITICAL)

**Less is more. The simplest solution is the best solution.** The action hierarchy for every change: **Delete > Replace > Add**.

1. **Solve at the owner**: Put behavior in the code path that owns or observes it. For fixes, never guard a symptom with a staleness check, initialization flag, skip-first-call branch, or `try/except` around broken logic; relocate the trigger and delete the wrong path. For features, extend the existing owner rather than creating a parallel abstraction.
2. **Search and reuse first**: Search the whole repository before creating a feature, component, helper, workflow, or utility. Reuse or adapt what exists, consolidate in-scope duplication in the shared owner, and delete duplicate paths. Three similar lines beat a helper nobody else calls.
3. **Delete and modify existing code before creating new code**: Bugfixes are net-negative by default unless deletion and relocation are demonstrably impossible. A new file must first prove it cannot fit cleanly in an existing owner.
4. **Keep scope minimal**: Implement only the simplest complete solution. Avoid impossible-state handling, speculative flags, compatibility shims, policy scaffolding, and unrelated cleanup. Tests are out of scope by default — rely on existing coverage and focused validation; only an uncovered, high-risk regression path justifies minimal new test code.
5. **Ship zero-regression, production-ready changes**: Understand what you remove instead of retaining broken code as insurance. Remove unused imports, functions, types, files, and comments; run relevant cleanup checks; and thoroughly debug and validate the changed owner. Do not break existing features or workflows unless the PR intentionally removes them with evidence.

**Review gate:** for every addition, the reviewer decides whether deleting or changing existing code would have fixed the problem instead — if it would, that is a blocking finding. A missing or thin PR description is never itself a finding.

NEVER push to `main`. NEVER force push. Always start work in a new git worktree (`git worktree add`) on a feature branch and open a PR — never edit the primary checkout directly, it may hold in-flight work.

## PR Workflow

After opening a PR:

1. Wait for the automated PR review and auto-format commit from Ultralytics Actions (`format.yml`), then pull and address every finding.
2. Review the full diff in-session against the Core Principles, performance, and the review gate above, then batch the fixes into one commit and push. After each round of bot or human commits, pull and resume the same reviewer on `<last-reviewed-sha>..HEAD` plus anything that delta could have invalidated. Repeat until the local head matches the live head.
3. Hand off or merge only on a clean final pass: one cold full-diff review returning LGTM with no findings, on a head that is still live at merge time.
4. Never fight other commits: Ultralytics Actions pushes auto-format and header commits, and multiple users may work on the same PR. `git pull --rebase` before pushing; never reset or revert commits you did not author.
5. After the PR merges, clean up: remove local worktrees and branches for it, then `git checkout main && git pull`.

## Commands

```bash
# Build (native; default features = annotate + visualize)
cargo build

# Run all tests as CI does on Linux/Windows (macOS CI uses "coreml,annotate")
cargo test --no-default-features --features annotate

# Run one test by name filter (add -- --ignored --exact for the network e2e tests)
cargo test --no-default-features --features annotate test_boxes_creation

# Lint exactly as CI (ci.yml `test` job; macOS swaps in "coreml,annotate")
cargo clippy --all-targets --no-default-features --features annotate -- -D warnings

# Format (checked with --check in ci.yml and format.yml)
cargo fmt --all

# Coverage exactly as CI (ci.yml `coverage` job: nightly toolchain, cargo-llvm-cov, FFmpeg dev libs)
cargo llvm-cov --features annotate,video,visualize --workspace --lcov --output-path lcov.info --ignore-filename-regex '(src/cuda_inference\.rs|src/visualizer/viewer\.rs|src/main\.rs|crates/web/)'

# Wasm checks (ci.yml `wasm` job)
cargo build -p ultralytics-inference --lib --no-default-features --target wasm32-unknown-unknown
cargo clippy -p ultralytics-inference-web --target wasm32-unknown-unknown -- -D warnings

# npm package build (wasm-pack + tsc)
cd web && npm ci && npm run build

# Fastest end-to-end smoke test (auto-downloads yolo26n.onnx and sample images)
cargo run -- predict
```

- CI matrix (`ci.yml`): `test` on ubuntu/macos/windows; `test-video` in FFmpeg 7.1/8.0 Linux containers (`--features annotate,video`); video builds on macOS/Windows; `wasm`; `coverage` (nightly) uploads to Codecov.
- MSRV is Rust 1.89 (`rust-version` in Cargo.toml), edition 2024.
- First native build downloads ONNX Runtime binaries (ort `download-binaries` feature), so builds need network once.

## Architecture

Rust workspace with two crates plus an npm wrapper, all versioned together from the root `Cargo.toml`:

- Root crate `ultralytics-inference`: YOLO inference library (`src/lib.rs`) and CLI binary (`src/main.rs`, thin wrapper over `src/cli/`). Pipeline: `source.rs` (images/dirs/globs/video/webcam) → `preprocessing.rs` (SIMD letterbox) → `model.rs` (`YOLOModel`, the ONNX Runtime session via `ort`, configured by `inference.rs`'s `InferenceConfig`) → `postprocessing.rs` → `results.rs` (`Results`/`Boxes`/`Masks`/`Keypoints`/`Probs`/`Obb`/`SemanticMask`/`DepthMap`/`Speed`, mirroring the Ultralytics Python API). `model.rs` reads embedded ONNX metadata (`metadata.rs`) and auto-downloads known YOLOv8/YOLO11/YOLO26 models and sample images (`download.rs`).
- `crates/web` (`ultralytics-inference-web`, `publish = false`): wasm32-only WebGPU bindings via `ort-web`. Excluded from `default-members`, so plain `cargo build`/`cargo test` from the root skip it; it only builds for `--target wasm32-unknown-unknown`.
- `web/`: npm package `@ultralytics/yolo` — TypeScript wrapper (`web/src/index.ts`) over the wasm-pack output of `crates/web`, with an optional LiteRT.js backend for `.tflite` models.
- GPU/accelerator features (`cuda`, `tensorrt`, `coreml`, …) gate no public API; docs.rs builds with `annotate,visualize,video` only (see `[package.metadata.docs.rs]`).
- Release gating: on every push to main, `publish.yml` reads the version from `Cargo.toml` — if tag `v{version}` does not exist it tags, creates a GitHub release, and publishes to crates.io; `npm-publish.yml` likewise publishes `@ultralytics/yolo` if that version is missing from npm. So merging a version bump to main releases both packages.

## Conventions

- Every source file starts with the `// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license` header — Ultralytics Actions adds them automatically; don't add or revert manually.
- Ultralytics Actions (`format.yml`) also runs prettier (YAML/JSON/Markdown), codespell, and a nightly `cargo fmt` check on PRs; expect bot commits on your PR branch. Format markdown exactly as the bot does, never with unpinned defaults: `npx prettier@3.8.5 --print-width 120 --write`.
- Lints are strict: clippy `all`/`pedantic`/`nursery`/`cargo` plus `missing_docs` and `unsafe_code` warn at the workspace level (CI promotes to errors with `-D warnings`), and `src/lib.rs` denies `dead_code` — document all public items and delete unused code.
- Unit tests live inline in `src/` modules; integration tests in `tests/integration_test.rs`. The e2e tests that download models/images (e.g. `test_run_prediction_e2e`) are `#[ignore]`d — run them explicitly with `-- --ignored`; macOS CI runs `test_coreml_model_loads_and_warms_up` this way. The `src/batch.rs` tests skip themselves when `yolo26n.onnx` is neither present nor downloadable, so the plain suite passes offline; with network they download the model once and run in full. The `src/annotate.rs` tests likewise try to fetch `Arial.ttf` from the Ultralytics asset CDN on a cold machine, caching it under `dirs::config_dir()/Ultralytics/`; they fall back to unfonted rendering when that fails, so label layout is only exercised once the font is cached.
- Version bumps update root `Cargo.toml`, `crates/web/Cargo.toml`, and `web/package.json` together; merging the bump to main auto-tags and publishes (see Architecture).
