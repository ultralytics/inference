# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in this repository. CLAUDE.md is a symlink to this file.

## Core Principles (CRITICAL)

Respecting these principles is critical for every PR.

**Less is more. The simplest solution is the best solution.**

The action hierarchy for every change: **Delete > Replace > Add**. The best code change is a deletion. The second best is modifying what exists. Adding new code is the last resort.

1. **Minimal**: The simplest solution that works. Do not over-engineer, over-abstract, or add code just in case. Three similar lines beat a premature abstraction. Avoid error handling for impossible states, feature flags, compatibility shims, or policy scaffolding unless they are truly required.
2. **Solve at the source**: Do not hack fixes. Solve problems at their root. If something is broken, fix or remove the broken thing. Never patch over a broken abstraction, add workarounds, or add synchronization code for state that should not be duplicated.
3. **Delete ruthlessly**: When replacing code, delete what it replaced. Remove unused imports, functions, types, files, and commented-out code. Git preserves history. Run the repo's relevant dead-code or cleanup check when available.
4. **Replace > Add**: Modify existing code over adding new code. Edit existing files, extend existing components or functions with minimal parameters, and reuse existing utilities. If creating a new file, first prove it cannot fit cleanly in an existing file.
5. **Check existing**: Search the entire repo before creating anything new. If a feature, component, helper, responder, workflow, or utility already solves a similar problem, reuse or adapt it and delete the duplicate path.
6. **Deduplicate**: Do not duplicate existing code when updating the repo. Consolidate or refactor duplicates you find when it is in scope and low risk.
7. **Zero Regression**: Do not break existing features or workflows unless the PR intentionally removes them with evidence.
8. **Production ready**: All changes must be thoroughly debugged, validated, and production ready.

**When fixing bugs, ask: "What can I delete?" before "What can I replace?" before "What should I add?"**

## PR Workflow

After opening a PR:

1. Wait for the automated PR review and auto-format commit from Ultralytics Actions (`format.yml`), then pull and address every finding.
2. Launch an independent adversarial review agent with cold context (just the PR diff and this file) to hunt for bugs, regressions, and Core Principles violations — use the Codex CLI, one fresh `codex exec` run per round. Fix, push, and repeat until a fresh run reports LGTM.
3. Never fight other commits: Ultralytics Actions pushes auto-format and header commits, and multiple users may work on the same PR. `git pull --rebase` before pushing; never force-push, reset, or revert commits you did not author.
4. After the PR merges, clean up: remove local worktrees and branches for it, then `git checkout main && git pull`.

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

# Python tooling (only used in CI, e.g. ultralytics-actions): always uv, never bare pip
uv pip install --system ultralytics-actions
```

- CI matrix (`ci.yml`): `test` on ubuntu/macos/windows; `test-video` in FFmpeg 7.1/8.0 Linux containers (`--features annotate,video`); video builds on macOS/Windows; `wasm`; `coverage` (nightly) uploads to Codecov.
- MSRV is Rust 1.89 (`rust-version` in Cargo.toml), edition 2024.
- First native build downloads ONNX Runtime binaries (ort `download-binaries` feature), so builds need network once.

## Architecture

Rust workspace with two crates plus an npm wrapper, all versioned together from the root `Cargo.toml`:

- Root crate `ultralytics-inference`: YOLO inference library (`src/lib.rs`) and CLI binary (`src/main.rs`, thin wrapper over `src/cli/`). Pipeline: `source.rs` (images/dirs/globs/video/webcam) → `preprocessing.rs` (SIMD letterbox) → `model.rs` (`YOLOModel`, the ONNX Runtime session via `ort`, configured by `inference.rs`'s `InferenceConfig`) → `postprocessing.rs` → `results.rs` (`Results`/`Boxes`/`Masks`/`Keypoints`/`Probs`/`SemanticMask`, mirroring the Ultralytics Python API). `model.rs` reads embedded ONNX metadata (`metadata.rs`) and auto-downloads known YOLOv8/YOLO11/YOLO26 models and sample images (`download.rs`).
- `crates/web` (`ultralytics-inference-web`, `publish = false`): wasm32-only WebGPU bindings via `ort-web`. Excluded from `default-members`, so plain `cargo build`/`cargo test` from the root skip it; it only builds for `--target wasm32-unknown-unknown`.
- `web/`: npm package `@ultralytics/yolo` — TypeScript wrapper (`web/src/index.ts`) over the wasm-pack output of `crates/web`, with an optional LiteRT.js backend for `.tflite` models.
- GPU/accelerator features (`cuda`, `tensorrt`, `coreml`, …) gate no public API; docs.rs builds with `annotate,visualize,video` only (see `[package.metadata.docs.rs]`).
- Release gating: on every push to main, `publish.yml` reads the version from `Cargo.toml` — if tag `v{version}` does not exist it tags, creates a GitHub release, and publishes to crates.io; `npm-publish.yml` likewise publishes `@ultralytics/yolo` if that version is missing from npm. So merging a version bump to main releases both packages.

## Conventions

- Every source file starts with the `Ultralytics 🚀 AGPL-3.0 License` header — Ultralytics Actions adds them automatically; don't add or revert manually.
- Ultralytics Actions (`format.yml`) also runs Prettier (YAML/JSON/Markdown), codespell, and a nightly `cargo fmt` check on PRs; expect it to push an auto-format commit to your PR branch.
- Lints are strict: clippy `all`/`pedantic`/`nursery`/`cargo` plus `missing_docs` and `unsafe_code` warn at the workspace level (CI promotes to errors with `-D warnings`), and `src/lib.rs` denies `dead_code` — document all public items and delete unused code.
- Unit tests live inline in `src/` modules; integration tests in `tests/integration_test.rs`. The e2e tests that download models/images (e.g. `test_run_prediction_e2e`) are `#[ignore]`d — run them explicitly with `-- --ignored`; macOS CI runs `test_coreml_model_loads_and_warms_up` this way. Note the non-ignored `src/batch.rs` tests also auto-download `yolo26n.onnx` on first run, so the plain test suite needs network until that file is cached.
- Version bumps update root `Cargo.toml`, `crates/web/Cargo.toml`, and `web/package.json` together; merging the bump to main auto-tags and publishes (see Architecture).
