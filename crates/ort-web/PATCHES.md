# ort-web 0.4.0+1.30 (patched copy)

A copy of `ort-web` 0.4.0+1.30 from crates.io (MIT OR Apache-2.0, pyke.io) with two fixes and one lint allowance, used through
`[patch.crates-io]` in the root `Cargo.toml`. Delete this folder and that entry once a release fixes both.

- `binding/tensor.rs`: the `dims` getter returned `Vec<i64>`, but ONNX Runtime Web hands back plain numbers, so every
  session failed with `Cannot convert 1 to a BigInt`. It reads `Vec<i32>` again, as in 0.3.1, and widens to `i64`.
- `api.rs`: `ort` 2.0.0-rc.14 registers the WebGPU provider as `WebGpuExecutionProvider`, which was rejected with
  `Provider 'WebGpuExecutionProvider' not supported`, so WebGPU always fell back to the CPU. Both names are accepted now.
- `lib.rs`: `#![allow(dead_code)]`, because a path dependency is not lint-capped the way a registry crate is and three unused bindings would warn.
