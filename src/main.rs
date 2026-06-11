// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#![allow(clippy::multiple_crate_versions)]
#![deny(dead_code)]

//! Ultralytics YOLO Inference CLI
//!
//! A command-line interface for running YOLO model inference on images and videos.
//!
//! # Usage
//!
//! ```bash
//! ultralytics-inference predict --model yolo26n.onnx --source image.jpg
//! ultralytics-inference predict --model yolo26n.onnx --source video.mp4
//! ultralytics-inference predict --model yolo26n.onnx --source 0 --conf 0.5
//! ultralytics-inference predict -m yolo26n.onnx -s assets/ --save --half
//! ultralytics-inference predict -m yolo26n.onnx -s video.mp4 --imgsz 1280 --show
//! ultralytics-inference version
//! ultralytics-inference help
//! ```

/// Entry point. The binary needs a `main` on every target, so this stays
/// unconditional and simply does nothing on `wasm32` (the CLI is native-only:
/// it depends on `clap`, the native ONNX Runtime, and the filesystem/source
/// modules; browser inference is the `ultralytics-inference-web` crate instead).
fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    run();
}

/// The actual CLI. All native-only imports live here, so a single `cfg` gates
/// the whole thing instead of one per `use`.
#[cfg(not(target_arch = "wasm32"))]
fn run() {
    use clap::Parser;
    use ultralytics_inference::cli::args::{Cli, Commands};
    use ultralytics_inference::cli::predict::run_prediction;
    use ultralytics_inference::logging::set_verbose;

    ultralytics_inference::io::init_logging();

    #[cfg(debug_assertions)]
    let _ = ort::init().commit();

    match &Cli::parse().command {
        Commands::Predict(args) => {
            set_verbose(args.verbose);
            run_prediction(args);
        }
        Commands::Version => {
            println!(
                "{} {}",
                ultralytics_inference::NAME,
                ultralytics_inference::VERSION
            );
        }
    }
}
