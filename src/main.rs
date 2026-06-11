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

// The CLI is native-only: it depends on `clap`, the native ONNX Runtime, and the
// filesystem/source modules, none of which exist on `wasm32`. Browser inference
// is provided by the `ultralytics-inference-web` crate instead.
#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
use clap::Parser;

#[cfg(not(target_arch = "wasm32"))]
use ultralytics_inference::cli::args::{Cli, Commands};
#[cfg(not(target_arch = "wasm32"))]
use ultralytics_inference::cli::predict::run_prediction;
#[cfg(not(target_arch = "wasm32"))]
use ultralytics_inference::logging::set_verbose;

/// Entry point for the Ultralytics YOLO Inference CLI.
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    ultralytics_inference::io::init_logging();

    #[cfg(debug_assertions)]
    let _ = ort::init().commit();

    let cli = Cli::parse();

    match &cli.command {
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
