// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#![allow(clippy::multiple_crate_versions)]

//! Ultralytics YOLO Inference CLI
//!
//! A command-line interface for running YOLO model inference on images and videos.
//!
//! # Usage
//!
//! ```bash
//! ultralytics-inference predict --model yolo11n.onnx --source image.jpg
//! ultralytics-inference predict --model yolo11n.onnx --source video.mp4
//! ultralytics-inference predict --model yolo11n.onnx --source 0 --conf 0.5
//! ultralytics-inference predict -m yolo11n.onnx -s assets/ --save --half
//! ultralytics-inference predict -m yolo11n.onnx -s video.mp4 --imgsz 1280 --show
//! ultralytics-inference version
//! ultralytics-inference help
//! ```

use clap::Parser;

use ultralytics_inference::cli::args::{Cli, Commands};
use ultralytics_inference::cli::logging::set_verbose;
use ultralytics_inference::cli::predict::run_prediction;

/// Entry point for the Ultralytics YOLO Inference CLI.
fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Predict(args) => {
            set_verbose(args.verbose);
            run_prediction(args);
        }
    }
}
