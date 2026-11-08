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
//! ultralytics-inference predict -m yolo26n.onnx -s assets/ --save --quantize 16
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
    use clap::{Arg, CommandFactory, FromArgMatches};
    use ultralytics_inference::cli::args::{Cli, Commands};
    use ultralytics_inference::cli::predict::run_prediction_with_tile;
    use ultralytics_inference::logging::set_verbose;

    ultralytics_inference::io::init_logging();

    // `--tile` is read here so the public `PredictArgs` struct stays as it is.
    let matches = Cli::command()
        .mut_subcommand("predict", |predict| {
            predict.arg(
                Arg::new("tile")
                    .long("tile")
                    .value_name("TILE")
                    .num_args(0..=1)
                    .default_missing_value("0")
                    .value_parser(clap::value_parser!(u32))
                    .help("Tile size for overlapping-window inference on large images; bare --tile uses --imgsz"),
            )
        })
        .get_matches();
    let cli = Cli::from_arg_matches(&matches).unwrap_or_else(|e| e.exit());
    let tile = matches
        .subcommand_matches("predict")
        .and_then(|m| m.get_one::<u32>("tile").copied());

    match &cli.command {
        Commands::Predict(args) => {
            set_verbose(args.verbose);
            run_prediction_with_tile(args, tile);
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
