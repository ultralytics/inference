// © 2014-2025 Ultralytics Inc. 🚀 All rights reserved. CONFIDENTIAL: Unauthorized use or distribution prohibited.

//! Ultralytics YOLO Inference CLI
//!
//! A command-line interface for running YOLO model inference on images and videos.
//!
//! # Usage
//!
//! ```bash
//! ultralytics-inference predict --model <model_path> --source <image_path>
//! ultralytics-inference version
//! ultralytics-inference help
//! ```
//!
//! # Examples
//!
//! ```bash
//! inference predict --model yolov8n.onnx --source image.jpg
//! inference predict --model yolov8n.onnx --source video.mp4
//! ```
//!
//! # Note
//!
//! This is a stub implementation. Full YOLO inference will be implemented
//! once dependencies (ONNX Runtime, image processing, etc.) are added.

use std::env;
use std::process;

/// Entry point for the Ultralytics YOLO Inference CLI.
///
/// Parses command-line arguments and dispatches to the appropriate handler.
///
/// # Supported Commands
///
/// - `predict` - Run inference on an image or video
/// - `version` - Print version information
/// - `help` - Show usage information
///
/// # Exit Codes
///
/// - `0` - Success
/// - `1` - Error (invalid arguments, unknown command, etc.)
fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    let command = &args[1];

    match command.as_str() {
        "predict" => run_prediction(&args[2..]),
        "version" => print_version(),
        "help" | "--help" | "-h" => print_usage(),
        _ => {
            eprintln!("Unknown command: {}", command);
            print_usage();
            process::exit(1);
        }
    }
}

/// Runs YOLO model inference on the specified source.
///
/// # Arguments
///
/// * `args` - A slice of command-line arguments following the `predict` command.
///            Expected format: `--model <path> --source <path>`
///
/// # Required Flags
///
/// - `--model` - Path to the ONNX model file (e.g., `yolov8n.onnx`)
/// - `--source` - Path to the input image or video file
///
/// # Panics
///
/// This function will exit the process with code `1` if:
/// - Required arguments are missing
/// - Unknown arguments are provided
/// - Flag values are not provided
fn run_prediction(args: &[String]) {
    if args.len() < 4 {
        eprintln!("Error: predict command requires --model and --source arguments");
        println!("\nUsage: inference predict --model <model_path> --source <image_path>");
        process::exit(1);
    }

    // Parse arguments
    let mut model_path = None;
    let mut source_path = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                if i + 1 < args.len() {
                    model_path = Some(&args[i + 1]);
                    i += 2;
                } else {
                    eprintln!("Error: --model requires a value");
                    process::exit(1);
                }
            }
            "--source" => {
                if i + 1 < args.len() {
                    source_path = Some(&args[i + 1]);
                    i += 2;
                } else {
                    eprintln!("Error: --source requires a value");
                    process::exit(1);
                }
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                process::exit(1);
            }
        }
    }

    if let (Some(model), Some(source)) = (model_path, source_path) {
        println!("🚀 Ultralytics YOLO Inference");
        println!("Model: {}", model);
        println!("Source: {}", source);
        println!("\n⚠️  Note: Full inference implementation coming soon!");
        println!("This will load the YOLO model and run inference on the source.");
    } else {
        eprintln!("Error: Both --model and --source are required");
        process::exit(1);
    }
}

/// Prints the current version of the Ultralytics Inference CLI.
///
/// The version is read from `CARGO_PKG_VERSION` at compile time.
fn print_version() {
    println!("Ultralytics Inference v{}", env!("CARGO_PKG_VERSION"));
}

/// Prints usage information and available commands.
///
/// Displays a help message with all supported commands, their descriptions,
/// and example usage patterns.
fn print_usage() {
    println!(r#"
    Ultralytics YOLO Inference CLI
    =============================
    Usage:
        inference predict --model <model_path> --source <image_path>
        inference version
        inference help

    Commands:
        predict    Run inference on an image or video
        version    Print version information
        help       Show this help message

    Examples:
        inference predict --model yolov8n.onnx --source image.jpg
        inference predict --model yolov8n.onnx --source video.mp4

    "#);
}
