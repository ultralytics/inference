// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

use crate::download::DEFAULT_MODEL;
use clap::{Args, Parser, Subcommand};

/// CLI arguments parser.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
#[command(after_help = r#"Predict Options:
    --model, -m <MODEL>    Path to ONNX model file [default: yolo11n.onnx]
    --source, -s <SOURCE>  Input source (image, directory, glob, video, webcam, or URL)
    --conf <CONF>          Confidence threshold [default: 0.25]
    --iou <IOU>            IoU threshold for NMS [default: 0.45]
    --imgsz <IMGSZ>        Inference image size
    --half                 Use FP16 half-precision inference
    --save                 Save annotated images to runs/<task>/predict
    --show                 Display results in a window
    --device <DEVICE>      Device (cpu, cuda:0, mps, coreml, directml:0, openvino, xnnpack)
    --verbose              Show verbose output

Examples:
    ultralytics-inference predict --model yolo11n.onnx --source image.jpg
    ultralytics-inference predict --model yolo11n.onnx --source video.mp4
    ultralytics-inference predict --model yolo11n.onnx --source 0 --conf 0.5
    ultralytics-inference predict -m yolo11n.onnx -s assets/ --save --half
    ultralytics-inference predict -m yolo11n.onnx -s video.mp4 --imgsz 1280 --show
    ultralytics-inference predict --model yolo11n.onnx --source image.jpg --device mps"#)]
pub struct Cli {
    #[command(subcommand)]
    /// Subcommand to execute.
    pub command: Commands,
}

/// Commands for the CLI.
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Run inference on an image, video, or stream
    Predict(PredictArgs),
}

/// Arguments for the predict command.
#[derive(Args, Debug)]
#[allow(clippy::struct_excessive_bools)]
pub struct PredictArgs {
    /// Path to ONNX model file
    #[arg(short, long, default_value = DEFAULT_MODEL)]
    pub model: String,

    /// Input source (image, directory, glob, video, webcam, or URL)
    #[arg(short, long)]
    pub source: Option<String>,

    /// Confidence threshold
    #[arg(long, default_value_t = 0.25)]
    pub conf: f32,

    /// `IoU` threshold for NMS
    #[arg(long, default_value_t = 0.45)]
    pub iou: f32,

    /// Inference image size
    #[arg(long)]
    pub imgsz: Option<usize>,

    /// Batch size for inference
    #[arg(long, default_value_t = 1)]
    pub batch: usize,

    /// Use FP16 half-precision inference
    #[arg(long, default_value_t = false)]
    pub half: bool,

    /// Save annotated images to runs/<task>/predict
    #[arg(long, default_value_t = false)]
    pub save: bool,

    /// Display results in a window
    #[arg(long, default_value_t = false)]
    pub show: bool,

    /// Device to use (cpu, cuda:0, mps, coreml, directml:0, openvino, tensorrt:0, etc.)
    #[arg(long)]
    pub device: Option<String>,

    /// Show verbose output
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub verbose: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_cli() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }

    #[test]
    fn test_predict_args_defaults() {
        let args = Cli::parse_from(["app", "predict", "--model", "yolo11n.onnx"]);
        match args.command {
            Commands::Predict(predict_args) => {
                assert_eq!(predict_args.model, "yolo11n.onnx");
                assert!((predict_args.conf - 0.25).abs() < f32::EPSILON);
                assert!((predict_args.iou - 0.45).abs() < f32::EPSILON);
                assert!(!predict_args.half);
                assert!(predict_args.verbose);
                assert!(predict_args.source.is_none());
            }
        }
    }

    #[test]
    fn test_predict_args_custom() {
        let args = Cli::parse_from([
            "app",
            "predict",
            "--model",
            "custom.onnx",
            "--source",
            "test.jpg",
            "--conf",
            "0.8",
            "--verbose",
            "false",
        ]);
        match args.command {
            Commands::Predict(predict_args) => {
                assert_eq!(predict_args.model, "custom.onnx");
                assert_eq!(predict_args.source, Some("test.jpg".to_string()));
                assert!((predict_args.conf - 0.8).abs() < f32::EPSILON);
                assert!(!predict_args.verbose);
            }
        }
    }
}
