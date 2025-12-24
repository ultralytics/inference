// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#![allow(clippy::multiple_crate_versions)]

//! Ultralytics YOLO Inference CLI
//!
//! A command-line interface for running YOLO model inference on images and videos.
//!
//! # Usage
//!
//! ```bash
//! inference predict --model yolo11n.onnx --source image.jpg
//! inference predict --model yolo11n.onnx --source video.mp4
//! inference predict --model yolo11n.onnx --source 0 --conf 0.5
//! inference predict -m yolo11n.onnx -s assets/ --save --half
//! inference predict -m yolo11n.onnx -s video.mp4 --imgsz 1280 --show
//! inference version
//! inference help
//! ```

use std::collections::HashMap;
use std::env;
use std::path::Path;
use std::process;

#[cfg(feature = "annotate")]
use std::fs;

#[cfg(feature = "annotate")]
use inference::annotate::{annotate_image, find_next_run_dir};

#[cfg(feature = "visualize")]
use inference::visualizer::Viewer;

use inference::{InferenceConfig, Results, VERSION, YOLOModel};

/// Default model path when not specified.
const DEFAULT_MODEL: &str = "yolo11n.onnx";

/// Entry point for the Ultralytics YOLO Inference CLI.
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
            eprintln!("Unknown command: {command}");
            print_usage();
            process::exit(1);
        }
    }
}

/// Run YOLO model inference.
#[allow(
    clippy::too_many_lines,
    clippy::option_if_let_else,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn run_prediction(args: &[String]) {
    // Parse arguments
    let mut model_path: Option<&String> = None;
    let mut source_path: Option<&String> = None;
    let mut conf_threshold = 0.25_f32;
    let mut iou_threshold = 0.45_f32;
    let mut imgsz: usize = 640;
    let mut save = false;
    let mut half = false;
    let mut verbose = true;
    #[cfg(feature = "visualize")]
    let mut show = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                if i + 1 < args.len() {
                    model_path = Some(&args[i + 1]);
                    i += 2;
                } else {
                    eprintln!("Error: --model requires a value");
                    process::exit(1);
                }
            }
            "--source" | "-s" => {
                if i + 1 < args.len() {
                    source_path = Some(&args[i + 1]);
                    i += 2;
                } else {
                    eprintln!("Error: --source requires a value");
                    process::exit(1);
                }
            }
            "--conf" => {
                if i + 1 < args.len() {
                    conf_threshold = args[i + 1].parse().unwrap_or(0.25);
                    i += 2;
                } else {
                    eprintln!("Error: --conf requires a value");
                    process::exit(1);
                }
            }
            "--iou" => {
                if i + 1 < args.len() {
                    iou_threshold = args[i + 1].parse().unwrap_or(0.45);
                    i += 2;
                } else {
                    eprintln!("Error: --iou requires a value");
                    process::exit(1);
                }
            }
            "--save" => {
                save = true;
                i += 1;
            }
            "--half" => {
                half = true;
                i += 1;
            }
            "--show" => {
                #[cfg(feature = "visualize")]
                {
                    show = true;
                }
                #[cfg(not(feature = "visualize"))]
                {
                    eprintln!(
                        "WARNING ⚠️ --show requires the 'visualize' feature. Compile with --features visualize to enable display."
                    );
                }
                i += 1;
            }
            "--imgsz" => {
                if i + 1 < args.len() {
                    imgsz = args[i + 1].parse().unwrap_or(640);
                    i += 2;
                } else {
                    eprintln!("Error: --imgsz requires a value");
                    process::exit(1);
                }
            }
            "--verbose" => {
                if i + 1 < args.len() {
                    let next_arg = &args[i + 1];
                    if let Ok(v) = next_arg.parse::<bool>() {
                        verbose = v;
                        i += 2;
                    } else {
                        verbose = true;
                        i += 1;
                    }
                } else {
                    // End of args
                    verbose = true;
                    i += 1;
                }
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                process::exit(1);
            }
        }
    }

    // Use defaults with warnings if not specified
    let default_model = DEFAULT_MODEL.to_string();

    let model_path = if let Some(m) = model_path {
        m.clone()
    } else {
        if verbose {
            eprintln!(
                "WARNING ⚠️ 'model' argument is missing. Using default 'model={DEFAULT_MODEL}'."
            );
        }
        default_model
    };

    // Load model first so we can determine appropriate default source based on task
    let mut config = InferenceConfig::new()
        .with_confidence(conf_threshold)
        .with_iou(iou_threshold)
        .with_half(half);

    // Apply imgsz (default 640)
    config = config.with_imgsz(imgsz, imgsz);

    let mut model = match YOLOModel::load_with_config(&model_path, config) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error loading model: {e}");
            process::exit(1);
        }
    };

    // Determine source
    let source = source_path.as_ref().map_or_else(
        || {
            // Select default images based on model task
            let default_urls = match model.task() {
                inference::task::Task::Obb => &[inference::download::DEFAULT_OBB_IMAGE],
                _ => inference::download::DEFAULT_IMAGES,
            };

            if verbose {
                eprintln!(
                    "WARNING ⚠️ 'source' argument is missing. Using default images: {}",
                    default_urls.join(", ")
                );
            }

            // Download images to current directory (skips if already exists)
            let downloaded_files = inference::download::download_images(default_urls);

            if downloaded_files.is_empty() {
                eprintln!("Error: Failed to download any images");
                process::exit(1);
            }

            // Convert to PathBufs for ImageList
            let paths = downloaded_files
                .into_iter()
                .map(std::path::PathBuf::from)
                .collect();

            inference::source::Source::ImageList(paths)
        },
        |s| inference::source::Source::from(s.as_str()),
    );

    #[cfg(feature = "annotate")]
    let save_dir = if save {
        let parent_dir = match model.task() {
            inference::task::Task::Detect => "runs/detect",
            inference::task::Task::Segment => "runs/segment",
            inference::task::Task::Pose => "runs/pose",
            inference::task::Task::Classify => "runs/classify",
            inference::task::Task::Obb => "runs/obb",
        };
        let dir = find_next_run_dir(parent_dir, "predict");
        fs::create_dir_all(&dir).expect("Failed to create save directory");
        Some(dir)
    } else {
        None
    };

    #[cfg(not(feature = "annotate"))]
    if save {
        eprintln!(
            "WARNING ⚠️ --save requires the 'annotate' feature. Compile with --features annotate to enable saving."
        );
    }

    let is_half = model.metadata().half || half;
    let precision = if is_half { "FP16" } else { "FP32" };
    println!("Ultralytics {VERSION} 🚀 Rust ONNX {precision} CPU");

    let imgsz = model.imgsz();
    println!(
        "YOLO11 summary: {} classes, imgsz=({}, {})",
        model.num_classes(),
        imgsz.0,
        imgsz.1
    );
    println!();

    // Source is already initialized above
    let is_video = source.is_video();
    let source_iter = match inference::source::SourceIterator::new(source) {
        Ok(iter) => iter,
        Err(e) => {
            eprintln!("Error initializing source: {e}");
            process::exit(1);
        }
    };

    // Process each image/frame
    let mut all_results: Vec<(String, Results)> = Vec::new();
    let mut total_preprocess = 0.0;
    let mut total_inference = 0.0;
    let mut total_postprocess = 0.0;
    let mut last_inference_shape = (0, 0);

    #[cfg(feature = "visualize")]
    let mut viewer: Option<Viewer> = None;

    #[cfg(feature = "visualize")]
    let mut frame_count = 0;
    for item in source_iter {
        let (img, meta) = match item {
            Ok(val) => val,
            Err(e) => {
                eprintln!("Error reading source: {e}");
                continue;
            }
        };

        let image_path = meta.path;
        let results = match model.predict_image(&img, image_path.clone()) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Error processing {image_path}: {e}");
                continue;
            }
        };

        for result in results {
            // Build detection summary
            let detection_summary = format_detection_summary(&result);

            // Get image dimensions from result
            // let orig_shape = result.orig_shape();
            let inference_shape = result.inference_shape();
            last_inference_shape = (inference_shape.0 as usize, inference_shape.1 as usize);

            // Format total frames for display
            let total_frames_str = meta
                .total_frames
                .map_or_else(|| "?".to_string(), |n| n.to_string());

            if verbose {
                if is_video {
                    // Assuming single video input for now as per CLI structure
                    // Use "video 1/1"
                    println!(
                        "video 1/1 (frame {}/{}) {}: {}x{} {}, {:.1}ms",
                        meta.frame_idx + 1,
                        total_frames_str,
                        image_path,
                        inference_shape.1,
                        inference_shape.0,
                        detection_summary,
                        result.speed.inference.unwrap_or(0.0)
                    );
                } else {
                    println!(
                        "image {}/{} {}: {}x{} {}, {:.1}ms",
                        meta.frame_idx + 1,
                        total_frames_str,
                        image_path,
                        inference_shape.1,
                        inference_shape.0,
                        detection_summary,
                        result.speed.inference.unwrap_or(0.0)
                    );
                }
            }

            // Save annotated image if --save is specified
            #[cfg(feature = "annotate")]
            if let Some(ref dir) = save_dir {
                let annotated = annotate_image(&img, &result, None);
                let filename = Path::new(&image_path)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy();
                let save_path = format!("{dir}/{filename}");
                if let Err(e) = annotated.save(&save_path) {
                    eprintln!("Error saving {save_path}: {e}");
                }
            }

            // Show result in viewer if enabled
            #[cfg(feature = "visualize")]
            if show {
                // Use inference shape for viewer dimensions
                let view_width = inference_shape.1 as usize;
                let view_height = inference_shape.0 as usize;

                // If viewer exists but dimensions don't match, drop it to recreate
                if let Some(ref v) = viewer
                    && (v.width != view_width || v.height != view_height)
                {
                    viewer = None;
                }

                // Initialize viewer lazily with inference dimensions
                if viewer.is_none() {
                    viewer = Some(
                        Viewer::new("Ultralytics Inference", view_width, view_height).unwrap(),
                    );
                }

                if let Some(ref mut v) = viewer {
                    let annotated = annotate_image(&img, &result, None);
                    // Resize annotated image to inference dimensions
                    let resized = annotated.resize_exact(
                        view_width as u32,
                        view_height as u32,
                        image::imageops::FilterType::Triangle,
                    );

                    // Update viewer
                    if v.update(&resized).is_ok() {
                        // Add delay logic based on source type
                        if is_video {
                            // 200ms delay for initial start of video
                            if frame_count == 0 {
                                let _ = v.wait(std::time::Duration::from_millis(200));
                            }
                        } else {
                            // 500ms delay for images (single or directory)
                            let _ = v.wait(std::time::Duration::from_millis(200));
                        }
                    }
                }
            }

            // Accumulate timings
            total_preprocess += result.speed.preprocess.unwrap_or(0.0);
            total_inference += result.speed.inference.unwrap_or(0.0);
            total_postprocess += result.speed.postprocess.unwrap_or(0.0);

            all_results.push((image_path.clone(), result));
        }
        #[cfg(feature = "visualize")]
        {
            frame_count += 1;
        }
    }

    // Print speed summary with inference tensor shape (after letterboxing)
    if verbose {
        let num_results = all_results.len().max(1) as f64;
        println!(
            "Speed: {:.1}ms preprocess, {:.1}ms inference, {:.1}ms postprocess per image at shape (1, 3, {}, {})",
            total_preprocess / num_results,
            total_inference / num_results,
            total_postprocess / num_results,
            last_inference_shape.0,
            last_inference_shape.1
        );
    }

    // Print save directory if --save was used
    #[cfg(feature = "annotate")]
    if let Some(ref dir) = save_dir {
        println!("Results saved to {dir}");
    }

    // Print footer
    println!("💡 Learn more at https://docs.ultralytics.com/modes/predict");
}

/// Format detection summary like "4 persons, 1 bus".
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::option_if_let_else
)]
fn format_detection_summary(result: &Results) -> String {
    if let Some(ref boxes) = result.boxes {
        if boxes.is_empty() {
            return String::new();
        }

        // Count detections per class
        let cls = boxes.cls();
        let mut counts: HashMap<usize, usize> = HashMap::new();

        for i in 0..boxes.len() {
            let class_id = cls[i] as usize;
            *counts.entry(class_id).or_insert(0) += 1;
        }

        // Sort by class ID for consistent output
        let mut sorted_counts: Vec<(usize, usize)> = counts.into_iter().collect();
        sorted_counts.sort_by_key(|(class_id, _)| *class_id);

        // Format each class count
        let parts: Vec<String> = sorted_counts
            .iter()
            .map(|(class_id, count)| {
                let class_name = result.names.get(class_id).map_or("object", String::as_str);
                // Pluralize if count > 1
                let name = if *count > 1 {
                    pluralize(class_name)
                } else {
                    class_name.to_string()
                };
                format!("{count} {name}")
            })
            .collect();

        parts.join(", ")
    } else if result.probs.is_some() {
        if let Some(probs) = &result.probs {
            let top5 = probs.top5();
            let parts: Vec<String> = top5
                .iter()
                .map(|&i| {
                    let name = result.names.get(&i).map_or("unknown", String::as_str);
                    format!("{} {:.2}", name, probs.data[[i]])
                })
                .collect();
            parts.join(", ")
        } else {
            "classification".to_string()
        }
    } else {
        String::new()
    }
}

/// Simple pluralization for common COCO class names.
fn pluralize(word: &str) -> String {
    match word {
        "person" => "persons".to_string(),
        "bus" => "buses".to_string(),
        "knife" => "knives".to_string(),
        "mouse" => "mice".to_string(),
        "sheep" => "sheep".to_string(),
        "skis" => "skis".to_string(),
        _ => {
            if word.ends_with('s') || word.ends_with("ch") || word.ends_with("sh") {
                format!("{word}es")
            } else if word.ends_with('y') && !word.ends_with("ey") && !word.ends_with("ay") {
                format!("{}ies", &word[..word.len() - 1])
            } else {
                format!("{word}s")
            }
        }
    }
}

/// Print version information.
fn print_version() {
    println!("Ultralytics Inference v{VERSION}");
}

/// Print usage information.
fn print_usage() {
    println!(
        r"Ultralytics YOLO Inference CLI
==============================

Usage:
    inference predict --model <model_path> --source <source>
    inference version
    inference help

Commands:
    predict    Run inference on an image, video, or stream
    version    Print version information
    help       Show this help message

Options:
    --model, -m     Path to ONNX model file
    --source, -s    Input source (image, directory, glob, video, webcam, or URL)
    --conf          Confidence threshold (default: 0.25)
    --iou           IoU threshold for NMS (default: 0.45)
    --imgsz         Inference image size (default: 640)
    --half          Use FP16 half-precision inference
    --save          Save annotated images to runs/<task>/predict
    --show          Display results in a window
    --verbose       Show verbose output (default: true)

Examples:
    inference predict --model yolo11n.onnx --source image.jpg
    inference predict --model yolo11n.onnx --source video.mp4
    inference predict --model yolo11n.onnx --source 0 --conf 0.5
    inference predict -m yolo11n.onnx -s assets/ --save --half
    inference predict -m yolo11n.onnx -s video.mp4 --imgsz 1280 --show"
    );
}
