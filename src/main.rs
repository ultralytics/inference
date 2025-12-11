// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Ultralytics YOLO Inference CLI
//!
//! A command-line interface for running YOLO model inference on images and videos.
//!
//! # Usage
//!
//! ```bash
//! inference predict --model yolo11n.onnx --source image.jpg
//! inference predict --model yolo11n.onnx --source video.mp4
//! inference predict --model yolo11n.onnx --source 0  # webcam
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
use ab_glyph::{FontRef, PxScale};
#[cfg(feature = "annotate")]
use image::{DynamicImage, Rgb};
#[cfg(feature = "annotate")]
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
#[cfg(feature = "annotate")]
use imageproc::rect::Rect;

use inference::{InferenceConfig, Results, YOLOModel, VERSION};

/// Default model path when not specified.
const DEFAULT_MODEL: &str = "yolo11n.onnx";
/// Default source path when not specified.
const DEFAULT_SOURCE: &str = "assets";

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
fn run_prediction(args: &[String]) {
    // Parse arguments
    let mut model_path: Option<&String> = None;
    let mut source_path: Option<&String> = None;
    let mut conf_threshold = 0.25_f32;
    let mut iou_threshold = 0.45_f32;
    let mut save = false;

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
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                process::exit(1);
            }
        }
    }

    // Use defaults with warnings if not specified
    let default_model = DEFAULT_MODEL.to_string();
    let default_source = DEFAULT_SOURCE.to_string();

    let model_path = match model_path {
        Some(m) => m.clone(),
        None => {
            eprintln!(
                "WARNING ⚠️ 'model' argument is missing. Using default 'model={DEFAULT_MODEL}'."
            );
            default_model
        }
    };

    let source_path = match source_path {
        Some(s) => s.clone(),
        None => {
            let cwd = env::current_dir()
                .map(|p| p.display().to_string())
                .unwrap_or_default();
            eprintln!(
                "WARNING ⚠️ 'source' argument is missing. Using default 'source={cwd}/{DEFAULT_SOURCE}'."
            );
            default_source
        }
    };

    // Print banner matching Ultralytics format
    println!("Ultralytics {} 🚀 Rust ONNX CPU", VERSION);

    // Create save directory if --save is specified
    #[cfg(feature = "annotate")]
    let save_dir = if save {
        let dir = find_next_run_dir("runs/detect", "predict");
        fs::create_dir_all(&dir).expect("Failed to create save directory");
        Some(dir)
    } else {
        None
    };

    // Warn if --save is used without annotate feature
    #[cfg(not(feature = "annotate"))]
    if save {
        eprintln!("WARNING: --save requires the 'annotate' feature. Rebuild with: cargo build --features annotate");
    }

    // Load model
    let config = InferenceConfig::new()
        .with_confidence(conf_threshold)
        .with_iou(iou_threshold);

    let mut model = match YOLOModel::load_with_config(&model_path, config) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error loading model: {e}");
            process::exit(1);
        }
    };

    // Print model summary
    let imgsz = model.imgsz();
    println!(
        "YOLO11 summary: {} classes, imgsz=({}, {})",
        model.num_classes(),
        imgsz.0,
        imgsz.1
    );
    println!();

    // Collect images to process
    let source_path_obj = Path::new(&source_path);
    let images: Vec<String> = if source_path_obj.is_dir() {
        collect_images_from_dir(source_path_obj)
    } else {
        vec![source_path.clone()]
    };

    let total_images = images.len();
    if total_images == 0 {
        eprintln!("No images found in source: {source_path}");
        process::exit(1);
    }

    // Process each image
    let mut all_results: Vec<(String, Results)> = Vec::new();
    let mut total_preprocess = 0.0;
    let mut total_inference = 0.0;
    let mut total_postprocess = 0.0;
    let mut last_inference_shape = (0, 0);

    for (idx, image_path) in images.iter().enumerate() {
        let results = match model.predict(image_path) {
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
            let orig_shape = result.orig_shape();
            let inference_shape = result.inference_shape();
            last_inference_shape = (inference_shape.0 as usize, inference_shape.1 as usize);

            // Print per-image output matching Ultralytics format
            // Use original image dimensions for the per-image output
            println!(
                "image {}/{} {}: {}x{} {}, {:.1}ms",
                idx + 1,
                total_images,
                image_path,
                orig_shape.1, // width first in output
                orig_shape.0, // then height
                detection_summary,
                result.speed.inference.unwrap_or(0.0)
            );

            // Save annotated image if --save is specified
            #[cfg(feature = "annotate")]
            if let Some(ref dir) = save_dir {
                if let Ok(img) = image::open(image_path) {
                    let annotated = annotate_image(&img, &result);
                    let filename = Path::new(image_path)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy();
                    let save_path = format!("{dir}/{filename}");
                    if let Err(e) = annotated.save(&save_path) {
                        eprintln!("Error saving {save_path}: {e}");
                    }
                }
            }

            // Accumulate timings
            total_preprocess += result.speed.preprocess.unwrap_or(0.0);
            total_inference += result.speed.inference.unwrap_or(0.0);
            total_postprocess += result.speed.postprocess.unwrap_or(0.0);

            all_results.push((image_path.clone(), result));
        }
    }

    // Print speed summary with inference tensor shape (after letterboxing)
    let num_results = all_results.len().max(1) as f64;
    println!(
        "Speed: {:.1}ms preprocess, {:.1}ms inference, {:.1}ms postprocess per image at shape (1, 3, {}, {})",
        total_preprocess / num_results,
        total_inference / num_results,
        total_postprocess / num_results,
        last_inference_shape.0,
        last_inference_shape.1
    );

    // Print save directory if --save was used
    #[cfg(feature = "annotate")]
    if let Some(ref dir) = save_dir {
        println!("Results saved to {dir}");
    }

    // Print footer
    println!("💡 Learn more at https://docs.ultralytics.com/modes/predict");
}

/// Format detection summary like "4 persons, 1 bus".
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
                let class_name = result
                    .names
                    .get(class_id)
                    .map(String::as_str)
                    .unwrap_or("object");
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
        "classification".to_string()
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

/// Collect image paths from a directory.
fn collect_images_from_dir(dir: &Path) -> Vec<String> {
    let mut paths = Vec::new();

    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                let ext = ext.to_string_lossy().to_lowercase();
                if matches!(
                    ext.as_str(),
                    "jpg" | "jpeg" | "png" | "bmp" | "gif" | "webp" | "tiff" | "tif"
                ) {
                    paths.push(path.to_string_lossy().to_string());
                }
            }
        }
    }

    paths.sort();
    paths
}

/// Print version information.
fn print_version() {
    println!("Ultralytics Inference v{VERSION}");
}

/// Print usage information.
fn print_usage() {
    println!(
        r#"Ultralytics YOLO Inference CLI
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
    --source, -s    Input source (image, video, webcam index, or URL)
    --conf          Confidence threshold (default: 0.25)
    --iou           IoU threshold for NMS (default: 0.45)
    --save          Save annotated images to runs/detect/predict

Examples:
    inference predict --model yolo11n.onnx --source image.jpg
    inference predict --model yolo11n.onnx --source video.mp4
    inference predict --model yolo11n.onnx --source 0 --conf 0.5
    inference predict -m yolo11n.onnx -s assets/ --save"#
    );
}

/// Find the next available run directory (predict, predict2, predict3, etc.)
#[cfg(feature = "annotate")]
fn find_next_run_dir(base: &str, prefix: &str) -> String {
    let base_path = Path::new(base);

    // First try without number
    let first = base_path.join(prefix);
    if !first.exists() {
        return first.to_string_lossy().to_string();
    }

    // Try with incrementing numbers
    for i in 2.. {
        let numbered = base_path.join(format!("{prefix}{i}"));
        if !numbered.exists() {
            return numbered.to_string_lossy().to_string();
        }
    }

    // Fallback (should never reach here)
    base_path.join(prefix).to_string_lossy().to_string()
}

/// Color palette for different classes (similar to Ultralytics)
#[cfg(feature = "annotate")]
const COLORS: [[u8; 3]; 20] = [
    [255, 56, 56],    // Red
    [255, 157, 151],  // Light red
    [255, 112, 31],   // Orange
    [255, 178, 29],   // Yellow-orange
    [207, 210, 49],   // Yellow-green
    [72, 249, 10],    // Green
    [146, 204, 23],   // Light green
    [61, 219, 134],   // Teal
    [26, 147, 52],    // Dark green
    [0, 212, 187],    // Cyan
    [44, 153, 168],   // Dark cyan
    [0, 194, 255],    // Light blue
    [52, 69, 147],    // Dark blue
    [100, 115, 255],  // Blue
    [0, 24, 236],     // Bright blue
    [132, 56, 255],   // Purple
    [82, 0, 133],     // Dark purple
    [203, 56, 255],   // Magenta
    [255, 149, 200],  // Pink
    [255, 55, 199],   // Hot pink
];

/// Get color for a class ID
#[cfg(feature = "annotate")]
fn get_class_color(class_id: usize) -> Rgb<u8> {
    let color = COLORS[class_id % COLORS.len()];
    Rgb(color)
}

/// Embedded font data (DejaVu Sans Mono - a free font)
/// Using a simple embedded approach for cross-platform compatibility
#[cfg(feature = "annotate")]
const FONT_DATA: &[u8] = include_bytes!("../assets/DejaVuSans.ttf");

/// Annotate an image with detection boxes and labels
#[cfg(feature = "annotate")]
fn annotate_image(image: &DynamicImage, result: &Results) -> DynamicImage {
    let mut img = image.to_rgb8();
    let (width, height) = img.dimensions();

    // Load font
    let font = FontRef::try_from_slice(FONT_DATA).ok();

    if let Some(ref boxes) = result.boxes {
        let xyxy = boxes.xyxy();
        let conf = boxes.conf();
        let cls = boxes.cls();

        for i in 0..boxes.len() {
            let class_id = cls[i] as usize;
            let confidence = conf[i];

            // Get box coordinates
            let x1 = xyxy[[i, 0]].round() as i32;
            let y1 = xyxy[[i, 1]].round() as i32;
            let x2 = xyxy[[i, 2]].round() as i32;
            let y2 = xyxy[[i, 3]].round() as i32;

            // Clamp to image bounds
            let x1 = x1.max(0).min(width as i32 - 1);
            let y1 = y1.max(0).min(height as i32 - 1);
            let x2 = x2.max(0).min(width as i32 - 1);
            let y2 = y2.max(0).min(height as i32 - 1);

            let color = get_class_color(class_id);

            // Draw bounding box (multiple times for thickness)
            for offset in 0..3_i32 {
                let rect_width = (x2 - x1 + 2 * offset).max(1) as u32;
                let rect_height = (y2 - y1 + 2 * offset).max(1) as u32;
                let rect = Rect::at(x1 - offset, y1 - offset).of_size(rect_width, rect_height);
                draw_hollow_rect_mut(&mut img, rect, color);
            }

            // Draw label
            let class_name = result
                .names
                .get(&class_id)
                .map(String::as_str)
                .unwrap_or("object");
            let label = format!("{} {:.2}", class_name, confidence);

            if let Some(ref f) = font {
                let scale = PxScale::from(20.0);
                let y_text = if y1 > 25 { y1 - 8 } else { y2 + 20 };
                draw_text_mut(&mut img, color, x1, y_text, scale, f, &label);
            }
        }
    }

    DynamicImage::ImageRgb8(img)
}
