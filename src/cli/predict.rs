// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

use std::collections::HashMap;
use std::path::Path;
use std::process;
#[cfg(feature = "visualize")]
use std::time::Duration;

#[cfg(feature = "annotate")]
use std::fs;

#[cfg(feature = "annotate")]
use crate::annotate::{annotate_image, find_next_run_dir};

#[cfg(feature = "visualize")]
use crate::visualizer::Viewer;

use crate::utils::pluralize;
use crate::{InferenceConfig, Results, VERSION, YOLOModel};

use crate::cli::args::PredictArgs;
use crate::{error, verbose, warn};

/// Run YOLO model inference.
#[allow(
    clippy::too_many_lines,
    clippy::option_if_let_else,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_panics_doc
)]
pub fn run_prediction(args: &PredictArgs) {
    // Parse arguments
    let model_path = &args.model;
    let source_path = &args.source;
    let conf_threshold = args.conf;
    let iou_threshold = args.iou;
    let imgsz = args.imgsz;
    let save = args.save;
    let half = args.half;
    let verbose = args.verbose;
    let device: Option<crate::Device> = args
        .device
        .as_ref()
        .map(|d| d.parse().expect("Invalid device"));
    #[cfg(feature = "visualize")]
    let show = args.show;

    // Use defaults with warnings if not specified
    // Clap handles default model path, so model_path is always set.

    // Load model first so we can determine appropriate default source based on task
    let mut config = InferenceConfig::new()
        .with_confidence(conf_threshold)
        .with_iou(iou_threshold)
        .with_half(half);

    // Apply imgsz if specified
    if let Some(sz) = imgsz {
        config = config.with_imgsz(sz, sz);
    }

    // Apply device if specified
    if let Some(d) = &device {
        config = config.with_device(d.clone());
    }

    let mut model = match YOLOModel::load_with_config(model_path, config) {
        Ok(m) => m,
        Err(e) => {
            error!("Error loading model: {e}");
            process::exit(1);
        }
    };

    // Determine source
    let source = source_path.as_ref().map_or_else(
        || {
            // Select default images based on model task
            let default_urls = match model.task() {
                crate::task::Task::Obb => &[crate::download::DEFAULT_OBB_IMAGE],
                _ => crate::download::DEFAULT_IMAGES,
            };

            if verbose {
                warn!(
                    "'source' argument is missing. Using default images: {}",
                    default_urls.join(", ")
                );
            }

            // Download images to current directory (skips if already exists)
            let downloaded_files = crate::download::download_images(default_urls);

            if downloaded_files.is_empty() {
                error!("Failed to download any images");
                process::exit(1);
            }

            // Convert to PathBufs for ImageList
            let paths = downloaded_files
                .into_iter()
                .map(std::path::PathBuf::from)
                .collect();

            crate::source::Source::ImageList(paths)
        },
        |s| crate::source::Source::from(s.as_str()),
    );

    #[cfg(feature = "annotate")]
    let save_dir = if save {
        let parent_dir = match model.task() {
            crate::task::Task::Detect => "runs/detect",
            crate::task::Task::Segment => "runs/segment",
            crate::task::Task::Pose => "runs/pose",
            crate::task::Task::Classify => "runs/classify",
            crate::task::Task::Obb => "runs/obb",
        };
        let dir = find_next_run_dir(parent_dir, "predict");
        fs::create_dir_all(&dir).expect("Failed to create save directory");
        Some(dir)
    } else {
        None
    };

    #[cfg(not(feature = "annotate"))]
    if save {
        warn!(
            "--save requires the 'annotate' feature. Compile with --features annotate to enable saving."
        );
    }

    let is_half = model.metadata().half || half;
    let precision = if is_half { "FP16" } else { "FP32" };
    let device_str = {
        let provider = model.execution_provider();
        if provider.contains("CoreML") {
            "MPS".to_string()
        } else if provider.contains("CUDA") {
            "CUDA".to_string()
        } else if provider.contains("TensorRT") {
            "TensorRT".to_string()
        } else if provider.contains("DirectML") {
            "DirectML".to_string()
        } else if provider.contains("ROCm") {
            "ROCm".to_string()
        } else if provider.contains("OpenVINO") {
            "OpenVINO".to_string()
        } else {
            "CPU".to_string()
        }
    };
    println!("Ultralytics {VERSION} 🚀 Rust ONNX {precision} {device_str}");
    println!("Using ONNX Runtime {}", model.execution_provider());

    let imgsz = model.imgsz();
    verbose!(
        "{} summary: {} classes, imgsz=({}, {})",
        model.metadata().model_name(),
        model.num_classes(),
        imgsz.0,
        imgsz.1
    );
    verbose!("");

    // Source is already initialized above
    let is_video = source.is_video();
    let source_iter = match crate::source::SourceIterator::new(source) {
        Ok(iter) => iter,
        Err(e) => {
            error!("Error initializing source: {e}");
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
                error!("Error reading source: {e}");
                continue;
            }
        };

        let image_path = meta.path;
        let results = match model.predict_image(&img, image_path.clone()) {
            Ok(r) => r,
            Err(e) => {
                error!("Error processing {image_path}: {e}");
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

            if is_video {
                // Assuming single video input for now as per CLI structure
                // Use "video 1/1"
                verbose!(
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
                verbose!(
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
                    error!("Error saving {save_path}: {e}");
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
                                let _ = v.wait(Duration::from_millis(200));
                            }
                        } else {
                            // 500ms delay for images (single or directory)
                            let _ = v.wait(Duration::from_millis(200));
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
    let num_results = all_results.len().max(1) as f64;
    verbose!(
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
        verbose!("Results saved to {dir}");
    }

    // Print footer
    verbose!("💡 Learn more at https://docs.ultralytics.com/modes/predict");
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
    } else if let Some(ref obb) = result.obb {
        if obb.is_empty() {
            return String::new();
        }

        // Count detections per class
        let cls = obb.cls();
        let mut counts: HashMap<usize, usize> = HashMap::new();

        for i in 0..obb.len() {
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
