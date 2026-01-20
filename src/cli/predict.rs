// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

use std::collections::HashMap;

use std::process;
#[cfg(feature = "visualize")]
use std::time::Duration;

#[cfg(feature = "annotate")]
use std::fs;

#[cfg(feature = "annotate")]
use crate::annotate::{annotate_image, find_next_run_dir};

#[cfg(feature = "visualize")]
use crate::visualizer::Viewer;
#[cfg(feature = "visualize")]
use image::GenericImageView;

use crate::utils::pluralize;
use crate::{InferenceConfig, Results, VERSION, YOLOModel};

use crate::batch::BatchProcessor;
use crate::cli::args::PredictArgs;
use crate::{error, verbose, warn};

/// Run YOLO model inference.
#[allow(
    clippy::too_many_lines,
    clippy::option_if_let_else,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_panics_doc,
    clippy::redundant_clone
)]
pub fn run_prediction(args: &PredictArgs) {
    // Parse arguments - use default model if not specified
    let model_is_default = args.model.is_none();
    let model_path = args
        .model
        .clone()
        .unwrap_or_else(|| crate::download::DEFAULT_MODEL.to_string());
    let source_path = &args.source;
    let conf_threshold = args.conf;
    let iou_threshold = args.iou;
    let imgsz = args.imgsz;
    let save = args.save;
    let save_frames = args.save_frames;
    let half = args.half;
    let verbose = args.verbose;
    let batch_size = args.batch as usize;
    let device: Option<crate::Device> = args
        .device
        .as_ref()
        .map(|d| d.parse().expect("Invalid device"));
    #[cfg(feature = "visualize")]
    let show = args.show;
    // Warn if using default model (like Python does)
    if model_is_default && verbose {
        warn!(
            "'model' argument is missing. Using default '--model={}'.",
            crate::download::DEFAULT_MODEL
        );
    }

    // Load model first so we can determine appropriate default source based on task
    let mut config = InferenceConfig::new()
        .with_confidence(conf_threshold)
        .with_iou(iou_threshold)
        .with_half(half)
        .with_batch(batch_size)
        .with_save_frames(save_frames)
        .with_rect(args.rect)
        .with_max_det(args.max_det);

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
        Some(std::path::PathBuf::from(dir))
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
    #[cfg(not(feature = "video"))]
    if is_video {
        warn!(
            "Video source detected but 'video' feature is not enabled. Please compile with '--features video'"
        );
        process::exit(1);
    }

    // Process each image/frame
    let mut all_results: Vec<(String, Results)> = Vec::new();
    let mut total_preprocess = 0.0;
    let mut total_inference = 0.0;
    let mut total_postprocess = 0.0;
    let mut last_inference_shape = (0, 0);

    #[cfg(feature = "visualize")]
    let mut viewer: Option<Viewer> = None;

    // Initialize ResultSaver if saving is enabled
    #[cfg(feature = "annotate")]
    let mut result_saver = save_dir
        .as_ref()
        .map(|d| crate::io::SaveResults::new(d.clone(), save_frames));
    #[cfg(not(feature = "annotate"))]
    let mut result_saver: Option<crate::io::SaveResults> = None;

    // Create a bounded channel for pipelined processing
    // Buffer size 2x batch size ensures we can decode the next batch while processing current one
    let channel_capacity = batch_size * 2;
    let (sender, receiver) = std::sync::mpsc::sync_channel(channel_capacity);

    // Spawn producer thread for frame decoding
    let source_clone = source.clone();
    std::thread::spawn(move || {
        let iter = match crate::source::SourceIterator::new(source_clone) {
            Ok(iter) => iter,
            Err(e) => {
                error!("Error initializing source in thread: {e}");
                return;
            }
        };

        for item in iter {
            if sender.send(item).is_err() {
                break; // Receiver dropped, stop decoding
            }
        }
    });

    // Use BatchProcessor for centralized batch management
    {
        let mut batch_processor = BatchProcessor::new(
            &mut model,
            batch_size,
            |batch_results: Vec<Vec<Results>>,
             images: &[image::DynamicImage],
             paths: &[String],
             metas: &[crate::source::SourceMeta]| {
                for (results, (meta, (image_path, img))) in batch_results
                    .into_iter()
                    .zip(metas.iter().zip(paths.iter().zip(images.iter())))
                {
                    for result in results {
                        // Build detection summary
                        let detection_summary = format_detection_summary(&result);

                        // Get image dimensions from result
                        let inference_shape = result.inference_shape();
                        last_inference_shape =
                            (inference_shape.0 as usize, inference_shape.1 as usize);

                        // Format total frames for display
                        let total_frames_str = meta
                            .total_frames
                            .map_or_else(|| "?".to_string(), |n| n.to_string());

                        if is_video {
                            verbose!(
                                "video 1/1 (frame {}/{}) {}: {}x{} {}, {:.1}ms",
                                meta.frame_idx + 1,
                                total_frames_str,
                                image_path,
                                inference_shape.0,
                                inference_shape.1,
                                detection_summary,
                                result.speed.inference.unwrap_or(0.0)
                            );
                        } else {
                            verbose!(
                                "image {}/{} {}: {}x{} {}, {:.1}ms",
                                meta.frame_idx + 1,
                                total_frames_str,
                                image_path,
                                inference_shape.0,
                                inference_shape.1,
                                detection_summary,
                                result.speed.inference.unwrap_or(0.0)
                            );
                        }

                        #[cfg(feature = "annotate")]
                        if save_dir.is_some() {
                            let annotated = annotate_image(img, &result, None);

                            #[allow(clippy::collapsible_if)]
                            if let Some(saver) = &mut result_saver {
                                if let Err(e) = saver.save(is_video, meta, &annotated) {
                                    error!("Failed to save result: {e}");
                                }
                            }
                        }

                        #[cfg(feature = "visualize")]
                        if show {
                            let (orig_w, orig_h) = img.dimensions();
                            let view_width = orig_w as usize;
                            let view_height = orig_h as usize;

                            if let Some(ref v) = viewer
                                && (v.width != view_width || v.height != view_height)
                            {
                                viewer = None;
                            }

                            if viewer.is_none() {
                                viewer = Some(
                                    Viewer::new("Ultralytics Inference", view_width, view_height)
                                        .unwrap(),
                                );
                            }

                            if let Some(ref mut v) = viewer {
                                let annotated = annotate_image(img, &result, None);

                                if v.update(&annotated).is_ok() {
                                    // Main thread is blocking on channel, so visualizer wait is less critical
                                    // but we keep a small wait to allow window events processing
                                    if !is_video {
                                        let _ = v.wait(Duration::from_millis(200));
                                    }
                                }
                            }
                        }

                        total_preprocess += result.speed.preprocess.unwrap_or(0.0);
                        total_inference += result.speed.inference.unwrap_or(0.0);
                        total_postprocess += result.speed.postprocess.unwrap_or(0.0);
                        all_results.push((image_path.clone(), result));
                    }
                }
            },
        );

        // Main thread: consume frames from channel and run inference
        for item in receiver {
            let (img, meta) = match item {
                Ok(val) => val,
                Err(e) => {
                    error!("Error reading source: {e}");
                    break;
                }
            };
            batch_processor.add(img, meta.path.clone(), meta);
        }
        batch_processor.flush();
    }

    #[allow(clippy::collapsible_if)]
    if let Some(saver) = result_saver {
        if let Err(e) = saver.finish() {
            error!("Failed to finish saving: {e}");
        }
    }

    // Print speed summary with inference tensor shape (after letterboxing)
    let num_results = all_results.len().max(1) as f64;
    verbose!(
        "Speed: {:.1}ms preprocess, {:.1}ms inference, {:.1}ms postprocess per image at shape ({}, 3, {}, {})",
        total_preprocess / num_results,
        total_inference / num_results,
        total_postprocess / num_results,
        batch_size,
        last_inference_shape.0,
        last_inference_shape.1
    );

    // Print save directory if --save was used
    #[cfg(feature = "annotate")]
    if let Some(ref dir) = save_dir {
        verbose!("Results saved to {}", dir.display());
    }

    // Print footer
    verbose!("💡 Learn more at https://docs.ultralytics.com/modes/predict");
}

/// Count detections per class and format as summary string (e.g., "4 persons, 1 bus").
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn format_class_counts(
    cls: &ndarray::ArrayView1<'_, f32>,
    count: usize,
    names: &HashMap<usize, String>,
) -> String {
    if count == 0 {
        return String::new();
    }

    // Count detections per class
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for i in 0..count {
        let class_id = cls[i] as usize;
        *counts.entry(class_id).or_insert(0) += 1;
    }

    // Sort by class ID for consistent output
    let mut sorted_counts: Vec<(usize, usize)> = counts.into_iter().collect();
    sorted_counts.sort_by_key(|(class_id, _)| *class_id);

    // Format each class count with pluralization
    let parts: Vec<String> = sorted_counts
        .iter()
        .map(|(class_id, count)| {
            let class_name = names.get(class_id).map_or("object", String::as_str);
            let name = if *count > 1 {
                pluralize(class_name)
            } else {
                class_name.to_string()
            };
            format!("{count} {name}")
        })
        .collect();

    parts.join(", ")
}

/// Format detection summary like "4 persons, 1 bus".
#[allow(clippy::option_if_let_else)]
fn format_detection_summary(result: &Results) -> String {
    if let Some(ref boxes) = result.boxes {
        format_class_counts(&boxes.cls(), boxes.len(), &result.names)
    } else if let Some(ref obb) = result.obb {
        format_class_counts(&obb.cls(), obb.len(), &result.names)
    } else if let Some(ref probs) = result.probs {
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
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::results::{Boxes, Obb, Probs, Results, Speed};
    use ndarray::{Array2, Array3};
    use std::collections::HashMap;

    fn create_names() -> HashMap<usize, String> {
        let mut names = HashMap::new();
        names.insert(0, "person".to_string());
        names.insert(1, "car".to_string());
        names.insert(2, "bus".to_string());
        names.insert(5, "bicycle".to_string());
        names
    }

    fn create_dummy_image() -> Array3<u8> {
        Array3::zeros((100, 100, 3))
    }

    /// Test `format_detection_summary` with boxes - single detection.
    #[test]
    fn test_format_summary_single_box() {
        // Boxes data: [x1, y1, x2, y2, conf, cls] - 6 columns
        let data =
            Array2::from_shape_vec((1, 6), vec![10.0, 10.0, 100.0, 100.0, 0.95, 0.0]).unwrap();
        let boxes = Boxes::new(data, (100, 100));

        let mut result = Results::new(
            create_dummy_image(),
            "test.jpg".to_string(),
            create_names(),
            Speed::default(),
            (640, 640),
        );
        result.boxes = Some(boxes);

        let summary = format_detection_summary(&result);
        assert_eq!(summary, "1 person");
    }

    /// Test `format_detection_summary` with boxes - multiple classes.
    #[test]
    fn test_format_summary_multiple_boxes() {
        // 3 boxes: 2 persons (class 0), 1 bus (class 2)
        let data = Array2::from_shape_vec(
            (3, 6),
            vec![
                10.0, 10.0, 100.0, 100.0, 0.95, 0.0, // person
                20.0, 20.0, 200.0, 200.0, 0.90, 0.0, // person
                30.0, 30.0, 300.0, 300.0, 0.85, 2.0, // bus
            ],
        )
        .unwrap();
        let boxes = Boxes::new(data, (640, 640));

        let mut result = Results::new(
            create_dummy_image(),
            "test.jpg".to_string(),
            create_names(),
            Speed::default(),
            (640, 640),
        );
        result.boxes = Some(boxes);

        let summary = format_detection_summary(&result);
        assert_eq!(summary, "2 persons, 1 bus");
    }

    /// Test `format_detection_summary` with empty boxes.
    #[test]
    fn test_format_summary_empty_boxes() {
        let data = Array2::from_shape_vec((0, 6), vec![]).unwrap();
        let boxes = Boxes::new(data, (640, 640));

        let mut result = Results::new(
            create_dummy_image(),
            "test.jpg".to_string(),
            create_names(),
            Speed::default(),
            (640, 640),
        );
        result.boxes = Some(boxes);

        let summary = format_detection_summary(&result);
        assert!(summary.is_empty());
    }

    /// Test `format_detection_summary` with OBB detections.
    #[test]
    fn test_format_summary_obb() {
        // OBB data: [x, y, w, h, rotation, conf, cls] - 7 columns
        let data = Array2::from_shape_vec(
            (2, 7),
            vec![
                50.0, 50.0, 100.0, 50.0, 0.5, 0.9, 1.0, // car
                150.0, 150.0, 80.0, 40.0, 0.3, 0.8, 1.0, // car
            ],
        )
        .unwrap();
        let obb = Obb::new(data, (640, 640));

        let mut result = Results::new(
            create_dummy_image(),
            "test.jpg".to_string(),
            create_names(),
            Speed::default(),
            (640, 640),
        );
        result.obb = Some(obb);

        let summary = format_detection_summary(&result);
        assert_eq!(summary, "2 cars");
    }

    /// Test `format_detection_summary` with empty OBB.
    #[test]
    fn test_format_summary_empty_obb() {
        let data = Array2::from_shape_vec((0, 7), vec![]).unwrap();
        let obb = Obb::new(data, (640, 640));

        let mut result = Results::new(
            create_dummy_image(),
            "test.jpg".to_string(),
            create_names(),
            Speed::default(),
            (640, 640),
        );
        result.obb = Some(obb);

        let summary = format_detection_summary(&result);
        assert!(summary.is_empty());
    }

    /// Test `format_detection_summary` with classification probs.
    #[test]
    fn test_format_summary_probs() {
        let data = ndarray::Array1::from_vec(vec![0.1, 0.7, 0.15, 0.03, 0.02]);
        let probs = Probs::new(data);

        let mut names = HashMap::new();
        names.insert(0, "cat".to_string());
        names.insert(1, "dog".to_string());
        names.insert(2, "bird".to_string());
        names.insert(3, "fish".to_string());
        names.insert(4, "hamster".to_string());

        let mut result = Results::new(
            create_dummy_image(),
            "test.jpg".to_string(),
            names,
            Speed::default(),
            (224, 224),
        );
        result.probs = Some(probs);

        let summary = format_detection_summary(&result);
        // Top5 should include dog (0.7)
        assert!(summary.contains("dog"));
        assert!(summary.contains("0.70"));
    }

    /// Test `format_detection_summary` with no results (empty result).
    #[test]
    fn test_format_summary_no_results() {
        let result = Results::new(
            create_dummy_image(),
            "test.jpg".to_string(),
            create_names(),
            Speed::default(),
            (640, 640),
        );

        let summary = format_detection_summary(&result);
        assert!(summary.is_empty());
    }

    /// Test `format_detection_summary` with unknown class (uses "object" fallback).
    #[test]
    fn test_format_summary_unknown_class() {
        // Class 99 doesn't exist in names
        let data =
            Array2::from_shape_vec((1, 6), vec![10.0, 10.0, 100.0, 100.0, 0.95, 99.0]).unwrap();
        let boxes = Boxes::new(data, (100, 100));

        let mut result = Results::new(
            create_dummy_image(),
            "test.jpg".to_string(),
            create_names(),
            Speed::default(),
            (640, 640),
        );
        result.boxes = Some(boxes);

        let summary = format_detection_summary(&result);
        assert_eq!(summary, "1 object");
    }
}
