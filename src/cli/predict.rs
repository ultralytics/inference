// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

use std::path::Path;
use std::process;
#[cfg(feature = "visualize")]
use std::time::Duration;

#[cfg(feature = "annotate")]
use crate::annotate::annotate_image;
use crate::io::find_next_run_dir;

#[cfg(feature = "visualize")]
use crate::visualizer::Viewer;
#[cfg(feature = "visualize")]
use image::GenericImageView;

use crate::{DISPLAY_NAME, InferenceConfig, Results, VERSION, YOLOModel};

use crate::batch::BatchProcessor;
use crate::cli::args::PredictArgs;
use crate::task::Task;
use crate::{error, verbose, warn};

const DEFAULT_OBB_IMAGES: &[&str] = &[crate::download::DEFAULT_OBB_IMAGE];

/// Run YOLO model inference.
#[cfg_attr(coverage_nightly, coverage(off))]
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
    let (model_path, model_is_default) = resolve_model_path(args);
    let source_path = &args.source;
    let save = args.save;
    let save_frames = args.save_frames;
    let save_json = args.save_json;
    let half = args.half;
    let verbose = args.verbose;
    let batch_size = args.batch as usize;
    let device = parse_device_arg(args.device.as_deref()).unwrap_or_else(|e| {
        error!("{e}");
        process::exit(1);
    });
    #[cfg(feature = "visualize")]
    let show = args.show;
    if model_is_default && verbose {
        warn!("'model' argument is missing. Using default '--model={model_path}'.");
    }

    // Load model first so we can determine appropriate default source based on task
    let config = build_inference_config(args, device).unwrap_or_else(|e| {
        error!("{e}");
        process::exit(1);
    });

    let mut model = match YOLOModel::load_with_config(model_path, config) {
        Ok(m) => m,
        Err(e) => {
            error!("Error loading model: {e}");
            process::exit(1);
        }
    };

    if let Some(task) = args.task {
        if model_is_default {
            // Model was auto-selected from --task, so metadata will always agree.
            // set_task is a no-op here but kept for explicitness.
            model.set_task(task);
        } else if task != model.task() {
            error!(
                "'--task={task}' conflicts with task '{}' detected from model metadata. \
                 Provide a model that matches the requested task, or omit --task.",
                model.task()
            );
            process::exit(1);
        }
        // task == model.task(): explicit model, matching task; nothing to do.
    }

    // Determine source
    let source = source_path.as_ref().map_or_else(
        || {
            // Select default images based on model task
            let default_urls = default_source_urls(model.task());

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

    if save_json && model.task() != crate::task::Task::Semantic {
        warn!(
            "--save-json is currently supported only for semantic segmentation; ignoring for task '{}'.",
            model.task()
        );
    }

    // Determine whether we need an incremented predict dir. `--save` always needs one
    // when annotation support is compiled in; semantic class-map export also needs one.
    let need_predict_dir = needs_predict_dir(save, save_json, model.task());

    let save_dir: Option<std::path::PathBuf> = if need_predict_dir {
        let parent_dir = predict_parent_dir(model.task());
        let dir = find_next_run_dir(parent_dir, "predict");
        if let Err(e) = crate::io::ensure_dir(Path::new(&dir)) {
            error!("Failed to create save directory '{dir}': {e}");
            process::exit(1);
        }
        Some(std::path::PathBuf::from(dir))
    } else {
        None
    };

    // Per-image semantic class maps go in `<save_dir>/results/<stem>.png`.
    let results_dir: Option<std::path::PathBuf> = save_dir.as_ref().and_then(|d| {
        if !needs_results_dir(save_json, model.task()) {
            return None;
        }
        let dir = d.join("results");
        if let Err(e) = crate::io::ensure_dir(&dir) {
            error!(
                "Failed to create results directory '{}': {e}",
                dir.display()
            );
            process::exit(1);
        }
        Some(dir)
    });

    #[cfg(not(feature = "annotate"))]
    if save {
        warn!(
            "--save requires the 'annotate' feature. Compile with --features annotate to enable saving."
        );
    }

    let is_half = model.metadata().half || half;
    let precision = precision_label(is_half);
    let device_str = provider_label(model.execution_provider());
    println!("{DISPLAY_NAME} {VERSION} 🚀 Rust ONNX {precision} {device_str}");
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
            "Video source detected but video support is not enabled. Please compile with '--features video'"
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
        .filter(|_| save)
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

                        if let Some(ref cdir) = results_dir
                            && let Some(ref sm) = result.semantic_mask
                        {
                            // For video/webcam sources every frame shares the same path stem,
                            // so append the frame index to avoid overwriting earlier frames.
                            let stem =
                                semantic_output_stem(image_path, meta.frame_idx, meta.total_frames);
                            let out_path = cdir.join(format!("{stem}.png"));
                            let (h, w) = (sm.data.shape()[0], sm.data.shape()[1]);
                            let max_id = sm.data.iter().copied().max().unwrap_or(0);
                            if max_id > 255 {
                                // Class IDs exceed 8-bit range; save as 16-bit grayscale PNG.
                                warn!(
                                    "Semantic class IDs exceed 255 (max={max_id}); saving 16-bit PNG: {}",
                                    out_path.display()
                                );
                                let buf: Vec<u16> = sm.data.iter().copied().collect();
                                if let Some(img16) =
                                    image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_raw(
                                        w as u32, h as u32, buf,
                                    )
                                    && let Err(e) = img16.save(&out_path)
                                {
                                    error!(
                                        "Failed to save semantic mask '{}': {e}",
                                        out_path.display()
                                    );
                                }
                            } else {
                                let buf: Vec<u8> = sm.data.iter().map(|&v| v as u8).collect();
                                if let Some(gray) =
                                    image::GrayImage::from_raw(w as u32, h as u32, buf)
                                    && let Err(e) = gray.save(&out_path)
                                {
                                    error!(
                                        "Failed to save semantic mask '{}': {e}",
                                        out_path.display()
                                    );
                                }
                            }
                        }

                        #[cfg(feature = "annotate")]
                        if save {
                            let annotated = annotate_image(img, &result, None);

                            if let Some(saver) = &mut result_saver
                                && let Err(e) = saver.save(is_video, meta, &annotated)
                            {
                                error!("Failed to save result: {e}");
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
                                    Viewer::new(DISPLAY_NAME, view_width, view_height).unwrap(),
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

    if let Some(saver) = result_saver
        && let Err(e) = saver.finish()
    {
        error!("Failed to finish saving: {e}");
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

fn resolve_model_path(args: &PredictArgs) -> (String, bool) {
    let model_is_default = args.model.is_none();
    let model_path = args
        .model
        .clone()
        .unwrap_or_else(|| args.task.unwrap_or(Task::Detect).default_model());
    (model_path, model_is_default)
}

fn parse_device_arg(device: Option<&str>) -> Result<Option<crate::Device>, String> {
    device
        .map(|d| d.parse().map_err(|e| format!("Invalid device '{d}': {e}")))
        .transpose()
}

fn build_inference_config(
    args: &PredictArgs,
    device: Option<crate::Device>,
) -> Result<InferenceConfig, String> {
    let mut config = InferenceConfig::new()
        .with_confidence(args.conf)
        .with_iou(args.iou)
        .with_half(args.half)
        .with_batch(args.batch as usize)
        .with_save_frames(args.save_frames)
        .with_rect(args.rect)
        .with_max_det(args.max_det);

    if let Some(sz) = args.imgsz {
        config = config.with_imgsz(sz, sz);
    }

    if let Some(d) = device {
        config = config.with_device(d);
    }

    if let Some(classes_str) = &args.classes {
        let classes = crate::cli::args::parse_classes(classes_str)
            .map_err(|e| format!("Error parsing classes: {e}"))?;
        if !classes.is_empty() {
            config = config.with_classes(classes);
        }
    }

    Ok(config)
}

const fn default_source_urls(task: Task) -> &'static [&'static str] {
    match task {
        Task::Obb => DEFAULT_OBB_IMAGES,
        _ => crate::download::DEFAULT_IMAGES,
    }
}

fn needs_predict_dir(save: bool, save_json: bool, task: Task) -> bool {
    #[cfg(feature = "annotate")]
    {
        save || needs_results_dir(save_json, task)
    }
    #[cfg(not(feature = "annotate"))]
    {
        let _ = save;
        needs_results_dir(save_json, task)
    }
}

const fn predict_parent_dir(task: Task) -> &'static str {
    match task {
        Task::Detect => "runs/detect",
        Task::Segment => "runs/segment",
        Task::Pose => "runs/pose",
        Task::Classify => "runs/classify",
        Task::Obb => "runs/obb",
        Task::Semantic => "runs/semantic",
        Task::Depth => "runs/depth",
    }
}

fn needs_results_dir(save_json: bool, task: Task) -> bool {
    save_json && task == Task::Semantic
}

const fn precision_label(is_half: bool) -> &'static str {
    if is_half { "FP16" } else { "FP32" }
}

fn provider_label(provider: &str) -> &'static str {
    let provider = provider.to_ascii_lowercase();
    if provider.contains("coreml") {
        "CoreML"
    } else if provider.contains("cuda") {
        "CUDA"
    } else if provider.contains("tensorrt") {
        "TensorRT"
    } else if provider.contains("directml") {
        "DirectML"
    } else if provider.contains("rocm") {
        "ROCm"
    } else if provider.contains("openvino") {
        "OpenVINO"
    } else {
        "CPU"
    }
}

fn semantic_output_stem(image_path: &str, frame_idx: usize, total_frames: Option<usize>) -> String {
    let base_stem = std::path::Path::new(image_path)
        .file_stem()
        .map_or_else(|| "frame".to_owned(), |s| s.to_string_lossy().into_owned());
    if total_frames == Some(1) {
        base_stem
    } else {
        format!("{base_stem}_{frame_idx:06}")
    }
}

/// Count detections per class and format as summary string (e.g., "4 persons, 1 bus").
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn format_detection_summary(result: &Results) -> String {
    result.detection_summary()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::results::{Boxes, Obb, Probs, Results, SemanticMask, Speed};
    use ndarray::{Array2, Array3};
    use std::collections::HashMap;
    use std::sync::Arc;

    fn create_names() -> Arc<HashMap<usize, String>> {
        let mut names = HashMap::new();
        names.insert(0, "person".to_string());
        names.insert(1, "car".to_string());
        names.insert(2, "bus".to_string());
        names.insert(5, "bicycle".to_string());
        Arc::new(names)
    }

    fn create_dummy_image() -> Array3<u8> {
        Array3::zeros((100, 100, 3))
    }

    fn predict_args() -> PredictArgs {
        PredictArgs {
            model: None,
            task: None,
            source: None,
            conf: InferenceConfig::DEFAULT_CONF,
            iou: InferenceConfig::DEFAULT_IOU,
            max_det: InferenceConfig::DEFAULT_MAX_DET,
            imgsz: None,
            rect: InferenceConfig::DEFAULT_RECT,
            batch: 1,
            half: InferenceConfig::DEFAULT_HALF,
            save: InferenceConfig::DEFAULT_SAVE,
            save_frames: InferenceConfig::DEFAULT_SAVE_FRAMES,
            save_json: false,
            show: false,
            device: None,
            verbose: true,
            classes: None,
        }
    }

    #[test]
    fn test_resolve_model_path_uses_detect_default() {
        let args = predict_args();

        let (model_path, model_is_default) = resolve_model_path(&args);

        assert_eq!(model_path, "yolo26n.onnx");
        assert!(model_is_default);
    }

    #[test]
    fn test_resolve_model_path_uses_task_default() {
        let args = PredictArgs {
            task: Some(Task::Semantic),
            ..predict_args()
        };

        let (model_path, model_is_default) = resolve_model_path(&args);

        assert_eq!(model_path, "yolo26n-sem.onnx");
        assert!(model_is_default);
    }

    #[test]
    fn test_resolve_model_path_keeps_explicit_model() {
        let args = PredictArgs {
            model: Some("custom.onnx".to_string()),
            task: Some(Task::Pose),
            ..predict_args()
        };

        let (model_path, model_is_default) = resolve_model_path(&args);

        assert_eq!(model_path, "custom.onnx");
        assert!(!model_is_default);
    }

    #[test]
    fn test_parse_device_arg() {
        assert_eq!(parse_device_arg(None).unwrap(), None);
        assert_eq!(
            parse_device_arg(Some("cuda:2")).unwrap(),
            Some(crate::Device::Cuda(2))
        );

        let err = parse_device_arg(Some("neural")).unwrap_err();
        assert!(err.contains("Invalid device 'neural'"));
        assert!(err.contains("Unknown device: neural"));
    }

    #[test]
    fn test_build_inference_config_from_cli_args() {
        let args = PredictArgs {
            conf: 0.42,
            iou: 0.55,
            max_det: 77,
            imgsz: Some(512),
            rect: false,
            batch: 4,
            half: true,
            save_frames: true,
            classes: Some("[0, 2, 5]".to_string()),
            ..predict_args()
        };

        let config = build_inference_config(&args, Some(crate::Device::OpenVino)).unwrap();

        assert!((config.confidence_threshold - 0.42).abs() < f32::EPSILON);
        assert!((config.iou_threshold - 0.55).abs() < f32::EPSILON);
        assert_eq!(config.max_det, 77);
        assert_eq!(config.imgsz, Some((512, 512)));
        assert_eq!(config.batch, Some(4));
        assert!(config.half);
        assert_eq!(config.device, Some(crate::Device::OpenVino));
        assert!(config.save_frames);
        assert!(!config.rect);
        assert_eq!(config.classes, Some(vec![0, 2, 5]));
    }

    #[test]
    fn test_build_inference_config_ignores_empty_class_filter() {
        let args = PredictArgs {
            classes: Some("[]".to_string()),
            ..predict_args()
        };

        let config = build_inference_config(&args, None).unwrap();

        assert!(config.classes.is_none());
    }

    #[test]
    fn test_build_inference_config_rejects_invalid_classes() {
        let args = PredictArgs {
            classes: Some("0,truck".to_string()),
            ..predict_args()
        };

        let err = build_inference_config(&args, None).unwrap_err();

        assert!(err.contains("Error parsing classes"));
        assert!(err.contains("Invalid class ID 'truck'"));
    }

    #[test]
    fn test_default_source_urls_are_task_specific() {
        assert_eq!(
            default_source_urls(Task::Detect),
            crate::download::DEFAULT_IMAGES
        );
        assert_eq!(
            default_source_urls(Task::Segment),
            crate::download::DEFAULT_IMAGES
        );
        assert_eq!(
            default_source_urls(Task::Obb),
            &[crate::download::DEFAULT_OBB_IMAGE]
        );
    }

    #[test]
    fn test_predict_parent_dir_by_task() {
        assert_eq!(predict_parent_dir(Task::Detect), "runs/detect");
        assert_eq!(predict_parent_dir(Task::Segment), "runs/segment");
        assert_eq!(predict_parent_dir(Task::Pose), "runs/pose");
        assert_eq!(predict_parent_dir(Task::Classify), "runs/classify");
        assert_eq!(predict_parent_dir(Task::Obb), "runs/obb");
        assert_eq!(predict_parent_dir(Task::Semantic), "runs/semantic");
        assert_eq!(predict_parent_dir(Task::Depth), "runs/depth");
    }

    #[test]
    fn test_predict_dir_needed_for_saves_and_semantic_json() {
        assert!(needs_predict_dir(false, true, Task::Semantic));
        assert!(!needs_predict_dir(false, true, Task::Detect));
        assert!(needs_results_dir(true, Task::Semantic));
        assert!(!needs_results_dir(true, Task::Segment));
        assert!(!needs_results_dir(false, Task::Semantic));

        #[cfg(feature = "annotate")]
        assert!(needs_predict_dir(true, false, Task::Detect));

        #[cfg(not(feature = "annotate"))]
        assert!(!needs_predict_dir(true, false, Task::Detect));
    }

    #[test]
    fn test_precision_and_provider_labels() {
        assert_eq!(precision_label(false), "FP32");
        assert_eq!(precision_label(true), "FP16");
        assert_eq!(provider_label("CoreMLExecutionProvider"), "CoreML");
        assert_eq!(provider_label("CUDAExecutionProvider"), "CUDA");
        assert_eq!(provider_label("TensorrtExecutionProvider"), "TensorRT");
        assert_eq!(provider_label("TensorRTExecutionProvider"), "TensorRT");
        assert_eq!(provider_label("DirectMLExecutionProvider"), "DirectML");
        assert_eq!(provider_label("ROCmExecutionProvider"), "ROCm");
        assert_eq!(provider_label("OpenVINOExecutionProvider"), "OpenVINO");
        assert_eq!(provider_label("CPUExecutionProvider"), "CPU");
    }

    #[test]
    fn test_semantic_output_stem_single_image_and_frames() {
        assert_eq!(
            semantic_output_stem("images/bus.jpg", 0, Some(1)),
            "bus".to_string()
        );
        assert_eq!(
            semantic_output_stem("stream", 42, Some(100)),
            "stream_000042".to_string()
        );
        assert_eq!(
            semantic_output_stem("", 3, None),
            "frame_000003".to_string()
        );
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
        assert_eq!(summary, "(no detections)");
    }

    #[test]
    fn test_format_summary_semantic_mask() {
        let mut result = Results::new(
            create_dummy_image(),
            "test.jpg".to_string(),
            create_names(),
            Speed::default(),
            (640, 640),
        );
        result.semantic_mask = Some(SemanticMask::new(
            Array2::from_shape_vec((2, 3), vec![0u16, 1, 1, 2, 2, 2]).unwrap(),
            (2, 3),
        ));

        let summary = format_detection_summary(&result);
        assert_eq!(summary, "person, car, bus");
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
        assert_eq!(summary, "(no detections)");
    }

    /// Test `format_detection_summary` with classification probs.
    #[test]
    fn test_format_summary_probs() {
        let data = ndarray::Array1::from_vec(vec![0.1, 0.7, 0.15, 0.03, 0.02]);
        let probs = Probs::new(data);

        let names = Arc::new({
            let mut n = HashMap::new();
            n.insert(0, "cat".to_string());
            n.insert(1, "dog".to_string());
            n.insert(2, "bird".to_string());
            n.insert(3, "fish".to_string());
            n.insert(4, "hamster".to_string());
            n
        });

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
        assert_eq!(summary, "(no detections)");
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
