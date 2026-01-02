// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Example script demonstrating how to save predictions from the library.
//!
//! This example covers:
//! 1. Predicting and saving for a single image.
//! 2. Predicting and saving a video source using the automated `predict_source` method.

use std::path::Path;
use ultralytics_inference::{Result, Source, YOLOModel};

fn main() -> Result<()> {
    // 1. Load the model (auto-downloads if missing)
    println!("Loading model...");
    let mut model = YOLOModel::load("yolo11n.onnx")?;

    // 2. Simple Image Prediction & Save
    let image_path = "bus.jpg";
    if Path::new(image_path).exists() {
        println!("Processing {image_path}...");
        // predict_source handles loading, inference, annotation, and saving
        // This will save "bus.jpg" to the current directory (annotated)
        model.predict_source(Source::from(image_path), true, Some(Path::new(".")))?;
        println!("Saved annotated image to ./bus.jpg");
    } else {
        println!("Skipping image test: '{image_path}' not found.");
    }

    // 3. Video Prediction & Saving
    // This demonstrates how to process a video and save the output automatically.
    // The library handles frame iteration, annotation, and video encoding.

    // Use local video file
    let video_path = "people.mp4";
    if Path::new(video_path).exists() {
        println!("\nProcessing video: {video_path}");

        // Create output directory
        std::fs::create_dir_all("runs")?;

        // Run inference and save video
        // This will create 'runs/people.mp4'
        model.predict_source(Source::from(video_path), true, Some(Path::new("runs")))?;

        println!("Saved output video to runs/people.mp4");
    } else {
        println!("Skipping video test: '{video_path}' not found.");
    }

    println!("\nDone! Check the output directory for saved results.");
    Ok(())
}
