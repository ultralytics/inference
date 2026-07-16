// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Annotate detections onto the image and save the result to disk.
//!
//! Requires the `annotate` feature (enabled by default):
//!
//! ```bash
//! cargo run --example annotate --features annotate
//! cargo run --example annotate --features annotate -- path/to/image.jpg
//! ```

use std::path::Path;

use ultralytics_inference::YOLOModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the model (auto-downloads a known Ultralytics YOLO26 model if missing).
    let mut model = YOLOModel::load("yolo26n.onnx")?;

    // Predict on the given image path, or an auto-downloaded sample when omitted.
    let results = match std::env::args().nth(1) {
        Some(path) => model.predict(path)?,
        None => model.predict_default()?,
    };

    // Draw boxes/labels onto each result and write it out. `Results::save`
    // annotates the original image and saves it in one call.
    let out_dir = Path::new("runs/predict");
    std::fs::create_dir_all(out_dir)?;
    for (i, result) in results.iter().enumerate() {
        let out = out_dir.join(format!("annotated_{i}.jpg"));
        result.save(&out)?;
        println!("Saved {}", out.display());
    }

    Ok(())
}
