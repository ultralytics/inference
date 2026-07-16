// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Configure inference with `InferenceConfig` before loading the model.
//!
//! ```bash
//! cargo run --example config
//! cargo run --example config -- path/to/image.jpg
//! ```

use ultralytics_inference::{Device, InferenceConfig, YOLOModel};

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build a configuration with the builder methods, then load the model with it.
    let config = InferenceConfig::new()
        .with_confidence(0.5) // keep detections at or above 0.5 confidence
        .with_iou(0.45) // NMS IoU threshold
        .with_imgsz(640, 640) // inference image size
        .with_device(Device::Cpu); // run on CPU

    let mut model = YOLOModel::load_with_config("yolo26n.onnx", config)?;

    let results = match std::env::args().nth(1) {
        Some(path) => model.predict(path)?,
        None => model.predict_default()?,
    };

    for result in &results {
        let Some(boxes) = &result.boxes else { continue };
        println!("Found {} detections", boxes.len());
        for i in 0..boxes.len() {
            let cls = boxes.cls()[i] as usize;
            let conf = boxes.conf()[i];
            let name = result.names.get(&cls).map_or("unknown", String::as_str);
            println!("  {name} {conf:.2}");
        }
    }

    Ok(())
}
