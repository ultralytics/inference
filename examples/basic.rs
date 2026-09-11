// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Basic quickstart: load a model, run inference, print detections.
//!
//! Takes an optional image path, then an optional model path. Run it
//! (auto-downloads `yolo26n.onnx` and a sample image on first use):
//!
//! ```bash
//! cargo run --example basic
//! cargo run --example basic -- path/to/image.jpg
//!
//! # a supported model, e.g. RT-DETR after `yolo export model=rtdetr-l.pt format=onnx`
//! cargo run --example basic -- path/to/image.jpg rtdetr-l.onnx
//! ```

use ultralytics_inference::YOLOModel;

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the model named by the second argument, else the default `yolo26n.onnx`, a known
    // Ultralytics name that is auto-downloaded when not already on disk. Model metadata (classes, task,
    // image size) is read from the file either way, so an RT-DETR export needs no extra
    // handling here.
    let mut args = std::env::args().skip(1);
    let image_path = args.next();
    let model_path = args.next().unwrap_or_else(|| "yolo26n.onnx".to_string());
    let mut model = YOLOModel::load(&model_path)?;

    // Predict on the image path passed as the first argument, or fall back to an
    // auto-downloaded sample image when none is given.
    let results = match image_path {
        Some(path) => model.predict(path)?,
        None => model.predict_default()?,
    };

    // Print every detection as: <name> <conf> [x1 y1 x2 y2].
    for result in &results {
        let Some(boxes) = &result.boxes else { continue };
        println!("Found {} detections", boxes.len());
        let xyxy = boxes.xyxy();
        for i in 0..boxes.len() {
            let cls = boxes.cls()[i] as usize;
            let conf = boxes.conf()[i];
            let name = result.names.get(&cls).map_or("unknown", String::as_str);
            let b = xyxy.row(i);
            println!(
                "  {name} {conf:.2} [{:.1} {:.1} {:.1} {:.1}]",
                b[0], b[1], b[2], b[3]
            );
        }
    }

    Ok(())
}
