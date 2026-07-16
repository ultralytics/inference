//! Basic quickstart: load a model, run inference, print detections.
//!
//! Run it (auto-downloads `yolo26n.onnx` and a sample image on first use):
//!
//! ```bash
//! cargo run --example basic
//! cargo run --example basic -- path/to/image.jpg
//! ```

use ultralytics_inference::YOLOModel;

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the model. A known Ultralytics YOLO26 name is auto-downloaded if the
    // file is not already on disk; model metadata (classes, task, image size) is
    // read automatically from the ONNX file.
    let mut model = YOLOModel::load("yolo26n.onnx")?;

    // Predict on the image path passed as the first argument, or fall back to an
    // auto-downloaded sample image when none is given.
    let results = match std::env::args().nth(1) {
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
