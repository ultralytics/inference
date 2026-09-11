// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Run an RT-DETR model.
//!
//! RT-DETR has no prebuilt ONNX to download, so export one first:
//!
//! ```bash
//! pip install -U "ultralytics[export-base]"
//! yolo export model=rtdetr-l.pt format=onnx
//! ```
//!
//! Then point this example at the exported file. The model path defaults to
//! `rtdetr-l.onnx` and the image to an auto-downloaded sample:
//!
//! ```bash
//! cargo run --example rtdetr
//! cargo run --example rtdetr -- rtdetr-x.onnx path/to/image.jpg
//! ```
//!
//! Nothing here is RT-DETR specific: the scale-fill input and the normalized
//! decoder output are picked up from the model's own metadata, so the code is
//! the same as for a YOLO model.

use ultralytics_inference::YOLOModel;

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().unwrap_or_else(|| "rtdetr-l.onnx".to_string());

    let mut model = YOLOModel::load(&model_path)?;
    println!(
        "{} on {}: {} classes",
        model.metadata().model_name(),
        model.execution_provider(),
        model.metadata().num_classes()
    );

    let results = match args.next() {
        Some(image) => model.predict(image)?,
        None => model.predict_default()?,
    };

    for result in &results {
        let Some(boxes) = &result.boxes else { continue };
        println!("{}: {} detections", result.path, boxes.len());
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
