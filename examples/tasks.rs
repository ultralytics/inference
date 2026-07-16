// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Print a short, task-appropriate summary for any YOLO model.
//!
//! The default model is detection. Pass another task's ONNX to try it:
//!
//! ```bash
//! cargo run --example tasks                        # detect (default)
//! cargo run --example tasks -- yolo26n-seg.onnx    # segment
//! cargo run --example tasks -- yolo26n-pose.onnx   # pose
//! cargo run --example tasks -- yolo26n-obb.onnx    # obb
//! cargo run --example tasks -- yolo26n-cls.onnx    # classify
//! cargo run --example tasks -- yolo26n-sem.onnx    # semantic (YOLO26)
//! cargo run --example tasks -- yolo26n-depth.onnx  # depth (YOLO26)
//! ```
//!
//! The segment, semantic, and depth branches also print the raw output array.
//! `ndarray` truncates large arrays automatically, showing the corners with `...`.

use ultralytics_inference::YOLOModel;

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A known model name is auto-downloaded if it is not already on disk.
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "yolo26n.onnx".to_string());
    let mut model = YOLOModel::load(&model_path)?;

    // predict_default downloads the sample image that matches the model's task.
    let results = model.predict_default()?;

    // Each task fills a different field of Results. Check the specific ones first,
    // since segment and pose models also carry boxes.
    for result in &results {
        if let Some(masks) = &result.masks {
            println!("segment: {} instance masks", masks.len());
            // masks.data is Array3<f32> of shape [N, H, W] (per-pixel mask probabilities).
            println!("  raw data shape {:?}", masks.data.shape());
            println!("{:?}", masks.data);
        } else if let Some(keypoints) = &result.keypoints {
            println!("pose: {} sets of keypoints", keypoints.len());
        } else if let Some(obb) = &result.obb {
            println!("obb: {} oriented boxes", obb.len());
        } else if let Some(probs) = &result.probs {
            let top1 = probs.top1();
            let name = result.names.get(&top1).map_or("unknown", String::as_str);
            println!("classify: top1 {name} {:.2}", probs.top1conf());
        } else if let Some(sem) = &result.semantic_mask {
            println!("semantic: class map shape {:?}", sem.data.shape());
            // sem.data is Array2<u16> of shape [H, W] (per-pixel class ids).
            println!("{:?}", sem.data);
        } else if let Some(depth) = &result.depth {
            println!(
                "depth: map shape {:?}, range {:?}..{:?}",
                depth.data.shape(),
                depth.min_depth(),
                depth.max_depth()
            );
            // depth.data is Array2<f32> of shape [H, W] (per-pixel meters).
            println!("{:?}", depth.data);
        } else if let Some(boxes) = &result.boxes {
            println!("detect: {} objects", boxes.len());
            for i in 0..boxes.len() {
                let cls = boxes.cls()[i] as usize;
                let name = result.names.get(&cls).map_or("unknown", String::as_str);
                println!("  {name} {:.2}", boxes.conf()[i]);
            }
        }
    }

    Ok(())
}
