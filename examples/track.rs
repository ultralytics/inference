// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Track objects across the frames of a video, like Ultralytics `model.track()`.
//!
//! Takes an optional video path or URL, then an optional tracker name or YAML file. Run it
//! (auto-downloads `yolo26n.onnx` and streams a sample video):
//!
//! ```bash
//! cargo run --example track --features track,video
//! cargo run --example track --features track,video -- path/to/video.mp4 bytetrack.yaml
//! ```

use ultralytics_inference::{Source, SourceIterator, YOLOModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let video = args.next().unwrap_or_else(|| {
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4"
            .into()
    });
    let tracker = args.next().unwrap_or_else(|| "botsort.yaml".to_string());
    let mut model = YOLOModel::load("yolo26n.onnx")?;

    // Tracks carry over from frame to frame, like `persist=True` in Ultralytics.
    for frame in SourceIterator::new(Source::from(video.as_str()))? {
        let (image, meta) = frame?;
        let results = model.track_image(&image, meta.path, &tracker)?;
        let ids = results[0].boxes.as_ref().and_then(|b| b.id());
        println!(
            "frame {}: ids {:?}",
            meta.frame_idx,
            ids.map(|i| i.to_vec())
        );
    }
    Ok(())
}
