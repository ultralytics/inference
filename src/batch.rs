// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Batch processing module for YOLO inference.
//!
//! This module provides the [`BatchProcessor`] struct, which abstracts the logic for
//! buffering images and running batch inference. It handles:
//!
//! - **Buffering**: Collects images until the batch size is reached
//! - **Batch inference**: Runs inference on the full batch
//! - **Automatic fallback**: Falls back to single-image inference if batch fails
//! - **Callback invocation**: Invokes a user-provided callback with results
//!
//! # Usage
//!
//! Feed a [`Source`](crate::Source) through the processor to run over a directory, a glob,
//! a video, or a webcam. The callback sees one batch at a time and the buffers are cleared
//! as soon as it returns, so the processor's own memory does not grow with the length of
//! the input; only what the callback keeps is retained. That makes it the right shape for
//! a long video or an open-ended stream, where collecting every frame's [`Results`] would
//! not be.
//!
//! The directory source below runs on the default features. A video, webcam, or RTSP source
//! additionally needs `--features video` and `FFmpeg`; without it the iterator yields
//! [`InferenceError::FeatureNotEnabled`](crate::InferenceError::FeatureNotEnabled) on the
//! first frame.
//!
//! ```no_run
//! use ultralytics_inference::{Source, SourceIterator, YOLOModel, batch::BatchProcessor};
//!
//! let mut model = YOLOModel::load("yolo26n.onnx")?;
//! let mut detections = 0usize;
//!
//! // The processor borrows the model, so scope it to release the borrow afterwards.
//! {
//!     let mut processor = BatchProcessor::new(&mut model, 4, |results, _images, paths, _metas| {
//!         for (path, per_image) in paths.iter().zip(&results) {
//!             for result in per_image {
//!                 let n = result.boxes.as_ref().map_or(0, ultralytics_inference::Boxes::len);
//!                 println!("{path}: {n} detections");
//!                 detections += n;
//!             }
//!         }
//!     });
//!
//!     for frame in SourceIterator::new(Source::from("images"))? {
//!         let (image, meta) = frame?;
//!         processor.add(image, meta.path.clone(), meta);
//!     }
//!
//!     // Runs the last partial batch; without this its images are never inferred.
//!     processor.flush();
//! }
//!
//! println!("{detections} total");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::{Results, YOLOModel, source::SourceMeta};
use image::DynamicImage;

/// A processor for handling batch inference.
///
/// This struct manages collecting images into batches, running inference (with fallback),
/// and invoking a callback with the results.
///
/// Images are dropped once their batch has been handed to the callback, so the buffers cost
/// one batch rather than growing with the input. See the [module docs](self) for a complete
/// source-to-callback loop.
pub struct BatchProcessor<'a, F>
where
    F: FnMut(Vec<Vec<Results>>, &[DynamicImage], &[String], &[SourceMeta]),
{
    model: &'a mut YOLOModel,
    batch_size: usize,
    images: Vec<DynamicImage>,
    paths: Vec<String>,
    metas: Vec<SourceMeta>,
    callback: F,
}

impl<'a, F> BatchProcessor<'a, F>
where
    F: FnMut(Vec<Vec<Results>>, &[DynamicImage], &[String], &[SourceMeta]),
{
    /// Create a new `BatchProcessor`.
    ///
    /// # Arguments
    ///
    /// * `model` - Mutable reference to the [`YOLOModel`] for inference.
    /// * `batch_size` - Maximum number of images to collect before processing.
    /// * `callback` - Closure invoked with batch results. Receives:
    ///   - `Vec<Vec<Results>>` - Results for each image in the batch
    ///   - `&[DynamicImage]` - The batch images
    ///   - `&[String]` - Paths for each image
    ///   - `&[SourceMeta]` - Metadata for each image
    ///
    /// # Returns
    ///
    /// A new `BatchProcessor` instance.
    pub fn new(model: &'a mut YOLOModel, batch_size: usize, callback: F) -> Self {
        Self {
            model,
            batch_size,
            images: Vec::with_capacity(batch_size),
            paths: Vec::with_capacity(batch_size),
            metas: Vec::with_capacity(batch_size),
            callback,
        }
    }

    /// Add an image to the batch.
    ///
    /// If the batch becomes full (reaches `batch_size`), it is automatically processed
    /// and the callback is invoked.
    ///
    /// # Arguments
    ///
    /// * `image` - The image to add.
    /// * `path` - Path or identifier for this image.
    /// * `meta` - Source metadata for this image.
    pub fn add(&mut self, image: DynamicImage, path: String, meta: SourceMeta) {
        self.images.push(image);
        self.paths.push(path);
        self.metas.push(meta);

        if self.images.len() >= self.batch_size {
            self.process();
        }
    }

    /// Process any remaining images in the batch.
    ///
    /// This should be called after all images have been added to ensure
    /// the last partial batch is processed. Has no effect if the batch is empty.
    pub fn flush(&mut self) {
        self.process();
    }

    /// Run the buffered batch through the model, hand it to the callback, and clear the buffers.
    fn process(&mut self) {
        if self.images.is_empty() {
            return;
        }

        let batch_results = self.run_inference();
        (self.callback)(batch_results, &self.images, &self.paths, &self.metas);

        self.images.clear();
        self.paths.clear();
        self.metas.clear();
    }

    /// Batch-infer the buffered images, falling back to one image at a time on failure.
    fn run_inference(&mut self) -> Vec<Vec<Results>> {
        if let Ok(batch_results) = self.model.predict_batch(&self.images, &self.paths) {
            return batch_results;
        }

        crate::warn!("Batch inference failed. Falling back to single-image inference...");

        let mut fallback_results = Vec::with_capacity(self.images.len());
        for (idx, img) in self.images.iter().enumerate() {
            let path = &self.paths[idx];
            match self.model.predict_image(img, path.clone()) {
                Ok(results) => fallback_results.push(results),
                Err(e) => {
                    eprintln!("Error processing {path}: {e}");
                    fallback_results.push(Vec::new());
                }
            }
        }
        fallback_results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// Helper to load a test image from assets.
    fn load_test_image() -> DynamicImage {
        // Use bus.jpg which should exist in assets/
        image::open("assets/bus.jpg")
            .or_else(|_| image::open("assets/zidane.jpg"))
            .unwrap_or_else(|_| DynamicImage::new_rgb8(640, 640))
    }

    /// Buffering, the per-batch callback, and the empty-flush no-op, against one session.
    ///
    /// `YOLOModel` wraps an ORT session and cannot be mocked, so this skips only when
    /// `yolo26n.onnx` is both absent and not downloadable; the default suite then runs
    /// offline while CI, which has network, still exercises it. `batch_size` is 1 because
    /// the default `yolo26n.onnx` only supports batch 1.
    #[test]
    #[serial]
    fn test_batch_processor_buffers_and_flushes() {
        let path = std::path::Path::new("yolo26n.onnx");
        let mut model = match YOLOModel::load(path) {
            Ok(model) => model,
            // Still absent after the attempt means the download could not happen, which is
            // the offline case worth skipping. `download_file` stages through a `.part` file
            // and removes it on failure, so a present file here is a real one: a corrupt
            // model, or a session build that failed after a successful download, must fail
            // the test rather than silently skip every assertion below.
            Err(_) if !path.exists() => return,
            Err(e) => panic!("yolo26n.onnx is present but failed to load: {e}"),
        };

        let calls = Rc::new(RefCell::new(0usize));
        let counter = Rc::clone(&calls);
        let mut processor =
            BatchProcessor::new(&mut model, 1, move |_results, _images, _paths, _metas| {
                *counter.borrow_mut() += 1;
            });

        let meta = SourceMeta {
            path: "test.jpg".to_string(),
            frame_idx: 0,
            total_frames: Some(1),
            fps: None,
        };

        // Nothing buffered, so the flush must not reach the callback.
        processor.flush();
        assert_eq!(*calls.borrow(), 0);

        // batch_size is 1, so every add fills the batch and fires the callback immediately.
        for i in 1..=3usize {
            processor.add(load_test_image(), format!("img{i}.jpg"), meta.clone());
            assert_eq!(*calls.borrow(), i);
        }

        // The last add emptied the buffer again, so this flush is another no-op.
        processor.flush();
        assert_eq!(*calls.borrow(), 3);
    }
}
