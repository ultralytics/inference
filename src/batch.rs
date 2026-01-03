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
//! ```no_run
//! use ultralytics_inference::{YOLOModel, batch::BatchProcessor};
//!
//! let mut model = YOLOModel::load("yolo11n.onnx")?;
//! let mut processor = BatchProcessor::new(&mut model, 4, |results, images, paths, metas| {
//!     for (idx, result_vec) in results.iter().enumerate() {
//!         println!("Image {}: {} detections", paths[idx], result_vec.len());
//!     }
//! });
//!
//! // Add images as they become available
//! // processor.add(image, path, meta);
//!
//! // Don't forget to flush remaining images
//! processor.flush();
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::{Results, YOLOModel, source::SourceMeta};
use image::DynamicImage;

/// A processor for handling batch inference.
///
/// This struct manages collecting images into batches, running inference (with fallback),
/// and invoking a callback with the results.
///
/// # Example
///
/// ```no_run
/// use ultralytics_inference::{YOLOModel, batch::BatchProcessor};
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let mut model = YOLOModel::load("yolo11n.onnx")?;
///     let batch_size = 4;
///     
///     let mut processor = BatchProcessor::new(&mut model, batch_size, |results, images, paths, metas| {
///         println!("Processed batch of {} images", results.len());
///     });
///     
///     // Add images...
///     // processor.add(image, path, meta);
///     
///     processor.flush();
///     Ok(())
/// }
/// ```
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

    fn run_inference(&mut self) -> Vec<Vec<Results>> {
        if let Ok(batch_results) = self.model.predict_batch(&self.images, &self.paths) {
            return batch_results;
        }

        eprintln!("WARNING ⚠️ Batch inference failed. Falling back to single-image inference...");

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

    /// Test that `BatchProcessor` correctly buffers images and invokes callback.
    ///
    /// Uses `batch_size=1` since the default yolo11n.onnx model only supports batch=1.
    /// The model is auto-downloaded if not present.
    #[test]
    #[serial]
    fn test_batch_processor_with_model() {
        let mut model = YOLOModel::load("yolo11n.onnx").expect("Model should load");

        let callback_count = Rc::new(RefCell::new(0));
        let callback_count_clone = Rc::clone(&callback_count);

        // Use batch_size=1 since default model only supports batch=1
        let mut processor =
            BatchProcessor::new(&mut model, 1, move |_results, _images, _paths, _metas| {
                *callback_count_clone.borrow_mut() += 1;
            });

        // Load real test images
        let img1 = load_test_image();
        let img2 = load_test_image();

        let meta = SourceMeta {
            path: "test.jpg".to_string(),
            frame_idx: 0,
            total_frames: Some(1),
            fps: None,
        };

        // Add first image - should trigger callback immediately (batch_size=1)
        processor.add(img1, "img1.jpg".to_string(), meta.clone());
        assert_eq!(*callback_count.borrow(), 1);

        // Add second image - should trigger another callback
        processor.add(img2, "img2.jpg".to_string(), meta);
        assert_eq!(*callback_count.borrow(), 2);

        // Flush should not trigger callback (batch is empty)
        processor.flush();
        assert_eq!(*callback_count.borrow(), 2);
    }

    /// Test that flush on empty processor does nothing.
    #[test]
    #[serial]
    fn test_batch_processor_empty_flush() {
        let mut model = YOLOModel::load("yolo11n.onnx").expect("Model should load");

        let callback_count = Rc::new(RefCell::new(0));
        let callback_count_clone = Rc::clone(&callback_count);

        let mut processor =
            BatchProcessor::new(&mut model, 1, move |_results, _images, _paths, _metas| {
                *callback_count_clone.borrow_mut() += 1;
            });

        // Flush without adding anything should not call callback
        processor.flush();
        assert_eq!(*callback_count.borrow(), 0);
    }

    /// Test that callback is invoked correct number of times with results.
    #[test]
    #[serial]
    fn test_batch_processor_callback_count() {
        let mut model = YOLOModel::load("yolo11n.onnx").expect("Model should load");

        let count = Rc::new(RefCell::new(0));
        let count_clone = Rc::clone(&count);

        // Use `batch_size=1` to work with default model (which only supports batch=1)
        let mut processor =
            BatchProcessor::new(&mut model, 1, move |_results, _images, _paths, _metas| {
                *count_clone.borrow_mut() += 1;
            });

        let meta = SourceMeta {
            path: "test.jpg".to_string(),
            frame_idx: 0,
            total_frames: Some(1),
            fps: None,
        };

        // Add 3 images with batch_size=1
        for i in 0..3 {
            let img = load_test_image();
            processor.add(img, format!("img{i}.jpg"), meta.clone());
        }
        processor.flush();

        // Should have 3 callbacks (one per image since batch_size=1)
        assert_eq!(*count.borrow(), 3);
    }
}
