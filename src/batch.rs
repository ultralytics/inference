// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Batch processing module.
//!
//! This module provides the [`BatchProcessor`] struct, which abstracts the logic for
//! buffering images and running batch inference.

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
///     let mut processor = BatchProcessor::new(&mut model, batch_size, |results, images, paths, metas, offset| {
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
    F: FnMut(Vec<Vec<Results>>, &[DynamicImage], &[String], &[SourceMeta], usize),
{
    model: &'a mut YOLOModel,
    batch_size: usize,
    images: Vec<DynamicImage>,
    paths: Vec<String>,
    metas: Vec<SourceMeta>,
    callback: F,
    frame_count: usize,
}

impl<'a, F> BatchProcessor<'a, F>
where
    F: FnMut(Vec<Vec<Results>>, &[DynamicImage], &[String], &[SourceMeta], usize),
{
    /// Create a new `BatchProcessor`.
    ///
    /// # Arguments
    ///
    /// * `model` - Mutable reference to the `YOLOModel`.
    /// * `batch_size` - The size of the batch.
    /// * `callback` - A closure to handle the results of each batch.
    pub fn new(model: &'a mut YOLOModel, batch_size: usize, callback: F) -> Self {
        Self {
            model,
            batch_size,
            images: Vec::with_capacity(batch_size),
            paths: Vec::with_capacity(batch_size),
            metas: Vec::with_capacity(batch_size),
            callback,
            frame_count: 0,
        }
    }

    /// Add an image to the batch.
    ///
    /// If the batch becomes full, it is automatically processed.
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
    /// This should be called after all images have been added to ensure the last partial batch is processed.
    pub fn flush(&mut self) {
        self.process();
    }

    fn process(&mut self) {
        if self.images.is_empty() {
            return;
        }

        let batch_results = self.run_inference();
        (self.callback)(
            batch_results,
            &self.images,
            &self.paths,
            &self.metas,
            self.frame_count,
        );

        self.frame_count += self.images.len();
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
    use std::cell::RefCell;
    use std::rc::Rc;

    /// Helper to load a test image from assets.
    fn load_test_image() -> DynamicImage {
        // Use bus.jpg which should exist in assets/
        image::open("assets/bus.jpg")
            .or_else(|_| image::open("assets/zidane.jpg"))
            .unwrap_or_else(|_| DynamicImage::new_rgb8(640, 640))
    }

    /// Test that BatchProcessor correctly buffers images until batch_size is reached.
    ///
    /// Note: The default yolo11n.onnx only supports batch=1, so when we use batch_size=2,
    /// the batch inference will fail and fall back to single-image inference.
    /// This test still validates the buffering and callback behavior.
    #[test]
    #[ignore = "requires yolo11n.onnx model file - run with --include-ignored"]
    fn test_batch_processor_with_model() {
        let mut model = YOLOModel::load("yolo11n.onnx").expect("Model should load");

        let callback_count = Rc::new(RefCell::new(0));
        let callback_count_clone = Rc::clone(&callback_count);

        let mut processor = BatchProcessor::new(
            &mut model,
            2,
            move |_results, _images, _paths, _metas, _offset| {
                *callback_count_clone.borrow_mut() += 1;
            },
        );

        // Load real test images
        let img1 = load_test_image();
        let img2 = load_test_image();
        let img3 = load_test_image();

        let meta = SourceMeta {
            path: "test.jpg".to_string(),
            frame_idx: 0,
            total_frames: Some(1),
            fps: None,
        };

        // Add first image - should not trigger callback
        processor.add(img1, "img1.jpg".to_string(), meta.clone());
        assert_eq!(*callback_count.borrow(), 0);

        // Add second image - should trigger callback (batch_size = 2)
        processor.add(img2, "img2.jpg".to_string(), meta.clone());
        assert_eq!(*callback_count.borrow(), 1);

        // Add third image - should not trigger callback
        processor.add(img3, "img3.jpg".to_string(), meta);
        assert_eq!(*callback_count.borrow(), 1);

        // Flush should trigger callback for remaining image
        processor.flush();
        assert_eq!(*callback_count.borrow(), 2);
    }

    /// Test that flush on empty processor does nothing.
    #[test]
    #[ignore = "requires yolo11n.onnx model file - run with --include-ignored"]
    fn test_batch_processor_empty_flush() {
        let mut model = YOLOModel::load("yolo11n.onnx").expect("Model should load");

        let callback_count = Rc::new(RefCell::new(0));
        let callback_count_clone = Rc::clone(&callback_count);

        let mut processor = BatchProcessor::new(
            &mut model,
            2,
            move |_results, _images, _paths, _metas, _offset| {
                *callback_count_clone.borrow_mut() += 1;
            },
        );

        // Flush without adding anything should not call callback
        processor.flush();
        assert_eq!(*callback_count.borrow(), 0);
    }

    /// Test frame_count accumulation across batches.
    ///
    /// Note: Uses batch_size=1 to avoid triggering fallback (since default model is batch=1).
    #[test]
    #[ignore = "requires yolo11n.onnx model file - run with --include-ignored"]
    fn test_batch_processor_frame_count() {
        let mut model = YOLOModel::load("yolo11n.onnx").expect("Model should load");

        let offsets = Rc::new(RefCell::new(Vec::new()));
        let offsets_clone = Rc::clone(&offsets);

        // Use batch_size=1 to work with default model (which only supports batch=1)
        let mut processor = BatchProcessor::new(
            &mut model,
            1,
            move |_results, _images, _paths, _metas, offset| {
                offsets_clone.borrow_mut().push(offset);
            },
        );

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
        let offsets = offsets.borrow();
        assert_eq!(offsets.len(), 3);
        assert_eq!(offsets[0], 0); // First batch starts at 0
        assert_eq!(offsets[1], 1); // Second batch starts at 1
        assert_eq!(offsets[2], 2); // Third batch starts at 2
    }
}
