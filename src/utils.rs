// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Utility functions for the inference library

/// Calculate IoU (Intersection over Union) between two bounding boxes
///
/// # Arguments
///
/// * `box1` - First bounding box [x1, y1, x2, y2]
/// * `box2` - Second bounding box [x1, y1, x2, y2]
///
/// # Returns
///
/// IoU value between 0.0 and 1.0
pub fn calculate_iou(box1: &[f32; 4], box2: &[f32; 4]) -> f32 {
    let x1 = box1[0].max(box2[0]);
    let y1 = box1[1].max(box2[1]);
    let x2 = box1[2].min(box2[2]);
    let y2 = box1[3].min(box2[3]);

    let intersection = ((x2 - x1).max(0.0)) * ((y2 - y1).max(0.0));

    let area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    let area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    let union = area1 + area2 - intersection;

    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

/// Non-Maximum Suppression (NMS) for filtering overlapping detections
///
/// # Arguments
///
/// * `boxes` - Vector of bounding boxes with scores [(bbox, score)]
/// * `iou_threshold` - IoU threshold for suppression
///
/// # Returns
///
/// Indices of boxes to keep
pub fn nms(boxes: &[([f32; 4], f32)], iou_threshold: f32) -> Vec<usize> {
    if boxes.is_empty() {
        return vec![];
    }

    // Sort by score (descending)
    let mut indices: Vec<usize> = (0..boxes.len()).collect();
    indices.sort_by(|&a, &b| boxes[b].1.partial_cmp(&boxes[a].1).unwrap());

    let mut keep = vec![];
    let mut suppressed = vec![false; boxes.len()];

    for &i in &indices {
        if suppressed[i] {
            continue;
        }
        keep.push(i);

        for &j in &indices {
            if !suppressed[j] && i != j {
                let iou = calculate_iou(&boxes[i].0, &boxes[j].0);
                if iou > iou_threshold {
                    suppressed[j] = true;
                }
            }
        }
    }

    keep
}

/// Per-class Non-Maximum Suppression (NMS) for filtering overlapping detections
///
/// Only suppresses boxes within the same class, matching Ultralytics behavior.
///
/// # Arguments
///
/// * `boxes` - Vector of bounding boxes with scores and class IDs [(bbox, score, class_id)]
/// * `iou_threshold` - IoU threshold for suppression
///
/// # Returns
///
/// Indices of boxes to keep
pub fn nms_per_class(boxes: &[([f32; 4], f32, usize)], iou_threshold: f32) -> Vec<usize> {
    if boxes.is_empty() {
        return vec![];
    }

    // Sort by score (descending)
    let mut indices: Vec<usize> = (0..boxes.len()).collect();
    indices.sort_by(|&a, &b| boxes[b].1.partial_cmp(&boxes[a].1).unwrap());

    let mut keep = vec![];
    let mut suppressed = vec![false; boxes.len()];

    for &i in &indices {
        if suppressed[i] {
            continue;
        }
        keep.push(i);

        let class_i = boxes[i].2;

        for &j in &indices {
            if !suppressed[j] && i != j {
                // Only suppress boxes of the same class
                if boxes[j].2 == class_i {
                    let iou = calculate_iou(&boxes[i].0, &boxes[j].0);
                    if iou > iou_threshold {
                        suppressed[j] = true;
                    }
                }
            }
        }
    }

    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_iou() {
        let box1 = [0.0, 0.0, 10.0, 10.0];
        let box2 = [5.0, 5.0, 15.0, 15.0];
        let iou = calculate_iou(&box1, &box2);
        assert!((iou - 0.142857).abs() < 0.001); // 25 / (100 + 100 - 25)
    }

    #[test]
    fn test_nms() {
        let boxes = vec![
            ([0.0, 0.0, 10.0, 10.0], 0.9),
            ([1.0, 1.0, 11.0, 11.0], 0.8),
            ([100.0, 100.0, 110.0, 110.0], 0.95),
        ];
        let keep = nms(&boxes, 0.5);
        assert_eq!(keep.len(), 2); // Should keep boxes at indices 2 and 0
        assert!(keep.contains(&2));
        assert!(keep.contains(&0));
    }

    #[test]
    fn test_nms_per_class() {
        // Two overlapping boxes of different classes should both be kept
        let boxes = vec![
            ([0.0, 0.0, 10.0, 10.0], 0.9, 0),        // class 0
            ([1.0, 1.0, 11.0, 11.0], 0.8, 1),        // class 1 (different class)
            ([100.0, 100.0, 110.0, 110.0], 0.95, 0), // class 0, non-overlapping
        ];
        let keep = nms_per_class(&boxes, 0.5);
        // All 3 boxes should be kept (overlapping boxes are different classes)
        assert_eq!(keep.len(), 3);
    }

    #[test]
    fn test_nms_per_class_suppression() {
        // Two overlapping boxes of the same class - lower score suppressed
        let boxes = vec![
            ([0.0, 0.0, 10.0, 10.0], 0.9, 0), // class 0, higher score
            ([1.0, 1.0, 11.0, 11.0], 0.8, 0), // class 0, lower score (suppressed)
        ];
        let keep = nms_per_class(&boxes, 0.5);
        assert_eq!(keep.len(), 1);
        assert!(keep.contains(&0)); // Keep higher score box
    }
}
