// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

use axum::{
    Router,
    extract::{Multipart, Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::env;
use std::sync::Arc;
use tokio::sync::Mutex;
use ultralytics_inference::YOLOModel;
use utoipa::{IntoParams, OpenApi, ToSchema};
use utoipa_swagger_ui::SwaggerUi;

// Shared application state
struct AppState {
    model: Mutex<YOLOModel>,
}

// Query parameters for inference configuration
#[derive(Debug, Deserialize, IntoParams)]
struct PredictParams {
    /// Confidence threshold (0.0 - 1.0). Default: 0.25
    #[param(minimum = 0.0, maximum = 1.0, example = 0.25)]
    conf: Option<f32>,
    /// Maximum number of detections to return. Default: 300
    #[param(minimum = 1, maximum = 10000, example = 300)]
    max_det: Option<usize>,
}

// Detection result
#[derive(Serialize, ToSchema)]
struct Detection {
    /// Class ID (0-79 for COCO)
    class_id: usize,
    /// Human-readable class name
    class_name: String,
    /// Detection confidence (0.0 - 1.0)
    confidence: f32,
    /// Bounding box [x1, y1, x2, y2] in pixels
    bbox: [f32; 4],
}

// Keypoint for pose estimation
#[derive(Serialize, ToSchema)]
struct KeypointData {
    /// X coordinate in pixels
    x: f32,
    /// Y coordinate in pixels
    y: f32,
    /// Keypoint confidence (0.0 - 1.0)
    confidence: f32,
}

// Pose result (detection + keypoints)
#[derive(Serialize, ToSchema)]
struct PoseResult {
    /// Class ID
    class_id: usize,
    /// Human-readable class name
    class_name: String,
    /// Detection confidence
    confidence: f32,
    /// Bounding box [x1, y1, x2, y2]
    bbox: [f32; 4],
    /// 17 body keypoints (COCO format)
    keypoints: Vec<KeypointData>,
}

// Segmentation result
#[derive(Serialize, ToSchema)]
struct SegmentationResult {
    /// Class ID
    class_id: usize,
    /// Human-readable class name
    class_name: String,
    /// Detection confidence
    confidence: f32,
    /// Bounding box [x1, y1, x2, y2]
    bbox: [f32; 4],
    /// Mask dimensions [height, width]
    mask_shape: [usize; 2],
}

// Classification result
#[derive(Serialize, ToSchema)]
struct ClassificationResult {
    /// Top-1 predicted class ID
    top1_class_id: usize,
    /// Top-1 predicted class name
    top1_class_name: String,
    /// Top-1 confidence
    top1_confidence: f32,
    /// Top-5 predictions as (class_id, class_name, confidence)
    top5: Vec<(usize, String, f32)>,
}

// OBB result (oriented bounding box)
#[derive(Serialize, ToSchema)]
struct ObbResult {
    /// Class ID
    class_id: usize,
    /// Human-readable class name
    class_name: String,
    /// Detection confidence
    confidence: f32,
    /// 4 corner points of the rotated bounding box [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    xyxyxyxy: [[f32; 2]; 4],
}

// Unified response that handles all task types
#[derive(Serialize, ToSchema)]
struct PredictResponse {
    /// Model task type: Detect, Segment, Pose, Classify, or Obb
    task: String,
    /// Inference time in milliseconds
    inference_time_ms: f32,
    /// Confidence threshold used for filtering
    conf_threshold: f32,
    /// Number of results returned
    count: usize,
    /// Detection results (for Detect task)
    #[serde(skip_serializing_if = "Option::is_none")]
    detections: Option<Vec<Detection>>,
    /// Pose estimation results (for Pose task)
    #[serde(skip_serializing_if = "Option::is_none")]
    poses: Option<Vec<PoseResult>>,
    /// Segmentation results (for Segment task)
    #[serde(skip_serializing_if = "Option::is_none")]
    segmentations: Option<Vec<SegmentationResult>>,
    /// Classification result (for Classify task)
    #[serde(skip_serializing_if = "Option::is_none")]
    classification: Option<ClassificationResult>,
    /// Oriented bounding box results (for Obb task)
    #[serde(skip_serializing_if = "Option::is_none")]
    obb_detections: Option<Vec<ObbResult>>,
}

#[derive(Serialize, ToSchema)]
struct ErrorResponse {
    /// Error message
    error: String,
}

#[derive(Serialize, ToSchema)]
struct InfoResponse {
    /// Path to the loaded model
    model_path: String,
    /// Model task type
    task: String,
    /// Number of classes
    num_classes: usize,
    /// Input image size (height, width)
    imgsz: (usize, usize),
}

#[derive(Serialize, ToSchema)]
struct HealthResponse {
    /// Server status
    status: String,
    /// API version
    version: String,
}

// OpenAPI Documentation
#[derive(OpenApi)]
#[openapi(
    info(
        title = "Ultralytics Inference Server",
        description = "High-performance YOLO inference API for detection, segmentation, pose estimation, classification, and OBB tasks.\n\n## Query Parameters\nFilter results with:\n- `conf`: Confidence threshold (0.0-1.0, default: 0.25)\n- `max_det`: Maximum detections (default: 300)",
        version = "0.1.0",
        license(name = "AGPL-3.0", url = "https://github.com/ultralytics/inference/blob/main/LICENSE"),
        contact(name = "Ultralytics", url = "https://ultralytics.com")
    ),
    paths(root, health, info, predict),
    components(schemas(
        Detection,
        KeypointData,
        PoseResult,
        SegmentationResult,
        ClassificationResult,
        ObbResult,
        PredictResponse,
        ErrorResponse,
        InfoResponse,
        HealthResponse
    )),
    tags(
        (name = "inference", description = "YOLO model inference endpoints"),
        (name = "health", description = "Health check endpoints")
    )
)]
struct ApiDoc;

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Get model path from environment variable (default: yolo11n.onnx for detection)
    let model_path = env::var("MODEL_PATH").unwrap_or_else(|_| "yolo11n.onnx".to_string());

    println!("Loading model: {}", model_path);

    // Load the model
    let model = YOLOModel::load(&model_path)
        .unwrap_or_else(|_| panic!("Failed to load model: {}", model_path));

    println!(
        "Model loaded - Task: {:?}, Classes: {}",
        model.task(),
        model.num_classes()
    );

    let state = Arc::new(AppState {
        model: Mutex::new(model),
    });

    // Build our application with routes
    let app = Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .route("/info", get(info))
        .route("/predict", post(predict))
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .with_state(state);

    // Get port from environment (default: 3000)
    let port = env::var("PORT").unwrap_or_else(|_| "3000".to_string());
    let addr = format!("0.0.0.0:{}", port);

    // Run our app
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    tracing::info!("Server listening on {}", listener.local_addr().unwrap());
    println!("Server listening on {}", listener.local_addr().unwrap());
    println!(
        "Swagger UI available at http://localhost:{}/swagger-ui/",
        port
    );
    axum::serve(listener, app).await.unwrap();
}

/// Root endpoint
///
/// Returns a welcome message and API information.
#[utoipa::path(
    get,
    path = "/",
    tag = "health",
    responses(
        (status = 200, description = "Welcome message", body = String)
    )
)]
async fn root() -> &'static str {
    "Ultralytics Inference Server - POST /predict with image file. Swagger UI at /swagger-ui/"
}

/// Health check endpoint
///
/// Returns server health status.
#[utoipa::path(
    get,
    path = "/health",
    tag = "health",
    responses(
        (status = 200, description = "Server is healthy", body = HealthResponse)
    )
)]
async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: "0.1.0".to_string(),
    })
}

/// Model information endpoint
///
/// Returns information about the loaded model including task type, number of classes, and input size.
#[utoipa::path(
    get,
    path = "/info",
    tag = "inference",
    responses(
        (status = 200, description = "Model information", body = InfoResponse)
    )
)]
async fn info(State(state): State<Arc<AppState>>) -> Json<InfoResponse> {
    let model = state.model.lock().await;
    Json(InfoResponse {
        model_path: model.model_path().to_string(),
        task: format!("{:?}", model.task()),
        num_classes: model.num_classes(),
        imgsz: model.imgsz(),
    })
}

/// Run inference on an image
///
/// Upload an image file and get predictions based on the loaded model.
/// The response format depends on the model task type (Detect, Segment, Pose, Classify, Obb).
///
/// ## Query Parameters
/// - `conf`: Confidence threshold to filter results (0.0-1.0, default: 0.25)
/// - `max_det`: Maximum number of detections to return (default: 300)
#[utoipa::path(
    post,
    path = "/predict",
    tag = "inference",
    params(PredictParams),
    request_body(content_type = "multipart/form-data", description = "Image file to analyze"),
    responses(
        (status = 200, description = "Inference successful", body = PredictResponse),
        (status = 400, description = "Bad request - invalid image or missing field", body = ErrorResponse),
        (status = 500, description = "Internal server error", body = ErrorResponse)
    )
)]
async fn predict(
    State(state): State<Arc<AppState>>,
    Query(params): Query<PredictParams>,
    mut multipart: Multipart,
) -> Result<Json<PredictResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Get filtering parameters
    let conf_threshold = params.conf.unwrap_or(0.25);
    let max_det = params.max_det.unwrap_or(300);

    // Extract image from multipart form
    while let Ok(Some(field)) = multipart.next_field().await {
        if field.name() == Some("image") {
            let data = match field.bytes().await {
                Ok(bytes) => bytes,
                Err(e) => {
                    return Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ErrorResponse {
                            error: format!("Failed to read field: {}", e),
                        }),
                    ));
                }
            };

            // Decode image from memory
            let img = match image::load_from_memory(&data) {
                Ok(img) => img,
                Err(e) => {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: format!("Invalid image: {}", e),
                        }),
                    ));
                }
            };

            // Run inference
            let mut model = state.model.lock().await;
            let task = model.task();

            let results = match model.predict_image(&img, "upload".to_string()) {
                Ok(r) => r,
                Err(e) => {
                    return Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ErrorResponse {
                            error: format!("Inference failed: {}", e),
                        }),
                    ));
                }
            };

            let result = &results[0];
            let inference_time_ms = result.speed.inference.unwrap_or(0.0) as f32;

            // Build response based on task type with filtering
            let mut response = PredictResponse {
                task: format!("{:?}", task),
                inference_time_ms,
                conf_threshold,
                count: 0,
                detections: None,
                poses: None,
                segmentations: None,
                classification: None,
                obb_detections: None,
            };

            match task {
                ultralytics_inference::Task::Detect => {
                    if let Some(ref boxes) = result.boxes {
                        let mut detections = Vec::new();
                        for i in 0..boxes.len() {
                            let conf = boxes.conf()[i];
                            if conf < conf_threshold {
                                continue;
                            }
                            if detections.len() >= max_det {
                                break;
                            }
                            let cls = boxes.cls()[i] as usize;
                            let bbox = boxes.xyxy();
                            detections.push(Detection {
                                class_id: cls,
                                class_name: result.names.get(&cls).cloned().unwrap_or_default(),
                                confidence: conf,
                                bbox: [bbox[[i, 0]], bbox[[i, 1]], bbox[[i, 2]], bbox[[i, 3]]],
                            });
                        }
                        response.count = detections.len();
                        response.detections = Some(detections);
                    }
                }
                ultralytics_inference::Task::Segment => {
                    if let Some(ref boxes) = result.boxes {
                        let mut segmentations = Vec::new();
                        for i in 0..boxes.len() {
                            let conf = boxes.conf()[i];
                            if conf < conf_threshold {
                                continue;
                            }
                            if segmentations.len() >= max_det {
                                break;
                            }
                            let cls = boxes.cls()[i] as usize;
                            let bbox = boxes.xyxy();

                            // Get mask shape if available
                            let mask_shape = if let Some(ref masks) = result.masks {
                                let shape = masks.data.shape();
                                [shape[1], shape[2]]
                            } else {
                                [0, 0]
                            };

                            segmentations.push(SegmentationResult {
                                class_id: cls,
                                class_name: result.names.get(&cls).cloned().unwrap_or_default(),
                                confidence: conf,
                                bbox: [bbox[[i, 0]], bbox[[i, 1]], bbox[[i, 2]], bbox[[i, 3]]],
                                mask_shape,
                            });
                        }
                        response.count = segmentations.len();
                        response.segmentations = Some(segmentations);
                    }
                }
                ultralytics_inference::Task::Pose => {
                    if let Some(ref boxes) = result.boxes {
                        let mut poses = Vec::new();
                        for i in 0..boxes.len() {
                            let conf = boxes.conf()[i];
                            if conf < conf_threshold {
                                continue;
                            }
                            if poses.len() >= max_det {
                                break;
                            }
                            let cls = boxes.cls()[i] as usize;
                            let bbox = boxes.xyxy();

                            // Extract keypoints for this detection
                            let mut keypoints = Vec::new();
                            if let Some(ref kpts) = result.keypoints {
                                let xy = kpts.xy();
                                let kpts_conf = kpts.conf(); // Returns Option<Array2<f32>>
                                let num_keypoints = xy.shape()[1];

                                for k in 0..num_keypoints {
                                    let kpt_conf = kpts_conf.as_ref().map_or(1.0, |c| c[[i, k]]);
                                    keypoints.push(KeypointData {
                                        x: xy[[i, k, 0]],
                                        y: xy[[i, k, 1]],
                                        confidence: kpt_conf,
                                    });
                                }
                            }

                            poses.push(PoseResult {
                                class_id: cls,
                                class_name: result.names.get(&cls).cloned().unwrap_or_default(),
                                confidence: conf,
                                bbox: [bbox[[i, 0]], bbox[[i, 1]], bbox[[i, 2]], bbox[[i, 3]]],
                                keypoints,
                            });
                        }
                        response.count = poses.len();
                        response.poses = Some(poses);
                    }
                }
                ultralytics_inference::Task::Classify => {
                    if let Some(ref probs) = result.probs {
                        let top1 = probs.top1();
                        let top5_indices = probs.top5();
                        let top5_confs = probs.top5conf();

                        let top5: Vec<(usize, String, f32)> = top5_indices
                            .iter()
                            .zip(top5_confs.iter())
                            .map(|(&idx, &c)| {
                                (idx, result.names.get(&idx).cloned().unwrap_or_default(), c)
                            })
                            .collect();

                        response.count = 1;
                        response.classification = Some(ClassificationResult {
                            top1_class_id: top1,
                            top1_class_name: result.names.get(&top1).cloned().unwrap_or_default(),
                            top1_confidence: probs.top1conf(),
                            top5,
                        });
                    }
                }
                ultralytics_inference::Task::Obb => {
                    if let Some(ref obb) = result.obb {
                        let mut obb_detections = Vec::new();
                        for i in 0..obb.len() {
                            let conf = obb.conf()[i];
                            if conf < conf_threshold {
                                continue;
                            }
                            if obb_detections.len() >= max_det {
                                break;
                            }
                            let cls = obb.cls()[i] as usize;
                            let corners = obb.xyxyxyxy();

                            obb_detections.push(ObbResult {
                                class_id: cls,
                                class_name: result.names.get(&cls).cloned().unwrap_or_default(),
                                confidence: conf,
                                xyxyxyxy: [
                                    [corners[[i, 0, 0]], corners[[i, 0, 1]]],
                                    [corners[[i, 1, 0]], corners[[i, 1, 1]]],
                                    [corners[[i, 2, 0]], corners[[i, 2, 1]]],
                                    [corners[[i, 3, 0]], corners[[i, 3, 1]]],
                                ],
                            });
                        }
                        response.count = obb_detections.len();
                        response.obb_detections = Some(obb_detections);
                    }
                }
            }

            return Ok(Json(response));
        }
    }

    Err((
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
            error: "Missing 'image' field".to_string(),
        }),
    ))
}
