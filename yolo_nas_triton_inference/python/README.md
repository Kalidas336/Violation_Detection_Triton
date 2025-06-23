# YOLO-NAS Small Triton Inference

This repository provides a Python implementation for running YOLO-NAS object detection inference using the Triton Inference Server. It supports both HTTP and gRPC clients. The code processes images, generates detections, visualizes results, and saves annotations in YOLO format.

## Features

- Image preprocessing with aspect ratio preservation
- Inference using Triton HTTP and gRPC clients
- Asynchronous streaming support with gRPC
- CUDA Shared Memory support for gRPC client
- Post-processing of detection results
- Visualization of detected objects
- YOLO format annotation generation
- Performance statistics tracking

## Requirements

- Docker Image: [nvcr.io/nvidia/tritonserver:24.01-py3]
- `uv`
- `numpy`
- `opencv-python` (cv2)
- `tritonclient[all]`
- `tqdm`

## Installation

1. Install the required packages:
   ```bash
   uv pip install numpy opencv-python tritonclient[all] tqdm
   ```

2. Ensure Triton Inference Server is running with the YOLO-NAS model loaded:
   - HTTP Server URL: `localhost:8000`
   - gRPC Server URL: `localhost:8001`
   - Model name: `yolo_nas`

## Setting up Triton Inference Server for YOLO-NAS

To run the Triton Inference Server with the YOLO-NAS model, execute the following command:

```bash
docker run --gpus all --rm -it -p 8000:8000 -p 8001:8001 -p 8002:8002 \
-v model_repository:/models \
nvcr.io/nvidia/tritonserver:24.01-py3 tritonserver --model-repository=/models
```

Ensure that the `model_repository` directory contains a properly formatted Triton model repository with the YOLO-NAS model.

## EfficientNMS_TRT Plugin

The YOLO-NAS TensorRT model includes an integrated Non-Maximum Suppression (NMS) plugin called `EfficientNMS_TRT` for filtering redundant detections. The plugin operates with the following attributes:

- `plugin_version`: '1'
- `background_class`: -1
- `max_output_boxes`: 1000
- `score_threshold`: 0.25
- `iou_threshold`: 0.7
- `score_activation`: False
- `box_coding`: 0

## Preprocessing Steps

Before inference, the client scripts preprocess images as follows:
1. **Image Loading**: The image is read from disk using OpenCV.
2. **Resizing and Padding**: The image is resized to (640, 640) while preserving aspect ratio. Padding is applied if necessary.
3. **Channel Rearrangement**: The image is converted from HWC (height, width, channels) to NCHW (batch, channels, height, width) format.
4. **Normalization**: The pixel values are converted to uint8 format.
5. **Batch Dimension Addition**: A batch dimension is added before sending the image for inference.

## Postprocessing Steps

After inference, the output undergoes postprocessing:
1. **Extraction of Predictions**: Bounding boxes, confidence scores, and class IDs are extracted from the model output.
2. **Thresholding**: Detections with confidence scores below a defined threshold (default 0.25) are filtered out.
3. **Coordinate Transformation**: Bounding box coordinates are transformed from the resized image back to the original input image dimensions.
4. **YOLO Format Conversion**: The bounding box coordinates are converted to YOLO format (center_x, center_y, width, height) and normalized.
5. **Visualization**: Bounding boxes are drawn on the original image with class labels and confidence scores.
6. **Annotation Saving**: The detections are saved in YOLO annotation format.

## Usage

### HTTP Client

Run the HTTP client script for synchronous inference:
```bash
uv run yolo_nas_trt_triton_http_client.py
```

### gRPC Client

Run the gRPC client script for asynchronous inference:
```bash
uv run yolo_nas_trt_triton_grpc_client.py
```

### gRPC Client with CUDA Shared Memory

For optimized performance using CUDA shared memory, run:
```bash
uv run yolo_nas_trt_triton_grpc_client_cuda_shm_async.py
```

## Output

Results will be saved in:
- **Visualizations:** `Triton_YOLONas_predictions/Visual`
- **YOLO annotations:** `Triton_YOLONas_predictions/YOLO`

## Supported Classes

The following object classes are supported:
- ped (pedestrian)
- bicycle
- car
- two-wheeler
- Mini-bus
- bus
- Mini-truck
- truck
- three-wheeler

## Code Structure

- `resize_image()`: Resizes and pads images while preserving aspect ratio
- `yolo_normalized()`: Converts bounding boxes to YOLO format
- `preprocess_image()`: Prepares images for inference
- `postprocess_output()`: Processes model outputs
- `visualize_detections()`: Draws bounding boxes on images
- `save_yolo_annotations()`: Saves detections in YOLO format
- `main()`: The inference pipeline

## Configuration

Modify these variables in `main()` as needed:
- `url`: Triton server URL (default: "localhost:8000" for HTTP, "localhost:8001" for gRPC)
- `model_name`: Model name in Triton (default: "yolo_nas")
- `input_dir`: Input images directory
- `output_dir`: Output results directory

## Performance Metrics

The script provides:
1. Visualized images with bounding boxes
2. YOLO format annotation files (.txt)
3. Performance statistics:
   - Total processing time
   - Average inference time
   - Min/max inference times
   - Throughput (FPS)

## Notes

- Images are resized (letterbox) to 640x640 by default.
- Created NMS plugin `EfficientNMS_TRT` as a part of tensorrt model 
- gRPC client supports asynchronous streaming for improved throughput.
- CUDA Shared Memory optimization is available for better performance.

