'''
YOLO-NAS Object Detection Inference using Triton Inference Server over gRPC Asynchronous Streaming
This code processes images, generates detections, visualizes results, and saves annotations in YOLO format.
'''
import os
import sys
import time
import queue
from functools import partial
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
from typing import Tuple, List, Dict, Any, Optional

# Configuration
MODEL_NAME = "yolo_nas"
SERVER_URL = "localhost:8001"
IMAGE_SIZE = (640, 640)
CONF_THRESHOLD = 0.25

# Class names for visualization
CLS_NAMES = [
    "person_sb",
    "person_nsb",
    "phone",
    "no_phone",
    "rider_phone",
    "rider_no_phone",
    "helmet_rider",
    "half_helmet_rider",
    "cap_rider",
    "no_helmet_rider",
    "helmet_pillion",
    "half_helmet_pillion",
    "cap_pillion",
    "no_helmet_pillion",
    "tr",
    "notr"
]

class UserData:
    """Class for managing asynchronous request data"""
    def __init__(self):
        self._completed_requests = queue.Queue()

def callback(user_data, result, error):
    """Callback function for handling async inference responses"""
    response_time = time.time()
    if error:
        user_data._completed_requests.put((error, response_time))
    else:
        user_data._completed_requests.put((result, response_time))

def resize_image(image, new_shape=(640, 640), auto=False, scale_fill=False, 
                scaleup=False, center=True, stride=32):
    """
    Resize and pad image while preserving aspect ratio.
    
    Args:
        image: Input image (numpy array)
        new_shape: Target shape (height, width) for resizing
        auto: Whether to use minimum rectangle
        scale_fill: Whether to stretch the image to new_shape
        scaleup: Whether to allow scaling up. If False, only scale down
        center: Whether to center the image or align to top-left
        stride: Stride for rounding padding
        
    Returns:
        Resized and padded image, resize ratio, padding info as (padded_img, ratio, (dw, dh))
    """
    # Get current shape
    shape = image.shape[:2]  # height, width
    
    # Handle new_shape as int or tuple
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
        
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Only scale down, do not scale up (for better results)
    if not scaleup:
        r = min(r, 1.0)
        
    # Compute padding
    ratio = (r, r)  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
    
    if center:
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        
    # Resize and pad
    if shape[::-1] != new_unpad:  # resize
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        
    top, bottom = int(round(dh - 0.1)) if center else 0, int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)) if center else 0, int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, 
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border
    
    pad = (dw, dh)
    
    return image, ratio, pad

def preprocess_image(image_path: str, target_size: tuple = IMAGE_SIZE) -> Tuple[Optional[np.ndarray], tuple, np.ndarray, tuple, tuple]:
    """
    Preprocess an image for YOLONas inference.
    
    Args:
        image_path: Path to the input image
        target_size: Target size for resizing (height, width)
        
    Returns:
        Tuple containing:
        - Preprocessed image in NCHW format
        - Original image shape
        - Original image (for visualization)
        - Resize ratio
        - Padding information
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # Get original image dimensions
        original_shape = image.shape[:2]  # (height, width)
        original_image = image.copy()
        
        # Resize and pad image
        resized_image, ratio, pad = resize_image(image, new_shape=target_size)
        
        # Convert to NCHW format for inference
        input_image = np.transpose(resized_image, (2, 0, 1)).astype(np.uint8)
        
        # Add batch dimension
        input_image = np.expand_dims(input_image, axis=0)
        
        return input_image, original_shape, original_image, ratio, pad
    
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {str(e)}")
        return None, None, None, None, None

def yolo_normalized(bbox, ratio, pad, image_width, image_height):
    """
    Convert bounding box from resized coordinates to YOLO format (normalized)
    
    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        ratio: Resize ratio (width_ratio, height_ratio)
        pad: Padding values (width_pad, height_pad)
        image_width: Original image width
        image_height: Original image height
        
    Returns:
        Normalized YOLO format coordinates [x_center, y_center, width, height]
    """
    # Convert from padded coordinates to original image space
    x1 = (bbox[0] - pad[0]) / ratio[0]
    y1 = (bbox[1] - pad[1]) / ratio[1]
    x2 = (bbox[2] - pad[0]) / ratio[0]
    y2 = (bbox[3] - pad[1]) / ratio[1]
    
    # Clip to image boundaries
    x1 = max(0, min(x1, image_width))
    y1 = max(0, min(y1, image_height))
    x2 = max(0, min(x2, image_width))
    y2 = max(0, min(y2, image_height))
    
    # Convert to YOLO format (normalized center, width, height)
    x_c = (x1 + x2) / (2 * image_width)
    y_c = (y1 + y2) / (2 * image_height)
    w = (x2 - x1) / image_width
    h = (y2 - y1) / image_height
    
    return [x_c, y_c, w, h]

def postprocess_output(num_detections: np.ndarray, boxes: np.ndarray, scores: np.ndarray, 
                      class_ids: np.ndarray, original_shape: tuple, ratio: tuple, 
                      pad: tuple, conf_threshold: float = CONF_THRESHOLD) -> List:
    """
    Postprocess detection outputs from YOLONas model.
    
    Args:
        num_detections: Number of detections
        boxes: Bounding box coordinates [x1, y1, x2, y2]
        scores: Confidence scores
        class_ids: Class IDs
        original_shape: Original image shape (height, width)
        ratio: Resize ratio (width_ratio, height_ratio)
        pad: Padding values (width_pad, height_pad)
        conf_threshold: Confidence threshold for filtering detections
        
    Returns:
        List of filtered detections
    """
    # Get the number of detections
    num_dets = int(num_detections[0].item())
    
    # Filter by confidence threshold
    mask = scores[:num_dets] >= conf_threshold
    filtered_boxes = boxes[:num_dets][mask]
    filtered_scores = scores[:num_dets][mask]
    filtered_class_ids = class_ids[:num_dets][mask]
    
    # Convert to list of detection tuples
    detections = []
    for box, score, class_id in zip(filtered_boxes, filtered_scores, filtered_class_ids):
        detections.append((box, score, class_id))
    
    return detections

def visualize_detections(image: np.ndarray, detections: list) -> np.ndarray:
    """
    Visualize detections on the image.
    
    Args:
        image: Original input image
        detections: List of detections (box, score, class_id)
        
    Returns:
        Image with visualized detections
    """
    result_image = image.copy()
    
    for box, score, class_id in detections:
        # Box coordinates as integers
        x1, y1, x2, y2 = map(int, box)
        print(x1,y1)
        # Draw bounding box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label with class name and score
        class_name = CLS_NAMES[int(class_id)] if int(class_id) < len(CLS_NAMES) else f"class_{class_id}"
        label = f"{class_name}: {score:.2f}"
        cv2.putText(result_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return result_image

def save_yolo_annotations(detections: list, image_path: str, output_dir: str, 
                         original_shape: tuple, ratio: tuple, pad: tuple):
    """
    Save detections in YOLO annotation format.
    
    Args:
        detections: List of detections (box, score, class_id)
        image_path: Path to the original image
        output_dir: Directory to save YOLO annotations
        original_shape: Original image shape (height, width)
        ratio: Resize ratio (width_ratio, height_ratio)
        pad: Padding values (width_pad, height_pad)
    """
    image_height, image_width = original_shape
    yolo_annotations = []
    
    for box, score, class_id in detections:
        # Convert to YOLO normalized format
        x_center, y_center, yolo_width, yolo_height = yolo_normalized(
            box, ratio, pad, image_width, image_height)
        
        if x_center and y_center and yolo_width and yolo_height:
            yolo_annotations.append(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {yolo_width:.6f} {yolo_height:.6f} {score:.6f}")
    
    # Save YOLO annotations
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    yolo_path = os.path.join(output_dir, f"{base_name}.txt")
    
    with open(yolo_path, "w") as yolo_file:
        yolo_file.write("\n".join(yolo_annotations))

def infer_image(triton_client, image_path, user_data):
    """
    Submit a single image for asynchronous inference
    
    Args:
        triton_client: Triton gRPC client
        image_path: Path to input image
        user_data: UserData instance for callbacks
        
    Returns:
        request_id, request_time, and image metadata if successful, None otherwise
    """
    try:
        # Preprocess image
        input_data, original_shape, original_image, ratio, pad = preprocess_image(image_path)
        
        if input_data is None:
            return None
        
        # Prepare inference inputs
        inputs = []
        inputs.append(grpcclient.InferInput("input", input_data.shape, "UINT8"))
        inputs[0].set_data_from_numpy(input_data)
        
        # Prepare inference outputs
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput("det_boxes"))
        outputs.append(grpcclient.InferRequestedOutput("det_scores"))
        outputs.append(grpcclient.InferRequestedOutput("det_classes"))
        outputs.append(grpcclient.InferRequestedOutput("num_dets"))
        
        # Generate unique request ID
        request_id = f"img_{os.path.basename(image_path)}"
        request_time = time.time()
        
        # Submit asynchronous inference request
        triton_client.async_stream_infer(
            model_name=MODEL_NAME,
            inputs=inputs,
            outputs=outputs,
            request_id=request_id
        )
        
        return request_id, request_time, original_shape, original_image, ratio, pad
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def main():
    # Set up directories
    current_path = os.getcwd()
    input_dir = r"/home/mtx003/kalidas/violation_det/cropped"
    output_dir = os.path.join(current_path, "YOLO_NAS_gRPC_Async_TensorRT")
    
    # Create output subdirectories
    yolo_dir = os.path.join(output_dir, "YOLO")
    vis_dir = os.path.join(output_dir, "Visual")
    os.makedirs(yolo_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(list(Path(input_dir).glob(f"*{ext}")))
        image_paths.extend(list(Path(input_dir).glob(f"*{ext.upper()}")))
    
    if not image_paths:
        print(f"No images found in folder '{input_dir}'")
        sys.exit(1)
    
    total_images = len(image_paths)
    print(f"Found {total_images} images in folder '{input_dir}'")
    
    # Initialize user data for callbacks
    user_data = UserData()
    start_time = time.time()
    
    try:
        # Set up Triton client with stream processing
        with grpcclient.InferenceServerClient(url=SERVER_URL) as triton_client:
            # Start the data stream
            triton_client.start_stream(callback=partial(callback, user_data))
            
            # Dictionary to track request information
            request_ids = {}
            
            # Send all inference requests
            print("Sending inference requests...")
            for image_path in tqdm(image_paths, desc="Submitting requests", unit="image"):
                image_path_str = str(image_path)
                result = infer_image(triton_client, image_path_str, user_data)
                                
                if result:
                    request_id, request_time, original_shape, original_image, ratio, pad = result
                    request_ids[request_id] = {
                        "path": image_path_str,
                        "request_time": request_time,
                        "original_shape": original_shape,
                        "original_image": original_image,
                        "ratio": ratio,
                        "pad": pad
                    }
            
            # Process results as they come back
            total_detections = 0
            inference_times = []
            
            # Create progress bar for results processing
            progress_bar = tqdm(total=len(request_ids), desc="Processing results", unit="image")
            
            # Process each response as it comes back
            for _ in range(len(request_ids)):
                # Get the next completed request
                data = user_data._completed_requests.get()
                
                if isinstance(data[0], InferenceServerException):
                    print(f"Inference error: {data[0]}")
                else:
                    result, response_time = data
                    this_id = result.get_response().id
                    
                    if this_id in request_ids:
                        request_time = request_ids[this_id]["request_time"]
                        inference_time = response_time - request_time
                        inference_times.append(inference_time)
                        
                        image_path = request_ids[this_id]["path"]
                        original_shape = request_ids[this_id]["original_shape"]
                        original_image = request_ids[this_id]["original_image"]
                        ratio = request_ids[this_id]["ratio"]
                        pad = request_ids[this_id]["pad"]
                        
                        # Extract detection results
                        boxes = result.as_numpy("det_boxes")
                        scores = result.as_numpy("det_scores")
                        class_ids = result.as_numpy("det_classes")
                        num_detections = result.as_numpy("num_dets")
                        
                        # Postprocess detections
                        detections = postprocess_output(
                            num_detections, boxes, scores, class_ids,
                            original_shape, ratio, pad
                        )
                        
                        total_detections += len(detections)
                        
                        # Visualize detections
                        result_image = visualize_detections(original_image, detections)
                        
                        # Save visualization
                        vis_path = os.path.join(vis_dir, os.path.basename(image_path))
                        cv2.imwrite(vis_path, result_image)
                        
                        # Save YOLO annotations
                        save_yolo_annotations(
                            detections, image_path, yolo_dir,
                            original_shape, ratio, pad
                        )
                    else:
                        print(f"Received unexpected request ID: {this_id}")
                
                # Update progress bar
                progress_bar.update(1)
            
            # Close progress bar
            progress_bar.close()
            
            # Calculate statistics
            total_time = time.time() - start_time
            
            if inference_times:
                avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
                min_inference_time = np.min(inference_times) * 1000
                max_inference_time = np.max(inference_times) * 1000
                fps = total_images / sum(inference_times)
            else:
                avg_inference_time = 0
                min_inference_time = 0
                max_inference_time = 0
                fps = 0
            
            overall_fps = total_images / total_time
            
            # Print statistics
            print("\nInference Statistics:")
            print(f"Processed {total_images} images with {total_detections} total detections")
            print(f"Total processing time: {total_time:.2f} seconds")
            print(f"Overall throughput: {overall_fps:.2f} images/second")
            print("\nInference Time Statistics:")
            print(f"Average inference time: {avg_inference_time:.2f} ms")
            print(f"Minimum inference time: {min_inference_time:.2f} ms")
            print(f"Maximum inference time: {max_inference_time:.2f} ms")
            print(f"Inference throughput: {fps:.2f} images/second")
            print(f"Visualizations saved to: {vis_dir}")
            print(f"YOLO annotations saved to: {yolo_dir}")
    
    except InferenceServerException as e:
        print(f"Inference server error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()