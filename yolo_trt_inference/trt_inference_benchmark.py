import time
import os
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from typing import List, Tuple
import cv2
import numpy as np

#-------------------------- LetterBox Resizing -------------------------------------------------------------
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

#---------------------------------- Function to convert bounding box from resized coordinates to YOLO format (normalized) --------------------------
def yolo_normalized(bbox, ratio, pad, image_width, image_height):
    
    #--------------------- Convert from padded coordinates to original image space ------------------------
    x1 = (bbox[0] - pad[0]) / ratio[0]
    y1 = (bbox[1] - pad[1]) / ratio[1]
    x2 = (bbox[2] - pad[0]) / ratio[0]
    y2 = (bbox[3] - pad[1]) / ratio[1]
    
    #----------------------------- Clip to image boundaries -----------------------------------------------
    x1 = max(0, min(x1, image_width))
    y1 = max(0, min(y1, image_height))
    x2 = max(0, min(x2, image_width))
    y2 = max(0, min(y2, image_height))
    
    #------------------------------- Convert to YOLO format (normalized center, width, height) ---------------------------------------
    x_c = (x1 + x2) / (2 * image_width)
    y_c = (y1 + y2) / (2 * image_height)
    w = (x2 - x1) / image_width
    h = (y2 - y1) / image_height
    
    return [x_c, y_c, w, h]

cls_names = ["ped", "bicycle", "car", "two-wheeler", "Mini-bus", "bus", "Mini-truck", "truck", "three-wheeler"]

#-------------------------- Check if the environment is correctly configured with CUDA -------------------------------------------------
print(trt.__version__)

trt_logger = trt.Logger(trt.Logger.VERBOSE)

device = cuda.Device(0)
print(device.compute_capability())

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for idx in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(idx)
        shape = engine.get_tensor_shape(tensor_name)
        size = trt.volume(shape) 
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
        
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        
        # Append the device buffer to device bindings
        bindings.append(int(device_mem))
        
        # Append to the appropriate list
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
            
    return inputs, outputs, bindings, stream



def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    trt.init_libnvinfer_plugins(trt_logger, "")

    with open(engine_file_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
        serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)

        # print("_"*80)
        # for idx in range(engine.num_layers):
        #     layer = engine.get_layer(idx)
        #     print(f"Layer {idx}: {layer.name}, Precision: {layer.precision}")
        # print("_"*80)

        return engine

#---------------- TensorRT inference part -----------------------------------------------------------------------
  
class InferenceSession:
    
    def __init__(self, engine_file, inference_shape: Tuple[int, int], warmup_iterations: int = 50):
        self.engine = load_engine(engine_file)
        self.context = None
        self.inference_shape = inference_shape
        self.warmup_iterations = warmup_iterations

    def __enter__(self):
        self.context = self.engine.create_execution_context()
        assert self.context

        self.context.set_input_shape('input', (1, 3, *self.inference_shape))
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)
   
        # Perform warmup if specified
        if self.warmup_iterations > 0:
            self.warmup()
        
        return self
    
    def warmup(self):
        """
        Warm up the TensorRT engine by running several dummy inferences.
        This helps eliminate the first-inference latency spike.
        """
        print(f"Warming up TensorRT engine with {self.warmup_iterations} iterations...")
        
        # Create a dummy input with the right shape
        dummy_input = np.ones((1, 3, *self.inference_shape), dtype=np.float32)
        
        # Run the model several times with dummy data
        for i in range(self.warmup_iterations):
            self.inputs[0].host[:np.prod(dummy_input.shape)] = dummy_input.ravel()
            [cuda.memcpy_htod(inp.device, inp.host) for inp in self.inputs]
            self.context.execute_v2(bindings=self.bindings)
            [cuda.memcpy_dtoh(out.host, out.device) for out in self.outputs]
        
        print("Warmup complete.")

    def preprocess(self, image):
        resized_image, ratio, pad = resize_image(image)
        original_shape = resized_image.shape[:2]
        return np.moveaxis(resized_image, 2, 0), original_shape, resized_image, ratio, pad
        
    def postprocess(self, detected_boxes, original_shape: Tuple[int, int]):
        sx = original_shape[1] / self.inference_shape[1]
        sy = original_shape[0] / self.inference_shape[0]
        detected_boxes[:, :, [0, 2]] *= sx
        detected_boxes[:, :, [1, 3]] *= sy
        return detected_boxes

    def __call__(self, image):
        input_image, original_shape, resized_image, ratio, pad = self.preprocess(image)

        self.inputs[0].host[:np.prod(input_image.shape)] = np.asarray(input_image).ravel()

        [cuda.memcpy_htod(inp.device, inp.host) for inp in self.inputs]
        start_time = time.time()  # Start timing
        success = self.context.execute_v2(bindings=self.bindings)
        inference_time = time.time() - start_time  # End timing
        assert success
        [cuda.memcpy_dtoh(out.host, out.device) for out in self.outputs]

        num_detections, detected_boxes, detected_scores, detected_labels = [o.host for o in self.outputs]

        num_detections = num_detections.reshape(-1)
        num_predictions_per_image = len(detected_scores) // 1  # Batch size is 1
        detected_boxes = detected_boxes.reshape(1, num_predictions_per_image, 4)
        detected_scores = detected_scores.reshape(1, num_predictions_per_image)
        detected_labels = detected_labels.reshape(1, num_predictions_per_image)

        detected_boxes = self.postprocess(detected_boxes, original_shape)  # Scale coordinates back to original image shape
        return num_detections, detected_boxes, detected_scores, detected_labels, resized_image, ratio, pad, inference_time

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.inputs, self.outputs, self.bindings, self.stream, self.context

def main(dataPar, yoloFolder, vis_dir):
    Total_detections = 0
    total_inference_time = 0
    total_images = 0
    with InferenceSession("engine_files/yolonas_s.trt", (640, 640)) as session:
        for image in os.listdir(dataPar):
            if not image.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue  # Skip non-image files

            imgPath = os.path.join(dataPar, image)
            src_image = cv2.imread(imgPath)

            if src_image is None:
                print(f"Warning: Unable to load image {imgPath}. Skipping.")
                continue

            image_height, image_width = src_image.shape[:2]

            
            result = session(src_image)  # Pass the OpenCV image

            # Unpack predictions
            num_detections, detected_boxes, detected_scores, detected_labels, resized_image, ratio, pad, inference_time = result

            # Accumulate inference time
            total_inference_time += inference_time
            total_images += 1

            # # Get detections for the first image in batch
            # boxes = detected_boxes[0]
            # scores = detected_scores[0]
            # classes = detected_labels[0]

            # yolo_annotations = []

            # for i, (box, score, class_id) in enumerate(zip(boxes, scores, classes)):
            #     x1, y1, x2, y2 = box.astype(int)
            #     x_center, y_center, yolo_width, yolo_height = yolo_normalized(box, ratio, pad, image_width, image_height)

            #     if x_center and y_center and yolo_width and yolo_height:
            #         yolo_annotations.append(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {yolo_width:.6f} {yolo_height:.6f} {score:.6f}")

            #     # Draw bounding box and label on the image
            #     label = f"{cls_names[int(class_id)]} {score:.2f}"
            #     color = (0, 255, 0)  # Green color for bounding box
            #     cv2.rectangle(resized_image, (x1, y1), (x2, y2), color, 2)
            #     cv2.putText(resized_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # # Save the visualized image
            # vis_path = os.path.join(vis_dir, image)
            # cv2.imwrite(vis_path, resized_image)

            # # Save YOLO annotations
            # yolo_path = os.path.join(yoloFolder, os.path.splitext(image)[0] + ".txt")

            # with open(yolo_path, "w") as yolo_file:
            #     yolo_file.write("\n".join(yolo_annotations))

            # Total_detections += len(boxes)

    print(f"Visualized images saved to {vis_dir}")
    # print("Total detections", Total_detections)

    # Calculate metrics
    average_inference_time = (total_inference_time / total_images)*1000
    fps = total_images / total_inference_time

    print(f"Average Inference Time per Image: {average_inference_time:.4f} Milli seconds")
    print(f"Total Inference Time: {total_inference_time:.4f} seconds")
    print(f"FPS: {fps:.2f}")

if __name__ == "__main__":
    cPath = os.getcwd()
    dataPar = os.path.join(cPath, r"./Test_dataset")
    outpath = os.path.join(cPath, "Tensort_INT8_prediction_INT8")

    yoloFolder = os.path.join(outpath, "YOLO")
    os.makedirs(yoloFolder, exist_ok=True)

    vis_dir = os.path.join(outpath, "Visual")
    os.makedirs(vis_dir, exist_ok=True)

    main(dataPar, yoloFolder, vis_dir)