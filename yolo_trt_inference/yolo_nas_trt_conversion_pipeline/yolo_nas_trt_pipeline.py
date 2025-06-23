import os
import sys
import time
import json
import argparse
import cv2
import numpy as np
import torch
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import defaultdict
from typing import List, Tuple
from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.conversion.conversion_enums import ExportTargetBackend

class YOLONAS_Pipeline:
    def __init__(self, config):
        self.config = config

        with open(self.config['class_file_path']) as f:
            self.cls_names=[cls.strip() for cls in f.readlines()]

        self.trt_logger = trt.Logger(trt.Logger.VERBOSE)
        self.setup_directories()
        
    def setup_directories(self):
        """Create all necessary output directories"""
        os.makedirs(self.config['output_folder'], exist_ok=True)
        
        # PyTorch inference directories
        self.pytorch_txt_dir = os.path.join(self.config['output_folder'], "PyTorch", "YOLO")
        self.pytorch_vis_dir = os.path.join(self.config['output_folder'], "PyTorch", "Visual")
        os.makedirs(self.pytorch_txt_dir, exist_ok=True)
        os.makedirs(self.pytorch_vis_dir, exist_ok=True)
        
        # TensorRT inference directories
        self.tensorrt_txt_dir = os.path.join(self.config['output_folder'], "TensorRT", "YOLO")
        self.tensorrt_vis_dir = os.path.join(self.config['output_folder'], "TensorRT", "Visual")
        os.makedirs(self.tensorrt_txt_dir, exist_ok=True)
        os.makedirs(self.tensorrt_vis_dir, exist_ok=True)
        
        # Metrics directory
        self.metrics_dir = os.path.join(self.config['output_folder'], "Metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Ground truth directory
        self.gt_dir = os.path.join(self.config['output_folder'], "GroundTruth")
        os.makedirs(self.gt_dir, exist_ok=True)
        
    def create_coco_ground_truth(self):
        """Create COCO format ground truth from YOLO labels"""
        
        print("Creating COCO ground truth from YOLO labels...")
        
        coco_gt = {
            "images": [],
            "annotations": [],
            "categories": [],
            "info": {
                "description": "COCO format dataset",
                "version": "1.0",
                "year": 2023,
                "contributor": "YOLO-NAS Pipeline"
            }
        }
        
        # Add categories
        for idx, cls_name in enumerate(self.cls_names):
            coco_gt["categories"].append({
                "id": idx + 1,
                "name": cls_name,
                "supercategory": "none"
            })
        
        annotation_id = 1
        image_files = [f for f in os.listdir(os.path.join(self.config['dataset_folder'],"images")) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_idx, image_file in enumerate(image_files):
            image_path = os.path.join(self.config['dataset_folder'],"images", image_file)
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {image_file}, skipping...")
                continue
                
            height, width = img.shape[:2]
            
            # Add image info
            image_id = img_idx + 1
            coco_gt["images"].append({
                "id": image_id,
                "file_name": image_file,
                "width": width,
                "height": height,
                "license": 1,
                "date_captured": ""
            })
            
            # Read corresponding YOLO label file
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(self.config['dataset_folder'],"labels", label_file)
            
            if not os.path.exists(label_path):
                continue
                
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:  # YOLO format: class x_center y_center width height
                    continue
                    
                class_id, x_center, y_center, bbox_width, bbox_height = map(float, parts)
                
                # Convert YOLO to COCO bbox format [x_min, y_min, width, height]
                x_min = (x_center - bbox_width/2) * width
                y_min = (y_center - bbox_height/2) * height
                bbox_width = bbox_width * width
                bbox_height = bbox_height * height
                
                # Clip coordinates to image boundaries
                x_min = max(0, min(x_min, width))
                y_min = max(0, min(y_min, height))
                bbox_width = min(bbox_width, width - x_min)
                bbox_height = min(bbox_height, height - y_min)
                
                coco_gt["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(class_id) + 1,  # COCO class IDs start at 1
                    "bbox": [x_min, y_min, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "iscrowd": 0,
                    "segmentation": []
                })
                annotation_id += 1
        
        # Save COCO ground truth
        gt_path = os.path.join(self.gt_dir, "ground_truth_coco.json")
        with open(gt_path, 'w') as f:
            json.dump(coco_gt, f, indent=4)
            
        print(f"COCO ground truth saved to {gt_path}")
        return gt_path
    
    def resize_image(self, image, new_shape=(640, 640), auto=False, scale_fill=False, 
                    scaleup=False, center=True, stride=32):
        """Resize and pad image while preserving aspect ratio"""
        shape = image.shape[:2]  # height, width
        
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
            
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        if not scaleup:
            r = min(r, 1.0)
            
        ratio = (r, r)
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        
        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        elif scale_fill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
        
        if center:
            dw /= 2
            dh /= 2
            
        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)) if center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if center else 0, int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, 
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        pad = (dw, dh)
        return image, ratio, pad
    
    def yolo_normalized(self, bbox, ratio, pad, image_width, image_height):
        """Convert bounding box to YOLO normalized format"""
        x1 = (bbox[0] - pad[0]) / ratio[0]
        y1 = (bbox[1] - pad[1]) / ratio[1]
        x2 = (bbox[2] - pad[0]) / ratio[0]
        y2 = (bbox[3] - pad[1]) / ratio[1]
        
        x1 = max(0, min(x1, image_width))
        y1 = max(0, min(y1, image_height))
        x2 = max(0, min(x2, image_width))
        y2 = max(0, min(y2, image_height))
        
        x_c = (x1 + x2) / (2 * image_width)
        y_c = (y1 + y2) / (2 * image_height)
        w = (x2 - x1) / image_width
        h = (y2 - y1) / image_height
        
        return [x_c, y_c, w, h]
    
    def draw_bboxes(self, image, bboxes, labels, confidences):
        """Draw bounding boxes on image"""
        for label, bbox, conf in zip(labels, bboxes, confidences):
            x1, y1, x2, y2 = map(int, bbox)
            class_name = self.cls_names[int(label)]
            color = (0, 255, 0)  # Green
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label_text = f"{class_name} {conf:.2f}"
            cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return image
    
    def convert_to_onnx_and_trt(self):
        """Convert PyTorch model to ONNX and TensorRT"""
        print("Loading PyTorch model...")
        model = models.get(Models.YOLO_NAS_S, num_classes=len(self.cls_names), 
                          checkpoint_path=self.config['model_path'])
        
        print("Exporting to ONNX...")
        model.export(
            os.path.join(self.config['output_folder'], "yolonas_s.onnx"),
            preprocessing=True,
            postprocessing=True,
            engine=ExportTargetBackend.TENSORRT,
            batch_size=self.config['batch_size']
        )
        
        print("Converting ONNX to TensorRT engine...")
        self.convert_onnx_to_trt_engine(
            os.path.join(self.config['output_folder'], "yolonas_s.onnx"),
            os.path.join(self.config['output_folder'], "yolonas_s.trt"),
            enable_int8_quantization=self.config['enable_int8']
        )
    
    def convert_onnx_to_trt_engine(self, onnx_file, trt_output_file, enable_int8_quantization=False):
        """Convert ONNX model to TensorRT engine"""
        EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        
        with trt.Builder(self.trt_logger) as builder, builder.create_network(EXPLICIT_BATCH) as network:
            config = builder.create_builder_config()
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
            
            if enable_int8_quantization:
                config.set_flag(trt.BuilderFlag.INT8)
            else:
                config.set_flag(trt.BuilderFlag.FP16)
                
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
            trt.init_libnvinfer_plugins(self.trt_logger, "")
            
            with trt.OnnxParser(network, self.trt_logger) as onnx_parser:
                with open(onnx_file, 'rb') as f:
                    parse_success = onnx_parser.parse(f.read())
                    if not parse_success:
                        errors = "\n".join(
                            [str(onnx_parser.get_error(error)) for error in range(onnx_parser.num_errors)]
                        )
                        raise RuntimeError(f"Failed to parse onnx model for trt conversion. Errors: \n{errors}")
                
                self.trt_logger.log(trt.ILogger.INFO, "Parsed ONNX model")
            
            serialized_engine = builder.build_serialized_network(network, config)
            with open(trt_output_file, "wb") as output_file:
                output_file.write(serialized_engine)
                self.trt_logger.log(trt.ILogger.INFO, "Serialization done")
    
    def pytorch_inference(self):
        """Run inference using PyTorch model"""
        print("Running PyTorch inference...")
        model = models.get(Models.YOLO_NAS_S, num_classes=len(self.cls_names), 
                          checkpoint_path=self.config['model_path'])
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")

        pipeline__=model._get_pipeline()
        
        image_files = [f for f in os.listdir(os.path.join(self.config['dataset_folder'],"images")) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        total_time = 0
        total_images = 0
        total_det=0
        
        for image_file in image_files:
            img_path = os.path.join(self.config['dataset_folder'],"images", image_file)
            src_image = cv2.imread(img_path)
            
            if src_image is None:
                print(f"Warning: Unable to load image {img_path}. Skipping.")
                continue
                
            # Read and resize image
            image_height, image_width, _ = src_image.shape
            resized_image, ratio, pad = self.resize_image(src_image)
            
            # Get predictions

            start_time = time.time()

            detection_predictions = pipeline__(resized_image)
            inference_time = time.time() - start_time
            total_time += inference_time
            total_images += 1
            
            # Extract predictions
            bboxes = detection_predictions.prediction.bboxes_xyxy
            confidence = detection_predictions.prediction.confidence
            labels = detection_predictions.prediction.labels
            
            # Save predictions in YOLO format
            output = []
            for label, bbox, conf in zip(labels, bboxes, confidence):
                bbox = self.yolo_normalized(bbox, ratio, pad, image_width, image_height)
                output.append(f"{int(label)} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {conf}\n")
            
            total_det+=len(output)
            
            # Write to file
            txt_file = os.path.join(self.pytorch_txt_dir, 
                                   os.path.splitext(image_file)[0] + ".txt")
            with open(txt_file, "w") as f:
                f.writelines(output)
            
            # Visualize and save
            visualized_image = self.draw_bboxes(resized_image, bboxes, labels, confidence)
            vis_path = os.path.join(self.pytorch_vis_dir, image_file)
            cv2.imwrite(vis_path, visualized_image)
        
        # Save PyTorch metrics
        metrics = {
            "average_inference_time": total_time / total_images,
            "total_inference_time": total_time,
            "fps": total_images / total_time,
            "total_images": total_images,
            "total_detections":total_det
        }
        
        with open(os.path.join(self.metrics_dir, "pytorch_metrics.txt"), "w") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")
    
    class HostDeviceMem:
        """Helper class for TensorRT memory management"""
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem
        
        def __str__(self):
            return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
        
        def __repr__(self):
            return self.__str__()
    
    def allocate_buffers(self, engine):
        """Allocate buffers for TensorRT inference"""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for idx in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(idx)
            shape = engine.get_tensor_shape(tensor_name)
            size = trt.volume(shape) 
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(self.HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(self.HostDeviceMem(host_mem, device_mem))
                
        return inputs, outputs, bindings, stream
    
    def load_engine(self, engine_file_path):
        """Load TensorRT engine"""
        assert os.path.exists(engine_file_path)
        print("Reading engine from file {}".format(engine_file_path))
        trt.init_libnvinfer_plugins(self.trt_logger, "")
        
        with open(engine_file_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    class TRTInferenceSession:
        """TensorRT inference session"""
        def __init__(self, engine_file, inference_shape: Tuple[int, int], parent, warmup_iterations: int = 50):
            self.parent = parent  # Reference to parent YOLONAS_Pipeline instance
            self.inference_shape = inference_shape
            self.warmup_iterations = warmup_iterations
            self.engine = self.parent.load_engine(engine_file)
            self.context = None
        
        def __enter__(self):
            self.context = self.engine.create_execution_context()
            assert self.context
            
            self.context.set_input_shape('input', (1, 3, *self.inference_shape))
            self.inputs, self.outputs, self.bindings, self.stream = self.parent.allocate_buffers(self.engine)
            
            if self.warmup_iterations > 0:
                self.warmup()
            
            return self
        
        def warmup(self):
            """Warm up TensorRT engine"""
            dummy_input = np.ones((1, 3, *self.inference_shape), dtype=np.float32)
            
            for _ in range(self.warmup_iterations):
                self.inputs[0].host[:np.prod(dummy_input.shape)] = dummy_input.ravel()
                [cuda.memcpy_htod(inp.device, inp.host) for inp in self.inputs]
                self.context.execute_v2(bindings=self.bindings)
                [cuda.memcpy_dtoh(out.host, out.device) for out in self.outputs]
        
        def preprocess(self, image):
            """Preprocess image for TensorRT inference"""
            resized_image, ratio, pad = self.parent.resize_image(image)
            original_shape = resized_image.shape[:2]
            return np.moveaxis(resized_image, 2, 0), original_shape, resized_image, ratio, pad
            
        def postprocess(self, detected_boxes, original_shape: Tuple[int, int]):
            """Postprocess TensorRT output"""
            sx = original_shape[1] / self.inference_shape[1]
            sy = original_shape[0] / self.inference_shape[0]
            detected_boxes[:, :, [0, 2]] *= sx
            detected_boxes[:, :, [1, 3]] *= sy
            return detected_boxes
        
        def __call__(self, image):
            """Run inference on single image"""
            input_image, original_shape, resized_image, ratio, pad = self.preprocess(image)
            
            self.inputs[0].host[:np.prod(input_image.shape)] = np.asarray(input_image).ravel()
            
            [cuda.memcpy_htod(inp.device, inp.host) for inp in self.inputs]
            start_time = time.time()
            success = self.context.execute_v2(bindings=self.bindings)
            inference_time = time.time() - start_time
            assert success
            [cuda.memcpy_dtoh(out.host, out.device) for out in self.outputs]
            
            num_detections, detected_boxes, detected_scores, detected_labels = [o.host for o in self.outputs]
            
            num_detections = num_detections.reshape(-1)
            num_predictions_per_image = len(detected_scores) // 1  # Batch size is 1
            detected_boxes = detected_boxes.reshape(1, num_predictions_per_image, 4)
            detected_scores = detected_scores.reshape(1, num_predictions_per_image)
            detected_labels = detected_labels.reshape(1, num_predictions_per_image)
            
            detected_boxes = self.postprocess(detected_boxes, original_shape)
            return num_detections, detected_boxes, detected_scores, detected_labels, resized_image, ratio, pad, inference_time
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            """Cleanup"""
            del self.inputs, self.outputs, self.bindings, self.stream, self.context

    def tensorrt_inference(self):
        """Run TensorRT inference"""
        print("Running TensorRT inference...")
        trt_engine_path = os.path.join(self.config['output_folder'], "yolonas_s.trt")
        
        total_time = 0
        total_images = 0
        total_detections = 0
        
        image_files = [f for f in os.listdir(os.path.join(self.config['dataset_folder'],"images")) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        with self.TRTInferenceSession(trt_engine_path, (640, 640), self) as session:
            for image_file in image_files:
                img_path = os.path.join(self.config['dataset_folder'], "images",image_file)
                src_image = cv2.imread(img_path)
                
                if src_image is None:
                    print(f"Warning: Unable to load image {img_path}. Skipping.")
                    continue
                
                image_height, image_width = src_image.shape[:2]
                
                # Run inference
                result = session(src_image)
                num_detections, detected_boxes, detected_scores, detected_labels, resized_image, ratio, pad, inference_time = result
                
                # Accumulate metrics
                total_time += inference_time
                total_images += 1
                
                # Process detections
                boxes = detected_boxes[0]
                scores = detected_scores[0]
                classes = detected_labels[0]
                
                yolo_annotations = []
                for i, (box, score, class_id) in enumerate(zip(boxes, scores, classes)):
                    x1, y1, x2, y2 = box.astype(int)
                    x_center, y_center, yolo_width, yolo_height = self.yolo_normalized(
                        box, ratio, pad, image_width, image_height)
                    
                    if x_center and y_center and yolo_width and yolo_height:
                        yolo_annotations.append(
                            f"{int(class_id)} {x_center:.6f} {y_center:.6f} {yolo_width:.6f} {yolo_height:.6f} {score:.6f}")
                    
                    # Draw bounding box
                    label = f"{self.cls_names[int(class_id)]} {score:.2f}"
                    color = (0, 255, 0)
                    cv2.rectangle(resized_image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(resized_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Save results
                vis_path = os.path.join(self.tensorrt_vis_dir, image_file)
                cv2.imwrite(vis_path, resized_image)
                
                yolo_path = os.path.join(self.tensorrt_txt_dir, os.path.splitext(image_file)[0] + ".txt")
                with open(yolo_path, "w") as yolo_file:
                    yolo_file.write("\n".join(yolo_annotations))
                
                total_detections += len(yolo_annotations)
        
        # Save TensorRT metrics
        metrics = {
            "average_inference_time": total_time / total_images,
            "total_inference_time": total_time,
            "fps": total_images / total_time,
            "total_images": total_images,
            "total_detections": total_detections
        }
        
        with open(os.path.join(self.metrics_dir, "tensorrt_metrics.txt"), "w") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")
    
    def generate_evaluation_metrics(self, mode="both"):
        """Generate evaluation metrics for both PyTorch and TensorRT predictions"""
        print("Generating evaluation metrics...")
        
        gt_path = os.path.join(self.gt_dir, "ground_truth_coco.json")
        
        if mode in ("both", "pytorch"):
            # Evaluate PyTorch predictions
            pytorch_pred_path = os.path.join(self.metrics_dir, "pytorch_predictions.json")
            self.yolo_to_coco(os.path.join(self.config['dataset_folder'], "images"),self.pytorch_txt_dir, pytorch_pred_path)
            
            coco_gt, coco_pred = self.load_coco_data(gt_path, pytorch_pred_path)
            coco_eval, metrics = self.evaluate_predictions(gt_path, pytorch_pred_path)
            pr_metrics = self.calculate_precision_recall_f1(coco_gt, coco_pred)
            
            # Save PyTorch evaluation metrics
            with open(os.path.join(self.metrics_dir, "pytorch_eval_metrics.txt"), "w") as f:
                f.write("PyTorch Evaluation Metrics:\n")
                f.write(f"mAP@0.5: {metrics['mAP@0.5']:.4f}\n")
                f.write(f"mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}\n")
                f.write(f"Precision@0.5: {pr_metrics['precision']:.4f}\n")
                f.write(f"Recall@0.5: {pr_metrics['recall']:.4f}\n")
                f.write(f"F1@0.5: {pr_metrics['f1']:.4f}\n")
        
        if mode in ("both", "tensorrt"):
            # Evaluate TensorRT predictions
            tensorrt_pred_path = os.path.join(self.metrics_dir, "tensorrt_predictions.json")
            self.yolo_to_coco(os.path.join(self.config['dataset_folder'], "images"),self.tensorrt_txt_dir, tensorrt_pred_path)
            
            coco_gt, coco_pred = self.load_coco_data(gt_path, tensorrt_pred_path)
            coco_eval, metrics = self.evaluate_predictions(gt_path, tensorrt_pred_path)
            pr_metrics = self.calculate_precision_recall_f1(coco_gt, coco_pred)
            
            # Save TensorRT evaluation metrics
            with open(os.path.join(self.metrics_dir, "tensorrt_eval_metrics.txt"), "w") as f:
                f.write("TensorRT Evaluation Metrics:\n")
                f.write(f"mAP@0.5: {metrics['mAP@0.5']:.4f}\n")
                f.write(f"mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}\n")
                f.write(f"Precision@0.5: {pr_metrics['precision']:.4f}\n")
                f.write(f"Recall@0.5: {pr_metrics['recall']:.4f}\n")
                f.write(f"F1@0.5: {pr_metrics['f1']:.4f}\n")

    def yolo_to_coco(self,image_folder,yolo_folder, output_json_path):

        count=0
        # Initialize COCO JSON structure
        coco_output = {
            "images": [],
            "annotations": [],
            "categories": []
        }

        # Add categories (class names)
        for idx, cls_name in enumerate(self.cls_names):
            coco_output["categories"].append({
                "id": idx + 1,  # COCO category IDs start from 1
                "name": cls_name,
                "supercategory": "none"
            })

        # Initialize annotation ID counter
        annotation_id = 1

        # Loop through all images in the folder
        for image_file in os.listdir(image_folder):

            if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue  # Skip non-image files

            # Load image to get dimensions
            image_path = os.path.join(image_folder, image_file)
            img = Image.open(image_path)
            img_width, img_height = img.size

            # Add image info to COCO JSON
            image_id = len(coco_output["images"]) + 1

            coco_output["images"].append({
                "id": image_id,
                "file_name": image_file,
                "width": img_width,
                "height": img_height
            })

            # Read corresponding YOLO annotation file
            yolo_file = os.path.splitext(image_file)[0] + ".txt"
            yolo_path = os.path.join(yolo_folder, yolo_file)

            if not os.path.exists(yolo_path):
                print(yolo_path)
                continue  # Skip if no annotation file exists

            with open(yolo_path, "r") as f:
                lines = f.readlines()

            # Process each annotation
            for line in lines:
                count+=1
                # print(count)
                parts = line.strip().split()

                if len(parts) != 6:
                    continue  # Skip invalid lines

                class_id, x_center, y_center, width, height,score = map(float, parts)

                # Convert YOLO format to COCO format
                x_min = (x_center - width / 2) * img_width
                y_min = (y_center - height / 2) * img_height
                width = width * img_width
                height = height * img_height

                xmax,ymax=x_min + width, y_min + height

                x_min = max(0, min(x_min, img_width))
                y_min = max(0, min(y_min, img_height))
                xmax = max(0, min(xmax, img_width))
                ymax = max(0, min(ymax, img_height))

                # Add annotation to COCO JSON
                coco_output["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(class_id) + 1,  # COCO category IDs start from 1
                    "bbox": [x_min, y_min, width, height],
                    "area": width * height,
                    "score":float(score),
                    "iscrowd": 0
                })

                annotation_id += 1

        # Save COCO JSON to file
        with open(output_json_path, "w") as f:
            json.dump(coco_output["annotations"], f, indent=4)

        print(f"COCO JSON saved to {output_json_path}")

    def load_coco_data(self, gt_path, pred_path):
        """Load COCO ground truth and predictions"""
        coco_gt = COCO(gt_path)
        with open(pred_path, 'r') as f:
            pred_json = json.load(f)
        coco_pred = coco_gt.loadRes(pred_json)
        return coco_gt, coco_pred
    
    def evaluate_predictions(self, gt_path, pred_path):
        """Evaluate predictions using COCO metrics"""
        coco_gt, coco_pred = self.load_coco_data(gt_path, pred_path)
        
        coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        metrics = {
            'mAP@0.5': coco_eval.stats[1],
            'mAP@0.5:0.95': coco_eval.stats[0],
        }
        
        return coco_eval, metrics
    
    def calculate_precision_recall_f1(self, coco_gt, coco_pred, iou_threshold=0.5):
        """Calculate precision, recall and F1 score"""
        img_ids = sorted(coco_gt.getImgIds())
        cat_ids = coco_gt.getCatIds()
        
        tp, fp, fn = 0, 0, 0
        
        for img_id in img_ids:

            for cat_id in cat_ids:
                gt_anns = coco_gt.getAnnIds(imgIds=img_id, catIds=[cat_id])
                gt_anns = coco_gt.loadAnns(gt_anns)
                
                pred_anns = coco_pred.getAnnIds(imgIds=img_id, catIds=[cat_id])
                pred_anns = coco_pred.loadAnns(pred_anns)
                
                matched_gt = set()
                
                for pred in pred_anns:
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for i, gt in enumerate(gt_anns):
                        if i in matched_gt:
                            continue
                        
                        # Calculate IoU
                        pred_bbox = pred['bbox']
                        gt_bbox = gt['bbox']
                        
                        pred_x1, pred_y1, pred_w, pred_h = pred_bbox
                        pred_x2, pred_y2 = pred_x1 + pred_w, pred_y1 + pred_h
                        
                        gt_x1, gt_y1, gt_w, gt_h = gt_bbox
                        gt_x2, gt_y2 = gt_x1 + gt_w, gt_y1 + gt_h
                        
                        # Intersection coordinates
                        x_left = max(pred_x1, gt_x1)
                        y_top = max(pred_y1, gt_y1)
                        x_right = min(pred_x2, gt_x2)
                        y_bottom = min(pred_y2, gt_y2)
                        
                        if x_right < x_left or y_bottom < y_top:
                            iou = 0
                        else:
                            intersection_area = (x_right - x_left) * (y_bottom - y_top)
                            pred_area = pred_w * pred_h
                            gt_area = gt_w * gt_h
                            union_area = pred_area + gt_area - intersection_area
                            iou = intersection_area / union_area
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = i
                    
                    if best_iou >= iou_threshold and best_gt_idx >= 0:
                        tp += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        fp += 1
                
                fn += len(gt_anns) - len(matched_gt)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
    
    def run_pipeline(self):
        """Run complete pipeline"""
        # Step 0: Create COCO ground truth from YOLO labels
        # gt_path = self.create_coco_ground_truth()
        # self.config['gt_path'] = gt_path  # Update config with generated ground truth path
        
        # # Step 1: Convert model to ONNX and TensorRT
        # self.convert_to_onnx_and_trt()
        
        # # Step 2: Run PyTorch inference
        # self.pytorch_inference()
        
        # # Step 3: Run TensorRT inference
        self.tensorrt_inference()
        
        # Step 4: Generate evaluation metrics
        # self.generate_evaluation_metrics()
        
        print("Pipeline completed successfully!")

def parse_arguments():
    parser = argparse.ArgumentParser(description='YOLONAS Pipeline Configuration')
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True, 
        help='Path to the model file (must be a string)'
    )
    
    parser.add_argument(
        '--dataset_folder', 
        type=str, 
        required=True, 
        help='Path to the dataset directory (must be a string)'
    )
    
    parser.add_argument(
        '--output_folder', 
        type=str, 
        required=True, 
        help='Path to save output files (must be a string)'
    )
    
    parser.add_argument(
        '--batch_size', 
        type=int, 
        required=True, 
        help='Batch size for inference (must be an integer)'
    )

    parser.add_argument(
        '--class_file_path', 
        type=str, 
        required=True, 
        help='Path of classes.txt file (must be a string)'
    )
    
    return parser.parse_args()

if __name__=="__main__":
    args = parse_arguments()

    # Configuration
    config = {
        'model_path': args.model_path,
        'dataset_folder': args.dataset_folder,
        'output_folder': args.output_folder,
        'batch_size': args.batch_size,
        "class_file_path":args.class_file_path,
        'enable_int8': False
    }
    # Run pipeline
    pipeline = YOLONAS_Pipeline(config)
    pipeline.run_pipeline()