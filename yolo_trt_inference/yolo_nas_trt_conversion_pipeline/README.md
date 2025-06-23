# YOLO-NAS EfficientNMSTRT_Plugin added FP16 TensorRT Conversion with performance benchmarking

Pipeline for converting YOLO-NAS models to TensorRT, performing inference, and benchmarking performance across PyTorch and TensorRT backends with detailed evaluation metrics. 

## Features

- **Model Conversion**
  - PyTorch to ONNX conversion
  - ONNX to TensorRT engine conversion
  - Support for FP16 and INT8 quantization

- **Inference Backends**
  - Native PyTorch inference
  - Optimized TensorRT inference
  - Custom pre/post-processing pipelines

- **Performance Metrics**
  - Detailed timing statistics (FPS, latency)
  - Memory utilization tracking
  - Warm-up iterations for stable benchmarking

- **Evaluation Framework**
  - COCO-style evaluation metrics (mAP@0.5, mAP@0.5:0.95)
  - Precision, Recall, F1 score calculations
  - Per-class performance breakdown
  - Support for custom class definitions

- **Data Handling**
  - YOLO to COCO format conversion
  - Automatic ground truth generation
  - Letterbox image resizing with aspect ratio preservation

- **Visualization**
  - Bounding box visualization with class labels
  - Confidence score display
  - Side-by-side comparison of PyTorch vs TensorRT results

## Docker Setup

### Pull the TensorRT Docker image
```bash
docker pull nvcr.io/nvidia/deepstream:7.0-triton-multiarch
```
### Depedencies 
- TensorRT 8.6.1.6-1+cuda12.0 and Corresponding python binding 
- cuda 12.2

### Run container with GPU support and volume mounting
```
docker run -it --gpus all --name yolo_nas_trt_conversion \
  -v /path/on/host:/path/in/container \
  nvcr.io/nvidia/deepstream:7.0-triton-multiarch
```

Inside container install the additional required dependencies

```bash
pip install cmake super-gradients==3.7.1 pycocotools==2.0.8 pillow  pycuda 
```

## pytorch to TensorRT conversion with performance and accuracy metrics

### Ground truth datset structure 
For accuracy metrics like precision, recall, and F1 score, a ground truth dataset with images and corresponding YOLO format labels is required:

```
dataset_folder/  
├── images/       # Contains image files
└── labels/       # Contains .txt annotation files (YOLO format)  (class_id xc yc bw bh conf)
```

### To Run the conversion
```
python yolo_nas_trt_pipeline.py \
    --model_path "/path/to/model" \
    --dataset_folder "/path/to/dataset" \
    --output_folder "/path/to/output" \
    --batch_size 1 \
    --class_file_path "/path/to/classes.txt"
``` 
- Performs pytorch inference,TensorRT inference and outputs corresponding yolo labels anmd its visualizations 
- Convert the given yolo files to coco files and consider it as groundtruth. With this groundtruth it will calculate the accuracy metrics of pytorch inference and TensorRT inference and the results will be avalibale in ```pytorch_eval_metrics.txt``` and ```tensorrt_eval_metrics.txt```
- Pytorch inference FPS as well as TensorRT inference FPS will be available in ```pytorch_metrics.txt``` and ```tensorrt_metrics.txt```
- The above code will generate output like the below

### Output structure 
```
MODEL-BENCHMARKING/
├── PyTorch/
│   ├── YOLO/            # YOLO-format prediction files
│   └── Visual/          # Visualized detection images
├── TensorRT/
│   ├── YOLO/            # YOLO-format prediction files
│   └── Visual/          # Visualized detection images
├── Metrics/
│   ├── pytorch_metrics.txt        # PyTorch timing metrics
│   ├── tensorrt_metrics.txt       # TensorRT timing metrics
│   ├── pytorch_eval_metrics.txt   # PyTorch evaluation metrics
│   └── tensorrt_eval_metrics.txt  # TensorRT evaluation metrics
│── GroundTruth/
│    └── ground_truth_coco.json     # Auto-generated COCO ground truth
│   
│── yolonas_s.onnx # Corresponding onnx model file 
│    
│── yolonas_s.trt  # Corresponding Trt model file 

```
### Known issues 
```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```
To solve the issue :
```
apt update
apt install -y libgl1-mesa-glx
```

## Pytorch to Fp16 TensorRT conversion 
- To convert Pytorch to Fp16 TensorRT without perfomance and evaluation metrices calculation use this script ```export_to_onnx_trt_fp16.py```  

To Run the conversion 

Inside the ```pytorch_trt_conversion``` folder

Required arguments 
- Number of classes (num_cls)
- Model checkpoint path( ckpt_path)
- Required batch size (batch_)

```
python export_to_onnx_trt_fp16.py --num_classes num_cls --model_path ckpt_path --batch_size batch_
```

example :
```python script.py --num_classes 9 --model_path ./ckpt_best.pth --batch_size 1```
