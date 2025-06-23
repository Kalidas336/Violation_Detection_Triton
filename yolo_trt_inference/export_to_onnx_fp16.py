import super_gradients
from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.conversion.conversion_enums import ExportTargetBackend, ExportQuantizationMode, DetectionOutputFormatMode

yolonas = models.get("yolo_nas_s", num_classes=9, checkpoint_path="/home/user/VARUN/Benchmarking_YOLONAS/ckpt_best.pth")
yolonas.export("yolonas_s.onnx", preprocessing=True, postprocessing=True, engine=ExportTargetBackend.TENSORRT)