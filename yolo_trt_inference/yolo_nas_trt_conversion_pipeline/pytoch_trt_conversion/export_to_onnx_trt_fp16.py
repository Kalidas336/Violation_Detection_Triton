import super_gradients
from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.conversion.conversion_enums import ExportTargetBackend, ExportQuantizationMode, DetectionOutputFormatMode
from typing import List, Tuple
import tensorrt as trt
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Convert YOLO-NAS model to TensorRT engine')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes in the model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint file')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for the model')
    parser.add_argument('--int8', action='store_true', help='Enable INT8 quantization')
    return parser.parse_args()

def convert_onnx_to_trt_engine(onnx_file, trt_output_file, enable_int8_quantization:bool = False):
    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    print(EXPLICIT_BATCH)

    with trt.Builder(trt_logger) as builder, builder.create_network(EXPLICIT_BATCH) as network, builder.create_builder_config() as config:

        config = builder.create_builder_config()
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        if enable_int8_quantization:
            config.set_flag(trt.BuilderFlag.INT8)
        else:
            config.set_flag(trt.BuilderFlag.FP16)

        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        # Register TensorRT plugins
        trt.init_libnvinfer_plugins(trt_logger, "")

        # Load your ONNX model
        with trt.OnnxParser(network, trt_logger) as onnx_parser:
            with open(onnx_file, 'rb') as f:
                parse_success = onnx_parser.parse(f.read())
                if not parse_success:
                    errors = "\n".join(
                        [str(onnx_parser.get_error(error)) for error in range(onnx_parser.num_errors)]
                    )
                    raise RuntimeError(f"Failed to parse onnx model for trt conversion. Errors: \n{errors}")

            trt_logger.log(trt.ILogger.INFO, "Parsed ONNX model")

        # Query input names and shapes from parsed TensorRT network
        network_inputs = [network.get_input(i) for i in range(network.num_inputs)]
        input_names = [_input.name for _input in network_inputs]  # ex: ["actual_input1"]

        assert input_names[0] == 'input'

        serialized_engine = builder.build_serialized_network(network, config)
        with open(trt_output_file, "wb") as output_file:
            output_file.write(serialized_engine)
            trt_logger.log(trt.ILogger.INFO, "Serialization done")

if __name__ == "__main__":
    args = parse_args()
    trt_logger = trt.Logger(trt.Logger.VERBOSE)
    
    # Load and export the model
    yolonas = models.get("yolo_nas_s", num_classes=args.num_classes, checkpoint_path=args.model_path)
    onnx_file = f"yolonas_s_b_{str(args.batch_size)}.onnx"
    yolonas.export(
        onnx_file, 
        preprocessing=True, 
        postprocessing=True, 
        engine=ExportTargetBackend.TENSORRT,
        batch_size=args.batch_size
    )
    
    # Convert to TensorRT engine
    trt_file = f"yolonas_s_b_{str(args.batch_size)}.trt"
    convert_onnx_to_trt_engine(onnx_file, trt_file, args.int8)
