import os
import time
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# You may replace these with actual class names
CLS_NAMES = ["person", "car", "bike", "truck"]

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))
    return inputs, outputs, bindings, stream

def preprocess_image(image, input_size=(640, 640)):
    h, w = image.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    image_resized = cv2.resize(image, (nw, nh))
    pad_top = (input_size[1] - nh) // 2
    pad_left = (input_size[0] - nw) // 2
    padded = np.full((input_size[1], input_size[0], 3), 114, dtype=np.uint8)
    padded[pad_top:pad_top+nh, pad_left:pad_left+nw] = image_resized
    img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_transposed = np.transpose(img_rgb, (2, 0, 1))[None, ...]
    return img_transposed, scale, (pad_left, pad_top), padded

def run_inference(engine, context, inputs, outputs, bindings, stream, image_np):
    np.copyto(inputs[0][0], image_np.ravel())
    cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0][0], outputs[0][1], stream)
    stream.synchronize()
    return outputs[0][0]

def yolo_normalized(box, scale, pad, orig_w, orig_h):
    x1, y1, x2, y2 = box
    pad_x, pad_y = pad
    x1 -= pad_x
    x2 -= pad_x
    y1 -= pad_y
    y2 -= pad_y
    x1 /= scale
    x2 /= scale
    y1 /= scale
    y2 /= scale
    x_center = (x1 + x2) / 2 / orig_w
    y_center = (y1 + y2) / 2 / orig_h
    width = (x2 - x1) / orig_w
    height = (y2 - y1) / orig_h
    return x_center, y_center, width, height

def main(trt_engine_path, image_dir, vis_dir, txt_dir, input_size=(640, 640)):
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)

    engine = load_engine(trt_engine_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_time = 0
    total_images = 0
    total_detections = 0

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: failed to load {img_path}")
            continue

        orig_h, orig_w = image.shape[:2]
        image_np, scale, pad, resized = preprocess_image(image, input_size)

        start = time.time()
        output = run_inference(engine, context, inputs, outputs, bindings, stream, image_np)
        inference_time = time.time() - start

        total_time += inference_time
        total_images += 1

        # Dummy output parsing — replace with your model's actual output format
        # Let's assume output is Nx7: [x1, y1, x2, y2, conf, class_id, keep_flag]
        output = output.reshape(-1, 7)
        boxes = output[:, :4]
        scores = output[:, 4]
        class_ids = output[:, 5]

        yolo_annotations = []
        for box, score, cls_id in zip(boxes, scores, class_ids):
            if score < 0.4:  # Confidence threshold
                continue
            box = box.astype(np.int32)
            x1, y1, x2, y2 = box
            x_center, y_center, w, h = yolo_normalized(box, scale, pad, orig_w, orig_h)

            yolo_annotations.append(f"{int(cls_id)} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} {score:.6f}")
            label = f"{CLS_NAMES[int(cls_id)]} {score:.2f}"
            cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Save image with bboxes
        cv2.imwrite(os.path.join(vis_dir, img_file), resized)

        # Save YOLO .txt annotation
        with open(os.path.join(txt_dir, img_file.replace(".jpg", ".txt")), "w") as f:
            f.write("\n".join(yolo_annotations))

        total_detections += len(yolo_annotations)

    print(f"\n--- Inference Summary ---")
    print(f"Total images: {total_images}")
    print(f"Total detections: {total_detections}")
    print(f"Avg Inference Time: {total_time/total_images:.4f} sec")
    print(f"FPS: {total_images/total_time:.2f}")

if __name__ == "__main__":
    main(
        trt_engine_path="/home/mtx003/shilpa/nvinferserver/model_repository/yolo_nas/1/model.plan",
        image_dir="/home/mtx003/kalidas/violation_det/cropped",
        vis_dir="/home/mtx003/kalidas/violation_det/efficient_nms/vis",
        txt_dir="/home/mtx003/kalidas/violation_det/efficient_nms/trt"
    )
