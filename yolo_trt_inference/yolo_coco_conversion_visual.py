import os
import json
from PIL import Image, ImageDraw, ImageFont

# Class names (must match the order used in YOLO annotations)
cls_names = ["ped", "bicycle", "car", "two-wheeler", "Mini-bus", "bus", "Mini-truck", "truck", "three-wheeler"]

def yolo_to_coco(yolo_folder, output_json_path, output_image_folder):

    count=0
    # Initialize COCO JSON structure
    coco_output = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Add categories (class names)
    for idx, cls_name in enumerate(cls_names):
        coco_output["categories"].append({
            "id": idx + 1,  # COCO category IDs start from 1
            "name": cls_name,
            "supercategory": "none"
        })

    # Initialize annotation ID counter
    annotation_id = 1

    # Create output image folder if it doesn't exist
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

    # Loop through all images in the folder
    for image_file in os.listdir(yolo_folder):

        if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue  # Skip non-image files

        # Load image to get dimensions
        image_path = os.path.join(yolo_folder, image_file)
        img = Image.open(image_path)
        img_width, img_height = img.size

        # Add image info to COCO JSON
        image_id = len(coco_output["images"]) + 1

        # print(image_id)

        coco_output["images"].append({
            "id": image_id,
            "file_name": image_file,
            "width": img_width,
            "height": img_height
        })

        # Read corresponding YOLO annotation file
        yolo_file = os.path.splitext(image_file)[0] + ".txt"
        yolo_path = os.path.join(yolo_folder_v2, yolo_file)

        if not os.path.exists(yolo_path):
            print(yolo_path)
            continue  # Skip if no annotation file exists

        with open(yolo_path, "r") as f:
            lines = f.readlines()

        # Create a drawing context
        draw = ImageDraw.Draw(img)

        # Load a font (you may need to adjust the path to the font file)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()

        # Process each annotation
        for line in lines:

            count+=1
            print(count)
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

            # # # Skip invalid boxes
            if width <= 0 or height <= 0 or x_min < 0 or y_min < 0 or xmax > img_width or ymax > img_height:

                print("Yes","*"*60)
                break
                continue

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

            # Draw the bounding box and label on the image
            label = cls_names[int(class_id)]
            draw.rectangle([x_min, y_min,xmax,ymax ], outline="red", width=2)
            draw.text((x_min, y_min - 15), label, fill="red", font=font)

            annotation_id += 1

        # Save the image with detections
        output_image_path = os.path.join(output_image_folder, image_file)
        img.save(output_image_path)

    # Save COCO JSON to file
    with open(output_json_path, "w") as f:
        json.dump(coco_output, f, indent=4)

    print(f"COCO JSON saved to {output_json_path}")
    print(f"Images with detections saved to {output_image_folder}")

# Example usage
imgFolder = r"/home/user/VARUN/Benchmarking_YOLONAS/tensorrt_inference/Test_dataset"  # Folder containing images and YOLO annotation files
yolo_folder_v2 = r"/home/user/VARUN/Benchmarking_YOLONAS/tensorrt_inference/Tensort_INT8_prediction_INT8/YOLO"
output_json_path = "int8_infer.json"
output_image_folder = "torch_int8_visual"

yolo_to_coco(imgFolder, output_json_path, output_image_folder)