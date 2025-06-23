import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def load_coco_data(gt_path, pred_path):
    """
    Load ground truth and prediction COCO JSON files
    """
    # Load ground truth COCO annotations
    coco_gt = COCO(gt_path)
    
    # Load predictions
    with open(pred_path, 'r') as f:
        pred_json = json.load(f)
    
    # Create COCO prediction object
    coco_pred = coco_gt.loadRes(pred_json)
    
    return coco_gt, coco_pred

def run_coco_evaluation(coco_gt, coco_pred, category_ids=None):
    """
    Run COCO evaluation and return metrics
    """
    # Initialize COCOeval object
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    
    # Set specific category IDs if provided
    if category_ids:
        coco_eval.params.catIds = category_ids
    
    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extract key metrics
    metrics = {
        'mAP@0.5': coco_eval.stats[1],  # AP at IoU=0.5
        'mAP@0.5:0.95': coco_eval.stats[0],  # AP at IoU=0.5:0.95
    }
    
    return coco_eval, metrics

def calculate_precision_recall_f1(coco_gt, coco_pred, iou_threshold=0.5):
    """
    Calculate precision, recall and F1 score at specified IoU threshold
    """
    # Get all images and categories
    img_ids = sorted(coco_gt.getImgIds())
    cat_ids = coco_gt.getCatIds()
    
    # Counters for true positives, false positives, and false negatives
    tp = 0
    fp = 0
    fn = 0
    
    # Process each image
    for img_id in img_ids:
        for cat_id in cat_ids:
            # Get ground truth annotations for this image and category
            gt_anns = coco_gt.getAnnIds(imgIds=img_id, catIds=[cat_id])
            gt_anns = coco_gt.loadAnns(gt_anns)
            
            # Get predicted annotations for this image and category
            pred_anns = coco_pred.getAnnIds(imgIds=img_id, catIds=[cat_id])
            pred_anns = coco_pred.loadAnns(pred_anns)
            
            # Match predictions to ground truth
            matched_gt = set()
            
            for pred in pred_anns:
                best_iou = 0
                best_gt_idx = -1
                
                for i, gt in enumerate(gt_anns):
                    if i in matched_gt:
                        continue
                    
                    # Calculate IoU between prediction and ground truth
                    pred_bbox = pred['bbox']
                    gt_bbox = gt['bbox']
                    
                    # Convert COCO bbox format [x, y, width, height] to [x1, y1, x2, y2]
                    pred_x1, pred_y1, pred_w, pred_h = pred_bbox
                    pred_x2, pred_y2 = pred_x1 + pred_w, pred_y1 + pred_h
                    
                    gt_x1, gt_y1, gt_w, gt_h = gt_bbox
                    gt_x2, gt_y2 = gt_x1 + gt_w, gt_y1 + gt_h
                    
                    # Calculate intersection coordinates
                    x_left = max(pred_x1, gt_x1)
                    y_top = max(pred_y1, gt_y1)
                    x_right = min(pred_x2, gt_x2)
                    y_bottom = min(pred_y2, gt_y2)
                    
                    # No overlap if intersection doesn't exist
                    if x_right < x_left or y_bottom < y_top:
                        iou = 0
                    else:
                        # Area of intersection
                        intersection_area = (x_right - x_left) * (y_bottom - y_top)
                        
                        # Area of both boxes
                        pred_area = pred_w * pred_h
                        gt_area = gt_w * gt_h
                        
                        # Calculate IoU
                        union_area = pred_area + gt_area - intersection_area
                        iou = intersection_area / union_area
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
                
                # If IoU exceeds threshold, count as true positive
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1
            
            # Count unmatched ground truths as false negatives
            fn += len(gt_anns) - len(matched_gt)
    
    # Calculate precision, recall and F1
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

def plot_precision_recall_curve(coco_eval):
    """
    Plot precision-recall curves from COCOeval object
    """
    precisions = coco_eval.eval['precision']
    # Precision is of shape [T, R, K, A, M]
    # T: IoU thresholds [0.5:0.05:0.95], size 10
    # R: recall thresholds [0:0.01:1], size 101
    # K: category, size 80 in COCO
    # A: area range, size 4 in COCO
    # M: max detections, size 3 in COCO
    
    # Extract precision at IoU=0.5 for all categories, all areas, and 100 detections max
    p = precisions[0, :, :, 0, 2]  # IoU=0.5, all categories, all areas, 100 detections
    
    # Average precision across categories
    p_mean = np.mean(p, axis=1)
    
    # Recall values (fixed for all precisions)
    r = np.arange(0, 1.01, 0.01)
    
    plt.figure(figsize=(10, 7))
    plt.plot(r, p_mean, 'b-', label='mean')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve at IoU=0.5')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.grid(True)
    plt.legend()
    plt.savefig('precision_recall_curve.png')
    plt.show()

def print_category_metrics(coco_gt, coco_eval):
    """
    Print per-category metrics
    """
    # Get category info
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    cat_names = {cat['id']: cat['name'] for cat in cats}
    
    # Per-category AP at IoU=0.5
    precision_50 = coco_eval.eval['precision'][0, :, :, 0, 2]  # IoU=0.5
    
    # Calculate AP per category
    ap_per_category = defaultdict(float)
    for cat_idx, cat_id in enumerate(coco_eval.params.catIds):
        ap = np.mean(precision_50[:, cat_idx])
        ap_per_category[cat_names[cat_id]] = ap
    
    # Print results
    print("\nPer-category AP at IoU=0.5:")
    for cat_name, ap in sorted(ap_per_category.items()):
        print(f"{cat_name}: {ap:.4f}")

def main():
    # File paths - replace with your actual paths
    gt_path = r'/home/user/VARUN/Benchmarking_YOLONAS/tensorrt_inference/yolo_to_coco.json'
    pred_path = r'/home/user/VARUN/Benchmarking_YOLONAS/tensorrt_inference/int8_infer.json'
    
    # Load COCO data
    print("Loading COCO data...")
    coco_gt, coco_pred = load_coco_data(gt_path, pred_path)
    
    # Run COCO evaluation
    print("Running COCO evaluation...")
    coco_eval, metrics = run_coco_evaluation(coco_gt, coco_pred)
    
    # Calculate precision, recall, F1
    print("Calculating precision, recall, and F1 score...")
    pr_metrics = calculate_precision_recall_f1(coco_gt, coco_pred, iou_threshold=0.5)
    
    # Print results
    print("\nOverall Metrics:")
    print(f"mAP@0.5: {metrics['mAP@0.5']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
    print(f"Precision@0.5: {pr_metrics['precision']:.4f}")
    print(f"Recall@0.5: {pr_metrics['recall']:.4f}")
    print(f"F1@0.5: {pr_metrics['f1']:.4f}")
    
    # Print per-category metrics
    print_category_metrics(coco_gt, coco_eval)
    
    # Plot precision-recall curve
    print("\nPlotting precision-recall curve...")
    plot_precision_recall_curve(coco_eval)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()