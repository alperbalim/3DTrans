import sys
import pickle
import numpy as np

sys.path.insert(0, '/root/3DTrans/')

from pcdet.datasets.kitti.kitti_object_eval_python.eval import do_eval, eval_class, get_mAP_R40
from pcdet.datasets.kitti.kitti_utils import transform_annotations_to_kitti_format
from pcdet.datasets.kitti.kitti_object_eval_python.kitti_common import filter_annos_low_score

print("pcdet evaluation module imported successfully.")

map_name_to_kitti = {'Car':'Car', 'Pedestrian':'Pedestrian', 'Cyclist':'Cyclist'}

result_path = '/root/3DTrans/output/cfgs/MDF/KNW/customnw_pvrcnn_feat_3_uni3d/default/eval/epoch_2/val/default/result.pkl'
with open(result_path, 'rb') as f:
    dt_data = pickle.load(f)

infos_path = '/root/3DTrans/data/custom_kitti2/custom_infos_test.pkl'
with open(infos_path, 'rb') as f:
    gt_data = pickle.load(f)

print(f"Successfully loaded data from {result_path} and {infos_path}")

gt_annos = []
for gt_info in gt_data:
    anno = {
        "bbox": gt_info["annos"]["gt_boxes_lidar"],
        "gt_boxes_lidar": gt_info["annos"]["gt_boxes_lidar"],
        "name": gt_info["annos"]["name"],
        "dimensions": gt_info["annos"]["dimensions"],
        "location": gt_info["annos"]["location"],
        "rotation_y": gt_info["annos"]["rotation_y"],
        "alpha": np.asarray([0.0] * len(gt_info["annos"]["gt_boxes_lidar"]), dtype=np.float32),
        "occluded": np.asarray( [0.0] * len(gt_info["annos"]["gt_boxes_lidar"]), dtype=np.int32),
        "truncated": np.asarray([0.0] * len(gt_info["annos"]["gt_boxes_lidar"]), dtype=np.float32)
    }
    anno["difficulty"] = np.asarray( [0.0]* len(gt_info["annos"]["gt_boxes_lidar"]), dtype=np.int32)
    gt_annos.append(anno)

gt_annos = transform_annotations_to_kitti_format(gt_annos, map_name_to_kitti=map_name_to_kitti)

dt_annos_raw = []
for dt in dt_data:
    dt_anno = {
        "bbox": dt["boxes_lidar"],
        "boxes_lidar": dt["boxes_lidar"],
        "score": dt["score"],
        "name": dt["name"],
        "alpha" : np.asarray([0 for i in range(len(dt["boxes_lidar"]))]),
        "occluded" : np.asarray([0 for i in range(len(dt["boxes_lidar"]))]),
        "truncated" : np.asarray([0 for i in range(len(dt["boxes_lidar"]))]),
        "location": dt["location"],
        "dimensions": dt["dimensions"],
        "rotation_y": dt["rotation_y"]
    }
    dt_annos_raw.append(dt_anno)

dt_annos_raw = transform_annotations_to_kitti_format(dt_annos_raw, map_name_to_kitti=map_name_to_kitti)

min_overlaps = np.array([0.3, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

class_ids_to_eval = [0] # Car class (usually 0 in KITTI)
difficulty_levels_to_eval = [0] # Single difficulty bin (all combined)

num_metrics = 4
min_overlaps_for_map = np.zeros((len(min_overlaps), num_metrics, len(class_ids_to_eval)), dtype=np.float32)
min_overlaps_for_map[:, 2, 0] = min_overlaps

print("\nCalculating standard mAP (do_eval)...")

#dt_annos = filter_annos_low_score(dt_annos_raw, 0.0) # Keep all detections for mAP calculation

mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = do_eval(
    gt_annos, dt_annos_raw, class_ids_to_eval, min_overlaps_for_map, compute_aos=False
)

print("\n--- Standard mAP Results (Car, All Difficulties Combined) ---")

c_idx = 0
d_idx = 0

print(f"Class: Car (ID: {class_ids_to_eval[c_idx]})")
print("All Difficulties Combined:")

for i_idx, iou_thresh in enumerate(min_overlaps):
    if i_idx < mAP3d.shape[1]:
      map_value = mAP3d[c_idx, d_idx, i_idx]
      print(f" 3D mAP @ IoU={iou_thresh:.2f}: {map_value:.4f}")

for i_idx, iou_thresh in enumerate(min_overlaps):
    if mAP3d_R40 is not None and i_idx < mAP3d_R40.shape[1]:
        map_r40_value = mAP3d_R40[c_idx, d_idx, i_idx]
        print(f" 3D mAP@R40 @ IoU:{iou_thresh:.2f}: {map_r40_value:.4f}")
print("-" * 25)

print("\n--- Confidence Score Threshold Analysis (Car, All Difficulties Combined) ---")

iou_thresholds_for_confidence_analysis = [0.5, 0.7]
confidence_thresholds_to_analyze = np.arange(0.05, 1.05 , 0.05)

analysis_class_id = 0
analysis_difficulty_id = 0

for current_iou_for_analysis in iou_thresholds_for_confidence_analysis:
    print(f"\nAnalyzing for IoU Threshold > {current_iou_for_analysis}...")

    min_overlaps_single_iou = np.ones((1, 3, 1), dtype=np.float32)
    min_overlaps_single_iou[:, 2, 0] = current_iou_for_analysis

    res_eval_class = eval_class(
        gt_annos, dt_annos_raw, [analysis_class_id], [analysis_difficulty_id], 2, min_overlaps_single_iou, compute_aos=False
    )

    eval_precision_points = res_eval_class['precision'][0, 0, 0, :]
    eval_recall_points = res_eval_class['recall'][0, 0, 0, :]
    eval_confidence_points = res_eval_class['thresholds']

    if len(eval_confidence_points) == 0:
        print(f"No evaluation points returned by eval_class for IoU > {current_iou_for_analysis}. P/R/F1 will be 0.")
        for conf_thresh in confidence_thresholds_to_analyze:
            print(f" {conf_thresh:.2f} | {0.0:.4f} | {0.0:.4f} | {0.0:.4f}")

    else:
        print(" Confidence | Precision | Recall | F1-Score")
        print(" -------------------------------------------------")

        for conf_thresh in confidence_thresholds_to_analyze:
            index = np.searchsorted(eval_confidence_points, conf_thresh, side='right') - 1

            if index >= 0:
                precision = eval_precision_points[index]
                recall = eval_recall_points[index]
            else:
                precision = 0.0
                recall = 0.0

            denominator = precision + recall
            f1 = np.where(denominator > 0, 2 * (precision * recall) / denominator, 0.0) if denominator > 1e-6 else 0.0

            print(f" {conf_thresh:.2f} | {precision:.4f} | {recall:.4f} | {f1:.4f}")

print("\nAnalysis Complete.")