import sys
sys.path.insert(0, '/root/3DTrans/')

import pickle
import numpy as np
import pandas as pd
from pcdet.datasets.kitti.kitti_object_eval_python.eval import do_eval, eval_class, calculate_iou_partly

# Tahminleri ve ground truth verilerini yükleme
with open('/root/3DTrans/output/root/3DTrans/tools/cfgs/custom/pv_rcnn/default/eval/epoch_80/test/default/result.pkl', 'rb') as f:
    predictions = pickle.load(f)

with open('/root/3DTrans/data/custom_kitti2/custom_infos_test.pkl', 'rb') as f:
    ground_truth = pickle.load(f)

gt_annos=[]
for gt in ground_truth:
    gt["annos"]["bbox"]= gt["annos"]["gt_boxes_lidar"]
    gt["annos"]["alpha"]= np.asarray([0 for i in range(len(gt["annos"]["gt_boxes_lidar"]))])
    gt["annos"]["occluded"] = [0 for i in range(len(gt["annos"]["gt_boxes_lidar"]))]
    gt["annos"]["truncated"] = [0 for i in range(len(gt["annos"]["gt_boxes_lidar"]))]
    gt_annos.append(gt["annos"])


dt_annos=[]
for dt in predictions:
    dt["bbox"]= dt["boxes_lidar"]
    dt_annos.append(dt)



# Threshold değerlerini belirleme
ths = np.linspace(0.05, 0.95, 19)  # 0.1 ile 0.9 arasında 9 eşit aralıklı değer
#[num_minoverlap, metric, num_class]
min_overlaps = np.ones((ths.shape[0],3,5))
min_overlaps[:,2,0] = ths

# do_eval(gt_annos,  dt_annos, current_classes, min_overlaps, compute_aos=False, PR_detail_dict=None):
#mAP result: [num_class, num_diff, num_minoverlap]
mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = do_eval(gt_annos, dt_annos, [0], min_overlaps, compute_aos=False)


# eval_class(gt_annos, dt_annos, current_classes, difficultys,  metric, min_overlaps, compute_aos=False, num_parts=100):
res =eval_class(gt_annos,  dt_annos, [0], [0],  2, min_overlaps, compute_aos=False, num_parts=100)


"""
    # Sonuçları saklama
    result = {
        'Threshold': threshold,
        'mAP_bbox_R40': mAPbbox_R40[0, 0, 0],
        'mAP_bev_R40': mAPbev_R40[0, 0, 0],
        'mAP_3d_R40': mAP3d_R40[0, 0, 0],
    }
    results.append(result)

# Sonuçları bir DataFrame'e çevirme
results_df = pd.DataFrame(results)

# Sonuçları kaydetme
results_df.to_csv("evaluation_results.csv", index=False)

print("Testler tamamlandı ve sonuçlar evaluation_results.csv dosyasına kaydedildi.")
"""