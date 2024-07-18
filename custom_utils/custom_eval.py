import sys
sys.path.insert(0, '/root/3DTrans/')
sys.path.insert(0, '/root/3DTrans/tools')
import torch
from pathlib import Path
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.datasets.custom.custom_dataset import CustomDataset

from pcdet.datasets.kitti import kitti_utils

from easydict import EasyDict
from pcdet.utils import common_utils
import numpy as np
import yaml
import pickle
import copy
from pcdet.datasets.kitti.kitti_object_eval_python.eval import *
from pcdet.datasets.kitti.kitti_object_eval_python.eval import _prepare_data

class_names = ['Car', 'Pedestrian', 'Cyclist']
root_path = Path('/root/3DTrans/data/custom_kitti2')
dataset_cfg_path = '/root/3DTrans/tools/cfgs/dataset_configs/custom/custom_dataset_custom.yaml'
dataset_cfg = EasyDict(yaml.safe_load(open(dataset_cfg_path)))
dataset_cfg.USE_ROAD_PLANE=False
log_file = ('log_dummy.txt')
logger = common_utils.create_logger(log_file, rank=0)
dataset = CustomDataset(dataset_cfg=dataset_cfg, class_names=class_names, training=False, root_path=root_path, logger =logger)


with open('/root/3DTrans/output/root/3DTrans/tools/cfgs/custom/pv_rcnn/default/eval/epoch_80/test/default/result.pkl', 'rb') as f:
    dt_annos = pickle.load(f)

#args ={"eval_metric":"kitti"}
#dataset.evaluation(dt_annos,class_names, eval_metric="kitti")

ths = np.linspace(0.05, 0.95, 19)  # 0.1 ile 0.9 arasında 9 eşit aralıklı değer
#[num_minoverlap, metric, num_class]
min_overlaps = np.ones((ths.shape[0],3,5))
min_overlaps[:,2,0] = ths

eval_det_annos = copy.deepcopy(dt_annos)
eval_det_annos = kitti_utils.transform_annotations_to_kitti_format(dt_annos, map_name_to_kitti=dataset.map_class_to_kitti)

eval_gt_annos = [copy.deepcopy(info['annos']) for info in dataset.custom_infos]
eval_gt_annos = kitti_utils.transform_annotations_to_kitti_format(eval_gt_annos, map_name_to_kitti=dataset.map_class_to_kitti, info_with_fakelidar= False)


current_classes =[0]
compute_aos=False

"""
mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = do_eval(
        eval_gt_annos, eval_det_annos, current_classes, min_overlaps, compute_aos, PR_detail_dict=None)
ret =eval_class(eval_gt_annos,  eval_det_annos, [0], [0],  2, min_overlaps, compute_aos=False, num_parts=100)
mAP_3d_R40 = get_mAP_R40(ret["precision"])
mAP_bev_R40 = get_mAP_R40(ret["precision"])
"""
rets = _prepare_data(eval_gt_annos, eval_det_annos, 0, 0)
(gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares, total_dc_num, total_num_valid_gt) = rets
overlaps, parted_overlaps, total_dt_num, total_gt_num  = calculate_iou_partly(eval_det_annos, eval_gt_annos, 2, 41)


from prettytable import PrettyTable
t = PrettyTable(['TH', 'Precisin', "Recall", "tot"])

t2 = PrettyTable(['TH', 'tp', "fp", "fn"])

for th in ths:
    tp_s = 0
    fp_s = 0
    fn_s = 0
    for i in range(len(dt_datas_list)):
        tp, fp, fn, similarity, thresholds = compute_statistics_jit(
                                overlaps[i],
                                gt_datas_list[i],
                                dt_datas_list[i],
                                ignored_gts[i],
                                ignored_dets[i],
                                dontcares[i],
                                2,
                                min_overlap=th,
                                thresh=0.7,
                                compute_fp=True)

        tp_s += tp 
        fp_s += fp
        fn_s += fn
        
    t.add_row([th, tp_s/(tp_s+fp_s), tp_s/(tp_s+fn_s), tp_s/(tp_s+fn_s+fp_s)])
    t2.add_row([th, tp_s, fp_s, fn_s])

print(t)

print(t2)


