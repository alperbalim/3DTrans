import sys
sys.path.insert(0, '/root/3DTrans/')
sys.path.insert(0, '/root/3DTrans/tools')
import torch
from pathlib import Path
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.datasets.custom.custom_dataset import CustomDataset
from pcdet.datasets.waymo.waymo_dataset import WaymoDataset
from pcdet.datasets.kitti import kitti_utils
from easydict import EasyDict
from pcdet.utils import common_utils
import numpy as np
import yaml
import pickle
import copy
from pcdet.datasets.kitti.kitti_object_eval_python.eval_partly import *
from pcdet.datasets.kitti.kitti_object_eval_python.eval_partly import _prepare_data

MAP_CLASS_TO_KITTI= {
    'Vehicle': 'Car',
    'Pedestrian': 'Pedestrian',
    'Cyclist': 'Cyclist',
    'Sign': 'Sign',
    'Car': 'Car'
}

class_names = ['Vehicle', 'Pedestrian', 'Cyclist']
root_path = Path('/root/3DTrans/data/waymo')
dataset_cfg_path = '/root/3DTrans/tools/cfgs/dataset_configs/waymo/OD/waymo_dataset.yaml'
dataset_cfg = EasyDict(yaml.safe_load(open(dataset_cfg_path)))
dataset_cfg.USE_ROAD_PLANE=False
log_file = ('log_dummy.txt')
logger = common_utils.create_logger(log_file, rank=0)
dataset = WaymoDataset(dataset_cfg=dataset_cfg, class_names=class_names, training=False, root_path=root_path, logger =logger)

with open('/root/3DTrans/output/root/3DTrans/tools/cfgs/waymo_models/pv_rcnn/default/eval/eval_with_train/epoch_30/val/result.pkl', 'rb') as f:
    dt_annos = pickle.load(f)

#eval_det_annos = copy.deepcopy(dt_annos)


kwargs={}
kwargs['eval_metric'] = 'kitti'
with torch.no_grad():
    ap_result_str, ap_dict = dataset.evaluation(dt_annos, class_names, output_path="/root/3DTrans/output/root/3DTrans/tools/cfgs/waymo_models/pv_rcnn/default/waymo_kitti_eval",**kwargs)
print(ap_result_str)
print(ap_dict)

#eval_gt_annos = [copy.deepcopy(info['annos']) for info in dataset.infos]
#mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = do_eval(eval_gt_annos[0:1000], eval_det_annos[0:1000], [0], min_overlaps, compute_aos=True)


"""
mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = do_eval(eval_gt_annos[0:1000], eval_det_annos[0:1000], [0], min_overlaps, compute_aos=True)

current_classes =[0]
difficultys =[0]
#metric: eval type. 0: bbox, 1: bev, 2: 3d
metric = 2
compute_aos=False
# eval_class(gt_annos, dt_annos, current_classes, difficultys,  metric, min_overlaps, compute_aos=True, num_parts=100):

#mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = do_eval(eval_gt_annos, eval_det_annos, [0], min_overlaps, compute_aos=True)
#res =eval_class(eval_gt_annos,  eval_det_annos, [0], [0],  2, min_overlaps, compute_aos=False, num_parts=100)


dataset.evalu
from prettytable import PrettyTable
t = PrettyTable(['TH', 'mAPbbox_R40', "mAPbev_R40", "mAP3d_R40", "mAPaos_R40", "recall"])


for i in range(0,ths.shape[0]):
    mAP = mAP3d_R40[0,0,i]
    rc = res["recall"][0,0,i,20]
    t.add_row([ths[i], mAPbbox_R40[0,0,i], mAPbev_R40[0,0,i], mAP3d_R40[0,0,i], mAPaos_R40[0,0,i],rc])
t.float_format = "10.2"
print(t)
"""