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

import glob
import os
import re
result_dir = "/root/3DTrans/output/root/3DTrans/tools/cfgs/custom/centerpoint/default/eval/eval_all_default/default"
result_list = glob.glob(os.path.join(result_dir, 'epoch_*/test/result.pkl'))


eval_gt_annos = [copy.deepcopy(info['annos']) for info in dataset.custom_infos]
eval_gt_annos = kitti_utils.transform_annotations_to_kitti_format(eval_gt_annos, map_name_to_kitti=dataset.map_class_to_kitti, info_with_fakelidar= False)
    

epoch_list=[]


from prettytable import PrettyTable
t = PrettyTable(['Epoch', "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"])

for res in result_list:
    epoch = re.findall('epoch_(.*)/test/result.pkl', res)[0]

    with open(res, 'rb') as f:
        dt_annos = pickle.load(f)

    ths = np.linspace(0.1, 0.9, 9)  # 0.1 ile 0.9 arasında 9 eşit aralıklı değer
    #[num_minoverlap, metric, num_class]
    min_overlaps = np.ones((ths.shape[0],3,3))
    for i in range(0,3):
        for j in range(0,3):
            min_overlaps[:,i,j] = ths

    eval_det_annos = copy.deepcopy(dt_annos)
    eval_det_annos = kitti_utils.transform_annotations_to_kitti_format(dt_annos, map_name_to_kitti=dataset.map_class_to_kitti)

    current_classes =[0]
    difficultys =[0]
    #metric: eval type. 0: bbox, 1: bev, 2: 3d
    metric = 2
    compute_aos=False
    # eval_class(gt_annos, dt_annos, current_classes, difficultys,  metric, min_overlaps, compute_aos=True, num_parts=100):

    mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = do_eval(eval_gt_annos, eval_det_annos, [0], min_overlaps, compute_aos=True)
    
    list(mAPbbox_R40[0,0,:])
    t.add_row([epoch]+list(mAPbbox_R40[0,0,:]))
    
t.float_format = "10.2"
print(t)
