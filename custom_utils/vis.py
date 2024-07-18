#from mayavi import mlab
#mlab.test_contour3d()
#mlab.savefig('example.png')

import sys
sys.path.insert(0, '/root/3DTrans/')
sys.path.insert(0, '/root/3DTrans/tools')

import torch
from pathlib import Path

from pcdet.datasets.custom.custom_dataset import CustomDataset

#from pcdet.datasets.kitti.kitti_dataset import KittiDataset as KittiCustomDataset

from easydict import EasyDict

from pcdet.utils import common_utils

import numpy as np
import yaml
def visualize_sample(dataset, index):
    sample = dataset[index]
    
    points = sample['points']
    gt_boxes = sample['gt_boxes']
    #gt_boxes[0]=np.asarray([-17.670000076293945, 0.3799999952316284, 1.7400000095367432, 4.39, 2.01, 2.64, -4.4, 1])
    ref_boxes = sample.get('ref_boxes', None)
    scores = sample.get('scores', None)
    use_fakelidar = sample.get('use_fakelidar', False)
    
    CustomDataset.__vis__(points, gt_boxes, ref_boxes, scores, use_fakelidar)

class_names = ['Car', 'Pedestrian', 'Cyclist']
root_path = Path('/root/3DTrans/data/custom_kitti2')
dataset_cfg_path = '/root/3DTrans/tools/cfgs/dataset_configs/custom/custom_dataset_custom.yaml'
#dataset_cfg_path = '/root/3DTrans/tools/cfgs/dataset_configs/custom/custom_dataset_org.yaml'
dataset_cfg = EasyDict(yaml.safe_load(open(dataset_cfg_path)))

log_file = ('log_train_0.txt')
logger = common_utils.create_logger(log_file, rank=0)

dataset = CustomDataset(dataset_cfg=dataset_cfg, class_names=class_names, training=False, root_path=root_path, logger =logger)

sample_index = 121

visualize_sample(dataset, sample_index)
