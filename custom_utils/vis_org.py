#from mayavi import mlab
#mlab.test_contour3d()
#mlab.savefig('example.png')

import sys
sys.path.insert(0, '/root/3DTrans/')
sys.path.insert(0, '/root/3DTrans/tools')

import torch
from pathlib import Path

from pcdet.datasets.kitti.kitti_dataset import KittiDataset


from easydict import EasyDict

from pcdet.utils import common_utils


import yaml
def visualize_sample(dataset, index):
    sample = dataset[index]
    
    points = sample['points']
    gt_boxes = sample['gt_boxes']
    ref_boxes = sample.get('ref_boxes', None)
    scores = sample.get('scores', None)
    use_fakelidar = sample.get('use_fakelidar', False)
    
    KittiDataset.__vis__(points, gt_boxes, ref_boxes, scores, use_fakelidar)

class_names = ['Car', 'Pedestrian', 'Cyclist']
root_path = Path('/root/3DTrans/data/kitti')
dataset_cfg_path = '/root/3DTrans/tools/cfgs/dataset_configs/kitti/OD/kitti_dataset.yaml'
dataset_cfg = EasyDict(yaml.safe_load(open(dataset_cfg_path)))
dataset_cfg.USE_ROAD_PLANE=False
log_file = ('log_train_0.txt')
logger = common_utils.create_logger(log_file, rank=0)

dataset = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names, training=False, root_path=root_path, logger =logger)

sample_index = 319

visualize_sample(dataset, sample_index)
