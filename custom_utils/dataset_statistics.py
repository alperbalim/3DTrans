
import sys
sys.path
sys.path.append('/root/3DTrans')
import random
import easyyaml
import numpy as np
from pcdet.datasets import build_dataloader
import logging
from torch.utils.data import RandomSampler
import random

logger = logging.getLogger("pcdet_logger")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

# Veri setleri ve etiket sınıfları tanımları
datasets = ['nuscenes','custom', 'awsim','waymo', 'nuscenes']
target_labels = {
    'waymo': ['Vehicle'],
    'nuscenes': ['car', 'truck', 'construction_vehicle','bus','trailer'],
    'custom': ['Car'],
    'awsim': ['Car']
}


cfg_files = {
    'waymo': '/root/3DTrans/tools/cfgs/dataset_configs/waymo/OD/waymo_dataset.yaml',
    'nuscenes': '/root/3DTrans/tools/cfgs/dataset_configs/nuscenes/OD/nuscenes_dataset.yaml',
    'custom': '/root/3DTrans/tools/cfgs/dataset_configs/custom/custom_dataset_custom.yaml',
    'awsim': '/root/3DTrans/tools/cfgs/dataset_configs/custom/awsim_dataset_2.yaml'
}

# Config dosyalarını okuyup yapılandırmaları yükleyin
cfg_dict = {}
for ds, filepath in cfg_files.items():
    cfg_dict[ds] = easyyaml.load(filepath)  # EasyYAML ile yükleme

sample_size = 100
# Dataloader'ları oluşturun
data_loaders = {}
sample_ind = {}
samples = {}
annos = {}
for ds in datasets:
    cfg = cfg_dict[ds]
    classes = target_labels[ds]
    dataloader = build_dataloader(cfg, classes, batch_size=1, dist=False, root_path=None, workers=4,
                                  logger=logger, training=False, merge_all_iters_to_one_epoch=False, total_epochs=0)
    dataset =dataloader[0]
    data_loaders[ds] = dataloader[1]
    sample_ind[ds] = random.sample(range(0,len(dataset)), sample_size)
    samples[ds] = [dataset[s_ind] for s_ind in sample_ind[ds]]
    if ds in ["custom", "awsim"]:
        annos[ds] = [dataset.custom_infos[s_ind]["annos"] for s_ind in sample_ind[ds]]
    elif ds =="waymo":
        annos[ds] = [dataset.infos[s_ind]["annos"] for s_ind  in sample_ind[ds]]
    else:
        dataset.transform_to_kitti_format()
        annos[ds] = [dataset.annos_kitti[s_ind] for s_ind  in sample_ind[ds]]



# İstatistikleri hesaplayın
stats = {}

for ds in datasets:
    point_counts=[]
    box_sizes = []
    points_in_box = []
    class_counts=0
    for sample, anno in zip(samples[ds], annos[ds]):
        point_counts.append(len(sample["points"]))
        labels = anno["name"]
        points = sample['points'][:, :3]  # Nokta bulutunun XYZ koordinatları
        transformed_points = points

        for i, label in enumerate(labels):
            if label in target_labels[ds]:
                class_counts = class_counts + 1
                box_center = anno["location"]
                box_dims = anno["dimensions"]
                box_sizes.append(box_dims)

                mask = (
                    (transformed_points[:, 0] >= (box_center[0] - box_dims[0] / 2)) & 
                    (transformed_points[:, 0] <= (box_center[0] + box_dims[0] / 2)) &
                    (transformed_points[:, 1] >= (box_center[1] - box_dims[1] / 2)) &
                    (transformed_points[:, 1] <= (box_center[1] + box_dims[1] / 2)) &
                    (transformed_points[:, 2] >= (box_center[2] - box_dims[2] / 2)) &
                    (transformed_points[:, 2] <= (box_center[2] + box_dims[2] / 2))
                )
                points_in_box.append(np.sum(mask))

    # İstatistikleri hesaplayın
    avg_points = sum(point_counts) / len(point_counts)
    min_box_size = min(box_sizes) if box_sizes else [0, 0, 0]
    max_box_size = max(box_sizes) if box_sizes else [0, 0, 0]
    avg_points_in_box = sum(points_in_box) / len(points_in_box) if points_in_box else 0

    stats[ds] = {
        'avg_points': avg_points,
        'min_box_size': min_box_size,
        'max_box_size': max_box_size,
        'class_distribution': class_counts,
        'total_samples': len(samples),
        f'{", ".join(target_labels[ds])}_points_in_box_avg': avg_points_in_box,
    }
    print(stats[ds])
# İstatistikleri tablo halinde yazdırın
import pandas as pd

df = pd.DataFrame(stats).T
print(df)
