import sys
sys.path.append('/root/3DTrans')
import random
import easyyaml
import numpy as np
import logging
from pcdet.datasets import build_dataloader
from torch.utils.data import RandomSampler
import pandas as pd

# Logger ayarları
logger = logging.getLogger("pcdet_logger")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

# Veri setleri ve etiket sınıfları tanımları
datasets = ['waymo', 'nuscenes', 'custom', 'awsim']
target_labels = {
    'waymo': ['Vehicle'],
    'nuscenes': ['car', 'truck', 'construction_vehicle', 'bus', 'trailer'],
    'custom': ['Car'],
    'awsim': ['Car']
}

# Config dosya yolları
cfg_files = {
    'waymo': '/root/3DTrans/tools/cfgs/dataset_configs/waymo/OD/waymo_dataset.yaml',
    'nuscenes': '/root/3DTrans/tools/cfgs/dataset_configs/nuscenes/OD/nuscenes_dataset.yaml',
    'custom': '/root/3DTrans/tools/cfgs/dataset_configs/custom/custom_dataset_custom.yaml',
    'awsim': '/root/3DTrans/tools/cfgs/dataset_configs/custom/awsim_dataset_2.yaml'
}

# Config dosyalarını okuyup yapılandırmaları yükleyin
cfg_dict = {}
for ds, filepath in cfg_files.items():
    cfg_dict[ds] = easyyaml.load(filepath)

sample_size = 100
# Veri seti bilgilerini yükle ve rastgele örnekleri seç
samples = {}
for ds in datasets:
    cfg = cfg_dict[ds]
    classes = target_labels[ds]

    # build_dataloader ile veri seti oluşturun
    dataloader = build_dataloader(
        cfg,
        class_names=classes,
        batch_size=1,
        dist=False,
        root_path=None,
        workers=4,
        logger=logger,
        training=False,
        merge_all_iters_to_one_epoch=False,
        total_epochs=0
    )

    # Veri setindeki tüm bilgileri alın
    dataset = dataloader[0]
    all_infos = dataset.get_infos()  # get_infos() ile tüm bilgiye erişin

    # Rastgele örnekleri seçin
    sample_ind = random.sample(range(len(all_infos)), sample_size)
    samples[ds] = [all_infos[i] for i in sample_ind]

# İstatistikleri hesaplayın
stats = {}

for ds, examples in samples.items():
    point_counts = [ex['points'].shape[0] for ex in examples]
    class_counts = {}
    box_sizes = []
    points_in_box = []
    target_label_set = target_labels[ds]

    for ex in examples:
        labels = ex['gt_names']
        points = ex['points'][:, :3]

        # Nokta bulutlarını olduğu gibi kullanarak dönüştürülmüş gibi işleyin
        transformed_points = points

        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1

        gt_boxes = ex['gt_boxes']
        for i, box in enumerate(gt_boxes):
            # Box'un XYZ ve boyut bilgileri
            box_center, box_dims = box[:3], box[3:6]
            label = labels[i]
            
            # Seçilen etiketlerden biri ise, box içine düşen noktaları sayın
            if label in target_label_set:
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
        'total_samples': len(examples),
        f'{", ".join(target_label_set)}_points_in_box_avg': avg_points_in_box,
    }

# İstatistikleri tablo halinde yazdırın
df = pd.DataFrame(stats).T
print(df)
