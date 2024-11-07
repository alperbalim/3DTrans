
import sys
sys.path
sys.path.append('/root/3DTrans')
sys.path.append('/root/3DTrans/custom_utils')
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
    'nuscenes': ['Car'],
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
        


stats = {}

for ds in datasets:
    point_counts = []
    box_sizes = []
    points_in_box = []
    class_counts = 0
    total_voxels = 0
    total_points_in_voxels = 0

    # Config dosyasındaki min ve max limitleri alın
    point_cloud_range = cfg_dict[ds]['POINT_CLOUD_RANGE']  # Örnek: [min_x, min_y, min_z, max_x, max_y, max_z]
    min_limit = np.array(point_cloud_range[:3])
    max_limit = np.array(point_cloud_range[3:])

    for sample, anno in zip(samples[ds], annos[ds]):
        points = sample['points'][:, :3]  # Nokta bulutunun XYZ koordinatları

        # Noktaları min ve max limitlere göre filtreleyin
        mask = np.all((points >= min_limit) & (points <= max_limit), axis=1)
        points = points[mask]
        point_counts.append(len(points))

        # Voxel grid oluşturma
        voxel_size = (0.1, 0.1, 0.2)  # x, y, z için ayrı boyutlar (örnek olarak)
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        grid_shape = np.ceil((max_bound - min_bound) / voxel_size).astype(int)

        voxel_grid = {}
        for point in points:
            voxel_coord = tuple(((point - min_bound) / voxel_size).astype(int))
            if voxel_coord in voxel_grid:
                voxel_grid[voxel_coord].append(point)
            else:
                voxel_grid[voxel_coord] = [point]

        # Voxel sayısını ve voxel başına ortalama nokta sayısını hesaplayın
        total_voxels += len(voxel_grid)
        total_points_in_voxels += sum(len(pts) for pts in voxel_grid.values())
        
        avg_distance = np.mean(points[:, 2])  # Ortalama Z mesafesi
        std_dev_distance = np.std(points[:, 2])  # Z mesafesi standart sapması
        resolution = avg_distance  # Çözünürlük olarak ortalama mesafe
        volume = (np.max(points[:, 0]) - np.min(points[:, 0])) * (np.max(points[:, 1]) - np.min(points[:, 1])) * (np.max(points[:, 2]) - np.min(points[:, 2]))
        density = len(points) / volume if volume > 0 else 0  # Yoğunluk hesaplama
        variances = np.var(points, axis=0)  # Nokta bulutunun varyansı
        homogeneity = np.mean(variances)  # Homojenlik ölçütü
        z_range = np.max(points[:, 2]) - np.min(points[:, 2])  # Z eksenindeki yükseklik aralığı
        
        labels = anno["name"]
        for i, label in enumerate(labels):
            if label in target_labels[ds]:
                class_counts += 1
                box_center = anno["location"][i]
                box_dims = anno["dimensions"][i]
                box_sizes.append(box_dims)

                mask = (
                    (points[:, 0] >= (box_center[0] - box_dims[0] / 2)) & 
                    (points[:, 0] <= (box_center[0] + box_dims[0] / 2)) &
                    (points[:, 1] >= (box_center[1] - box_dims[1] / 2)) &
                    (points[:, 1] <= (box_center[1] + box_dims[1] / 2)) &
                    (points[:, 2] >= (box_center[2] - box_dims[2] / 2)) &
                    (points[:, 2] <= (box_center[2] + box_dims[2] / 2))
                )
                points_in_box.append(np.sum(mask))
                # İstatistikleri hesaplayın




    avg_points = sum(point_counts) / len(point_counts)
    min_box_size = box_sizes[np.argmin((np.prod(np.asarray(box_sizes), axis=1)))]
    max_box_size = box_sizes[np.argmax((np.prod(np.asarray(box_sizes), axis=1)))]
    mean_box_size = np.mean(np.asarray(box_sizes), axis=0)
    avg_points_in_box = sum(points_in_box) / len(points_in_box) if points_in_box else 0
    avg_points_per_voxel = total_points_in_voxels / total_voxels if total_voxels > 0 else 0

    stats[ds] = {
        'avg_points': np.round(avg_points, 2),
        'min_box_size': np.round(min_box_size, 2),
        'max_box_size': np.round(max_box_size, 2),
        'mean_box_size': np.round(mean_box_size, 2),
        'class_distribution': np.round(class_counts / len(samples[ds]), 2),
        'total_samples': len(samples[ds]),
        'points_in_box_avg': int(avg_points_in_box),
        'total_voxels': total_voxels,
        'avg_points_per_voxel': np.round(avg_points_per_voxel, 2)
    }

# İstatistikleri tablo halinde yazdırın
import pandas as pd

df = pd.DataFrame(stats).T
print(df)
df