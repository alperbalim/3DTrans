import sys
sys.path.append('/root/3DTrans')
sys.path.append('/root/3DTrans/custom_utils')
import random
import easyyaml
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pcdet.datasets import build_dataloader
from pcdet.utils import box_utils  # PCDet'in box_utils modülü

# Logger setup
logger = logging.getLogger("pcdet_logger")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

# Dataset and label definitions
datasets = ['nuscenes', 'custom', 'awsim', 'waymo']
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

# Load configurations
cfg_dict = {}
for ds, filepath in cfg_files.items():
    cfg_dict[ds] = easyyaml.load(filepath)

sample_size = 500

# Load datasets and create samples
samples = {}
annos = {}
for ds in datasets:
    cfg = cfg_dict[ds]
    classes = target_labels[ds]
    dataloader = build_dataloader(cfg, classes, batch_size=1, dist=False, root_path=None, workers=4,
                                  logger=logger, training=False, merge_all_iters_to_one_epoch=False, total_epochs=0)
    dataset = dataloader[0]
    sample_indices = random.sample(range(0, len(dataset)), sample_size)
    samples[ds] = [dataset[i] for i in sample_indices]

    if ds in ["custom", "awsim"]:
        annos[ds] = [dataset.custom_infos[i]["annos"] for i in sample_indices]
    elif ds == "waymo":
        annos[ds] = [dataset.infos[i]["annos"] for i in sample_indices]
    else:
        dataset.transform_to_kitti_format()
        annos[ds] = [dataset.annos_kitti[i] for i in sample_indices]

# Calculate points in boxes using PCDet box_utils
points_in_box_data = {ds: [] for ds in datasets}

for ds in datasets:
    for sample, anno in zip(samples[ds], annos[ds]):
        points = sample['points'][:, :3]  # Noktaların XYZ koordinatları
        
        # Kutuların koordinatları (location, dimensions, yaw rotasyonu)
        box_centers = np.array(anno["location"])
        box_dims = np.array(anno["dimensions"])
        
        if "rotation_y" in anno.keys():
            box_yaws = np.array(anno["rotation_y"])
        elif "heading_angle" in anno.keys():
            box_yaws = np.array(anno["heading_angle"])
        else:
            box_yaws = np.zeros(len(box_centers))  # Varsayılan rotasyon 0.0
        
        # Kutuları köşe koordinatlarına dönüştür
        boxes = np.concatenate((box_centers, box_dims, box_yaws[:, None]), axis=1)

        # PCDet kullanarak noktaların kutular içinde olup olmadığını kontrol et
        box_idxs = box_utils.points_in_boxes_cpu(points, boxes)  # (N, M) şeklinde sonuç döner

        for i in range(len(boxes)):
            points_in_box = (box_idxs == i).sum()  # Kutudaki nokta sayısını hesapla
            points_in_box_data[ds].append(points_in_box)

# Violin plot için veriyi filtrele
filtered_points_in_box_data = {ds: np.array(points_in_box_data[ds])[(np.array(points_in_box_data[ds]) > 10) &
                                                                    (np.array(points_in_box_data[ds]) < 5000)]
                               for ds in datasets}

# Violin Plot
fig, ax = plt.subplots(figsize=(12, 6))

datasets_ordered = ['nuscenes', 'waymo', 'custom', 'awsim']
violin_data = [filtered_points_in_box_data[ds] for ds in datasets_ordered]

sns.violinplot(data=violin_data, ax=ax)
ax.set_xticks(range(len(datasets_ordered)))
ax.set_xticklabels(['nuScenes', 'WAYMO', 'Our Real', 'Our Sim.'])
ax.set_ylabel("Number of Points per Vehicle")
plt.savefig("object_points_limited_pc.png", dpi=450, bbox_inches='tight')

# Histogram (örnek için custom dataset)
fig, ax = plt.subplots(figsize=(12, 6))
plt.hist(filtered_points_in_box_data["custom"], bins=50, alpha=0.7)
plt.title("Distribution of Points per Object in Custom Dataset")
plt.xlabel("Number of Points")
plt.ylabel("Frequency")
plt.savefig("histogram_custom_dataset_pc.png", dpi=450, bbox_inches='tight')
