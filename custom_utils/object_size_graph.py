import sys

sys.path

# Setup
sys.path.append('/root/3DTrans')
sys.path.append('/root/3DTrans/custom_utils')
import random
import easyyaml
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pcdet.datasets import build_dataloader
from torch.utils.data import RandomSampler


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
data_loaders = {}
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

# Extract object sizes and create violin plots
object_sizes = {ds: {"length": [], "width": [], "height": []} for ds in datasets}

datasets_ordered =['nuscenes', 'waymo', 'custom', 'awsim']

for ds in datasets_ordered:
    for anno in annos[ds]:
        for i in range(len(anno["dimensions"])):
            box_dims = anno["dimensions"][i]
            object_sizes[ds]["length"].append(box_dims[0])
            object_sizes[ds]["width"].append(box_dims[1])
            object_sizes[ds]["height"].append(box_dims[2])

# Create violin plots for Length, Width, and Height
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
dimensions = ["length", "width", "height"]
titles = ["Length", "Width", "Height"]
ranges = [(2.0, 7.0), (1.0, 3.0), (0.5, 3.0)]

for idx, (dim, title, value_range) in enumerate(zip(dimensions, titles, ranges)):
    data = [np.clip(object_sizes[ds][dim], value_range[0], value_range[1]) for ds in datasets]
    sns.violinplot(data=data, ax=axes[idx])
    axes[idx].set_title(title)
    axes[idx].set_xticks(range(len(datasets)))
    axes[idx].set_xticklabels(['nuScenes', 'WAYMO','Our Real','Our Sim.' ])
    axes[idx].set_ylim(value_range)

plt.tight_layout()
#plt.show()
#plt.title("UMAP Visualization of Point Clouds")
plt.legend()
plt.savefig("object_sizes2.png", dpi=450, bbox_inches='tight')  # Yüksek çözünürlükte PNG olarak kaydet