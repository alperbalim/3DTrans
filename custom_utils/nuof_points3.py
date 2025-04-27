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

# Calculate points falling on each object
points_in_box_data = {ds: [] for ds in datasets}

for ds in datasets:
    for sample, anno in zip(samples[ds], annos[ds]):
        points = sample['points'][:, :3]
        for i in range(len(anno["dimensions"])):
            box_center = anno["location"][i]
            box_dims = anno["dimensions"][i]
            if "rotation_y" in annos[ds][0].keys():
                box_yaw = anno["rotation_y"][i]
            elif "heading_angle" in annos[ds][0].keys():
                box_yaw = anno["heading_angle"][i]
            shifted_points = points - box_center
            cos_yaw = np.cos(-box_yaw)  # Yaw için ters dönüş
            sin_yaw = np.sin(-box_yaw)
            rotation_matrix = np.array([
                [cos_yaw, -sin_yaw, 0],
                [sin_yaw,  cos_yaw, 0],
                [0,        0,       1]])
            aligned_points = np.dot(shifted_points, rotation_matrix.T)
            mask = ((aligned_points[:, 0] >= -box_dims[0] / 2) &
                    (aligned_points[:, 0] <=  box_dims[0] / 2) &
                    (aligned_points[:, 1] >= -box_dims[1] / 2) &
                    (aligned_points[:, 1] <=  box_dims[1] / 2) &
                    (aligned_points[:, 2] >= -box_dims[2] / 2) &
                    (aligned_points[:, 2] <=  box_dims[2] / 2))
            points_in_box_data[ds].append(np.abs(np.sum(mask)))
            

# Filter data to include only points within the range [10, 5000]
filtered_points_in_box_data = {ds: [] for ds in datasets}

for ds in datasets:
    data = np.array(points_in_box_data[ds])
    for dat,i in enumerate(data):
        if dat>0:
            if dat<10000:
                filtered_points_in_box_data[ds].append(dat)
            else:
                filtered_points_in_box_data[ds].append(10000)
    ind=random.sample(range(0, len(filtered_points_in_box_data[ds])), 2400)
    filtered_points_in_box_data[ds] =np.asarray(filtered_points_in_box_data[ds])[ind]        
#filtered_points_in_box_data[ds][i] = [dat if data (data >= 1) & (data <= 10000) else ]
    
# Create violin plot for the filtered data
fig, ax = plt.subplots(figsize=(12, 6))

datasets_ordered =['nuscenes', 'waymo', 'custom', 'awsim']
# Prepare the data for plotting
violin_data = [filtered_points_in_box_data[ds] for ds in datasets_ordered]

sns.violinplot(data=violin_data, ax=ax)#,log_scale=True)
#ax.set_ylim(0, 1000)  # Adjust this range as needed to fit your data

#ax.set_title("Number of Points per Object Across Datasets ")
ax.set_xticks(range(len(datasets)))
ax.set_xticklabels(['nuScenes', 'WAYMO','Our Real','Our Sim.' ])
ax.set_ylabel("Number of Points per Vehicle")
#plt.yscale('log')
plt.rcParams["font.size"] = 24
plt.tight_layout()
plt.show()
plt.legend()
plt.savefig("object_points_limited11.png", dpi=450, bbox_inches='tight')  # Yüksek çözünürlükte PNG olarak kaydet


fig, ax = plt.subplots(figsize=(12, 6))
plt.hist(filtered_points_in_box_data["custom"])
plt.savefig("hist.png", dpi=450, bbox_inches='tight')  # Yüksek çözünürlükte PNG olarak kaydet