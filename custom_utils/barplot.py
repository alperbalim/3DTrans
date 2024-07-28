import matplotlib.pyplot as plt
import numpy as np

# Veriler
models = ['Voxel-RCNN', 'PV-RCNN', 'Centerpoint']
metrics = ['BEV', '2D', '3D']
datasets = ['KITTI', 'Waymo', 'nuScenes', 'Real C.', 'Sim. C.']

# Tablo verileri
data = {
    'Voxel-RCNN': {
        'BEV': [96.8, 0.0, 0.0, 68.0, 30.7],
        '2D': [91.3, 0.0, 0.0, 87.0, 48.5],
        '3D': [96.7, 0.0, 0.0, 66.6, 28.7]
    },
    'PV-RCNN': {
        'BEV': [96.5, 0.0, 0.0, 67.9, 30.4],
        '2D': [94.3, 0.0, 0.0, 87.4, 56.8],
        '3D': [96.3, 0.0, 0.0, 66.0, 30.2]
    },
    'Centerpoint': {
        'BEV': [94.4, 0.0, 0.0, 61.1, 12.1],
        '2D': [87.7, 0.0, 0.0, 86.5, 45.1],
        '3D': [94.0, 0.0, 0.0, 58.7, 11.9]
    }
}

# Bar plot oluşturma
fig, axs = plt.subplots(3, 1, figsize=(15, 18))

for i, metric in enumerate(metrics):
    ax = axs[i]
    for model in models:
        values = data[model][metric]
        ax.bar(np.arange(len(datasets)) + (0.25 * models.index(model)), values, width=0.25, label=model)

    ax.set_title(f'{metric} Metrics')
    ax.set_xlabel('Datasets')
    ax.set_ylabel('Values')
    ax.set_xticks(np.arange(len(datasets)) + 0.25)
    ax.set_xticklabels(datasets)
    ax.legend()

plt.tight_layout()
plt.show()
