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
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import entropy

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Kontrollü olarak UMAP import etmeye çalışın
try:
    from umap import UMAP
    umap_available = True
except ImportError:
    umap_available = False

logger = logging.getLogger("pcdet_logger")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

# Veri setleri ve etiket sınıfları tanımları
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

# Config dosyalarını okuyup yapılandırmaları yükleyin
cfg_dict = {}
for ds, filepath in cfg_files.items():
    cfg_dict[ds] = easyyaml.load(filepath)

sample_size = 100
data_loaders = {}
sample_ind = {}
samples = {}
annos = {}
for ds in datasets:
    cfg = cfg_dict[ds]
    classes = target_labels[ds]
    dataloader = build_dataloader(cfg, classes, batch_size=1, dist=False, root_path=None, workers=4,
                                  logger=logger, training=False, merge_all_iters_to_one_epoch=False, total_epochs=0)
    dataset = dataloader[0]
    data_loaders[ds] = dataloader[1]
    sample_ind[ds] = random.sample(range(0, len(dataset)), sample_size)
    samples[ds] = [dataset[s_ind] for s_ind in sample_ind[ds]]
    if ds in ["custom", "awsim"]:
        annos[ds] = [dataset.custom_infos[s_ind]["annos"] for s_ind in sample_ind[ds]]
    elif ds == "waymo":
        annos[ds] = [dataset.infos[s_ind]["annos"] for s_ind in sample_ind[ds]]
    else:
        dataset.transform_to_kitti_format()
        annos[ds] = [dataset.annos_kitti[s_ind] for s_ind in sample_ind[ds]]

# Fonksiyonlar
def compute_mmd(X, Y):
    XX = cdist(X, X, 'euclidean') ** 2
    YY = cdist(Y, Y, 'euclidean') ** 2
    XY = cdist(X, Y, 'euclidean') ** 2
    return np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)

# MMD hesaplamaları için
n_iterations = 2  # Rastgele örnekleme sayısı
domain_stats = {}
"""
for i, ds1 in enumerate(datasets):
    for j, ds2 in enumerate(datasets):
        if i < j:  # Sadece bir kez her çifti hesapla
            mmd_values = []

            for _ in range(n_iterations):
                # Her iki veri setinden rastgele örnekler seç
                sample_indices_ds1 = np.random.choice(len(samples[ds1]), sample_size, replace=False)
                sample_indices_ds2 = np.random.choice(len(samples[ds2]), sample_size, replace=False)

                # İki domain için örnekleri tam olarak al
                for samp in range(sample_size):
                    
                    points_ds1 = samples[ds1][sample_indices_ds1[samp]]['points'][:, :3]
                    points_ds2 = samples[ds2][sample_indices_ds2[samp]]['points'][:, :3]
                    
                    poins_sub1 = np.random.choice(len(points_ds1), 10000, replace=False)
                    poins_sub2 = np.random.choice(len(points_ds2), 10000, replace=False)
                    
                    p1 = points_ds1[poins_sub1]
                    p2 = points_ds2[poins_sub2]
                    
                    mmd = compute_mmd(p1, p2)
                    mmd_values.append(mmd)

            # MMD ortalamasını hesapla
            domain_stats[f"{ds1} vs {ds2}"] = {
                "MMD": np.round(np.mean(mmd_values), 4)
            }

# Domainler arası ayrışmayı gösteren istatistikler
domain_stats_df = pd.DataFrame(domain_stats).T
print(domain_stats_df)
"""
# t-SNE ve (mümkünse) UMAP ile verileri görselleştirmek için verileri hazırlama
all_points = []
labels = []

# Her veri setindeki örnekleri topla ve etiketleri belirle
for ds in datasets:
    points = np.concatenate([sample['points'][:, :3] for sample in samples[ds][0:10]], axis=0)
    all_points.append(points)
    labels.extend([ds] * len(points))

print(" Verileri tek bir diziye birleştir")
all_points = np.concatenate(all_points, axis=0)


print("t-SNE ile verileri düşük boyuta indir")# 
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
tsne_result = tsne.fit_transform(all_points)

# t-SNE Görselleştirme ve Kaydetme
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for ds in datasets:
    indices = [i for i, label in enumerate(labels) if label == ds]
    plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], label=ds, alpha=0.5)
plt.title("t-SNE Visualization of Point Clouds")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.savefig("tsne_visualization.png", dpi=300, bbox_inches='tight')  # Yüksek çözünürlükte PNG olarak kaydet

# UMAP Görselleştirme ve Kaydetme (eğer UMAP yüklü ise)
if umap_available:
    plt.figure(figsize=(12, 6))
    umap = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_result = umap.fit_transform(all_points)
    
    for ds in datasets:
        indices = [i for i, label in enumerate(labels) if label == ds]
        plt.scatter(umap_result[indices, 0], umap_result[indices, 1], label=ds, alpha=0.5)
    plt.title("UMAP Visualization of Point Clouds")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.legend()
    plt.savefig("umap_visualization.png", dpi=300, bbox_inches='tight')  # Yüksek çözünürlükte PNG olarak kaydet
