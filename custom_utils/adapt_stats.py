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
sample_size_per_dataset = 10000  # Her veri seti için seçilecek rastgele nokta sayısı
n_iterations = 100 
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

stats = {}

# Fonksiyonlar
def compute_mmd(X, Y):
    XX = cdist(X, X, 'euclidean') ** 2
    YY = cdist(Y, Y, 'euclidean') ** 2
    XY = cdist(X, Y, 'euclidean') ** 2
    return np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)

def compute_kl_divergence(p, q):
    return entropy(p, q)

def sample_points_from_dataset(samples, sample_size):
    all_points = np.concatenate([sample['points'][:, :3] for sample in samples], axis=0)
    if len(all_points) > sample_size:
        indices = np.random.choice(len(all_points), sample_size, replace=False)
        return all_points[indices]
    else:
        return all_points

# Domain Adaptation Metrikleri
domain_stats = {}
for i, ds1 in enumerate(datasets):
    for j, ds2 in enumerate(datasets):
        if i < j:  # Sadece bir kez her çifti hesapla
            mmd_values = []
            kl_divergence_values = []

            for _ in range(n_iterations):
                # Her iki veri setinden rastgele örnekler seç
                points_ds1 = sample_points_from_dataset(samples[ds1], sample_size_per_dataset)
                points_ds2 = sample_points_from_dataset(samples[ds2], sample_size_per_dataset)

                # MMD Hesapla
                mmd = compute_mmd(points_ds1, points_ds2)
                mmd_values.append(mmd)

                # KL Divergence Hesapla
                hist_ds1, _ = np.histogramdd(points_ds1, bins=20)
                hist_ds2, _ = np.histogramdd(points_ds2, bins=20)
                hist_ds1 = hist_ds1 / np.sum(hist_ds1)
                hist_ds2 = hist_ds2 / np.sum(hist_ds2)
                kl_divergence = compute_kl_divergence(hist_ds1.flatten(), hist_ds2.flatten())
                kl_divergence_values.append(kl_divergence)

            # MMD ve KL Divergence ortalamalarını hesapla
            domain_stats[f"{ds1} vs {ds2}"] = {
                "MMD": np.round(np.mean(mmd_values), 4),
                "KL Divergence": np.round(np.mean(kl_divergence_values), 4)
            }


# Domainler arası ayrışmayı gösteren istatistikler
domain_stats_df = pd.DataFrame(domain_stats).T
print(domain_stats_df)


# t-SNE ve (mümkünse) UMAP ile verileri görselleştirmek için verileri hazırlama
all_points = []
labels = []

# Her veri setindeki örnekleri topla ve etiketleri belirle
for ds in datasets:
    points = np.concatenate([sample['points'][:, :3] for sample in samples[ds]], axis=0)
    all_points.append(points)
    labels.extend([ds] * len(points))

# Verileri tek bir diziye birleştir
all_points = np.concatenate(all_points, axis=0)

# t-SNE ile verileri düşük boyuta indir
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
tsne_result = tsne.fit_transform(all_points)

# t-SNE Görselleştirme
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for ds in datasets:
    indices = [i for i, label in enumerate(labels) if label == ds]
    plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], label=ds, alpha=0.5)
plt.title("t-SNE Visualization of Point Clouds")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()

# UMAP ile verileri düşük boyuta indir ve görselleştir (eğer UMAP yüklü ise)
if umap_available:
    umap = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_result = umap.fit_transform(all_points)

    # UMAP Görselleştirme
    plt.subplot(1, 2, 2)
    for ds in datasets:
        indices = [i for i, label in enumerate(labels) if label == ds]
        plt.scatter(umap_result[indices, 0], umap_result[indices, 1], label=ds, alpha=0.5)
    plt.title("UMAP Visualization of Point Clouds")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.legend()

plt.show()
