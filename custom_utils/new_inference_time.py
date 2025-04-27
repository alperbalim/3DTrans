import sys
sys.path.insert(0, '/root/3DTrans/')
sys.path.insert(0, '/root/3DTrans/tools')

from pcdet.datasets import build_dataloader, __all__  # Eğer `__all__` gerekli değilse çıkarabilirsiniz.
from pcdet.datasets import WaymoDataset, NuScenesDataset, CustomDataset  # Kendi veri kümenize uygun dataset sınıflarını içe aktarın.
import time
import numpy as np
import torch
from pcdet.models import build_network
from pcdet.datasets import build_dataloader
from pcdet.config import cfg, cfg_from_yaml_file

def measure_inference_time(config_path, model_name, checkpoint_path, dataset_config, num_samples=100):
    """
    Belirtilen model ve veri kümesi ile inference sürelerini ölçer.
    """
    # Config dosyasını yükle
    cfg_from_yaml_file(config_path, cfg)

    # Dataset Config'in uygun formatta olduğundan emin olun
    dataset_class = __all__[dataset_config['dataset']]
    dataset = dataset_class(
        dataset_config,
        class_names=dataset_config['class_names'],
        root_path=dataset_config['root_path'],
        training=False,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        pin_memory=True,
        num_workers=0,
        shuffle=False
    )

    # Modeli oluştur
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=checkpoint_path, logger=None)
    model.cuda()
    model.eval()

    # Inference sürelerini ölç
    inference_times = []
    for i, data in enumerate(dataloader):
        if i >= num_samples:  # Ölçüm için belirlenen numune sayısını aşınca durdur
            break
        
        # Veriyi GPU'ya taşı
        data = {key: val.cuda() for key, val in data.items() if isinstance(val, torch.Tensor)}
        
        # Inference başlat ve süreyi ölç
        start_time = time.time()
        with torch.no_grad():
            _ = model.forward(data)
        end_time = time.time()
        
        inference_times.append((end_time - start_time) * 1000)  # ms cinsinden süre
        
    return np.mean(inference_times), np.std(inference_times)


# Model ve veri kümesi yapılandırmaları
configs = [
    {
        "model_name": "PVRCNN",
        "config_path": "/root/3DTrans/tools/cfgs/DA/nusc_custom/pvrcnn_old_anchor.yaml",
        "checkpoint_path": "output/cfgs/DA/nusc_custom/pvrcnn_old_anchor_sn_kitti/default/ckpt/checkpoint_epoch_3.pth",
        "dataset_config": {
            "dataset": "CustomDataset",
            "class_names": ["Car"],
            "root_path": "/root/3DTrans/data/custom_kitti2",
        }
    },
    {
        "model_name": "VoxelRCNN",
        "config_path": "/root/3DTrans/tools/cfgs/DA/nusc_custom/voxelrcnn/voxel_rcnn_feat_3_vehi.yaml",
        "checkpoint_path": "/root/3DTrans/output/cfgs/DA/nusc_custom/voxelrcnn_st3d_feat_3_vehi/default/ckpt/checkpoint_epoch_2.pth",
        "dataset_config": {
            "dataset": "CustomDataset",
            "class_names": ["Car"],
            "root_path": "/root/3DTrans/data/custom_kitti2",
        }
    }
]

# Modelleri test et
for config in configs:
    avg_time, std_time = measure_inference_time(
        config_path=config["config_path"],
        model_name=config["model_name"],
        checkpoint_path=config["checkpoint_path"],
        dataset_config=config["dataset_config"],
        num_samples=100  # Ölçüm için örnek sayısı
    )
    print(f"{config['model_name']} Modeli - Ortalama Inference Süresi: {avg_time:.2f} ms, Standart Sapma: {std_time:.2f} ms")
