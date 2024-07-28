import time
import torch
from openpcdet.datasets import build_dataloader
from openpcdet.models import build_network, load_data_to_gpu
from openpcdet.config import cfg, cfg_from_yaml_file
from pathlib import Path

# Config dosyanızın yolunu buraya yazın
cfg_file = 'configs/your_model_config.yaml'
cfg_from_yaml_file(cfg_file, cfg)

# Modeli yükleyin
model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=None)
model.load_params_from_file(filename='path/to/your/trained_model.pth', to_cpu=False)
model.cuda()
model.eval()

# Veri yükleyiciyi oluşturun
dataset, dataloader, _ = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    batch_size=1,  # Batch size 1 olarak ayarlandı
    dist=False,  # Non-distributed inference
    workers=4,  # İşçi sayısı
    training=False
)

# Inference zamanını ölçmek için bir liste oluşturun
inference_times = []

# Inference zamanını ölçün
with torch.no_grad():
    for idx, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        start_time = time.time()
        pred_dicts, _ = model.forward(batch_dict)
        end_time = time.time()
        
        inference_time = end_time - start_time
        inference_times.append(inference_time)
        
        # İsteğe bağlı olarak, belli bir sayıda örnek üzerinde ölçüm yapabilirsiniz
        if idx >= 99:  # Örneğin, ilk 100 örneği kullan
            break

# Ortalama inference zamanını hesaplayın
average_inference_time = sum(inference_times) / len(inference_times)
print(f"Average inference time: {average_inference_time:.4f} seconds")