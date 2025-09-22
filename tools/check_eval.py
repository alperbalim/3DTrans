import os
import re
import pandas as pd

# Kullanıcıdan ana klasör yolunu al

import argparse

# Komut satırı argümanlarını tanımlayın
parser = argparse.ArgumentParser(description="3DTrans Eval Log Analyzer")
parser.add_argument("eval_root", type=str, help="Ana eval klasör yolunu belirtin")
args = parser.parse_args()

# Ana klasör yolunu argparse ile al
eval_root = args.eval_root if args.eval_root else input("Ana klasör yolunu girin: ")
# Sonuçları toplamak için bir liste
results = []

results.append({
    "epoch": 0,
    "bbox": 0,
    "bev": 0,
    "3d": 0,
    "aos": 0,
    "mean_ap": 0
})

# `eval` klasöründeki tüm epoch alt klasörlerini dolaş
eval_list=os.listdir(eval_root)
if 'eval_with_train' in eval_list:
    eval_list.remove('eval_with_train')
for epoch_dir in eval_list:
    if os.path.isdir(os.path.join(eval_root, epoch_dir, "test")):
        epoch_path = os.path.join(eval_root, epoch_dir, "test", "default")
    elif os.path.isdir(os.path.join(eval_root, epoch_dir, "val")):
        epoch_path = os.path.join(eval_root, epoch_dir, "val", "default")
    else:
        continue
    if os.path.isdir(epoch_path):
        # `log_eval***` dosyalarını bul
        for file_name in os.listdir(epoch_path):
            if file_name.startswith("log_eval"):
                file_path = os.path.join(epoch_path, file_name)
                
                with open(file_path, "r") as f:
                    content = f.read()
                    
                    # `Car AP_R40@0.70, 0.50, 0.50` bloğunu bul
                    match = re.search(r"Car AP_R40@0.70, 0.50, 0.50:\s*bbox AP:(\d+\.\d+),.*\n.*bev\s+AP:(\d+\.\d+),.*\n.*3d\s+AP:(\d+\.\d+),.*\n.*aos\s+AP:(\d+\.\d+)", content)
                    
                    if match:
                        bbox_ap, bev_ap, three_d_ap, aos_ap = map(float, match.groups())
                        mean_ap = (bbox_ap + bev_ap + three_d_ap + aos_ap) / 4
                        
                        results.append({
                            "epoch": epoch_dir,
                            "bbox": bbox_ap,
                            "bev": bev_ap,
                            "3d": three_d_ap,
                            "aos": aos_ap,
                            "mean_ap": mean_ap
                        })

# Verileri DataFrame'e dönüştür
df = pd.DataFrame(results)

# En yüksek ortalama AP değerine sahip epoch'u bul
best_epoch = df.loc[df["mean_ap"].idxmax()]
best_bbox= df.loc[df["bbox"].idxmax()]
best_bev= df.loc[df["bev"].idxmax()]
best_3d= df.loc[df["3d"].idxmax()]
best_aos= df.loc[df["aos"].idxmax()]

# Sonuçları ana klasöre kaydet
output_file = os.path.join(eval_root, "combined_results.csv")
df.to_csv(output_file, index=False)

# En iyi epoch sonucunu yazdır
print("En yüksek ortalama AP değerine sahip epoch bilgisi:")
print(best_epoch)

print("En yüksek 3d AP değerine sahip epoch bilgisi:")
print(best_3d)

print(f"Tüm sonuçlar '{output_file}' dosyasına kaydedildi.")
