import pandas as pd
import numpy as np

# CSV'yi oku
csv_file = "/root/3DTrans/output/cfgs/SSDA/nuscenes_awsim/voxelrcnn/voxelrcnn_feat_3_vehi_01_finetune/default/eval/combined_results.csv"

files=["/root/3DTrans/output/cfgs/SSDA/nuscenes_awsim/pvrcnn/pvrcnn_feat_3_vehi_01_finetune/default/eval/combined_results.csv",
"/root/3DTrans/output/cfgs/SSDA/nuscenes_awsim/voxelrcnn/voxelrcnn_feat_3_vehi_01_finetune/default/eval/combined_results.csv",
"/root/3DTrans/output/cfgs/SSDA/nuscenes_custom/pvrcnn/pvrcnn_feat_3_vehi_01_finetune/default/eval/combined_results.csv",
"/root/3DTrans/output/cfgs/SSDA/nuscenes_custom/voxelrcnn/voxelrcnn_feat_3_vehi_01_finetune/default/eval/combined_results.csv",
"/root/3DTrans/output/cfgs/SSDA/waymo_awsim/pvrcnn/pvrcnn_feat_3_vehi_01_finetune/default/eval/combined_results.csv",
"/root/3DTrans/output/cfgs/SSDA/waymo_awsim/voxelrcnn/voxelrcnn_feat_3_vehi_01_finetune/default/eval/combined_results.csv",
"/root/3DTrans/output/cfgs/SSDA/waymo_custom/pvrcnn/pvrcnn_feat_3_vehi_01_finetune/default/eval/combined_results.csv",
"/root/3DTrans/output/cfgs/SSDA/waymo_custom/voxelrcnn/voxelrcnn_feat_3_vehi_01_finetune/default/eval/combined_results.csv"]

for file in files:
    df = pd.read_csv(csv_file)

    # mean_ap'e göre en iyi 5 epoch'u seç
    top_df = df.sort_values(by="mean_ap", ascending=False).head(5).reset_index(drop=True)

    print("Top 5 epochs selected based on mean_ap:")
    print(top_df[["epoch", "mean_ap", "3d", "bbox", "bev", "aos"]])
    print("\n")

    # Analiz yapılacak metrikler
    metrics = ["mean_ap", "3d", "bbox", "bev", "aos"]

    # Tablo oluştur
    summary_table = []

    for metric in metrics:
        scores = top_df[metric].values
        metric_max = np.max(scores)
        metric_min = np.min(scores)
        metric_mean = np.mean(scores)
        metric_std = np.std(scores)
        metric_var = np.var(scores)
        
        summary_table.append({
            "Metric": metric,
            "Max": round(metric_max, 2),
            "Mean": round(metric_mean, 2),
            "Min": round(metric_min, 2),
            "Std": round(metric_std, 2),
            "Var": round(metric_var, 2)
        })

    # DataFrame'e çevir
    summary_df = pd.DataFrame(summary_table)

    print(file)
    # Virgül ayrılmış şekilde yazdır
    print("Metric,Max,Mean,Min,Std,Var")
    for _, row in summary_df.iterrows():
        print(f"{row['Metric']},{row['Max']},{row['Mean']},{row['Min']},{row['Std']},{row['Var']}")
