import pandas as pd
import numpy as np
from scipy.stats import ttest_rel

# CSV dosya yolları
baseline_csv = "/root/3DTrans/output/cfgs/DA/waymo_custom/voxel_rcnn_sn_custom/default/eval/combined_results.csv"
adaptation_csv = "/root/3DTrans/output/cfgs/DA/waymo_custom/UDA/voxel_rcnn_pre_SN_feat_3/default/eval/combined_results.csv"

# CSV'leri oku
df_base = pd.read_csv(baseline_csv)
df_adapt = pd.read_csv(adaptation_csv)

# En iyi 5 epoch'u seç (mean_ap'e göre)
df_base_top = df_base.sort_values(by="mean_ap", ascending=False).head(5).reset_index(drop=True)
df_adapt_top = df_adapt.sort_values(by="mean_ap", ascending=False).head(5).reset_index(drop=True)

print("Baseline Top 5:")
print(df_base_top[["epoch", "mean_ap", "3d", "bbox", "bev", "aos"]])
print("\nAdaptation Top 5:")
print(df_adapt_top[["epoch", "mean_ap", "3d", "bbox", "bev", "aos"]])
print("\n")

# Metriğe göre p-test hesapla
metrics = ["mean_ap", "3d", "bbox", "bev", "aos"]

print("Metric, p-value, Baseline Mean, Adaptation Mean, Significant (<0.05)")
for metric in metrics:
    base_scores = df_base_top[metric].values
    adapt_scores = df_adapt_top[metric].values

    # Paired t-test
    t_stat, p_val = ttest_rel(adapt_scores, base_scores)
    base_mean = np.mean(base_scores)
    adapt_mean = np.mean(adapt_scores)
    significant = "YES" if p_val < 0.05 else "NO"

    print(f"{metric}, {p_val:.4f}, {base_mean:.2f}, {adapt_mean:.2f}, {significant}")
