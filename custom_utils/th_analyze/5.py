import sys

import pickle

import numpy as np

# Ortamınıza göre yolu ayarlayın
sys.path.insert(0, '/root/3DTrans/') # Burayı kendi repo yolunuzla değiştirin

# Değerlendirme modülünü import etme
# Bu import, 3DTrans veya OpenPCDet kurulumunuzdaki eval modülünü bulmalıdır.
from pcdet.datasets.kitti.kitti_object_eval_python.eval import do_eval, eval_class, get_mAP_R40 # [1, 2]
from pcdet.datasets.kitti.kitti_utils import *

print("pcdet evaluation module imported successfully.") # [2]
map_name_to_kitti = {'Car':'Car', 'Pedestrian':'Pedestrian', 'Cyclist':'Cyclist'}

# Tahminleri ve ground truth verilerini yükleme
# Dosya yollarının doğru olduğundan emin olun # [2]
result_path = '/root/3DTrans/output/cfgs/MDF/KNW/customnw_pvrcnn_feat_3_uni3d/default/eval/epoch_2/val/default/result.pkl' # <--- Burayı KENDİ result.pkl dosyanızın yolu ile DEĞİŞTİRİN # [2]
with open(result_path, 'rb') as f: # [3]
    dt_data = pickle.load(f)
result_path = '/root/3DTrans/data/custom_kitti2/custom_infos_test.pkl' # <--- Burayı KENDİ infos dosyanızın yolu ile DEĞİŞTİRİN # [3]
with open(result_path, 'rb') as f: # [3]
    gt_data = pickle.load(f) # [3]
print(f"Successfully loaded data from {result_path}") # [3]

# Ground truth verilerini değerlendirme formatına dönüştürme (KITTI stili)
gt_annos = []
for gt_info in gt_data: # [3]
    anno = {
    "bbox": gt_info["annos"]["gt_boxes_lidar"], # [3]
    "gt_boxes_lidar": gt_info["annos"]["gt_boxes_lidar"], # [3]
    "name": gt_info["annos"]["name"], # Sınıf isimleri # [3]
    "dimensions": gt_info["annos"]["dimensions"], # Boyutlar # [4]
    "location": gt_info["annos"]["location"], # Konum # [4]
    "rotation_y": gt_info["annos"]["rotation_y"], # Dönüş # [4]
    "alpha": np.asarray([0.0] * len(gt_info["annos"]["gt_boxes_lidar"]), dtype=np.float32), # [4]
    "occluded": np.asarray( [0.0] * len(gt_info["annos"]["gt_boxes_lidar"]), dtype=np.int32), # [4]
    "truncated": np.asarray([0.0] * len(gt_info["annos"]["gt_boxes_lidar"]), dtype=np.float32) # [4]
    }
    # Tüm GT nesnelerine tek bir zorluk seviyesi (0) ata - Zorlukları göz ardı etmek için # [5]
    anno["difficulty"] = np.asarray( [0.0]* len(gt_info["annos"]["gt_boxes_lidar"]), dtype=np.int32) # [5]
    gt_annos.append(anno) # [5]

gt_annos = transform_annotations_to_kitti_format(gt_annos, map_name_to_kitti=map_name_to_kitti)

# Tahmin verilerini değerlendirme formatına dönüştürme
dt_annos_raw = []
for dt in dt_data: # [5]
    # Tahminlerde güven skoru ('score') ve sınıf ('name') bilgilerinin olması beklenir. # [6]
    dt_anno = {
    "bbox": dt["boxes_lidar"], # 3D sınırlayıcı kutu # [6]
    "boxes_lidar": dt["boxes_lidar"], # 3D sınırlayıcı kutu # [6]
    "score": dt["score"], # Güven skoru # [6]
    "name": dt["name"], # Sınıf ismi # [6]
    "alpha" : np.asarray([0 for i in range(len(dt["boxes_lidar"]))]), # [6]
    "occluded" : np.asarray([0 for i in range(len(dt["boxes_lidar"]))]), # [6]
    "truncated" : np.asarray([0 for i in range(len(dt["boxes_lidar"]))]), # [6]
    "location": dt["location"], # [6]
    "dimensions": dt["dimensions"], # [6]
    "rotation_y": dt["rotation_y"], # [6]
    "score": dt["score"] # [6]
    }
    dt_annos_raw.append(dt_anno) # [7]

dt_annos_raw = transform_annotations_to_kitti_format(dt_annos_raw, map_name_to_kitti=map_name_to_kitti)

from pcdet.datasets.kitti.kitti_object_eval_python.kitti_common import filter_annos_low_score

# KITTI stilinde mAP hesaplaması için IoU eşiklerini belirleme # [7]
min_overlaps = np.array([0.3, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]) # [7]

# Sadece Car (ID 0) sınıfını ve tüm zorlukları gruplayan tek zorluk seviyesini (indeks 0) değerlendir # [8]
class_ids_to_eval = [0] # Araba sınıfı (KITTI'de genellikle 0) # [8]
difficulty_levels_to_eval = [0] # Tek zorluk kovası (tümü birleştirildi) # [8]

# do_eval için min_overlaps formatını ayarla: [num_minoverlap, num_metric, num_class] # [9]
# Metrik sıralaması varsayımı: 0=Image, 1=BEV, 2=3D, 3=AOS # [9]
num_metrics = 4 # [9]
min_overlaps_for_map = np.zeros((len(min_overlaps), num_metrics, len(class_ids_to_eval)), dtype=np.float32) # [9]
min_overlaps_for_map[:, 2, 0] = min_overlaps # IoU eşiklerini 3D metriği (index 2) ve Car (sınıf 0) için uygula # [9]

# Standart mAP (11 recall noktası) ve mAP@R40 hesaplama # [8]
print("\nCalculating standard mAP (do_eval)...") # [10]

# do_eval çağrısı: gt_annos, dt_annos, class_ids, difficulty_ids, min_overlaps, compute_aos # [11]
# Döndürür: mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 # [12, 13]
dt_annos = filter_annos_low_score(dt_annos_raw, 0.7)
mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = do_eval(
gt_annos, dt_annos, class_ids_to_eval, min_overlaps_for_map, compute_aos=False # [13]
) # [13]

print("\n--- Standard mAP Results (Car, All Difficulties Combined) ---") # [13]
# Sonuçları yazdır (Car sınıfı (indeks 0), tek zorluk kovası (indeks 0))
# mAP3d boyutu varsayımı: [num_class, num_difficulty, num_minoverlap] # [13]
c_idx = 0 # Car indeksi # [13]
d_idx = 0 # Tek zorluk kovası indeksi # [13]

print(f"Class: Car (ID: {class_ids_to_eval[c_idx]})") # [13]
print(" All Difficulties Combined:") # [14]

for i_idx, iou_thresh in enumerate(min_overlaps): # [14]
    map_value = mAP3d[c_idx, d_idx, i_idx] # [14]
    print(f" 3D mAP @ IoU={iou_thresh:.2f}: {map_value:.4f}") # [14]

for i_idx, iou_thresh in enumerate(min_overlaps): # [14]
    if mAP3d_R40 is not None: # [14]
        map_r40_value = mAP3d_R40[c_idx, d_idx, i_idx] # [14]
        print(f" 3D mAP@R40 @ IoU:{iou_thresh:.2f}: {map_r40_value:.4f}") # [14]

print("-" * 25) # [15]

# --- Güven Skoru Eşiği Analizi ---
# eval_class fonksiyonunu kullanarak farklı güven skoru eşiklerindeki P/R/F1 değerlerini hesaplama # [15]
iou_thresholds_for_confidence_analysis = [0.5, 0.7] # Analiz için IoU eşikleri # [16]
confidence_thresholds_to_analyze = np.arange(0.05, 1.05 , 0.05) # Analiz edilecek güven skoru eşikleri # [16]

# eval_class için sınıf ve zorluk seviyesini seç (tek sınıf, tek zorluk kovası) # [16]
analysis_class_id = 0 # Araba # [16]
analysis_difficulty_id = 0 # Tek zorluk kovası # [16]

print(f"\n--- Confidence Score Threshold Analysis (Car, All Difficulties Combined) ---") # [17]

for current_iou_for_analysis in iou_thresholds_for_confidence_analysis: # [17]
    print(f"\nAnalyzing for IoU Threshold > {current_iou_for_analysis}...") # [17]

    # eval_class için tek bir IoU eşiği içeren min_overlaps yapısı oluştur # [17]
    # Yapı: [num_minoverlap=1, metric, num_class=1] # [17]
    # Metrik 2: 3D Box # [17]
    min_overlaps_single_iou = np.ones((1, 3, 1), dtype=np.float32) # [17]
    min_overlaps_single_iou[:, 2, 0] = current_iou_for_analysis # 3D metriği için IoU eşiği [18]

    # eval_class çağrısı # [18]
    res_eval_class = eval_class(
    gt_annos, dt_annos_raw, [analysis_class_id], [analysis_difficulty_id], 2, min_overlaps_single_iou, compute_aos=False # [18]
    ) # [18]

    # eval_class sonuçlarından hassasiyet, geri çağırma ve bunlara karşılık gelen güven eşiklerini al # [18]
    # [19]
    
    eval_precision_points = res_eval_class['precision'] # [19]
    eval_recall_points = res_eval_class['recall'] # [20]
    eval_confidence_points = res_eval_class['thresholds'] # Güven skorları (azalan sırada) # [20]

    # Eğer eval_class herhangi bir nokta döndürmediyse (örneğin hiç TP yoksa) # [20]
    if len(eval_confidence_points) == 0: # [20]
        print(f" No evaluation points returned by eval_class for IoU > {current_iou_for_analysis}. Setting P/R/F1 to 0 for all confidence thresholds.") # [20]
        # [20]
        # [21]
    else: # [21]
        # Belirlenen güven skoru eşikleri için P/R/F1 değerlerini hesapla # [21]
        print(" Confidence | Precision | Recall | F1-Score") # [22]
        print(" -------------------------------------------------") # [22]

    for conf_thresh in confidence_thresholds_to_analyze: # [22]
        index = np.searchsorted(eval_confidence_points, conf_thresh, side='right') - 1 # [29]
        if index >= 0: # Geçerli bir indeks bulunduysa # [29]
            precision = eval_precision_points[index] # [29]
            recall = eval_recall_points[index] # [29]
        else: # conf_thresh en yüksek skordan bile büyükse # [30]
            precision = 0.0 # [30]
            recall = 0.0 # [30]

        # F1-skor hesaplama, paydanın sıfır olma durumunu ele al # [30]
        denominator = precision + recall # [30]
        print(denominator)
        f1 = np.where(denominator > 0, 2 * (precision * recall) / denominator, 0.0) # [30]

        # Mevcut güven skoru eşiği için sonuçları yazdır # [30]
        print(f" {conf_thresh:.2f} | {precision:.4f} | {recall:.4f} | {f1:.4f}") # [30]