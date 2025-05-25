import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt # Histogram çizmek için matplotlib'e ihtiyacımız olacak (kaynaklarda yok)
sys.path.insert(0, '/root/3DTrans/')

from pcdet.datasets.kitti.kitti_object_eval_python.eval import calculate_iou_partly



# Tahminleri ve ground truth verilerini yükleme
# Dosya yollarının doğru olduğundan emin olun [3, 4]
# Bu yolları KENDİ result.pkl ve infos.pkl dosyalarınızın yollarıyla DEĞİŞTİRİN.
result_path = '/root/3DTrans/output/cfgs/MDF/KNW/customnw_pvrcnn_feat_3_uni3d/default/eval/epoch_2/val/default/result.pkl' # <--- Tahminlerinizin result.pkl yolu
gt_infos_path = '/root/3DTrans/data/custom_kitti2/custom_infos_test.pkl' # <--- Ground truth infos.pkl yolu

try:
    with open(result_path, 'rb') as f:
        dt_data = pickle.load(f) # Tahmin verileri [4-7]
    print(f"Successfully loaded detection data from {result_path}")

    with open(gt_infos_path, 'rb') as f:
        gt_data = pickle.load(f) # Ground truth infos verileri [4, 8-15]
    print(f"Successfully loaded ground truth data from {gt_infos_path}")

except FileNotFoundError:
    print("Error: result.pkl or infos.pkl file not found. Please check the paths.")
    sys.exit("Exiting.")
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit("Exiting.")


# Ground truth annos listesini hazırlama (daha önceki scriptteki gibi) [4, 16, 17]
gt_annos = []
for gt_info in gt_data:
    anno = {
        "bbox": gt_info["annos"]["gt_boxes_lidar"], # Lidar koordinatındaki 3D kutular [8-16]
        "name": gt_info["annos"]["name"], # Sınıf isimleri [4, 8-16]
        # Diğer alanlar histogram için doğrudan kullanılmasa da uyumluluk için eklenebilir [16]
        "dimensions": gt_info["annos"].get("dimensions", np.array([])),
        "location": gt_info["annos"].get("location", np.array([])),
        "rotation_y": gt_info["annos"].get("rotation_y", np.array([])),
        "alpha": gt_info["annos"].get("alpha", np.zeros(len(gt_info["annos"]["name"]), dtype=np.float32)),
        "occluded": gt_info["annos"].get("occluded", np.zeros(len(gt_info["annos"]["name"]), dtype=np.int32)),
        "truncated": gt_info["annos"].get("truncated", np.zeros(len(gt_info["annos"]["name"]), dtype=np.float32)),
        "difficulty": gt_info["annos"].get("difficulty", np.zeros(len(gt_info["annos"]["name"]), dtype=np.int32))
    }
    gt_annos.append(anno)

# Tahmin verilerini değerlendirme formatına dönüştürme (daha önceki scriptteki gibi) [18, 19]
dt_annos_raw = []
for dt in dt_data:
     dt_anno = {
        "bbox": dt["boxes_lidar"], # Tahmin edilen 3D kutular [5-7, 18]
        "score": dt["score"],     # Güven skoru [5-7, 18]
        "name": dt["name"],       # Sınıf ismi [5-7, 18]
        # Diğer gerekli alanlar, eğer mevcutsa veya IoU hesaplaması için gerekliyse eklenmeli [18, 19]
        "alpha" : dt.get("alpha", np.asarray([0 for i in range(len(dt["boxes_lidar"]))], dtype=np.float32)),
        "location": dt.get("location", np.array([])),
        "dimensions": dt.get("dimensions", np.array([])),
        "rotation_y": dt.get("rotation_y", np.array([]))
    }
     dt_annos_raw.append(dt_anno)


# --- Tüm Detectionlar İçin Max IoU Hesaplama ---

all_max_ious = []

# Hangi metrik için IoU hesaplayacağımızı belirtelim. 3D kutu için 2 [4, 20]
metric_type = 2 # 0: Bbox (2D), 1: BEV, 2: 3D

print(f"\nCalculating max IoU for all detections using metric {metric_type}...")

# Her bir frame için GT ve DT verilerini alıp IoU matrisini hesaplayalım.
# calculate_iou_partly fonksiyonu, dt_annos ve gt_annos listelerini input alır [1, 21, 22].
# Bu listeler her bir frame'in annotationlarını içerir.

# calculate_iou_partly, num_parts parametresi alır [1, 21]. eval_class'ta 100 kullanılmış [21].
num_parts_for_iou_calc = 100


# calculate_iou_partly fonksiyonu dt_annos ve gt_annos listelerini bekler,
# bu listelerin her elemanı tek bir frame'e ait annoları içerir.
# Bizim gt_annos ve dt_annos_raw listelerimiz zaten bu formatta [4, 18].

# calculate_iou_partly tüm frameler için topluca overlap hesaplar [22].
# Döndürdüğü overlaps değişkeni bir listedir, her elemanı bir frame'e ait IoU matrisidir.
# Boyut: [num_examples] list of [num_dt_in_frame, num_gt_in_frame]
overlaps_list, _, _, _ = calculate_iou_partly(dt_annos_raw, gt_annos, metric_type, num_parts_for_iou_calc)
print("IoU calculation complete for all frames.")

# Her frame'in IoU matrisini işleyerek her detection için max IoU'yu bulalım.
for frame_overlaps in overlaps_list:
    # frame_overlaps boyutu: [num_dt_in_frame, num_gt_in_frame]
    num_dt_in_frame = frame_overlaps.shape[0]
    num_gt_in_frame = frame_overlaps.shape[1]

    if num_dt_in_frame > 0:
        if num_gt_in_frame > 0:
            # Her detection (satır) için en yüksek IoU'yu bul
            max_ious_this_frame = np.max(frame_overlaps, axis=1)
            all_max_ious.extend(max_ious_this_frame)
        else:
            # Bu frame'de hiç GT yoksa, tüm detectionlar için max IoU 0'dır.
            all_max_ious.extend([0.0] * num_dt_in_frame)
    # else: Bu frame'de hiç detection yoksa eklenecek bir şey yok.


all_max_ious_array = np.array(all_max_ious)

print(f"Collected max IoUs for {len(all_max_ious_array)} detections.")

# --- Histogram Çizimi ---

# Matplotlib kütüphanesi kaynaklarda yok, bu yüzden harici bilgidir.
# Histogram binlerini (çubuklarını) belirleyelim, örneğin 0.0'dan 1.0'a 0.05'lik adımlarla.
bins = np.arange(0.05, 1.05, 0.05)

plt.figure(figsize=(10, 6))
plt.hist(all_max_ious_array, bins=bins, edgecolor='black', alpha=0.7)
plt.title('Distribution of Maximum IoUs for All Detections (3D Box Metric)')
plt.xlabel('Maximum IoU with any Ground Truth Box')
plt.ylabel('Number of Detections')
plt.grid(axis='y', alpha=0.75)
plt.xlim(0, 1.0)
plt.xticks(bins) # X eksenindeki etiketleri belirleyelim

print("\nDisplaying histogram...")
plt.show()

