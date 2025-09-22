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

# --- Güven Skoru Eşiği Analizi (IoU Eşiklerine Göre) ---
# Bu bölüm, farklı SABİT güven skoru eşiklerinin modelin performansına (Precision, Recall, F1-Score) etkisini analiz eder.
# Analiz, belirli bir IoU eşiği (True Positive belirlemek için) altında yapılır.
# Kaynaklar, güven skoru eşiğinin performans üzerinde önemli bir etkisi olduğunu ve optimal eşiğin
# sınıfa, detektöre ve senaryoya göre değişebileceğini belirtir [10-14].
# Bu analiz, hassasiyet ve geri çağırma arasındaki dengeyi bulmak için kritiktir [13, 15].

print(f"\n--- Güven Skoru Eşiği Analizi ---")

# Analiz edeceğimiz IoU eşikleri. Bunlar, bir tahminin True Positive sayılması için
# gerçek doğruluk kutusuyla minimum örtüşme miktarını (IoU) belirler.
# KITTI Car için 0.7, diğer sınıflar için 0.5 yaygındır [16]. COCO gibi setler için 0.5'ten 0.95'e kadar aralıklar kullanılır [6, 17, 18].
iou_thresholds_for_confidence_analysis = [0.5, 0.7] # Kaynak 35 ve 62'de benzer eşiklerden bahsediliyor [19, 20].

# Analiz edeceğimiz SABİT güven skoru eşikleri. Modelin çıktığı skorlar bu eşiklere göre filtrelenmiş gibi davranılır.
# Genellikle 0.0'dan 1.0'a kadar düzenli aralıklarla seçilir [19, 20].
confidence_thresholds_to_analyze = np.arange(0.0, 1.05, 0.05) # 0.0, 0.05, ..., 1.00 değerleri [20].

# Sonuçları saklayacak sözlük: {iou_thresh: {conf_thresh: {P, R, F1}}}
pr_f1_at_confidence_thresholds = {}

print(f"Analiz Edilen IoU Eşikleri: {iou_thresholds_for_confidence_analysis}")


print("Analiz Edilen Güven Skoru Eşikleri Aralığı:", f"{confidence_thresholds_to_analyze[0]:.2f}","-", f"{confidence_thresholds_to_analyze[-1]:.2f}","(", f"{len(confidence_thresholds_to_analyze)}", "adım)")

# Her bir belirleyici IoU eşiği için güven skoru analizini yap
for current_iou_for_analysis in iou_thresholds_for_confidence_analysis:
    print(f"\nIoU Eşiği > {current_iou_for_analysis:.2f} için güven skoru analizi yapılıyor...")

    # pcdet'in eval_class fonksiyonu, tüm ham tahminleri (dt_annos_raw) alır
    # ve bunları güven skoruna göre azalan sırada sıralayarak bir PR eğrisi hesaplar [2, 6, 7].
    # Bu fonksiyon, farklı geri çağırma noktalarına karşılık gelen Hassasiyet (Precision), Geri Çağırma (Recall)
    # ve bu noktalara ulaşmak için kullanılan Güven Skoru eşiklerini (thresholds) döndürür [2, 8].
    # Döndürülen 'thresholds' dizisi, aslında sıralanmış her bir tahminin veya P/R'nin değiştiği noktadaki skor değeridir [2, 3].
    # Bu dizideki k'inci değere thresholds[k] diyelim. Bu değer, skoru >= thresholds[k] olan tüm tahminler
    # dikkate alındığında elde edilen hassasiyet (precision_points[k]) ve geri çağırmayı (recall_points[k]) temsil eder [21].

    # eval_class için tek bir IoU eşiği içeren min_overlaps yapısı oluşturma
    # Yapı: [num_minoverlap=1, metric, num_class=1]
    # Metrik 2: 3D Box değerlendirmesi için [7, 22].
    min_overlaps_single_iou = np.ones((1, 3, 1), dtype=np.float32)
    min_overlaps_single_iou[:, 2, 0] = current_iou_for_analysis # Sınıf 0 (Araba) ve tüm metrikler için bu IoU eşiğini ayarla (eval_class sadece belirtilen metriği kullanır) [23].

    # eval_class çağrısı: Ham tahminleri (dt_annos_raw), gerçek doğruluk verileriyle (gt_annos)
    # belirli bir sınıf ( -> Araba) ve zorluk seviyesi ( -> Easy) için,
    # yukarıda tanımlanan tek IoU eşiği (min_overlaps_single_iou) ile değerlendirir [7].
    res_eval_class = eval_class(
        gt_annos, dt_annos_raw, [0], [0], 2, min_overlaps_single_iou, compute_aos=False
    )

    # eval_class sonuçlarından Hassasiyet, Geri Çağırma ve bunlara karşılık gelen Güven Skorlarını alalım [8].
    # Bu diziler genellikle [num_class, num_difficulty, num_minoverlap, num_recall_points] şeklindedir.
    # Bizim çağrımızda num_class=1 (sınıf 0), num_difficulty=1 (zorluk 0), num_minoverlap=1 (tek IoU eşiği) olduğu için
    # diziler [1, 1, 1, num_recall_points] şeklinde olacaktır. İlk 3 boyut 0 indeksli olacaktır.
    precision_points = res_eval_class['precision'] # Shape: [num_recall_points]
    recall_points = res_eval_class['recall']    # Shape: [num_recall_points]
    confidence_points = res_eval_class['thresholds'] # Shape: [num_recall_points], azalan sırada [2, 5].

    # Belirlediğimiz SABİT güven skoru eşikleri için P/R/F1 değerlerini bu PR noktalarından 'çekelim'.
    # pr_f1_at_confidence_thresholds sözlüğünü bu IoU eşiği için hazırla.
    pr_f1_at_confidence_thresholds[current_iou_for_analysis] = {}

    # Analiz edilecek her bir güven skoru eşiği için döngü yap.
    for conf_threshold in confidence_thresholds_to_analyze:
        # Hedefimiz: Skoru >= conf_threshold olan tüm tahminleri dikkate alarak elde edilen P/R/F1'i bulmak.
        # eval_class'ın döndürdüğü confidence_points dizisi azalan sıradadır ve PR noktalarına karşılık gelir.
        # confidence_points[k] skoru, k'inci detection'ın (azalan sırada) skorudur.
        # Skoru >= conf_threshold olan detection'lar kümesi, thresholds array'inde değeri
        # conf_threshold'dan büyük veya eşit olan detection'ları kapsar.
        # threshold array'i azalan sırada olduğu için, conf_threshold'dan BÜYÜK VEYA EŞİT olan ilk değeri (en yüksek skora sahip detection) bulmamız gerekiyor [9].
        # Bu, np.where ile conf_threshold'dan >= olan tüm indeksleri bulup, bunlardan en küçüğünü (ilkini) alarak yapılır.
        valid_indices = np.where(confidence_points >= conf_threshold)

        if len(valid_indices) > 0:
            # thresholds azalan sırada. valid_indices'deki ilk eleman, conf_threshold'dan >= olan en yüksek skora sahip detection'ın indeksidir.
            # Bu indeksteki precision_points ve recall_points değerleri, skoru >= confidence_points[first_valid_index] olan tüm detection'lar dikkate alındığında elde edilen P/R'dir.
            # Bu, istediğimiz conf_threshold eşiğiyle elde edilen performansa en yakın, eval_class çıktısından doğrudan alınabilen noktadır.
            first_valid_index = np.squeeze(valid_indices)
            p = np.squeeze(precision_points)[first_valid_index]
            r = np.squeeze(recall_points)[first_valid_index]

            # F1-skoru hesapla: 2 * (P * R) / (P + R). P+R sıfırsa, F1 sıfırdır [13, 19, 24].
            denominator = p + r
            f1 = 2 * (p * r) / denominator if denominator.all() > 1e-6 else 0.0 # Float hassasiyeti için küçük eşik kullan [24].

            pr_f1_at_confidence_thresholds[current_iou_for_analysis][conf_threshold] = {
                'Precision': p,
                'Recall': r,
                'F1-Score': f1
            }
        else:
            # Eğer hiçbir detection belirlenen güven skoru eşiğini (veya daha yükseklerini) sağlamıyorsa,
            # o zaman Doğru Pozitif (TP) ve Yanlış Pozitif (FP) sayısı 0'dır.
            # Hassasiyet = 0 / (0 + 0) = 0 (tanımsız veya 0 kabul edilir).
            # Geri Çağırma = 0 / (0 + Toplam GT Sayısı) = 0.
            # F1-Skoru = 0.
            pr_f1_at_confidence_thresholds[current_iou_for_analysis][conf_threshold] = {
                'Precision': 0.0,
                'Recall': 0.0,
                'F1-Score': 0.0
            }

# Hesaplanan P/R/F1 sonuçlarını güven skoru eşiklerine göre yazdırma [25].
print("\nHesaplanan Güven Skoru Eşiklerine Göre P/R/F1 Sonuçları:")

for iou_thresh, results in pr_f1_at_confidence_thresholds.items():
    print(f"\n--- IoU Eşiği > {iou_thresh:.2f} ---")
    print("Güven Eşiği | Precision | Recall | F1-Score")
    print("-------------------------------------------------")
    # Güven eşiklerini artan sırada sıralayarak yazdıralım
    for conf_thresh in sorted(results.keys()):
        metrics = results[conf_thresh]
        precision = metrics['Precision']
        recall = metrics['Recall']
        f1_score = metrics['F1-Score']
        # NumPy skalerlerini doğru formatta yazdırmak için .item() kullanılabilir veya doğrudan yazdırılabilir.
        # Eğer results[conf_thresh]['Precision'] bir NumPy array ise, .item() kullanmak onu Python skalerine dönüştürür.
        # eval_class'tan gelen precision_points ve recall_points dizilerindeki değerler NumPy skalerleri gibidir.
        print(f" {conf_thresh:.2f} | {precision.item():.4f} | {recall.item():.4f} | {f1_score.item():.4f}")

# --- Güven Skoru ve Lokalizasyon Kalitesi (IoU) İlişkisi Analizi (Ek Analiz İçin) ---
# Bu kısım, her bir DOĞRU TESPİT (True Positive) için, tahminin güven skoru ile
# gerçek doğruluk kutusuyla olan IoU değeri arasındaki ilişkiyi inceler [26-28].
# Bu analiz, modelin tahmin ettiği güven skorunun, kutunun gerçekte ne kadar doğru lokalize edildiğini
# (IoU değeri) ne kadar iyi yansıttığını anlamak için önemlidir [26-29].
# Kaynaklar, bazı modellerde bu korelasyonun zayıf olabileceğini, yüksek güvenle düşük IoU'lu tahminlerin olabileceğini belirtir [28, 30].
# Bu analizi tam olarak yapmak için, değerlendirme süreci sırasında hangi tahminlerin TP olduğu,
# güven skorları ve eşleştikleri GT ile olan IoU değerleri bilgisini toplamamız gerekir.
# pcdet'in eval_class çıktıları bu bilgiyi doğrudan sağlamaz, kütüphanenin iç mekanizmasına
# daha derinlemesine inmek veya özel bir değerlendirme döngüsü yazmak gerekebilir [27, 31].

print("\n--- Güven Skoru ve Lokalizasyon Kalitesi (IoU) İlişkisi (Kavramsal Analiz) ---")
print("Bu analiz, her bir DOĞRU TESPİT (True Positive) için, tahminin güven skoru ile")
print("gerçek doğruluk kutusuyla olan IoU değeri arasındaki ilişkiyi inceler. [26-28]")
print("pcdet'in eval_class çıktılarından doğrudan bu eşleştirme detaylarına erişmek")
print("kod üzerinde ek geliştirmeler gerektirir. [27, 31]")
print("Ancak, bu analizi yapmak için kavramsal adımlar şunlardır:")
print("1. Belirli bir IoU eşiği altında True Positive olarak belirlenen her bir tahmini (detection) kaydedin.")
print("2. Kaydederken, bu TP tahminlerinin güven skorlarını ve eşleştikleri gerçek doğruluk kutularıyla olan IoU değerlerini not alın. [26]")
print("3. Toplanan güven skoru-IoU çiftlerini kullanarak dağılım grafikleri (scatter plot) çizin. [26]")
print("4. Güven skoru ile IoU arasındaki korelasyon katsayısını hesaplayın. [26]")
print("Beklenti: Yüksek güven skoruna sahip tahminlerin, ortalama olarak, yüksek IoU değerlerine sahip olmasıdır. [28]")
print("Ancak, bazı modellerde bu korelasyon zayıf olabilir [30], yani model yüksek güvenle yanlış lokalizasyonlar yapabilir [28]. Bu analizin amacı bu ilişkiyi ortaya koymaktır [26].")

# Orijinal scriptte saklanan mAP sonuçları (ilk IoU eşiği ve Easy zorluk için)
# result_map_standard sözlüğünde saklandı ve yazdırıldı [32].