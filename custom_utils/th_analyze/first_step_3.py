import sys
import pickle
import numpy as np
import os # Dosya yolu kontrolü için os kütüphanesini ekleyelim

# 3DTrans veya OpenPCDet ortamınızın kök dizinine giden yolu ayarlayın [5]
# BU SATIRI KENDİ 3DTrans VEYA OpenPCDet REPO YOLUNUZLA DEĞİŞTİRMELİSİNİZ!
sys.path.insert(0, '/root/3DTrans/')
pcdet_root_path = '/root/3DTrans/'
# pcdet değerlendirme modülünü import etme [5]
# Bu import, yukarıdaki yolda 'pcdet/datasets/kitti/kitti_object_eval_python/eval.py' dosyasını bulmalıdır.

from pcdet.datasets.kitti.kitti_object_eval_python.eval import do_eval, eval_class, get_mAP_R40
print("pcdet evaluation module imported successfully.")


# --- Veri Seti ve Inference Sonuçlarının Yüklenmesi --- [2, 3, 39, 40]

# Tahminleri (detections) ve ground truth verilerini yükleme
# BU YOL KENDİ result.pkl DOSYANIZIN YOLU İLE DEĞİŞTİRİLMELİDİR! [39]
# Bu dosya, modelinizin çıkarım (inference) sonuçlarını içerir (genellikle boxlar, skorlar, isimler).
result_path = '/root/3DTrans/output/cfgs/MDF/KNW/customnw_pvrcnn_feat_3_uni3d/default/eval/epoch_2/val/default/result.pkl' # Örnek: '/root/3DTrans/output/cfgs/.../result.pkl'

# BU YOL KENDİ GROUND TRUTH infos DOSYANIZIN YOLU İLE DEĞİŞTİRİLMELİDİR! [40]
# Bu dosya, değerlendirme yapılacak veri setinin ground truth bilgilerini içerir.
# KAYNAK ÖRNEĞİNE GÖRE custom_infos_test.pkl kullanıldı [40].
gt_infos_path = '/root/3DTrans/data/custom_kitti2/custom_infos_test.pkl' # Örnek: '/root/3DTrans/data/custom_kitti2/custom_infos_test.pkl'



# Tahmin verilerini yükleme
with open(result_path, 'rb') as f:
    dt_data = pickle.load(f)
print(f"Successfully loaded detection data from {result_path}") 

# Ground truth verilerini yükleme
with open(gt_infos_path, 'rb') as f:
    gt_data = pickle.load(f)
print(f"Successfully loaded ground truth data from {gt_infos_path}") 



# --- Veri Formatını Dönüştürme (KITTI formatına uygun hale getirme) --- [2, 40-44]

# Değerlendirme modülleri genellikle KITTI formatında organize edilmiş annotationları bekler.
# Sağlanan kaynak kodu, .pkl dosyasındaki veriyi bu formata dönüştürüyor. [40-44]
# 'gt_data.txt' kaynağı, ground truth verilerinin beklenen yapısını doğrular. [18-38]

gt_annos = []
for gt_info in gt_data:
    # Ground truth annotationları için gerekli alanları ekleme/dönüştürme [40-42]
    # KITTI formatı için 'bbox' (2D), 'dimensions', 'location' (3D), 'rotation_y', 'name' gerekir.
    # 'occluded', 'truncated', 'difficulty' gibi alanlar da değerlendirme tarafından kullanılabilir.
    # Kaynak koddaki gibi varsayılan değerler atandı, ancak gerçek veriden gelmesi daha doğru olur [42].
    
    anno = {
        "bbox": gt_info["annos"]["gt_boxes_lidar"],
        "name": gt_info["annos"]["name"], # Sınıf isimleri genellikle 'name' altında bulunur
        "dimensions": gt_info["annos"]["dimensions"], # Boyutlar
        "location": gt_info["annos"]["location"], # Konum
        "rotation_y": gt_info["annos"]["rotation_y"], # Dönüş
        "alpha": np.asarray([0.0] * len(gt_info["annos"]["gt_boxes_lidar"]), dtype=np.float32), # Örnek koddaki gibi 0.0 olarak ayarlandı
        "occluded": np.asarray( [0.0] * len(gt_info["annos"]["gt_boxes_lidar"]), dtype=np.int32), # Örnek koddaki gibi 0 olarak ayarlandı
        "truncated": np.asarray([0.0] * len(gt_info["annos"]["gt_boxes_lidar"]), dtype=np.float32) # Örnek koddaki gibi 0.0 olarak ayarlandı
    }
    # Sınıf adları ve box sayıları tutarlı olmalı
    assert len(anno["name"]) == len(anno["bbox"]), "GT name and box counts mismatch!"
    gt_annos.append(anno)

dt_annos_raw = []
for dt in dt_data:
    # Tahmin annotationları için gerekli alanları ekleme/dönüştürme [42-44]
    # 'boxes_lidar' (3D box), 'score' (güven skoru), 'name' (sınıf) beklenir [42, 43].
    # 'dimensions', 'location', 'rotation_y' de model çıktısında varsa eklenir [43].
    # Kaynak kod örneğinde 'location', 'dimensions', 'rotation_y' dt'den alınmış [43], bu doğru.
    dt_anno = {
        "bbox": dt["boxes_lidar"],
        "score": dt["score"], # Güven skoru [3]
        "name": dt["name"], # Sınıf ismi [3]
        "alpha" : np.asarray([0 for i in range(len(dt["boxes_lidar"]))]),  # Add alpha key
        "occluded" : np.asarray([0 for i in range(len(dt["boxes_lidar"]))]),  # Add alpha key
        "truncated" : np.asarray([0 for i in range(len(dt["boxes_lidar"]))]),  # Add alpha key
        "location": dt["location"],
        "dimensions": dt["dimensions"],
        "rotation_y": dt["rotation_y"]
        # Diğer gerekli alanlar (dimensions, location, rotation_y) eğer mevcutsa eklenebilir
    }
    # Sınıf adları ve box sayıları tutarlı olmalı
    assert len(dt_anno["name"]) == len(dt_anno["bbox"]), "DT name and box counts mismatch!"
    dt_annos_raw.append(dt_anno)

print("Data conversion to KITTI format complete.")

# --- Standart mAP Hesaplaması (do_eval kullanarak) --- [3, 6, 45-52]

# KITTI stilinde mAP hesaplaması için IoU eşiklerini belirleme [44, 49]
# Bu eşikler `do_eval` fonksiyonu içindir. Kaynak kodda 11 farklı eşik örneği verilmiş. [44]
standard_map_iou_thresholds = np.array([0.3, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

# Değerlendirme yapılacak sınıfları ve zorluk seviyelerini belirleme [45]
# KITTI için yaygın olarak Car (0), Pedestrian (1), Cyclist (2) kullanılır. [45]
# Zorluk seviyeleri Easy (0), Moderate (1), Hard (2) [45]
class_ids_to_eval = [0] # Örnek: Sadece Araba (Car) sınıfı [45] - İhtiyaca göre [72, 73] olarak değiştirilebilir.
difficulty_levels_to_eval = [0] # Örnek: Tüm zorluk seviyeleri (Easy, Moderate, Hard) [45] - İhtiyaca göre ayarlanabilir.

# `do_eval` fonksiyonunun beklediği `min_overlaps` yapısını oluşturma. [46-49]
# Yapı genellikle [num_minoverlap, num_metric, num_class] şeklindedir, metrikler (Image, BEV, 3D, AOS) için [48].
# Kaynak [48] metric 2'nin 3D olduğunu belirtir. Metric sayısı pcdet sürümüne göre değişebilir, yaygın olarak 4 kullanılır.
# Boyut: [len(standard_map_iou_thresholds), num_metrics (örn. 4), len(class_ids_to_eval)]
num_metrics = 4 # Varsayılan KITTI metrik sayısı (Image, BEV, 3D, AOS)
min_overlaps_for_map = np.zeros((len(standard_map_iou_thresholds), num_metrics, len(class_ids_to_eval)), dtype=np.float32)

# Her bir IoU eşiğini tüm metrikler ve sınıflar için ayarlama. [44]
# KITTI değerlendirmesinde genellikle farklı sınıflar için farklı IoU eşikleri kullanılır (örn. Car için 0.7, Ped/Cyc için 0.5).
# Kaynak kod [44] tüm eşikleri tek bir diziye koymuş. Kaynak [49] ise tek bir IoU için 3D metriğini ayarlamış.
# Standart mAP için tüm 11 eşiği kullanacağız, ancak belirli metrikler için doğru sütuna ayarlamalıyız.
# Yaygın KITTI config'e göre 3D (metric index 2) ve BEV (metric index 1) değerlendirilir.
# Örneğin, Car (class 0) için 3D eşiği 0.7, BEV eşiği 0.7 olabilir. Ped/Cyc için 3D 0.5, BEV 0.5 olabilir.
# Kaynak kodun genel yapısı [44] tüm 11 IoU'yu `min_overlaps_for_map`'e koymayı amaçlıyor gibi.
# Pcdet'in `do_eval` fonksiyonunun iç implementasyonuna bağlı olarak bu yapı değişir.
# Kaynak [49] ve [74] 3D metrik (index 2) için tek bir eşik ayarlama örneği sunmuş: `min_overlaps_single_iou[:, 2, 0] = current_iou_for_analysis`.
# Buna dayanarak, tüm 11 eşiği sadece 3D metrik sütununa (index 2) atayalım ve tüm sınıflar için tekrarlayalım.
for i, iou_thresh in enumerate(standard_map_iou_thresholds):
    min_overlaps_for_map[i, 2, :] = iou_thresh # 3D Metriği (index 2) için tüm sınıflara IoU eşiğini ata.
    # İsteğe bağlı: BEV metriği (index 1) için de farklı eşikler atayabilirsiniz.
    # min_overlaps_for_map[i, 1, :] = iou_thresh # BEV Metriği (index 1) için tüm sınıflara IoU eşiğini ata.


print("\nCalculating standard mAP (do_eval)...")

# do_eval çağrısı. Argümanlar: gt_annos, dt_annos, class_ids, min_overlaps, compute_aos [46, 49]
# compute_aos (Angle Orientation Similarity) genellikle 2D veya BEV değerlendirmesiyle ilgilidir. 3D için False bırakılabilir.
# do_eval'ın döndürdüğü sonuçların yapısı pcdet sürümüne göre değişebilir. [50, 51]
# Kaynak [51] 8 değer döndürdüğünü gösteriyor: mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40
# Bu değerler genellikle [num_class, num_difficulty, num_minoverlap] boyutunda numpy dizileridir. [51, 52]

# `do_eval` çağrısı `difficulty_levels_to_eval` argümanını doğrudan almayabilir.
# Genellikle GT annotationlarındaki 'difficulty' alanına bakar.
# Çağrı argümanları pcdet/datasets/kitti/kitti_object_eval_python/eval.py dosyasından kontrol edilmelidir.
# Kaynak [49] çağrıda `class_ids` ve `min_overlaps_for_map` kullanmış.
# Basitlik adına, bu çağrının tüm zorluk seviyelerini GT annotationlarındaki 'difficulty' alanına göre işlediğini varsayalım.

mAP_results = do_eval(
    gt_annos, dt_annos_raw, class_ids_to_eval, min_overlaps_for_map, compute_aos=False)

# Sonuçları çıkarma (8 ayrı değişken olarak) [51]
mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = mAP_results

print("\n--- Standard mAP Results ---")
# Sonuçları yazdırma (Örnek: Car (0) sınıfı için 3D mAP sonuçları)
# mAP3d dizisi [num_class, num_difficulty, num_minoverlap] boyutundadır [51, 52].

class_names = ['Car'] # KITTI için yaygın isimler
difficulty_names = ['Easy'] # KITTI zorluk isimleri

for c_idx, class_id in enumerate(class_ids_to_eval):
    print(f"Class: {class_names[class_id]} (ID: {class_id})")

    # mAP3d boyutu: [num_class, num_difficulty, num_minoverlap]
    num_difficulties = mAP3d.shape[1]
    if num_difficulties != 1:
        print(f"  Warning: More than one difficulty level detected in results ({num_difficulties}). Only index 0 will be used.")

    d_idx = 0  # Zorluk seviyesi sabit: sadece Easy (veya tek difficulty varsa o)
    print(f"  Difficulty: Ignored (Index {d_idx}):")

    for i_idx, iou_thresh in enumerate(standard_map_iou_thresholds):
        map_value = mAP3d[c_idx, d_idx, i_idx]
        print(f"    3D mAP @ IoU={iou_thresh:.2f}: {map_value:.4f}")

    if mAP3d_R40 is not None and mAP3d_R40.shape[1] > d_idx:
        map_r40_value = mAP3d_R40[c_idx, d_idx]
        print(f"    3D mAP@R40: {map_r40_value:.4f}")
        
print("-" * 25)


# --- Güven Skoru Eşiği Analizi (eval_class kullanarak) --- [13, 14, 15, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, Previous Conversation]

# `eval_class` fonksiyonunu kullanarak farklı güven skoru eşiklerindeki P/R/F1 değerlerini hesaplama. [6]
# Bu analiz, belirli bir IoU eşiği ve belirli bir sınıf/zorluk seviyesi için yapılır. [53, 55]

# Güven skoru analizini yapacağımız IoU eşikleri [7, 53]
# Bu eşikler genellikle mAP hesaplaması için kullanılanlardan daha standarttır (örn. 0.5, 0.7).
iou_thresholds_for_confidence_analysis = [0.5, 0.7]

# Analiz edeceğimiz güven skoru eşikleri [4, 53]
# Kaynaklar 0.1'den 0.95'e artan adımları önerir [4]. Kaynak kodda 0.0'dan 1.05'e 0.05 adım kullanılmış. [53]
confidence_thresholds_to_analyze = np.arange(0.0, 1.01, 0.01) # 0.0'dan 1.0'a 0.01 adımla daha ince analiz

# Güven skoru analizi yapılacak tek bir sınıf ve zorluk seviyesi seçin [11, 53]
# KITTI için Car=0, Pedestrian=1, Cyclist=2; Easy=0, Moderate=1, Hard=2.
analysis_class_id = 0 # Örnek: Car sınıfı [53] - İhtiyaca göre değiştirilebilir.
analysis_difficulty_id = 1 # Örnek: Moderate zorluk seviyesi [53] - İhtiyaca göre değiştirilebilir.

pr_f1_at_confidence_thresholds = {} # Sonuçları saklayacak sözlük: {iou_thresh: {conf_thresh: {'precision': p, 'recall': r, 'f1': f1}}} [53]

print(f"\n--- Confidence Score Threshold Analysis (Class: {class_names[analysis_class_id]}, Difficulty: {difficulty_names[analysis_difficulty_id]}) ---")

for current_iou_for_analysis in iou_thresholds_for_confidence_analysis:
    print(f"\nAnalyzing for IoU Threshold > {current_iou_for_analysis}...")

    # `eval_class` için tek bir IoU eşiği içeren `min_overlaps` yapısı oluşturma [54, 74]
    # Yapı: [num_minoverlap=1, metric, num_class=1]. Metrik 2: 3D Box [54].
    min_overlaps_single_iou = np.ones((1, num_metrics, 1), dtype=np.float32) # 1 IoU, num_metrics, 1 Class
    min_overlaps_single_iou[:, 2, 0] = current_iou_for_analysis # Sadece 3D metrik (index 2) ve tek sınıf (index 0) için ayarla [54]
    # Eğer BEV metriği (index 1) için de analiz yapmak isterseniz:
    # min_overlaps_single_iou[:, 1, 0] = current_iou_for_analysis

    # `eval_class` çağrısı. Bu, detectionları güven skoruna göre sıralar ve farklı recall noktalarındaki P/R değerlerini hesaplar. [55]
    # `eval_class(gt_annos, dt_annos, class_ids, difficulty_ids, metric_index, min_overlaps, compute_aos)` [55]
    # metric_index: Hangi metriğe (Image=0, BEV=1, 3D=2, AOS=3) göre değerlendirme yapılacağını belirtir. Biz 3D (2) metriğini kullanıyoruz. [55]
    try:
        # eval_class tek bir sınıf ve tek bir zorluk seviyesi için çağrılır.
        res_eval_class = eval_class(
            gt_annos, dt_annos_raw, [analysis_class_id], [analysis_difficulty_id], 2, min_overlaps_single_iou, compute_aos=False
        )

        # eval_class sonuçlarından hassasiyet (Precision), geri çağırma (Recall) ve bunlara karşılık gelen güven eşiklerini alalım. [56, 57]
        # Boyutlar genellikle [num_class=1, num_difficulty=1, num_minoverlap=1, num_recall_points] [56]
        eval_precision_points = res_eval_class['precision'][0, 0, 0, :]
        eval_recall_points = res_eval_class['recall'][0, 0, 0, :]
        eval_confidence_points = res_eval_class['thresholds'][0, 0, 0, :] # Tahmin skorları, azalan sırada [57]

        # Eğer eval_class herhangi bir nokta döndürmediyse (örneğin hiç TP yoksa) [57]
        if len(eval_confidence_points) == 0:
            print(f" No evaluation points returned by eval_class for IoU > {current_iou_for_analysis}. Setting P/R/F1 to 0 for all confidence thresholds.")
            results_for_this_iou = {
                ct: {'precision': 0.0, 'recall': 0.0, 'f1': 0.0} for ct in confidence_thresholds_to_analyze
            } 
        else:
            # Belirlediğimiz güven skoru eşikleri için P/R/F1 değerlerini bulalım. [58]
            # `eval_confidence_points` azalan sırada. Her bir `conf_thresh` T için, skoru >= T olan tüm detectionlar
            # alınarak hesaplanan P/R değerini bulmalıyız.
            # Kaynak kod yorumlarındaki doğru mantık: `eval_confidence_points` dizisinde, `conf_thresh`'e en yakın *ve ondan büyük eşit* olan *en küçük skora* karşılık gelen indeksi bulmalıyız. [64, 65]
            # Bu, `eval_confidence_points` azalan dizisinde `conf_thresh` değerinin *ilk kez aşıldığı* veya *tam eşitlendiği* noktanın indeksi olacaktır.
            # `np.searchsorted(..., side='right') - 1` bu indeksi bulmak için kullanılır. [66]

            results_for_this_iou = {}
            print("\n Confidence | Precision | Recall | F1-Score")
            print(" -------------------------------------------------")

            for conf_thresh in confidence_thresholds_to_analyze:
                # Azalan dizide `conf_thresh`'ten büyük veya eşit olan son elemanın indeksini bul. [66]
                # Bu indeks, `>= conf_thresh` eşiğini uyguladığımızda dahil edilen detectionların sonuncusuna karşılık gelir.
                index = np.searchsorted(eval_confidence_points, conf_thresh, side='right') - 1

                if index >= 0: # Geçerli bir indeks bulunduysa
                    precision = eval_precision_points[index]
                    recall = eval_recall_points[index]
                else: # conf_thresh en yüksek skordan bile büyükse, hiçbir detection dahil edilmez (TP=0, FP=0).
                    precision = 0.0
                    recall = 0.0

                # F1-skor hesaplama [4, 8, 67]
                # Paydanın sıfır olma durumunu ele al (recall ve precision aynı anda 0 olduğunda).
                denominator = precision + recall
                f1 = np.where(denominator > 0, 2 * (precision * recall) / denominator, 0.0)

                results_for_this_iou[float(conf_thresh)] = {'precision': float(precision), 'recall': float(recall), 'f1': float(f1)} # Dictionary'e ekle
                print(f" {conf_thresh:.2f} | {precision:.4f} | {recall:.4f} | {f1:.4f}")

        pr_f1_at_confidence_thresholds[current_iou_for_analysis] = results_for_this_iou

    except Exception as e:
        print(f"Error during eval_class or confidence analysis for IoU > {current_iou_for_analysis}: {e}")
        print(f"Skipping confidence threshold analysis for IoU > {current_iou_for_analysis}.")

# --- Güven Skoru ve Lokalizasyon Kalitesi (IoU) İlişkisi Analizi (Kavramsal) --- [9, 10, 68-71]

# Bu analiz, her bir DOĞRU POZİTİF (TP) tespit için tahmin edilen güven skoru ile
# eşleştiği ground truth nesnesiyle olan GERÇEK IoU değeri arasındaki ilişkiyi inceler. [9, 68, 69]
# Kaynaklardaki analiz betiğinde BU ANALİZİ YAPACAK KOD BULUNMAMAKTADIR, sadece yorumlar ve açıklamalar mevcuttur. [68, 69]
# Bu, pcdet'in `eval_class` fonksiyonunun iç eşleştirme detaylarına doğrudan erişim gerektirir. [69]

print("\n--- Confidence Score vs. Localization Quality (IoU) Analysis (Conceptual) ---")
print("This analysis examines the relationship between the confidence score of each TRUE POSITIVE detection")
print("and the actual IoU value with its matched ground truth box. [9, 68, 69]")
print("The provided code snippet does NOT perform this analysis directly.")
print("It requires access to the internal matching results of the evaluation library.")

print("\nConceptually, to perform this analysis:")
print("1. During the evaluation process, identify and record each detection that is a True Positive (TP) based on a specific IoU threshold.")
print("2. For each recorded TP detection, store its confidence score and the calculated IoU value with the ground truth box it matched.")
print("3. Create scatter plots with confidence scores on one axis and corresponding IoU values on the other. [10, 70]")
print("4. Calculate correlation coefficients between confidence scores and IoU values. [10, 71]")
print("This analysis helps understand if the model's confidence score is a reliable indicator of localization accuracy. [9, 10, 71]")

print("\nAnalysis Script Finished.")

# İsteğe bağlı: Hesaplanan P/R/F1 skorlarını veya mAP sonuçlarını bir dosyaya kaydedebilirsiniz.
# import json
# with open('evaluation_results.json', 'w') as f:
#     # NumPy float'ları JSON'a doğrudan yazılamayabilir, stringe veya listeye çevirmek gerekebilir.
#     # JSON serileştirme için daha gelişmiş bir yaklaşıma ihtiyaç duyulabilir.
#     json.dump(pr_f1_at_confidence_thresholds, f, indent=4)
# print("Confidence analysis results saved to evaluation_results.json")
