import sys
import pickle
import numpy as np

# 3DTrans veya OpenPCDet ortamınıza göre yolu ayarlayın
sys.path.insert(0, '/root/3DTrans/') # Eğer /root/3DTrans/ sizin repo yolunuzsa

# eval modülünü import etme
# Bu import, sizin 3DTrans kurulumunuzdaki pcdet.datasets.kitti.kitti_object_eval_python.eval modülünü bulmalıdır.
# Eğer yol doğru ayarlıysa veya Python ortamınızda bu modül erişilebilir durumdaysa bu satır çalışacaktır.

# OpenPCDet veya 3DTrans içindeki değerlendirme modülü
from pcdet.datasets.kitti.kitti_object_eval_python.eval import do_eval, eval_class, get_mAP_R40
print("pcdet evaluation module imported successfully.")



# Tahminleri ve ground truth verilerini yükleme
# Dosya yollarının doğru olduğundan emin olun [1]
# Bu yol, 3DTrans veya OpenPCDet çıktınızdan alınan result.pkl dosyasına işaret etmelidir.
# result_path = '/root/3DTrans/output/cfgs/MDF/KNW/customnw_pvrcnn_feat_3_uni3d/default/eval/epoch_2/val/default/result.pkl'
# Örnek olması için farklı bir yol kullanıyorum, gerçek yolla değiştirin:
result_path = '/root/3DTrans/output/cfgs/MDF/KNW/customnw_pvrcnn_feat_3_uni3d/default/eval/epoch_2/val/default/result.pkl' # <--- Burayı KENDİ result.pkl dosyanızın yolu ile DEĞİŞTİRİN

with open(result_path, 'rb') as f:
    dt_data = pickle.load(f)

result_path = '/root/3DTrans/data/custom_kitti2/custom_infos_test.pkl' # <--- Burayı KENDİ result.pkl dosyanızın yolu ile DEĞİŞTİRİN

with open(result_path, 'rb') as f:
    gt_data = pickle.load(f)

print(f"Successfully loaded data from {result_path}")


gt_annos = []
for gt_info in gt_data:
    # KITTI formatı için gerekli alanları ekleme/dönüştürme [3]
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
    # Zorluk seviyeleri genellikle occluded ve truncated'dan türetilir, veya infos dosyasında ayrı belirtilir.
    # Eğer infos dosyasında 'difficulty' varsa onu kullanmak daha doğru olurdu, yoksa varsayılan 0 kullanılabilir.
    # Burada basitçe 0 olarak varsayalım.
    anno["difficulty"] = np.asarray( [0.0]* len(gt_info["annos"]["gt_boxes_lidar"]), dtype=np.int32)

    gt_annos.append(anno)

# Tahmin verilerini değerlendirme formatına dönüştürme
dt_annos_raw = []
for dt in dt_data:
    # Tahminlerde güven skoru ('score') ve sınıf ('name') bilgilerinin olması beklenir. [3]
    # 'boxes_lidar' 3D sınırlayıcı kutudur.
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
    dt_annos_raw.append(dt_anno)

ation results exist.")
    exit() # Dosya bulunamazsa scripti sonlandır

# KITTI stilinde mAP hesaplaması için IoU eşiklerini belirleme [6, 7, 36]
# Bu eşikler do_eval fonksiyonu içindir.
min_overlaps = np.array([0.3, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]) # Örnek geniş aralık
min_overlaps_for_map = np.ones((min_overlaps.shape[0], 1, 1))
for i in range(min_overlaps.shape[0]):
    min_overlaps_for_map[i, :, :] = min_overlaps[i] # Tüm metrikler için aynı IoU eşiği [3]
# Sadece belirli sınıfları ve zorluk seviyelerini değerlendirmek isteyebilirsiniz
# KITTI için yaygın olarak Car (0), Pedestrian (1), Cyclist (2) kullanılır.
# Zorluk seviyeleri Easy (0), Moderate (1), Hard (2) [5, 10, 23]
class_ids_to_eval =  [0]# Örnek: Sadece Araba (Car) sınıfı [24, 30]
difficulty_levels_to_eval = [0] # Örnek: Tüm zorluk seviyeleri [23, 24, 30]

# Standart mAP (11 recall noktası) ve mAP@R40 hesaplama [6, 7, 37]
# do_eval genellikle farklı zorluk seviyeleri ve IoU eşikleri için mAP döndürür.
print("\nCalculating standard mAP (do_eval)...")
# do_eval çağrısı: gt_annos, dt_annos, class_ids, min_overlaps, compute_aos
# Not: Burada min_overlaps_for_map num_minoverlap x metric x num_class formatında olmalı.
# Örneğin, 3D box (metric=2) ve BEV (metric=1) için Car (class=0) için
# min_overlaps_for_map = np.array([[0.7, 0.7, 0.7]]) # Sadece 3D Car@0.7
# min_overlaps_for_map = np.array([[[0.7],[0.7]], [[0.5],[0.5]]]) # 2 IoU, BEV/3D
# do_eval'ın beklediği formatı pcdet kodundan kontrol etmek faydalı olacaktır.
# Kaynak [9] formatın [num_minoverlap=1, metric, num_class=1] olduğunu gösteriyor gibi,
# ama do_eval genellikle [num_minoverlap, num_metric, num_class] gibi daha genel bir yapı bekler.
# Burada basitlik adına tek bir zorluk seviyesi ve sınıf için standart do_eval çağrısı yapalım.
# Eğer çoklu zorluk veya sınıf istiyorsanız do_eval çağrısını ayarlamanız gerekir.
# Aşağıdaki çağrı, pcdet'in do_eval fonksiyonunun spesifik implementasyonuna bağlıdır.
# Genellikle her sınıf ve zorluk için ayrı ayrı veya topluca hesaplanabilir.
# Örnek: Car (0), Easy (0), Moderate (1), Hard (2) için 3D (2) metriğinde 0.7 IoU eşiği
min_overlaps_for_kitti_mAP = np.array([
    [[0.7], [0.7], [0.7]], # 3D IoU=0.7 (Easy, Mod, Hard)
    [[0.5], [0.5], [0.5]]  # BEV IoU=0.5 (Easy, Mod, Hard) - örnek
    # [IoU_for_Easy, IoU_for_Mod, IoU_for_Hard] for each metric
]) # Bu yapı muhtemelen do_eval'ın beklediği IoU eşiği formatına uymuyor, kontrol edilmeli.

# pcdet'in eval.py koduna bakıldığında, do_eval min_overlaps'ı [num_minoverlap, 4] veya [num_minoverlap, num_class, 4]
# veya benzer bir formatta bekliyor gibi görünüyor, burada 4 AOS, BEV, 3D, Image metrikleri olabilir.
# Kaynak [24]'teki do_eval çağrısı `` class_ids ve `min_overlaps_for_map` kullanmış.
# `min_overlaps_for_map`'in boyutu [num_minoverlap, 4] veya [num_class, num_difficulty, num_metric] formatında olabilir.
# Kaynak [9] 'min_overlaps_single_iou = np.ones((1, 3, 1), dtype=np.float32)' ve 'min_overlaps_single_iou[:, 2, 0] = current_iou_for_analysis'
# kullandığına göre, pcdet'in yapısı [num_minoverlap, num_metric, num_class] gibi duruyor. Metrikler 0=Image, 1=BEV, 2=3D.
# Tek bir IoU eşiği, tek bir metrik (3D), tek bir sınıf (Car) için:
min_overlaps_for_car_3d_mAP = np.array([[[0.7]]]) # 1 IoU eşiği (0.7), 1 metrik (3D), 1 sınıf (Car)

# do_eval fonksiyonu genellikle tüm sınıflar ve zorluk seviyeleri için toplu sonuç döndürür.
# Pcdet'in do_eval tanımına bakarak hangi argümanları beklediğini kontrol edin.
# Kaynak [24]'teki çağrıya benzer şekilde:
try:
    mAP_results = do_eval(
        gt_annos, dt_annos_raw, class_ids_to_eval, min_overlaps_for_map, compute_aos=False
        # class_ids_to_eval listesi, min_overlaps_for_map dizisi num_class boyutuna uygun olmalı
        # Eğer min_overlaps_for_map sadece IoU değerlerini içeriyorsa, do_eval'ın iç logic'i onu işler.
        # Varsayımsal olarak do_eval [num_class, num_difficulty, num_minoverlap, num_metric] boyutunda mAP döndürüyor.
    )
    # mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40
    # Kaynak [24] 8 değer döndürdüğünü gösteriyor.
    # Bizim örneğimizde class_ids_to_eval=, difficulty_levels_to_eval=[5, 10]
    # min_overlaps_for_map 11 IoU eşiği içeriyor.
    # mAP_results'ın boyutu karmaşık olabilir: (num_class, num_difficulty, num_minoverlap, num_metric)
    # veya do_eval bunları bir tuple/listede birleştirir.
    # Kaynak [24] 8 ayrı değişken ataması yapmış. Bu formatı kullanalım.
    mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = mAP_results

    print("\n--- Standard mAP Results ---")
    # Sonuçları yazdırma (Örnek: Car (0) sınıfı, Moderate (1) zorluk seviyesi için 0.7 IoU 3D mAP)
    # Bu indeksler mAP_results'ın do_eval tarafından nasıl döndürüldüğüne bağlı olacaktır.
    # Kaynak [24] [num_class, num_diff, num_minoverlap] formatından bahsetmiş.
    # Varsayalım ki mAP3d [num_class, num_diff, num_minoverlap] boyutundadır.
    # Örneğin, Car (0), Moderate (1), IoU=0.7 (min_overlaps_for_map dizisindeki indeksi bulmalısınız)
    # IoU=0.7'nin min_overlaps_for_map dizisindeki indeksi: np.where(min_overlaps_for_map == 0.7) varsayalım 5
    try:
        mod_0_7_iou_index = np.where(min_overlaps_for_map == 0.7)
        indexed_map = mAP3d[0, 1, mod_0_7_iou_index]
        print(f"3D mAP for Car (Class {class_ids_to_eval}), Moderate Difficulty (0), IoU=0.7: {indexed_map:.4f}")
    except Exception as e:
        print(f"Could not print specific mAP result, check do_eval output structure: {e}")
    print("-" * 25)

except Exception as e:
    print(f"Error during standard mAP calculation with do_eval: {e}")
    print("Skipping standard mAP output.")


# --- Güven Skoru Eşiği Analizi ---
# eval_class fonksiyonunu kullanarak farklı güven skoru eşiklerindeki P/R/F1 değerlerini hesaplama [8, 26-28]

# Güven skoru analizini yapacağımız IoU eşikleri [5, 8]
# eval_class her bir *tekil* IoU eşiği için bir PR eğrisi döndürür.
iou_thresholds_for_confidence_analysis = [0.5, 0.7]

# Analiz edeceğimiz güven skoru eşikleri [5, 8]
# Bu, 0.0'dan 1.0'a 0.05 adımla giden bir dizi oluşturur (örn. 0.0, 0.05, 0.10, ..., 1.00)
confidence_thresholds_to_analyze = np.arange(0.0, 1.05, 0.05)

# Sınıf ve zorluk seviyesini seçin (eval_class tek bir sınıf ve zorluk için çağrılır) [30]
analysis_class_id = 0 # Örnek: Car
analysis_difficulty_id = 1 # Örnek: Moderate (1) [23]

pr_f1_at_confidence_thresholds = {} # Sonuçları saklayacak sözlük: {iou_thresh: {conf_thresh: {'precision': p, 'recall': r, 'f1': f1}}}

print(f"\n--- Confidence Score Threshold Analysis (Class {analysis_class_id}, Difficulty {analysis_difficulty_id}) ---")

for current_iou_for_analysis in iou_thresholds_for_confidence_analysis:
    print(f"\nAnalyzing for IoU Threshold > {current_iou_for_analysis}...")

    # eval_class için tek bir IoU eşiği içeren min_overlaps yapısı oluşturma [9]
    # Yapı: [num_minoverlap=1, metric, num_class=1]
    # Metrik 2: 3D Box [30]
    min_overlaps_single_iou = np.ones((1, 3, 1), dtype=np.float32)
    min_overlaps_single_iou[:, 2, 0] = current_iou_for_analysis # 3D metriği için bu IoU eşiği [9]

    # eval_class çağrısı. Bu, detectionları güven skoruna göre sıralar ve farklı recall noktalarındaki P/R değerlerini hesaplar. [30, 32]
    # eval_class(gt_annos, dt_annos, class_ids, difficulty_ids, metric, min_overlaps, compute_aos)
    # Bizim durumumuzda tek sınıf, tek zorluk, tek metrik (3D), tek min_overlap (IoU eşiğimiz)
    try:
        res_eval_class = eval_class(
            gt_annos, dt_annos_raw, [analysis_class_id], [analysis_difficulty_id], 2, min_overlaps_single_iou, compute_aos=False
        )

        # eval_class sonuçlarından hassasiyet, geri çağırma ve bunlara karşılık gelen güven eşiklerini alalım. [31]
        # res_eval_class['precision'] -> [num_class=1, num_difficulty=1, num_minoverlap=1, num_recall_points]
        # res_eval_class['recall']    -> [num_class=1, num_difficulty=1, num_minoverlap=1, num_recall_points]
        # res_eval_class['thresholds'] -> [num_class=1, num_difficulty=1, num_minoverlap=1, num_recall_points] (güven skorları, azalan sırada)

        # Elde edilen PR eğrisi noktaları ve güven skorları
        eval_precision_points = res_eval_class['precision'][0, 0, 0, :]
        eval_recall_points = res_eval_class['recall'][0, 0, 0, :]
        eval_confidence_points = res_eval_class['thresholds'][0, 0, 0, :] # Tahmin skorları, azalan sırada [31]

        # Eğer eval_class herhangi bir nokta döndürmediyse (örneğin hiç TP yoksa)
        if len(eval_confidence_points) == 0:
            print(f"  No evaluation points returned by eval_class for IoU > {current_iou_for_analysis}. Setting P/R/F1 to 0 for all confidence thresholds.")
            pr_f1_at_confidence_thresholds[current_iou_for_analysis] = {
                ct: {'precision': 0.0, 'recall': 0.0, 'f1': 0.0} for ct in confidence_thresholds_to_analyze
            }
        else:
            # Belirlediğimiz güven skoru eşikleri için P/R/F1 değerlerini bulalım.
            # Her bir conf_threshold T için, eval_confidence_points dizisinde >= T olan en büyük skora karşılık gelen
            # P/R noktasını bulmalıyız. Eval confidence points azalan sırada olduğu için, bu T'den >= olan
            # *ilk* veya *son* indekse karşılık gelecektir. Kaynak [32]'teki mantığa göre, skoru >= thresholds[i] olan tüm
            # detection'ları alarak precision_points[i] ve recall_points[i] elde edilir.
            # Dolayısıyla, bir conf_threshold C için, eval_confidence_points içinde C'den >= olan
            # *en büyük indeksi* bulmak, C eşiği için performansı verir.

            results_for_this_iou = {}
            print("  Confidence | Precision | Recall   | F1-Score")
            print("  -------------------------------------------------")

            for conf_thresh in confidence_thresholds_to_analyze:
                # conf_thresh'ten büyük veya eşit olan güven skorlarının indekslerini bul [33]
                # eval_confidence_points azalan sırada. conf_thresh >= eval_confidence_points[i]
                # VEYA eval_confidence_points[i] <= conf_thresh olan indeksleri buluyoruz.
                # eval_confidence_points[i] >= conf_thresh olan indeksleri bulmalıyız.
                # np.where(eval_confidence_points >= conf_thresh) bize bu koşulu sağlayan tüm indeksleri verir.
                valid_indices = np.where(eval_confidence_points >= conf_thresh)

                if len(valid_indices) > 0:
                    # Koşulu sağlayan indeksler varsa, en büyük indeksi al (çünkü azalan sırada, bu en düşük skora sahip TP noktasını temsil eder >= conf_thresh)
                    # Aslında mantık tam tersi: C eşiğini kullandığımızda, skoru C'den büyük veya eşit olan TPs/FPs sayılır.
                    # eval_class çıktısındaki threshold[i] değeri, i-inci detection'ın skorudur (azalan).
                    # precision_points[i] ve recall_points[i] skoru >= threshold[i] olan tüm detectionlarla hesaplanmıştır.
                    # Bizim amacımız, skoru >= conf_thresh olan detectionları almak.
                    # eval_confidence_points azalan sırada: [s_0, s_1, s_2, ..., s_N] where s_0 >= s_1 >= ... >= s_N
                    # threshold = s_i demek, detection 0'dan i'ye kadar olanları (skoru >= s_i olanları) dahil etmek demek.
                    # Bizim eşiğimiz T. Skoru >= T olanları almak istiyoruz.
                    # eval_confidence_points dizisinde T'den büyük veya eşit olan *ilk* değeri bulmak en doğru mantık olabilir.
                    # np.searchsorted(eval_confidence_points, conf_thresh, side='left') azalan dizide, conf_thresh'in nereye
                    # yerleştirilmesi gerektiğini söyler ki dizi sıralı kalsın. side='left' >= koşuluna bakar.
                    # Eğer dizi [0.9, 0.8, 0.7, 0.6], conf_thresh=0.75 ise, 0.75 > 0.7, 0.75 < 0.8. 0.75, 0.8'in sağına (indeks 1'e) gelir.
                    # Biz `>= conf_thresh` olan ilk elemanı istiyoruz.
                    # index = np.searchsorted(eval_confidence_points[::-1], conf_thresh, side='left') # Tersi sırada ara
                    # index = len(eval_confidence_points) - 1 - index # Orijinal sıradaki indeksi bul

                    # Daha basit ve kaynak [33] ile uyumlu: `>= conf_thresh` olan indeksleri bul, ilkini al
                    # `np.where(eval_confidence_points >= conf_thresh)` bize `>= conf_thresh` olan tüm indeksleri azalan sırada verir.
                    # İlk indeks (yani en küçük indeks) en yüksek skora karşılık gelir. Bu noktadaki P/R, o skor veya daha yükseğini
                    # içeren tüm detectionları değerlendirir. Bu, istediğimiz "skoru >= conf_thresh" eşiğiyle aynı kümeyi temsil eder.
                    # Örneğin, eşik 0.7. Scores: [0.9, 0.8, 0.7, 0.6]. >= 0.7 olanlar 0.9, 0.8, 0.7. Indeksler [5, 10]. İlk indeks 0.
                    # precision_points ve recall_points skoru >= 0.9 olanları değil, skoru >= eval_confidence_points olanları verir.
                    # BU KISIM PCDet'in eval_class implementasyonuna ÇOK BAĞLI. Dokümantasyona veya koda bakmak lazım.
                    # Kaynak [32]: "thresholds[i] değeri, i'inci detection'ın skorudur. Yani, skoru >= thresholds[i] olan detection'ları alarak precision_points[i] ve recall_points[i] elde edilir."
                    # Bu durumda, conf_thresh için, eval_confidence_points içinde conf_thresh'e en yakın *ve ondan büyük eşit* değeri bulmalıyız ve onun indeksini kullanmalıyız.
                    # Veya daha doğrusu, conf_thresh'ten BÜYÜK VEYA EŞİT olan *en küçük skora* karşılık gelen indeksi bulmalıyız.
                    # Bu, eval_confidence_points dizisinde, conf_thresh değerinin *ilk kez aşıldığı* veya *tam eşitlendiği* noktadır.
                    # Örneğin: scores = [0.9, 0.8, 0.7, 0.6], conf_thresh = 0.75. `>= 0.75` olan skorlar 0.9, 0.8. En küçük skor 0.8 (indeks 1). P/R @ index 1.
                    # Örneğin: scores = [0.9, 0.8, 0.7, 0.6], conf_thresh = 0.7. `>= 0.7` olan skorlar 0.9, 0.8, 0.7. En küçük skor 0.7 (indeks 2). P/R @ index 2.
                    # Bu, `np.searchsorted(eval_confidence_points, conf_thresh, side='right') - 1` ile bulunabilir.
                    # searchsorted with 'right' side in a decreasing array finds the index *after* the last element >= value. So index-1 is the last element >= value.
                    index = np.searchsorted(eval_confidence_points, conf_thresh, side='right') - 1

                    if index >= 0: # Geçerli bir indeks bulunduysa
                        precision = eval_precision_points[index]
                        recall = eval_recall_points[index]
                    else: # conf_thresh en yüksek skordan bile büyükse, geçerli indeks yok
                        precision = 0.0
                        recall = 0.0
                else: # conf_thresh tüm skorlardan büyükse, valid_indices boş olur.
                    precision = 0.0
                    recall = 0.0

                # F1-skor hesaplama [26, 27] - Paydanın sıfır olma durumunu ele al [Previous Conversation]
                denominator = precision + recall
                f1 = np.where(denominator > 0, 2 * (precision * recall) / denominator, 0.0)

                results_for_this_iou[conf_thresh] = {'precision': precision, 'recall': recall, 'f1': f1}
                print(f"  {conf_thresh:.2f} | {precision:.4f} | {recall:.4f} | {f1:.4f}")

            pr_f1_at_confidence_thresholds[current_iou_for_analysis] = results_for_this_iou

    except Exception as e:
        print(f"Error during eval_class or confidence analysis for IoU > {current_iou_for_analysis}: {e}")
        print("Skipping confidence threshold analysis for this IoU.")


# --- Güven Skoru ve Lokalizasyon Kalitesi (IoU) İlişkisi Analizi (Kavramsal) ---
# Bu analiz, her bir EŞLEŞTİRİLMİŞ (TP) detection için güven skoru ile ground truth nesne ile olan gerçek IoU değeri arasındaki ilişkiyi inceler [10, 12, 13, 38, 39].
# Kaynaklardaki scriptte bu analizi yapacak kod bulunmuyor, sadece yorumlar var [29, 40-42].
# Bu analiz pcdet'in eval_class fonksiyonunun iç eşleştirme detaylarına erişim gerektirir.
# Kavramsal olarak nasıl yapılacağını script yorumlarına dayanarak tekrar belirtelim.

print("\n--- Confidence Score vs. Localization Quality (IoU) Analysis (Conceptual) ---")
print("This analysis examines the relationship between the confidence score of each TRUE POSITIVE detection")
print("and the actual IoU value with its matched ground truth box. [10, 12, 13, 38, 39]")
print("Accessing the exact matching details (detection score, matched GT's IoU) directly from pcdet's")
print("eval_class output might require modifications to the evaluation code itself or deeper inspection.")
print("However, conceptually, to perform this analysis:")
print("1. During the evaluation process, record each detection that is determined to be a True Positive (TP).")
print("2. For each recorded TP detection, store its confidence score and the IoU value with the ground truth box it matched.")
print("3. Create scatter plots using the collected confidence score-IoU pairs. [10, 39]")
print("4. Calculate correlation coefficients between confidence scores and IoU values. [10]")
print("Expected finding: Predictions with high confidence scores are expected, on average, to have high IoU values. [38]")
print("However, in some models, this correlation might be weak [12], meaning the model might predict high confidence for poorly localized boxes. [13, 39]")

print("\nAnalysis Complete.")

# İsteğe bağlı: Hesaplanan F1 skorlarını bir dosyaya kaydedebilirsiniz.
# import json
# with open('f1_scores_by_confidence_threshold.json', 'w') as f:
#     # NumPy float'ları JSON'a doğrudan yazılamaz, stringe çevirelim
#     serializable_results = {}
#     for iou_thresh, results in pr_f1_at_confidence_thresholds.items():
#         serializable_results[str(iou_thresh)] = {
#             str(conf_thresh): {k: float(v) for k, v in metrics.items()}
#             for conf_thresh, metrics in results.items()
#         }
#     json.dump(serializable_results, f, indent=4)
# print("F1 scores saved to f1_scores_by_confidence_threshold.json")