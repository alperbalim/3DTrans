import sys
sys.path.insert(0, '/root/3DTrans/')
import pickle
import numpy as np
from pcdet.datasets.kitti.kitti_object_eval_python.eval import do_eval, eval_class, get_mAP_R40
# custom_utils.loss_analyze import IoULoss # IoULoss bu senaryoda kullanılmıyor, yoruma alındı

# Tahminleri ve ground truth verilerini yükleme
# Dosya yollarının doğru olduğundan emin olun
try:
    with open('/root/3DTrans/output/cfgs/MDF/KNW/customnw_pvrcnn_feat_3_uni3d/default/eval/epoch_2/val/default/result.pkl', 'rb') as f:
        predictions = pickle.load(f)
    print(f"Tahmin verileri yüklendi: {len(predictions)} tespit mevcut.")
except FileNotFoundError:
    print("Hata: Tahmin dosyası bulunamadı. Lütfen yolu kontrol edin.")
    exit()

try:
    with open('/root/3DTrans/data/custom_kitti2/custom_infos_test.pkl', 'rb') as f:
        ground_truth = pickle.load(f)
    print(f"Ground Truth verileri yüklendi: {len(ground_truth)} sahne mevcut.")
except FileNotFoundError:
    print("Hata: Ground Truth dosyası bulunamadı. Lütfen yolu kontrol edin.")
    exit()

# Ground Truth verilerini değerlendirme formatına dönüştürme
gt_annos = []
for gt_info in ground_truth:
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
for dt in predictions:
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

print("Veri hazırlığı tamamlandı.")

# --- Standart mAP Değerlendirmesi (Orijinal Kodun Bir Kısmı) ---
print("\n--- Standart mAP Değerlendirmesi (IoU Eşiklerine Göre) ---")

# KITTI stilinde mAP hesaplaması için IoU eşiklerini belirleme [3, 4]
# Bu eşikler IoU > eşik kriterini belirler.
# mAP_R40 için genellikle 40 geri çağırma noktası kullanılır. [3, 5]
# KITTI için yaygın IoU eşikleri: Araba için 0.7, Yaya/Bisiklet için 0.5'tir.
# Örnek kodunuz 0.05-0.95 aralığını kullanmış, bu geniş bir analiz veya farklı bir veri seti için olabilir.
# Standart KITTI Car mAP için 0.7 eşiği yaygın olarak kullanılır.
# Burada örnek kodunuzdaki gibi geniş bir aralık kullanmaya devam edelim.
ths_iou_for_map = np.linspace(0.05, 0.95, 19) # IoU eşikleri [3, 4]
print(ths_iou_for_map,ths_iou_for_map.shape)
# min_overlaps yapısı: [num_minoverlap, metric, num_class]
# Metrikler: 0: Bbox, 1: BEV, 2: 3D [3]
# Sınıflar:  tek bir sınıf için (muhtemelen Araba)
# Zorluklar:  tek bir zorluk seviyesi için (örn. Easy)
# Bu yapı, her bir IoU eşiği için tüm metrikler ve tüm (burada 1) sınıf için aynı eşik değerini atar.
min_overlaps_for_map = np.ones((ths_iou_for_map.shape[0], 3, 1))
for i in range(ths_iou_for_map.shape[0]):
    min_overlaps_for_map[i, :, :] = ths_iou_for_map[i] # Tüm metrikler için aynı IoU eşiği [3]

# do_eval(gt_annos, dt_annos, current_classes, min_overlaps, compute_aos=False, PR_detail_dict=None):
# mAP result: [num_class, num_diff, num_minoverlap]
# Sınıf 0 (örn. Araba) için, zorluk seviyesi 0 (örn. Easy) için değerlendirme
# current_classes = 
# difficultys =  # do_eval zorluk seviyelerini içsel olarak ele alır, min_overlaps yapısı zorluk seviyesini temsil etmez, num_diff sonucu etkiler.

# do_eval fonksiyonu genellikle farklı zorluk seviyeleri için sonuçlar döndürür.
# GT hazırlarken zorluk seviyelerini ekledik.
# Varsayılan olarak KITTI değerlendirmesi Easy, Moderate, Hard için yapılır.
# difficultys_to_evaluate = [6, 7] # 0: Easy, 1: Moderate, 2: Hard
# Ancak gt_annos'ta difficulty alanını 0 olarak ayarladık, bu yüzden sadece zorluk 0 geçerli olacaktır.
# do_eval, GT'deki difficulty alanını kullanarak zorluk seviyelerine ayırır.

first_iou_threshold = ths_iou_for_map[0]
last_iou_threshold = ths_iou_for_map[-1]

print(f"mAP hesaplanıyor (IoU Eşikleri: {first_iou_threshold:.2f} - {last_iou_threshold:.2f})...")

mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = do_eval(
    gt_annos, dt_annos_raw,[0] , min_overlaps_for_map, compute_aos=False)

# Sonuçları saklama ve yazdırma (örn. Easy zorluk seviyesi için, ilk IoU eşiğinde)
# Sonuç matrisleri [num_class, num_diff, num_minoverlap] şeklindedir.
# Biz sınıf 0, zorluk 0 (Easy) için sonuçlara bakalım.
# IoU eşiği olarak da ths_iou_for_map (yani 0.05) değerine bakalım.
# Not: Genellikle raporlanan mAP3d değeri Moderate veya Easy/Moderate/Hard ortalamasıdır,
# ve Araba için 0.7 IoU eşiği kullanılır. Burada sadece ilk eşikteki değeri alıyoruz.
print("mAPbbox_R40.shape",mAPbbox_R40.shape)
"""
result_map_standard = {
    'IoU_Threshold_for_Report': ths_iou_for_map, # Raporlanan IoU eşiği
    'Difficulty_Level_for_Report': 0, # Raporlanan zorluk seviyesi (Easy)
    'mAP_bbox_R40_Easy_at_IoU_0.05': mAPbbox_R40 if mAPbbox_R40.shape[0] > 0 and mAPbbox_R40.shape[6] > 0 and mAPbbox_R40.shape[7] > 0 else None,
    'mAP_bev_R40_Easy_at_IoU_0.05': mAPbev_R40 if mAPbev_R40.shape[0] > 0 and mAPbev_R40.shape[6] > 0 and mAPbev_R40.shape[7] > 0 else None,
    'mAP_3d_R40_Easy_at_IoU_0.05': mAP3d_R40 if mAP3d_R40.shape[0] > 0 and mAP3d_R40.shape[6] > 0 and mAP3d_R40.shape[7] > 0 else None,
}
"""
result_map_standard = {
    'IoU_Threshold_for_Report': ths_iou_for_map,  # Raporlanan IoU eşiği
    'Difficulty_Level_for_Report': 0,  # Raporlanan zorluk seviyesi (Easy)
    'mAP_bbox_R40_Easy_at_IoU_0.05': mAPbbox_R40[0, 0, 0] if mAPbbox_R40.shape[0] > 0 and mAPbbox_R40.shape[1] > 0 and mAPbbox_R40.shape[2] > 0 else None,
    'mAP_bev_R40_Easy_at_IoU_0.05': mAPbev_R40[0, 0, 0] if mAPbev_R40.shape[0] > 0 and mAPbev_R40.shape[1] > 0 and mAPbev_R40.shape[2] > 0 else None,
    'mAP_3d_R40_Easy_at_IoU_0.05': mAP3d_R40[0, 0, 0] if mAP3d_R40.shape[0] > 0 and mAP3d_R40.shape[1] > 0 and mAP3d_R40.shape[2] > 0 else None,
}

print("\nStandart mAP R40 Sonuçları (Sınıf 0 - Easy Zorluk - IoU Eşiği 0.05):")
print(result_map_standard)

# eval_class çağrısı (PR eğrisi detayları için) [8]
# eval_class(gt_annos, dt_annos, current_classes, difficultys, metric, min_overlaps, compute_aos=False, num_parts=100):
# Metrik 2: 3D
# Sınıf 0, Zorluk 0 (Easy) için, tüm ths_iou_for_map eşiklerinde 3D değerlendirmesi.
# Bu çağrı, esas olarak her bir IoU eşiği için bir dizi hassasiyet/geri çağırma noktası (PR eğrisi) döndürür.
# Bizim amacımız, bu PR eğrilerini kullanarak belirli GÜVEN SKORU eşiklerindeki P/R/F1 değerlerini bulmak.
print("\nPR Eğrisi Detayları Hesaplama (eval_class)...")
# Sadece 3D metriği ve sınıf 0, zorluk 0 (Easy) için eval_class'ı çağıralım.
# eval_class her bir min_overlap eşiği için PR eğrisini döndürür.
# Bizim amacımız, farklı GÜVEN skoru eşiklerinde P/R/F1 hesaplamak.
# Bu nedenle, eval_class'ı belirli BİR IoU eşiği için çağırıp, dönen PR noktalarını kullanalım.
# Örneğin, KITTI Araba için standart IoU eşiği olan 0.7'yi veya COCO için 0.5'i kullanalım. [3, 4, 9]
# Burada hem 0.5 hem de 0.7 IoU eşikleri için güven skoru analizi yapalım.

iou_thresholds_for_confidence_analysis = [0.5, 0.7] # Güven skoru analizini yapacağımız IoU eşikleri [1]
confidence_thresholds_to_analyze = np.arange(0.0, 1.05, 0.05) # Analiz edeceğimiz güven skoru eşikleri [1] (örn. 0.0'dan 1.0'a 0.05 adımla)

pr_f1_at_confidence_thresholds = {} # Sonuçları saklayacak sözlük

print(f"\n--- Güven Skoru Eşiği Analizi (IoU Eşikleri: {iou_thresholds_for_confidence_analysis}) ---")

# dt_annos_raw zaten güven skorlarını içeriyor.
# eval_class'ı farklı IoU eşiklerinde çağırarak PR eğrilerini alacağız.

for current_iou_for_analysis in iou_thresholds_for_confidence_analysis:
    print(f"\nIoU Eşiği > {current_iou_for_analysis} için analiz yapılıyor...")

    # eval_class için tek bir IoU eşiği içeren min_overlaps yapısı oluşturma
    # Yapı: [num_minoverlap=1, metric, num_class=1]
    # Metrik 2: 3D Box
    min_overlaps_single_iou = np.ones((1, 3, 1), dtype=np.float32)
    min_overlaps_single_iou[:, 2, 0] = current_iou_for_analysis # 3D metriği için bu IoU eşiği

    # eval_class çağrısı. Bu, detectionları güven skoruna göre sıralar ve farklı recall noktalarındaki P/R değerlerini hesaplar. [8]
    # Sınıf 0 (Araba), Zorluk 0 (Easy) için değerlendirme yapılıyor.
    res_eval_class = eval_class(
        gt_annos, dt_annos_raw, [0], [0], 2, min_overlaps_single_iou, compute_aos=False
    )

    # eval_class sonuçlarından hassasiyet, geri çağırma ve bunlara karşılık gelen güven eşiklerini alalım.
    # res_eval_class['precision'] -> [num_class, num_difficulty, num_minoverlap, num_recall_points]
    # res_eval_class['recall']    -> [num_class, num_difficulty, num_minoverlap, num_recall_points]
    # res_eval_class['thresholds'] -> [num_class, num_difficulty, num_minoverlap, num_recall_points] (Güven skorları)

    # Sadece Sınıf 0, Zorluk 0 (Easy), Tek IoU eşiği (index 0) için verilere bakalım
    precision_points = res_eval_class['precision']
    recall_points = res_eval_class['recall']
    confidence_points = res_eval_class['thresholds']

    # Bu noktalar, güven skorları azalan şekilde sıralanmış detectionlara karşılık gelir.
    # Yani confidence_points[k] güven skoruna sahip detection(lar) ve ondan yüksek skorluların tümü dikkate alındığında
    # elde edilen geri çağırma recall_points[k] ve hassasiyet precision_points[k]'dır.

    # Belirlediğimiz güven skoru eşikleri için P/R/F1 değerlerini bulalım.
    # Her bir conf_threshold için, confidence_points içinde o eşiğe en yakın veya ondan büyük
    # en düşük güven skorunu bulup, ilgili P/R değerlerini alacağız.
    # NOT: eval_class'ın 'thresholds' dizisi aslında hassasiyet/geri çağırma noktalarına karşılık gelen confidence değerleridir.
    # Azalan sırada olmalıdır.
    # Bizim conf_thresholds değerlerimiz artan sırada.
    # Eşik T için, skoru >= T olan detectionları alırız. eval_class'ın döndürdüğü noktalarda,
    # confidence_points[k] değeri, k'inci detection'ın skoruna (azalan sırada) veya hassasiyet/geri çağırmanın değiştiği bir noktaya karşılık gelir.
    # En doğru yaklaşım, her conf_threshold T için, dt_annos_raw listesini T'ye göre filtrelemek ve sonra o filtrelenmiş liste için P/R/F1'i manuel hesaplamaktır.
    # Ancak bu, pcdet'in iç mekanizmasını yeniden uygulamayı gerektirir.
    # eval_class çıktısındaki 'thresholds' array'i, o recall/precision noktasına ulaşmak için gereken en düşük güven skorunu temsil eder.
    # Dolayısıyla, belirli bir güven skoru eşiği için, bu eşiğe karşılık gelen veya yakın olan recall/precision noktasını bulabiliriz.

    pr_f1_at_confidence_thresholds[current_iou_for_analysis] = {}

    for conf_threshold in confidence_thresholds_to_analyze:
        # Güven skoru eşiğine en yakın veya ondan yüksek olan confidence_points indeksini bulalım
        # confidence_points azalan sırada olmalı. Bizim eşiklerimiz artan sırada.
        # Örneğin, eşik 0.8 için, confidence_points içinde >= 0.8 olan en büyük indeksi (veya o noktayı) bulmalıyız.
        # np.searchsorted (azalan dizi için), veya np.where kullanabiliriz.
        # Veya en basit yol: confidence_points'teki değerler içinde eşiğimize en yakın olanı bulmak.
        # Veya: Eşiğimizden BÜYÜK VEYA EŞİT olan confidence_points değerlerinin İLKİni bulmak (çünkü azalan sırada).
        # Eğer böyle bir nokta yoksa (örn. tüm skorlar eşiğin altında), o zaman TP=0, FP=0, FN=tüm GT.
        # conf_thresholds artan sırada, confidence_points azalan sırada.

        # Eşik T için: detection_score >= T olanları al.
        # eval_class'ın noktaları detection'lar sıralandıktan sonra elde edilir.
        # confidence_points[k] değeri k'inci detection'ın (azalan sırada) skorudur.
        # Eğer conf_threshold = confidence_points[k] ise, bu k'inci detection'ı dahil ederek elde edilen P/R değerleridir.

        # Belirli bir güven skoru eşiği C için P/R/F1 hesaplamak için:
        # 1. Skoru >= C olan tüm detection'ları seç.
        # 2. Bu detection'ları GT ile belirli bir IoU eşiği I kullanarak eşleştir.
        # 3. TP, FP, FN sayılarını bul.
        # 4. P = TP / (TP+FP), R = TP / (TP+FN), F1 = 2PR / (P+R) hesapla.

        # pcdet eval kodunu kullanıyorsak, eval_class'ın içinde zaten bu hesaplama yapılıyor.
        # res_eval_class['thresholds'] array'i tam olarak bu "skoru >= eşik" mantığıyla elde edilen noktalara karşılık gelir.
        # Dolayısıyla, yapmamız gereken:
        # 1. res_eval_class['thresholds'] içinde belirlediğimiz conf_threshold değerine en yakın noktayı (veya <= conf_threshold olan ilk noktayı?) bulmak.
        # thresholds azalan sırada olduğuna göre, conf_threshold'dan >= olan *en düşük* skora sahip detection'ı bulmak için
        # thresholds array'i üzerinde arama yapabiliriz.
        # Örneğin, conf_threshold = 0.6. thresholds = [0.9, 0.8, 0.7, 0.5, ...]
        # 0.7 noktası, skoru >= 0.7 olan tüm detection'ları içerir.
        # Biz skoru >= 0.6 olanları istiyoruz. Bunun için confidence_points array'inde 0.6'dan >= olan en düşük skoru (yani 0.7'yi) bulmalıyız.
        # threshold array'indeki değerler azalan sırada olduğu için, conf_threshold değerinden BÜYÜK VEYA EŞİT olan ilk indeksi bulmak doğru olmaz.
        # Tam tersine, conf_threshold'dan KÜÇÜK olan ilk indeksi bulup, bir önceki indeksi almalıyız.
        # threshold array'inde conf_threshold değerinin kendisi veya ona çok yakın bir değer olmayabilir.
        # En güvenli yol, her conf_threshold için filtrelenmiş listeyi geçici olarak oluşturup P/R/F1 hesaplamaktır.
        # Ancak bu pcdet kütüphanesinin içindeki eşleştirme mantığını kopyalamayı gerektirir ki bu karmaşık.

        # Basit yaklaşım: confidence_points array'inde conf_threshold'a en yakın indexi bulmak.
        # Bu tam olarak "skoru >= eşik" durumunu yansıtmayabilir ama bir yaklaşımdır.
        # np.abs(confidence_points - conf_threshold).argmin() her zaman bir index döndürür.
        # Veya daha doğru: thresholds azalan sırada. Conf_threshold T için, skoru >= T olanları alıyoruz.
        # Bu, thresholds array'inde T'den büyük veya eşit olan tüm skorlara karşılık gelir.
        # thresholds[k] >= T koşulunu sağlayan en BÜYÜK k'yı bulmalıyız.

        valid_indices = np.where(confidence_points >= conf_threshold)[0]

        if len(valid_indices) > 0:
            # thresholds azalan sırada. Skoru >= conf_threshold olanlar valid_indices içinde.
            # Bizim istediğimiz P/R/F1, tam olarak conf_threshold eşiği kullanıldığında elde edilen değerdir.
            # eval_class'ın PR noktaları, detection'lar azalan skora göre sıralandıktan sonraki kümülatif TP/FP sayılarından gelir.
            # thresholds[i] değeri, i'inci detection'ın skorudur.
            # Yani, skoru >= thresholds[i] olan detection'ları alarak precision_points[i] ve recall_points[i] elde edilir.
            # Bizim istediğimiz, skoru >= conf_threshold olan detectionları almaktır.
            # Dolayısıyla, thresholds array'inde conf_threshold'dan >= olan tüm indekslerin en küçüğünü bulmalıyız.
            # valid_indices = np.where(confidence_points >= conf_threshold)
            # Eğer thresholds array'inde tam olarak conf_threshold yoksa, buna en yakın ve >= olanı bulalım.
            # Veya basitçe, valid_indices'deki en küçük indeksi (ilk detection) kullanalım.

            closest_index = valid_indices if len(valid_indices) > 0 else -1 # conf_threshold'dan >= olan ilk detection'ın indeksi

            if closest_index.all() != -1:
                p = precision_points[0,0,0,closest_index]
                r = recall_points[0,0,0,closest_index]
                # F1 = 2 * (P * R) / (P + R) [1]
                
                # F1-skoru formülünün paydası: p + r
                denominator = p + r

                # Paydanın sıfırdan büyük olduğu (veya sıfır olmadığı) koşulu eleman-bazlı kontrol et
                # Bu, True/False değerlerinden oluşan bir boolean dizi döndürür.
                condition = denominator > 0

                # np.where fonksiyonunu kullanarak F1-skorunu eleman-bazlı hesapla
                # np.where(koşul_dizisi, koşul_True_iken_yapılacak_işlem, koşul_False_iken_yapılacak_işlem)
                # Koşul doğruysa (payda > 0), F1 formülünü uygula: 2 * (p * r) / (p + r)
                # Koşul yanlışsa (payda <= 0, genellikle 0), sonucu 0.0 yap
                f1 = np.where(condition, 2 * (p * r) / denominator, 0.0)
                
                # f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

                pr_f1_at_confidence_thresholds[current_iou_for_analysis][conf_threshold] = {
                    'Precision': p,
                    'Recall': r,
                    'F1-Score': f1
                }
            else:
                # Hiçbir detection bu güven skoru eşiğini sağlamıyorsa (veya daha yükseklerini), P=0, R=0, F1=0
                 pr_f1_at_confidence_thresholds[current_iou_for_analysis][conf_threshold] = {
                     'Precision': 0.0,
                     'Recall': 0.0,
                     'F1-Score': 0.0
                 }
        else:
            # Hiçbir detection bu güven skoru eşiğini sağlamıyorsa, P=0, R=0, F1=0
             pr_f1_at_confidence_thresholds[current_iou_for_analysis][conf_threshold] = {
                 'Precision': 0.0,
                 'Recall': 0.0,
                 'F1-Score': 0.0
             }


# Sonuçları yazdırma
print("\nGüven Skoru Eşiklerine Göre P/R/F1 Sonuçları:")
for iou_thresh, results in pr_f1_at_confidence_thresholds.items():
    print(f"\n--- IoU Eşiği > {iou_thresh:.2f} ---")
    print("Güven Eşiği | Precision |   Recall  | F1-Score")
    print("-------------------------------------------------")
    # Güven eşiklerini artan sırada sıralayarak yazdıralım
    for conf_thresh in sorted(results.keys()):
         metrics = results[conf_thresh]
         precision = metrics['Precision']
         print(precision)
         print(precision.shape)
         recall = metrics['Recall']
         f1_score = metrics['F1-Score']
         print(f"    {conf_thresh:.2f}   | {precision:.4f}  | {recall:.4f}  | {f1_score:.4f}")


# --- Güven Skoru ve IoU İlişkisi Analizi (Ek Analiz İçin) ---
# Bu kısım, her bir EŞLEŞTİRİLMİŞ (TP) detection için güven skoru ile IoU arasındaki ilişkiyi analiz eder.
# Bu, eval_class'ın iç çıktılarından TP'leri ve eşleşen GT'leri alarak yapılabilir.
# pcdet kütüphanesinin iç mekanizmasına daha derinlemesine inmek gerekir.
# eval_class çalıştırıldığında, hangi detection'ların hangi GT'lerle eşleştiği ve hangi IoU değerleriyle eşleştiği bilgisi hesaplanır.
# Bu bilgilere doğrudan erişimimiz olmasa da, kavramsal olarak bu analizin nasıl yapılacağını açıklayalım.

print("\n--- Güven Skoru ve Lokalizasyon Kalitesi (IoU) İlişkisi (Kavramsal Analiz) ---")
print("Bu analiz, her bir DOĞRU TESPİT (True Positive) için, tahminin güven skoru ile")
print("gerçek doğruluk kutusuyla olan IoU değeri arasındaki ilişkiyi inceler. [2, 10, 11]")
print("pcdet'in eval_class çıktılarından doğrudan bu eşleştirme detaylarına erişmek")
print("kod üzerinde ek geliştirmeler gerektirir.")
print("Ancak, bu analizi yapmak için:")
print("1. Değerlendirme sürecinde True Positive olarak belirlenen her bir tahmini (detection) kaydedin.")
print("2. Kaydederken, tahminin güven skorunu ve eşleştiği gerçek doğruluk kutusuyla olan IoU değerini not alın.")
print("3. Toplanan güven skoru-IoU çiftlerini kullanarak dağılım grafikleri (scatter plot) çizin. [2]")
print("4. Güven skoru ile IoU arasındaki korelasyonu hesaplayın. [2]")
print("Beklenti: Yüksek güven skoruna sahip tahminlerin, ortalama olarak, yüksek IoU değerlerine sahip olmasıdır. [10]")
print("Ancak, bazı modellerde bu korelasyon zayıf olabilir [11], yani model yüksek güvenle yanlış lokalizasyonlar yapabilir.")

# Örnek kodun sakladığı orijinal mAP sonuçları (ilk IoU eşiği ve Easy zorluk için)
# result_map_standard sözlüğünde zaten saklandı ve yazdırıldı.