import sys
import pickle
import numpy as np
import pandas as pd # Tablo çıktısı için pandas kullanacağız

# 3DTrans veya OpenPCDet ortamınıza göre yolu ayarlayın (önceki koddan alınmıştır) [11]
# Eğer /root/3DTrans/ sizin repo yolunuzsa ve değerlendirme modüllerini import etmek istiyorsanız bu satırı kullanabilirsiniz.
# Bu script sadece veriyi yükleyip işleyeceği için eval modüllerini doğrudan import etmek ZORUNLU değildir.
# Ancak, 3D IoU fonksiyonunu kullanmak isterseniz, ilgili kütüphaneyi yolunuza eklemeniz gerekebilir.
# sys.path.insert(0, '/root/3DTrans/')

# --- YER TUTUCU: 3D Kutular İçin IoU Hesaplama Fonksiyonu ---
# Bu fonksiyon, verilen kaynaklarda tam olarak sağlanmamıştır. [5, 6]
# Kendi 3D IoU (Intersection over Union) hesaplama fonksiyonunuzu buraya eklemeniz GEREKMEKTEDİR.
# Kutuların formatı genellikle [x, y, z, length, width, height, yaw] şeklindedir. [12, 13]
# Bu sadece bir ÖRNEKTİR ve gerçek 3D IoU hesaplamasını YAPMAZ.
def calculate_iou_3d_box(box1, box2):
    """
    YER TUTUCU FONKSİYONU: İki 3D kutu arasındaki IoU'yu hesaplar.
    Gerçek bir implementasyon, 3D kutu formatına (genellikle [x, y, z, l, w, h, yaw]) [12, 13]
    ve dönme bilgisine [12, 13] dikkat etmeli ve kompleks geometrik hesaplamalar [5] yapmalıdır.
    Lütfen burayı gerçek 3D IoU fonksiyonunuzla DEĞİŞTİRİN.
    Şimdilik sadece basitleştirilmiş bir değer döndürüyor (örneğin 0.5 gibi) veya sıfır döndürüyor.
    Eğer gerçek kutulara ihtiyacınız varsa, box1 ve box2 numpy dizileridir.
    """
    # Gerçek IoU hesaplama kodu buraya gelecek.
    # Örnek: Basit bir BEV (Kuş Bakışı) IoU başlangıç noktası olabilir, ancak 3D tam farklıdır.
    # Kaynaklar karmaşık 3D IoU'ya işaret ediyor [5].
    # Bu yer tutucu fonksiyonu, sadece bir dummy değer döndürecektir.
    # Gerçek kullanımda, burası doğru 3D IoU'yu hesaplamalıdır.
    # Geçerli bir IoU değeri döndürebilmek için kutu boyutlarını kullanabiliriz,
    # ama doğru çakışmayı hesaplamak dönme gerektirir.
    # Basitlik adına, kutular birbiriyle çakışıyormuş gibi rastgele veya sabit bir değer atamıyoruz.
    # Gerçekte burada geometrik bir hesaplama olmalı.
    # Placeholder olarak, kutuların merkezlerinin belirli bir mesafeden yakınsa ve boyutları benzerse
    # yüksek bir IoU döndüren *yanlış* bir mantık bile eklemeyelim, çünkü bu yanıltıcı olur.
    # Doğru yaklaşım, kütüphanedeki rotate_iou_gpu_eval benzeri bir fonksiyonu kullanmaktır.
    # return np.random.rand() # Yanlış: Rastgele IoU döndürmemeli
    # return 0.7 # Yanlış: Sabit IoU döndürmemeli
    # return 0.0 # En güvenli yer tutucu: Başlangıçta çakışma yok varsayalım

    # Eğer OpenPCDet veya 3DTrans'in CPU tabanlı bir 3D IoU fonksiyonu varsa onu import edip burada kullanın.
    # Örneğin: from pcdet.utils.box_utils import boxes3d_lidar_pairwise_iou
    # return boxes3d_lidar_pairwise_iou(np.array([box1]), np.array([box2]))

    # Yer tutucu olarak, gerçek bir hesaplama yapılmadığını belirtelim:
    # print("Uyarı: calculate_iou_3d_box yer tutucu fonksiyonu kullanılıyor. Gerçek IoU hesaplanmıyor.")
    # Gerçek IoU hesaplaması olmadan bu tablo ANLAMLI OLMAYACAKTIR.

    # --- GERÇEK IoU İMPLEMENTASYONU GEREKLİ ---
    # Örnek bir IoU hesaplama kütüphanesinin çağrısı:
    # try:
    #     from your_iou_library import calculate_iou_3d # Varsayımsal kütüphane
    #     return calculate_iou_3d(box1, box2)
    # except ImportError:
    #     print("Gerçek IoU kütüphanesi bulunamadı. Lütfen calculate_iou_3d_box fonksiyonunu güncelleyin.")
    #     return 0.0 # Hata durumunda 0 döndür

    # --- GEÇİCİ ÇÖZÜM: BASİT BEV IOUsunu YER TUTUCU OLARAK KULLANALIM (UYARI: 3D DEĞİL!) ---
    # Bu sadece kodun yapısını göstermek içindir, gerçek 3D IoU değildir.
    # 3D kutular [x, y, z, l, w, h, yaw] formatındaysa, BEV için [x, y, l, w, yaw] alınabilir.
    # Ancak, bu yine de dönmeli BEV IoU hesaplaması gerektirir.
    # Basit bir Axis-Aligned BEV (dönme olmayan) IoU bile doğru IoU'yu temsil etmeyecektir.
    # En iyisi, gerçek 3D IoU implementasyonunun buraya eklenmesi gerektiğini vurgulamak.
    return 0.0 # Gerçek hesaplama olmadan 0 döndürmek en dürüst yaklaşım.


# IoU eşiği: Bu, bir detection'ın bir ground truth ile eşleşmesi için gereken minimum IoU değeridir.
# KITTI 3D Car için genellikle 0.7 kullanılır. [14, 15]
# Diğer sınıflar veya metrikler için farklı olabilir. [15]
iou_matching_threshold = 0.7 # Örnek eşik

# Tahminleri ve ground truth verilerini yükleme (önceki koddan alınmıştır) [16]
# Dosya yollarının doğru olduğundan emin olun
dt_result_path = '/root/3DTrans/output/cfgs/MDF/KNW/customnw_pvrcnn_feat_3_uni3d/default/eval/epoch_2/val/default/result.pkl' # KENDİ result.pkl yolunuz
gt_info_path = '/root/3DTrans/data/custom_kitti2/custom_infos_test.pkl' # KENDİ infos.pkl yolunuz

try:
    with open(dt_result_path, 'rb') as f:
        dt_data = pickle.load(f)
    print(f"Successfully loaded detection data from {dt_result_path}")

    with open(gt_info_path, 'rb') as f:
        gt_data = pickle.load(f)
    print(f"Successfully loaded ground truth data from {gt_info_path}")

except FileNotFoundError as e:
    print(f"Error loading data files: {e}")
    print("Please ensure the result.pkl and infos.pkl paths are correct.")
    sys.exit(1) # Dosya bulunamazsa programdan çık


# Tahmin verilerini değerlendirme formatına dönüştürme (önceki koddan alınmıştır) [13]
dt_annos_raw = []
for dt in dt_data:
    # Tahminlerde güven skoru ('score') ve sınıf ('name') bilgilerinin olması beklenir. [13]
    # 'boxes_lidar' 3D sınırlayıcı kutudur. [13]
    dt_anno = {
        "bbox": dt.get("boxes_lidar", np.array([])), # Kutu verisi, yoksa boş array
        "score": dt.get("score", np.array([])), # Güven skoru [13]
        "name": dt.get("name", np.array([])), # Sınıf ismi [13]
        # Diğer alanları da ekleyebiliriz ama bu script için bbox, score, name yeterli
    }
    # Sadece kutu, skor ve isim içeren tahminleri al
    if dt_anno["bbox"].shape[0] > 0:
         dt_annos_raw.append(dt_anno)
    else:
        # Eğer frame'de hiç detection yoksa, boş bir anno ekleyebiliriz
        dt_annos_raw.append({
             "bbox": np.array([]).reshape(-1, 7), # 7 sütunlu boş array (x,y,z,l,w,h,yaw)
             "score": np.array([]),
             "name": np.array([])
         })


# Ground truth verilerini değerlendirme formatına dönüştürme (önceki koddan alınmıştır) [12, 16, 17]
gt_annos = []
for gt_info in gt_data:
     # 'annos' anahtarının varlığını kontrol et
    if "annos" in gt_info and gt_info["annos"] is not None:
        annos = gt_info["annos"]
        anno = {
            # 'gt_boxes_lidar' 3D kutulardır. [12]
            "bbox": annos.get("gt_boxes_lidar", np.array([])),
            "name": annos.get("name", np.array([])), # Sınıf isimleri [12]
            # Diğer alanlar bu script için gerekli değil, ama önceki kodda eklenmişti:
            # "dimensions": annos.get("dimensions"),
            # "location": annos.get("location"),
            # "rotation_y": annos.get("rotation_y"),
            # "alpha": annos.get("alpha", np.asarray([0.0] * len(annos.get("gt_boxes_lidar", [])), dtype=np.float32)), # Örnek koddaki gibi [12]
            # "occluded": annos.get("occluded", np.asarray( * len(annos.get("gt_boxes_lidar", [])), dtype=np.int32)), # Örnek koddaki gibi [12]
            # "truncated": annos.get("truncated", np.asarray([0.0] * len(annos.get("gt_boxes_lidar", [])), dtype=np.float32)), # Örnek koddaki gibi [12]
            # "difficulty": annos.get("difficulty", np.asarray( * len(annos.get("gt_boxes_lidar", [])), dtype=np.int32)), # Örnek koddaki gibi [17]
        }
        # Sadece kutu ve isim içeren GT'leri al
        if anno["bbox"].shape[0] > 0:
            gt_annos.append(anno)
        else:
             # Eğer frame'de hiç GT yoksa, boş bir anno ekleyebiliriz
            gt_annos.append({
                 "bbox": np.array([]).reshape(-1, 7), # 7 sütunlu boş array (x,y,z,l,w,h,yaw)
                 "name": np.array([])
             })
    else:
         # Eğer infos dosyasında 'annos' yoksa (nadiren olur ama kontrol etmek iyi)
        gt_annos.append({
            "bbox": np.array([]).reshape(-1, 7),
            "name": np.array([])
        })


# Her bir ground truth için sonuçları saklayacak liste
results_list = []
gt_global_id_counter = 0 # Tüm veri seti boyunca GT'ler için benzersiz ID [Previous Conversation]

print("\nAnalyzing frames and calculating IoU/Confidence for each Ground Truth...")

# Her frame için GT ve DT verilerini işle
# gt_annos ve dt_annos_raw listeleri aynı sayıda öğe (frame) içermeli [18]
assert len(gt_annos) == len(dt_annos_raw), "GT ve DT annotasyon listeleri farklı sayıda frame içeriyor!"

for frame_idx in range(len(gt_annos)):
    gt_frame = gt_annos[frame_idx]
    dt_frame = dt_annos_raw[frame_idx]

    gt_boxes = gt_frame.get("bbox", np.array([]))
    dt_boxes = dt_frame.get("bbox", np.array([]))
    dt_scores = dt_frame.get("score", np.array([]))

    num_gt_in_frame = gt_boxes.shape
    num_dt_in_frame = dt_boxes.shape

    # Frame'deki her bir Ground Truth nesnesini döngüye al
    for gt_local_idx in range(num_gt_in_frame[0]):
        current_gt_box = gt_boxes[gt_local_idx]
        # current_gt_name = gt_frame["name"][gt_local_idx] # İsterseniz sınıf ismini de kullanabilirsiniz

        best_iou = 0.0
        best_dt_score = 0.0
        match_found = False

        # Current GT kutusu ile bu frame'deki tüm Detection kutuları arasındaki IoU'ları hesapla
        for dt_local_idx in range(num_dt_in_frame[0]):
            current_dt_box = dt_boxes[dt_local_idx]
            current_dt_score = dt_scores[dt_local_idx]

            # --- YER TUTUCU FONKSİYON ÇAĞRISI ---
            # Burası gerçek calculate_iou_3d_box implementasyonunuz olmalı
            iou = calculate_iou_3d_box(current_gt_box, current_dt_box)

            # En iyi eşleşmeyi (en yüksek IoU) bul
            if iou > best_iou:
                best_iou = iou
                best_dt_score = current_dt_score

        # En yüksek IoU belirli bir eşiğin üzerindeyse, bu GT'yi eşleşmiş kabul et [10, 18]
        if best_iou >= iou_matching_threshold:
            match_found = True
            # Tabloya eşleşen detection'ın IoU ve skorunu ekle
            results_list.append({
                'id': gt_global_id_counter,
                'iou': float(best_iou), # NumPy float'ı Python float'a çevir
                'confidence': float(best_dt_score) # NumPy float'ı Python float'a çevir
            })
        else:
            # Eşleşme bulunamadıysa (IoU eşiğin altında veya hiç detection yok)
            # Tabloya IoU ve confidence'ı 0 olarak ekle, istediğiniz gibi [User Request]
            results_list.append({
                'id': gt_global_id_counter,
                'iou': 0.0,
                'confidence': 0.0
            })

        # Global GT ID'yi artır
        gt_global_id_counter += 1

print(f"\nAnalysis complete. Processed {gt_global_id_counter} ground truth objects.")

# Sonuç listesini Pandas DataFrame'e dönüştür ve yazdır
if results_list:
    results_df = pd.DataFrame(results_list)
    print("\n--- Ground Truth Analysis Table ---")
    print(results_df)
    print("-----------------------------------")
    print("Note: The 'iou' values might be inaccurate as a placeholder function was used for 3D IoU calculation.")
    print(f"A detection was considered matched to a GT if its IoU was >= {iou_matching_threshold}")
    print("If no match above the threshold was found (or no detections), IoU and Confidence are 0.0")

else:
    print("\nNo ground truth objects found or processed to generate the table.")