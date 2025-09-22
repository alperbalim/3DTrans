import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import copy
import datetime
from pathlib import Path

sys.path.insert(0, '/root/3DTrans/') # Kullanıcı tarafından sağlandı

try:
    from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval
    from pcdet.datasets.kitti.kitti_utils import transform_annotations_to_kitti_format
    from pcdet.ops.iou3d_nms import iou3d_nms_utils
    from pcdet.utils import common_utils
except ImportError as e:
    print(f"Hata: OpenPCDet/3DTrans kütüphane dosyaları bulunamadı veya içe aktarılamadı. Detay: {e}")
    exit(1)

### SABİT PARAMETRELER ###
DT_RESULTS_PKL_PATH = '/root/3DTrans/output/cfgs/MDF/KNW/customnw_pvrcnn_feat_3_uni3d/default/eval/epoch_2/val/default/result.pkl'
GT_INFOS_PKL_PATH = '/root/3DTrans/data/custom_kitti2/custom_infos_test.pkl'
OUTPUT_DIR_PATH = 'analiz_sonuclari/esik_deger_analizi_dt_gt_v5_debug'

CLASS_MAPPING_TO_KITTI = {'Car':'Car', 'Pedestrian':'Pedestrian', 'Cyclist':'Cyclist'}
CURRENT_CLASS_NAME_FOR_ANALYSIS = 'Car'
KITTI_CLASS_NAMES_ORDERED = ['Car', 'Pedestrian', 'Cyclist']
EVALUATION_IOU_THRESHOLD=0.7
try:
    CURRENT_CLASS_ID_FOR_ANALYSIS = KITTI_CLASS_NAMES_ORDERED.index(CURRENT_CLASS_NAME_FOR_ANALYSIS)
except ValueError:
    print(f"HATA: '{CURRENT_CLASS_NAME_FOR_ANALYSIS}' sınıfı KITTI_CLASS_NAMES_ORDERED listesinde bulunamadı.")
    exit(1)

CONFIDENCE_THRESHOLDS_STR = '0.1,0.3' # Debug için azaltıldı
NMS_IOU_THRESHOLDS_STR = '0.5,0.7'     # Debug için azaltıldı
TARGET_AP_IOU_THRESHOLD = 0.7

NMS_PRE_MAXSIZE_PER_CLASS = 4096
NMS_POST_MAXSIZE_PER_CLASS = 500
NMS_TYPE_FOR_SINGLE_CLASS = 'nms_gpu'
##########################

logger = None

# Ana mantık main() fonksiyonu içine taşınacak.
# load_and_prepare_gt_annos ve load_and_prepare_dt_annos fonksiyonları kalabilir,
# çünkü bunlar veri yükleme ve ilk formatlama için nispeten standart.

def load_and_prepare_gt_annos(gt_infos_path, class_mapping, logger_func):
    logger_func.info(f"Referans (GT) verisi yükleniyor: {gt_infos_path}")
    if not Path(gt_infos_path).exists():
        logger_func.error(f"GT info dosyası bulunamadı: {gt_infos_path}")
        raise FileNotFoundError(f"GT info dosyası bulunamadı: {gt_infos_path}")
    with open(gt_infos_path, 'rb') as f:
        gt_data_list = pickle.load(f)
    
    gt_annos_list = []
    for idx, gt_info_dict in enumerate(gt_data_list):
        annos = gt_info_dict.get('annos', {})
        required_gt_keys = ["gt_boxes_lidar", "name", "dimensions", "location", "rotation_y"]
        # GT kutuları boş olsa bile örneği ekle, böylece DT ile sayı tutar
        if not all(key in annos for key in required_gt_keys):
            logger_func.warning(f"GT infos dosyasındaki {idx}. örnekte eksik 'annos' anahtarları. Boş anno oluşturuluyor.")
            # Create an empty annotation structure consistent with KITTI format
            empty_anno = {
                "bbox": np.array([]).reshape(0, 4).astype(np.float32), 
                "alpha": np.array([]).astype(np.float32),
                "dimensions": np.array([]).reshape(0, 3).astype(np.float32), 
                "location": np.array([]).reshape(0, 3).astype(np.float32),
                "rotation_y": np.array([]).astype(np.float32), 
                "name": np.array([], dtype='<U10'), # Empty string array
                "gt_boxes_lidar": np.array([]).reshape(0, 7).astype(np.float32),
                "difficulty": np.array([], dtype=np.int32), 
                "truncated": np.array([], dtype=np.float32),
                "occluded": np.array([], dtype=np.int32),
            }
            gt_annos_list.append(empty_anno)
            continue

        num_gt_boxes = 0
        if "gt_boxes_lidar" in annos and isinstance(annos["gt_boxes_lidar"], np.ndarray):
            num_gt_boxes = annos["gt_boxes_lidar"].shape[0]

        anno = {
            "bbox": annos.get("bbox", np.zeros((num_gt_boxes, 4), dtype=np.float32)),
            "gt_boxes_lidar": annos.get("gt_boxes_lidar", np.empty((0,7), dtype=np.float32)),
            "name": annos.get("name", np.array([], dtype='<U10')),
            "dimensions": annos.get("dimensions", np.empty((num_gt_boxes,3), dtype=np.float32)),
            "location": annos.get("location", np.empty((num_gt_boxes,3), dtype=np.float32)),
            "rotation_y": annos.get("rotation_y", np.empty(num_gt_boxes, dtype=np.float32)),
            "alpha": annos.get("alpha", np.zeros(num_gt_boxes, dtype=np.float32)),
            "occluded": annos.get("occluded", np.zeros(num_gt_boxes, dtype=np.int32)),
            "truncated": annos.get("truncated", np.zeros(num_gt_boxes, dtype=np.float32)),
            "difficulty": annos.get("difficulty", np.zeros(num_gt_boxes, dtype=np.int32)),
        }
        gt_annos_list.append(anno)
    
    logger_func.info(f"{len(gt_annos_list)} GT örneği yüklendi ve ham formatta hazırlandı.")
    return transform_annotations_to_kitti_format(gt_annos_list, map_name_to_kitti=class_mapping)

def load_and_prepare_dt_annos(dt_results_path, class_mapping, logger_func):
    logger_func.info(f"Tahmin (DT) verisi yükleniyor: {dt_results_path}")
    if not Path(dt_results_path).exists():
        logger_func.error(f"DT result dosyası bulunamadı: {dt_results_path}")
        raise FileNotFoundError(f"DT result dosyası bulunamadı: {dt_results_path}")
    with open(dt_results_path, 'rb') as f:
        dt_data_list = pickle.load(f)

    if isinstance(dt_data_list, dict) and 'model_state' in dt_data_list:
        logger_func.error(f"HATA: {dt_results_path} dosyası model checkpoint'i içeriyor, tahmin (detection) listesi değil!")
        raise ValueError("Yanlış DT .pkl dosyası formatı: Model checkpoint'i algılandı.")
    if not isinstance(dt_data_list, list):
        logger_func.error(f"HATA: {dt_results_path} dosyası beklenen liste formatında değil. İçerik tipi: {type(dt_data_list)}")
        raise ValueError("Yanlış DT .pkl dosyası formatı: Liste bekleniyordu.")
    
    if len(dt_data_list) > 0:
        logger_func.info(f"DT .pkl ilk eleman tipi: {type(dt_data_list[0])}")
        if isinstance(dt_data_list[0], dict):
            logger_func.info(f"DT .pkl ilk eleman anahtarları: {list(dt_data_list[0].keys())}")
            # İlk elemanın içeriğini daha detaylı logla (debug için)
            # for key, value in dt_data_list[0].items():
            #     if isinstance(value, np.ndarray):
            #         logger_func.debug(f"  Anahtar: {key}, Şekil: {value.shape}, Dtype: {value.dtype}, İlk 5: {value[:5]}")
            #     else:
            #         logger_func.debug(f"  Anahtar: {key}, Değer: {str(value)[:100]}")


    dt_annos_raw_list = []
    for idx, dt_sample in enumerate(dt_data_list):
        if not isinstance(dt_sample, dict):
            logger_func.warning(f"DT .pkl dosyasındaki {idx}. eleman bir sözlük değil (tip: {type(dt_sample)}). Boş anno oluşturuluyor.")
            dt_annos_raw_list.append({
                "bbox": np.array([]).reshape(0, 4), "alpha": np.array([]),
                "dimensions": np.array([]).reshape(0, 3), "location": np.array([]).reshape(0, 3),
                "rotation_y": np.array([]), "name": np.array([]), "score": np.array([]),
                "boxes_lidar": np.array([]).reshape(0, 7),
                "occluded": np.array([], dtype=np.int32), "truncated": np.array([])
            })
            continue
        
        frame_id_log = dt_sample.get('frame_id', dt_sample.get('metadata', {}).get('token', f"idx_{idx}"))
        boxes_lidar = dt_sample.get("boxes_lidar", dt_sample.get("pred_boxes"))
        score = dt_sample.get("score", dt_sample.get("pred_scores"))
        name_arr = dt_sample.get("name")
        if name_arr is None:
            pred_labels_arr = dt_sample.get("pred_labels")
            if pred_labels_arr is not None and isinstance(pred_labels_arr, np.ndarray) : # ndarray kontrolü eklendi
                try:
                    name_list = []
                    valid_pred_labels = []
                    for label_val in pred_labels_arr:
                        class_idx = int(label_val) -1 # 1-indexed to 0-indexed
                        if 0 <= class_idx < len(KITTI_CLASS_NAMES_ORDERED):
                            name_list.append(KITTI_CLASS_NAMES_ORDERED[class_idx])
                            valid_pred_labels.append(label_val)
                        else:
                            logger_func.warning(f"DT örneği {frame_id_log}: 'pred_labels' içinde geçersiz etiket değeri: {label_val}. Atlanıyor.")
                    name_arr = np.array(name_list, dtype='<U10') # KITTI formatına uygun dtype
                    # Eğer bazı etiketler geçersizse, eşleşen kutu ve skorları da filtrelemek gerekir.
                    # Bu kısım karmaşıklaşabilir. Şimdilik, eğer `name_arr` boş kalırsa, bu örneği atlayalım.
                    if len(name_arr) == 0 and len(pred_labels_arr) > 0 : # Hiç geçerli isim dönüştürülemediyse
                        logger_func.warning(f"DT örneği {frame_id_log}: Geçerli 'pred_labels' dönüştürülemedi. Atlanıyor.")
                        continue
                    elif len(name_arr) != len(pred_labels_arr) : # Bazı etiketler atıldıysa, kutu/skorları da güncelle
                        logger_func.warning(f"DT örneği {frame_id_log}: Bazı 'pred_labels' atıldı. Kutu/skorlar güncellenmeli.")
                        # Bu durum için daha detaylı filtreleme gerekir. Şimdilik basit tutalım.
                        # Eğer böyle bir durumla sık karşılaşıyorsanız, .pkl dosyasının 'name' içermesi daha iyi olur.


                except (IndexError, ValueError, TypeError) as e_label:
                    logger_func.warning(f"DT örneği {frame_id_log}: 'pred_labels' dönüştürülemedi. Etiketler: {pred_labels_arr}. Hata: {e_label}. Atlanıyor.")
                    continue
            else:
                logger_func.warning(f"DT örneği {frame_id_log}: 'name' veya 'pred_labels' anahtarı yok. Boş anno oluşturuluyor.")
                dt_annos_raw_list.append({ "bbox": np.array([]).reshape(0, 4), "alpha": np.array([]), "dimensions": np.array([]).reshape(0, 3), "location": np.array([]).reshape(0, 3), "rotation_y": np.array([]), "name": np.array([]), "score": np.array([]), "boxes_lidar": np.array([]).reshape(0, 7), "occluded": np.array([], dtype=np.int32), "truncated": np.array([])})
                continue
        
        location = dt_sample.get("location")
        dimensions = dt_sample.get("dimensions")
        rotation_y = dt_sample.get("rotation_y")

        if (location is None or dimensions is None or rotation_y is None) and \
           isinstance(boxes_lidar, np.ndarray) and boxes_lidar.ndim == 2 and boxes_lidar.shape[0] > 0 and boxes_lidar.shape[1] >= 7:
            location = boxes_lidar[:, 0:3]
            dimensions = boxes_lidar[:, 3:6]
            rotation_y = boxes_lidar[:, 6]
        elif (location is None or dimensions is None or rotation_y is None) and not (isinstance(boxes_lidar, np.ndarray) and boxes_lidar.shape[0] == 0): # Kutular varsa ama diğerleri yoksa
            logger_func.warning(f"DT örneği {frame_id_log}: 'location', 'dimensions' veya 'rotation_y' eksik ve boxes_lidar'dan çıkarılamıyor. Boş anno oluşturuluyor.")
            dt_annos_raw_list.append({ "bbox": np.array([]).reshape(0, 4), "alpha": np.array([]), "dimensions": np.array([]).reshape(0, 3), "location": np.array([]).reshape(0, 3), "rotation_y": np.array([]), "name": np.array([]), "score": np.array([]), "boxes_lidar": np.array([]).reshape(0, 7), "occluded": np.array([], dtype=np.int32), "truncated": np.array([])})
            continue


        if boxes_lidar is None or score is None or name_arr is None:
            logger_func.warning(f"DT örneği {frame_id_log}: 'boxes_lidar', 'score' veya 'name' temel anahtarlarından biri None. Boş anno oluşturuluyor.")
            dt_annos_raw_list.append({ "bbox": np.array([]).reshape(0, 4), "alpha": np.array([]), "dimensions": np.array([]).reshape(0, 3), "location": np.array([]).reshape(0, 3), "rotation_y": np.array([]), "name": np.array([]), "score": np.array([]), "boxes_lidar": np.array([]).reshape(0, 7), "occluded": np.array([], dtype=np.int32), "truncated": np.array([])})
            continue
            
        num_dt_boxes = boxes_lidar.shape[0] if isinstance(boxes_lidar, np.ndarray) else 0
        num_scores = score.shape[0] if isinstance(score, np.ndarray) else 0
        num_names = name_arr.shape[0] if isinstance(name_arr, np.ndarray) else 0

        if num_dt_boxes == 0:
            if num_scores == 0 and num_names == 0 :
                 dt_annos_raw_list.append({
                    "bbox": np.array([]).reshape(0, 4), "alpha": np.array([]), "dimensions": np.array([]).reshape(0, 3), 
                    "location": np.array([]).reshape(0, 3), "rotation_y": np.array([]), "name": np.array([]), 
                    "score": np.array([]), "boxes_lidar": np.array([]).reshape(0, 7),
                    "occluded": np.array([], dtype=np.int32), "truncated": np.array([])
                })
            else: logger_func.warning(f"DT örneği {frame_id_log}: Kutu sayısı 0 ama skor/isim var. Uyumsuzluk. Boş anno oluşturuluyor.")
            continue
        
        if not (num_dt_boxes == num_scores == num_names):
            logger_func.warning(f"DT örneği {frame_id_log}: Kutu ({num_dt_boxes}), skor ({num_scores}) veya isim ({num_names}) sayıları uyumsuz. Boş anno oluşturuluyor.")
            dt_annos_raw_list.append({ "bbox": np.array([]).reshape(0, 4), "alpha": np.array([]), "dimensions": np.array([]).reshape(0, 3), "location": np.array([]).reshape(0, 3), "rotation_y": np.array([]), "name": np.array([]), "score": np.array([]), "boxes_lidar": np.array([]).reshape(0, 7), "occluded": np.array([], dtype=np.int32), "truncated": np.array([])})
            continue

        dt_anno = {
            "bbox": dt_sample.get("bbox", np.zeros((num_dt_boxes, 4), dtype=np.float32)),
            "boxes_lidar": boxes_lidar, "score": score, "name": name_arr,
            "alpha" : dt_sample.get("alpha", np.zeros(num_dt_boxes, dtype=np.float32)),
            "occluded" : dt_sample.get("occluded", np.zeros(num_dt_boxes, dtype=np.int32)),
            "truncated" : dt_sample.get("truncated", np.zeros(num_dt_boxes, dtype=np.float32)),
            "location": location, "dimensions": dimensions, "rotation_y": rotation_y
        }
        dt_annos_raw_list.append(dt_anno)
    
    logger_func.info(f"{len(dt_annos_raw_list)} DT örneği yüklendi ve ham formatta hazırlandı.")
    if not dt_annos_raw_list:
        logger_func.error("Hiç geçerli DT örneği yüklenemedi.")
        return []
    return transform_annotations_to_kitti_format(dt_annos_raw_list, map_name_to_kitti=class_mapping)


def main():
    global logger
    
    dt_pkl_path = DT_RESULTS_PKL_PATH
    gt_pkl_path = GT_INFOS_PKL_PATH
    output_dir_str = OUTPUT_DIR_PATH
    conf_thresholds_str = CONFIDENCE_THRESHOLDS_STR
    nms_iou_thresholds_str = NMS_IOU_THRESHOLDS_STR
    class_mapping = CLASS_MAPPING_TO_KITTI
    current_class_name = CURRENT_CLASS_NAME_FOR_ANALYSIS
    current_class_id = CURRENT_CLASS_ID_FOR_ANALYSIS # Örn: Car için 0
    # eval_iou: AP'nin hangi IoU'da hesaplanacağını belirler (kitti_eval.eval_class'a min_overlaps olarak geçer)
    target_ap_iou = EVALUATION_IOU_THRESHOLD 
    
    nms_pre_max = NMS_PRE_MAXSIZE_PER_CLASS
    nms_post_max = NMS_POST_MAXSIZE_PER_CLASS
    nms_type = NMS_TYPE_FOR_SINGLE_CLASS

    conf_thresholds_list = [float(x.strip()) for x in conf_thresholds_str.split(',')]
    nms_iou_thresholds_list = [float(x.strip()) for x in nms_iou_thresholds_str.split(',')]

    output_path = Path(output_dir_str)
    output_path.mkdir(parents=True, exist_ok=True)
    
    main_log_file_path = output_path / f'log_esik_analizi_dt_gt_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.txt'
    logger = common_utils.create_logger(main_log_file_path, rank=0)
    logger.info("Eşik değeri analizi başlatılıyor (doğrudan DT/GT .pkl dosyaları ile, daha az fonksiyonlu)...")
    # ... (diğer başlangıç logları) ...
    logger.info(f"DT PKL Dosyası: {dt_pkl_path}")
    logger.info(f"GT PKL Dosyası: {gt_pkl_path}")
    logger.info(f"Çıktı Dizini: {output_path}")
    logger.info(f"Analiz Edilecek Sınıf: {current_class_name} (ID: {current_class_id})")
    logger.info(f"Hedeflenen AP IoU Eşiği (Grafikler için): {target_ap_iou}")
    logger.info(f"Test edilecek Güven Eşikleri: {conf_thresholds_list}")
    logger.info(f"Test edilecek NMS IoU Eşikleri: {nms_iou_thresholds_list}")
    logger.info(f"NMS Ayarları: PRE_MAX={nms_pre_max}, POST_MAX={nms_post_max}, TYPE={nms_type}")


    try:
        gt_annos_kitti_transformed = load_and_prepare_gt_annos(gt_pkl_path, class_mapping, logger)
        dt_annos_raw_kitti_transformed_original = load_and_prepare_dt_annos(dt_pkl_path, class_mapping, logger)
        if not dt_annos_raw_kitti_transformed_original: 
            logger.error("DT annoları yüklenemedi veya boş. Analiz durduruluyor.")
            return
    except Exception as e:
        logger.error(f"GT veya DT dosyaları yüklenirken/hazırlanırken hata: {e}", exc_info=True)
        return

    if len(gt_annos_kitti_transformed) != len(dt_annos_raw_kitti_transformed_original):
        logger.warning(f"GT ({len(gt_annos_kitti_transformed)}) ve DT ({len(dt_annos_raw_kitti_transformed_original)}) örnek sayıları eşleşmiyor! Sonuçlar hatalı olabilir.")
        # Gerekirse burada çıkış yapılabilir veya eşleşmeyenler atılabilir.
        # Şimdilik devam edelim. En kısa listeye göre kırpmak bir seçenek olabilir.
        min_len = min(len(gt_annos_kitti_transformed), len(dt_annos_raw_kitti_transformed_original))
        gt_annos_kitti_transformed = gt_annos_kitti_transformed[:min_len]
        dt_annos_raw_kitti_transformed_original = dt_annos_raw_kitti_transformed_original[:min_len]
        logger.info(f"  GT ve DT listeleri {min_len} örneğe eşitlendi.")


    results_list = []

    # ---- ANA DÖNGÜ BAŞLANGICI ----
    for conf_val in conf_thresholds_list:
        for nms_iou_val in nms_iou_thresholds_list:
            logger.info(f"İşleniyor: Güven Eşiği={conf_val:.2f}, NMS IoU Eşiği={nms_iou_val:.2f}")
            
            # ** apply_custom_postprocessing içeriği buraya taşınacak **
            current_dt_annos_for_processing = copy.deepcopy(dt_annos_raw_kitti_transformed_original)
            processed_dt_annos_for_eval = []

            logger.info(f"  apply_custom_postprocessing BAŞLADI (conf={conf_val:.2f}, nms_iou={nms_iou_val:.2f})")
            for sample_idx, sample_dt_original_kitti_fmt in enumerate(current_dt_annos_for_processing):
                sample_dt_current_iter = copy.deepcopy(sample_dt_original_kitti_fmt) # Her zaman orijinalden kopyala
                
                if 'score' not in sample_dt_current_iter or sample_dt_current_iter['score'] is None or len(sample_dt_current_iter['score']) == 0:
                    logger.debug(f"    Örnek {sample_idx}: Skor yok veya boş, olduğu gibi ekleniyor.")
                    processed_dt_annos_for_eval.append(sample_dt_current_iter)
                    continue

                original_num_scores_in_sample = len(sample_dt_current_iter['score'])
                logger.debug(f"    Örnek {sample_idx}: Skor eşiği öncesi kutu sayısı: {original_num_scores_in_sample}")

                # 1. Güven eşiğini uygula
                score_mask_current = sample_dt_current_iter['score'] >= float(conf_val)
                
                keys_to_mask_in_sample = [k for k, v_arr in sample_dt_current_iter.items() if isinstance(v_arr, np.ndarray) and len(v_arr) == original_num_scores_in_sample]
                
                for key_m in keys_to_mask_in_sample:
                    try:
                        sample_dt_current_iter[key_m] = sample_dt_current_iter[key_m][score_mask_current]
                    except IndexError:
                        logger.warning(f"    Örnek {sample_idx}, anahtar '{key_m}': score_mask ile indeksleme hatası.")
                        sample_dt_current_iter[key_m] = np.array([]) # Hata durumunda boşalt

                num_boxes_after_score_thresh = len(sample_dt_current_iter.get('score', []))
                logger.debug(f"    Örnek {sample_idx}: Skor eşiği ({conf_val:.2f}) sonrası kutu sayısı: {num_boxes_after_score_thresh}")

                if num_boxes_after_score_thresh == 0:
                    # Güven eşiğinden sonra kutu kalmazsa, tüm array'leri uygun şekilde boşalt
                    for key_to_empty_in_sample in list(sample_dt_current_iter.keys()):
                        original_val_in_sample = sample_dt_original_kitti_fmt.get(key_to_empty) # Orijinalden şekil ve dtype al
                        if isinstance(original_val_in_sample, np.ndarray):
                            original_shape_in_sample = original_val_in_sample.shape
                            dtype_to_use_in_sample = original_val_in_sample.dtype
                            if len(original_shape_in_sample) > 1:
                                sample_dt_current_iter[key_to_empty_in_sample] = np.empty((0, original_shape_in_sample[1]), dtype=dtype_to_use_in_sample)
                            else:
                                sample_dt_current_iter[key_to_empty_in_sample] = np.array([], dtype=dtype_to_use_in_sample)
                    processed_dt_annos_for_eval.append(sample_dt_current_iter)
                    continue

                # 2. NMS Uygula (Her sınıf için ayrı ayrı)
                props_collected_after_nms_for_sample = {key: [] for key in sample_dt_current_iter.keys() if isinstance(sample_dt_current_iter[key], np.ndarray)}
                
                unique_names_in_current_sample = np.unique(sample_dt_current_iter['name'])
                logger.debug(f"    Örnek {sample_idx}: NMS için sınıflar: {unique_names_in_current_sample}")

                for class_name_iter in unique_names_in_current_sample:
                    class_mask_for_nms_iter = (sample_dt_current_iter['name'] == class_name_iter)
                    if not np.any(class_mask_for_nms_iter): continue

                    boxes_this_class_iter = sample_dt_current_iter['boxes_lidar'][class_mask_for_nms_iter]
                    scores_this_class_iter = sample_dt_current_iter['score'][class_mask_for_nms_iter]
                    
                    logger.debug(f"      Sınıf '{class_name_iter}': NMS öncesi kutu sayısı: {len(scores_this_class_iter)}")
                    if len(scores_this_class_iter) == 0: continue
                        
                    boxes_tensor_cls_iter = torch.from_numpy(boxes_this_class_iter).float().cuda()
                    scores_tensor_cls_iter = torch.from_numpy(scores_this_class_iter).float().cuda()
                    
                    current_pre_max_iter = min(nms_pre_max, scores_tensor_cls_iter.shape[0])
                    if current_pre_max_iter <= 0 : continue

                    scores_topk_cls_iter, indices_topk_cls_iter = torch.topk(scores_tensor_cls_iter, k=current_pre_max_iter)
                    boxes_topk_cls_iter = boxes_tensor_cls_iter[indices_topk_cls_iter]

                    selected_indices_in_topk_iter = torch.arange(len(scores_topk_cls_iter), device=scores_topk_cls_iter.device) # Varsayılan (NMS yok)
                    try:
                        nms_func_iter = getattr(iou3d_nms_utils, nms_type) # nms_type globalden geliyor
                        if nms_type == 'nms_gpu':
                            selected_indices_in_topk_iter, _ = nms_func_iter(
                                boxes_topk_cls_iter[:, 0:7], scores_topk_cls_iter, float(nms_iou_val)
                            )
                        elif nms_type == 'nms_normal_gpu':
                             selected_indices_in_topk_iter = nms_func_iter(
                                boxes_topk_cls_iter[:, 0:7], scores_topk_cls_iter, float(nms_iou_val)
                            )
                        else: logger.error(f"Desteklenmeyen NMS_TYPE: {nms_type}")
                    except Exception as e_nms_iter:
                        logger.error(f"NMS sırasında hata (sınıf: {class_name_iter}, tip: {nms_type}): {e_nms_iter}", exc_info=True)

                    current_post_max_iter = min(nms_post_max, len(selected_indices_in_topk_iter))
                    final_sel_indices_in_topk_for_class_iter = selected_indices_in_topk_iter[:current_post_max_iter]
                    orig_indices_for_class_after_nms_iter = indices_topk_cls_iter[final_sel_indices_in_topk_for_class_iter].cpu().numpy()
                    
                    logger.debug(f"      Sınıf '{class_name_iter}': NMS sonrası kutu sayısı: {len(orig_indices_for_class_after_nms_iter)}")

                    for key_p in props_collected_after_nms_for_sample.keys():
                        if key_p in sample_dt_current_iter and isinstance(sample_dt_current_iter[key_p], np.ndarray):
                            array_for_class_iter = sample_dt_current_iter[key_p][class_mask_for_nms_iter]
                            if len(array_for_class_iter) > 0:
                                 props_collected_after_nms_for_sample[key_p].append(array_for_class_iter[orig_indices_for_class_after_nms_iter])
                
                sample_dt_processed_final = {}
                any_box_left_in_sample_final = False
                for key_f in props_collected_after_nms_for_sample.keys():
                    if props_collected_after_nms_for_sample[key_f]:
                        valid_arrays_to_concat_f = [arr for arr in props_collected_after_nms_for_sample[key_f] if arr.size > 0]
                        if valid_arrays_to_concat_f:
                            concatenated_array_f = np.concatenate(valid_arrays_to_concat_f, axis=0)
                            sample_dt_processed_final[key_f] = concatenated_array_f
                            if key_f == 'score' and len(concatenated_array_f) > 0:
                                any_box_left_in_sample_final = True
                        else:
                            original_val_key_f = sample_dt_original_kitti_fmt.get(key_f)
                            if isinstance(original_val_key_f, np.ndarray):
                                original_shape_f = original_val_key_f.shape; dtype_to_use_f = original_val_key_f.dtype
                                if len(original_shape_f) > 1: sample_dt_processed_final[key_f] = np.empty((0, original_shape_f[1]), dtype=dtype_to_use_f)
                                else: sample_dt_processed_final[key_f] = np.array([], dtype=dtype_to_use_f)
                            else: sample_dt_processed_final[key_f] = np.array([])
                    else:
                        original_val_key_f = sample_dt_original_kitti_fmt.get(key_f)
                        if isinstance(original_val_key_f, np.ndarray):
                            original_shape_f = original_val_key_f.shape; dtype_to_use_f = original_val_key_f.dtype
                            if len(original_shape_f) > 1: sample_dt_processed_final[key_f] = np.empty((0, original_shape_f[1]), dtype=dtype_to_use_f)
                            else: sample_dt_processed_final[key_f] = np.array([], dtype=dtype_to_use_f)
                        else: sample_dt_processed_final[key_f] = np.array([])
                
                if not any_box_left_in_sample_final: # Tüm anahtarları boşalt (yukarıda yapıldı ama garanti)
                    logger.debug(f"    Örnek {sample_idx}: NMS sonrası hiç kutu kalmadı.")
                    for key_check_f in sample_dt_original_kitti_fmt.keys():
                        if isinstance(sample_dt_original_kitti_fmt.get(key_check_f), np.ndarray):
                            original_shape_f = sample_dt_original_kitti_fmt[key_check_f].shape
                            dtype_to_use_f = sample_dt_original_kitti_fmt[key_check_f].dtype
                            if len(original_shape_f) > 1: sample_dt_processed_final[key_check_f] = np.empty((0, original_shape_f[1]), dtype=dtype_to_use_f)
                            else: sample_dt_processed_final[key_check_f] = np.array([], dtype=dtype_to_use_f)
                        elif key_check_f not in sample_dt_processed_final : # Eğer props_collected'da yoksa ve ndarray değilse
                             sample_dt_processed_final[key_check_f] = sample_dt_original_kitti_fmt.get(key_check_f) # Orijinal değeri koru (frame_id vs.)


                processed_dt_annos_for_eval.append(sample_dt_processed_final)
                logger.debug(f"    Örnek {sample_idx}: Post-processing sonrası kutu sayısı: {len(sample_dt_processed_final.get('score',[]))}")

            # ** evaluate_kitti_directamente içeriği buraya taşınacak **
            ap_metric_value = 0.0
            logger.info(f"  evaluate_kitti_directamente BAŞLADI (IoU={target_ap_iou:.2f})")
            
            # DT annolarının geçerliliğini kontrol et
            valid_dt_annos_exist_for_eval = False
            if processed_dt_annos_for_eval: # Bu artık tüm dataset için işlenmiş DT'leri tutmalı
                for anno_eval in processed_dt_annos_for_eval: # Bu döngü yanlış yerde, tüm dt listesi için olmalı
                    if 'score' in anno_eval and anno_eval['score'] is not None and len(anno_eval['score']) > 0:
                        valid_dt_annos_exist_for_eval = True
                        break
            
            num_dt_boxes_for_eval = sum(len(anno.get('score',[])) for anno in processed_dt_annos_for_eval)
            logger.info(f"    Değerlendirmeye giren toplam DT kutu sayısı: {num_dt_boxes_for_eval}")

            if not valid_dt_annos_exist_for_eval or num_dt_boxes_for_eval == 0:
                logger.warning(f"    Değerlendirme için geçerli DT annosu bulunamadı. AP 0 olacak.")
                ap_metric_value = 0.0
            else:
                # min_overlaps matrisini oluştur (sadece hedeflenen IoU için)
                # `eval_class` fonksiyonu `min_overlaps` (çoğul 's' ile) argümanını bekler.
                # Şekli: [num_classes_to_evaluate, num_difficulty_levels, num_iou_thresholds_for_this_metric]
                # Biz tek sınıf, tek zorluk, tek IoU eşiği için değerlendirme yapıyoruz.
                # Ancak `eval_class` içindeki `compute_statistics_jit` fonksiyonu,
                # `min_overlap` argümanını [num_iou_thresholds_for_this_metric, num_metrics_like_bev_3d] şeklinde bekler.
                # Ve `metric` argümanı (0:bbox, 1:bev, 2:3d) hangi metriğin hesaplanacağını seçer.
                # `kitti_eval.py` içindeki `eval_class` `min_overlap` argümanını şöyle kullanır:
                # `min_overlap_class = min_overlap[:, metric_i]`
                # Bu durumda `min_overlap` [num_iou_thresholds, num_total_metrics_config_can_handle] şeklinde olmalı.
                # Biz sadece 3D (metric=2) için ve tek bir IoU eşiği ile ilgileniyoruz.
                # `min_overlaps_default = np.array([0.7, 0.5, 0.5, 0.7, 0.5, 0.5, 0.7, 0.5, 0.5]).reshape(3, 3)`
                # Bu default matris [metric_idx, class_idx_in_CarPedCyc_order] -> iou_thresh verir.
                # Eğer `min_overlaps` argümanını `eval_class`'a verirsek, bu default'u ezer.
                
                # `eval_class`a geçilecek `min_overlaps` argümanı için doğru şekil:
                # [num_classes_IN_THIS_CALL, num_difficulty_IN_THIS_CALL, num_iou_threshold_points_TO_CALCULATE_AP_OVER]
                # `metric` argümanı hangi metriğe (2D, BEV, 3D) bakılacağını söyler.
                # `average_precision` fonksiyonu 11 recall noktasındaki precision'ı bekler.
                # `eval_class` bu 11 noktayı, `min_overlaps`'taki *her bir IoU eşiği için* hesaplar.
                # Biz sadece tek bir IoU eşiğinde (target_ap_iou) AP istiyoruz.
                
                # Bu nedenle, eval_class'a [1,1,1] boyutunda bir min_overlaps ve tek bir IoU değeri vermeliyiz.
                min_overlaps_for_eval_class = np.array([[[target_ap_iou]]], dtype=np.float32) # Shape: [1,1,1] (class, diff, iou_thresh_idx)
                                                                                            # Bu, tek bir IoU eşiğinde AP hesapla demek.


                difficulties_to_eval_for_eval_class = [0] # Tek bir zorluk seviyesi (tümünü birleştir)
                num_parts_for_eval_class = 1 
                
                try:
                    logger.debug(f"    eval_class çağrılıyor. GT örnek: {len(gt_annos_kitti_transformed)}, DT örnek: {len(processed_dt_annos_for_eval)}")
                    if len(processed_dt_annos_for_eval) > 0 and isinstance(processed_dt_annos_for_eval[0].get('score'), np.ndarray) and len(processed_dt_annos_for_eval[0].get('score',[])) > 0:
                         logger.debug(f"      İlk DT örneğindeki ilk 5 skor (eval_class'a girmeden): {processed_dt_annos_for_eval[0]['score'][:5]}")
                         logger.debug(f"      İlk DT örneğindeki ilk 5 isim (eval_class'a girmeden): {processed_dt_annos_for_eval[0]['name'][:5]}")
                         logger.debug(f"      İlk DT örneğindeki ilk kutu (eval_class'a girmeden): {processed_dt_annos_for_eval[0]['boxes_lidar'][0] if len(processed_dt_annos_for_eval[0]['boxes_lidar']) > 0 else 'Kutu Yok'}")


                    res_eval_class_dict_current = kitti_eval.eval_class(
                        gt_annos_kitti_transformed, processed_dt_annos_for_eval, 
                        [current_class_id], difficulties_to_eval_for_eval_class, 
                        metric=2, # 3D
                        min_overlaps=min_overlaps_for_eval_class, # Düzeltilmiş argüman ismi
                        compute_aos=False, num_parts=num_parts_for_eval_class
                    )
                    
                    logger.info(f"    eval_class sonuç anahtarları: {list(res_eval_class_dict_current.keys())}")
                    # Örnek: ['recall', 'precision', 'thresholds', 'mAP_bev', 'mAP_3d', ...]

                    if 'precision' not in res_eval_class_dict_current or \
                       res_eval_class_dict_current['precision'] is None or \
                       res_eval_class_dict_current['precision'].size == 0:
                        logger.warning(f"    eval_class'tan 'precision' alınamadı veya boş. Sınıf: {current_class_name}. AP 0 olacak.")
                        ap_metric_value = 0.0
                    else:
                        # precision şekli: [num_classes_eval=1, num_diff_eval=1, num_iou_thresh_passed_in_min_overlaps=1, 11_recall_points]
                        precision_points_current = res_eval_class_dict_current['precision'][0, 0, 0, :] 
                        logger.info(f"      Precision Puanları ({current_class_name} @ IoU={target_ap_iou:.2f}): {precision_points_current}")
                        
                        if precision_points_current is None or len(precision_points_current) == 0:
                            ap_metric_value = 0.0
                        elif len(precision_points_current) == 11 : 
                            ap_metric_value = kitti_eval.average_precision(precision_points_current)
                            if ap_metric_value is None : ap_metric_value = 0.0 # average_precision None dönebilir
                        else:
                            logger.warning(f"      Beklenen 11 kesinlik noktası yerine {len(precision_points_current)} nokta alındı. AP hesaplanamıyor, 0 olarak ayarlandı.")
                            ap_metric_value = 0.0
                except Exception as e_eval:
                    logger.error(f"    kitti_eval.eval_class veya AP hesaplama sırasında hata: {e_eval}", exc_info=True)
                    ap_metric_value = 0.0

            results_list.append({
                'confidence_threshold': conf_val, 'nms_iou_threshold': nms_iou_val,
                'metric_value': ap_metric_value
            })
            logger.info(f"SONUÇ: Güven={conf_val:.2f}, NMS IoU={nms_iou_val:.2f}, AP(3D,IoU={target_ap_iou:.2f})[{current_class_name}] = {ap_metric_value:.4f}")
    # ---- ANA DÖNGÜ SONU ----

    results_dataframe = pd.DataFrame(results_list)
    # ... (Grafik çizimi ve CSV kaydı önceki gibi devam eder) ...
    csv_save_path = output_path / 'esik_analizi_dt_gt_metrikleri.csv'
    results_dataframe.to_csv(csv_save_path, index=False)
    logger.info(f"Sonuçlar CSV dosyasına kaydedildi: {csv_save_path}")

    if not results_dataframe.empty and not results_dataframe['metric_value'].isnull().all():
        plt.figure(figsize=(12, 8))
        for nms_val_plot in nms_iou_thresholds_list:
            subset = results_dataframe[results_dataframe['nms_iou_threshold'] == nms_val_plot]
            if not subset.empty: plt.plot(subset['confidence_threshold'], subset['metric_value'], marker='o', label=f'NMS IoU = {nms_val_plot:.2f}')
        plt.xlabel('Güven Eşiği (SCORE_THRESH)')
        plt.ylabel(f'AP (3D, IoU={target_ap_iou:.2f}) - Sınıf: {current_class_name}')
        plt.title(f'Metrik vs. Güven Eşiği ({current_class_name}) - Doğrudan DT/GT .pkl')
        plt.legend(); plt.grid(True)
        plt.savefig(output_path / f'metrik_vs_guven_esigi_{current_class_name}_dt_gt.png')
        plt.close()

        plt.figure(figsize=(12, 8))
        for conf_val_plot in conf_thresholds_list:
            subset = results_dataframe[results_dataframe['confidence_threshold'] == conf_val_plot]
            if not subset.empty: plt.plot(subset['nms_iou_threshold'], subset['metric_value'], marker='o', label=f'Güven = {conf_val_plot:.2f}')
        plt.xlabel('NMS IoU Eşiği')
        plt.ylabel(f'AP (3D, IoU={target_ap_iou:.2f}) - Sınıf: {current_class_name}')
        plt.title(f'Metrik vs. NMS IoU Eşiği ({current_class_name}) - Doğrudan DT/GT .pkl')
        plt.legend(); plt.grid(True)
        plt.savefig(output_path / f'metrik_vs_nms_iou_esigi_{current_class_name}_dt_gt.png')
        plt.close()

        if len(conf_thresholds_list) > 1 and len(nms_iou_thresholds_list) > 1:
            try:
                if results_dataframe['metric_value'].notnull().any():
                    pivot_data = results_dataframe.pivot(index='nms_iou_threshold', columns='confidence_threshold', values='metric_value')
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(pivot_data, annot=True, fmt=".4f", cmap="viridis", cbar_kws={'label': f'AP (3D, IoU={target_ap_iou:.2f}) - {current_class_name}'})
                    plt.title(f'Performans Isı Haritası ({current_class_name}) - Doğrudan DT/GT .pkl')
                    plt.xlabel('Güven Eşiği (SCORE_THRESH)'); plt.ylabel('NMS IoU Eşiği')
                    plt.savefig(output_path / f'metrik_isi_haritasi_{current_class_name}_dt_gt.png')
                    plt.close()
                else: logger.warning("Isı haritası için geçerli metrik değeri bulunamadı (tümü NaN).")
            except Exception as e: logger.error(f"Isı haritası oluşturulamadı: {e}")
    elif results_dataframe.empty: logger.info("Çizilecek sonuç bulunamadı (DataFrame boş).")
    else: logger.warning("Çizilecek geçerli metrik değeri bulunamadı (tüm değerler NaN). Grafik oluşturma atlanıyor.")

    logger.info("Doğrudan DT/GT .pkl dosyaları ile eşik değeri analizi tamamlandı.")


if __name__ == '__main__':
    main()