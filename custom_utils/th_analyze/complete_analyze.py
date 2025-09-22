import datetime
import os
import shutil
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import copy # For deep copying predictions
import sys
sys.path.insert(0, '/root/3DTrans/')

# OpenPCDet/3DTrans kütüphanesinden gerekli modüller
try:
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    # Model build_network ve load_params_from_file artık doğrudan kullanılmayacak
    # ama NMS veya diğer yardımcı fonksiyonlar için model_utils gerekebilir.
    from pcdet.models.model_utils import model_nms_utils
    from pcdet.utils import common_utils
    # tools.eval_utils.eval_utils içindeki standart değerlendirme fonksiyonu
    # dataset.evaluation() çağırmak için kullanılacak
except ImportError as e:
    print(f"Hata: OpenPCDet/3DTrans kütüphane dosyaları bulunamadı veya içe aktarılamadı. Lütfen PYTHONPATH'ınızı kontrol edin. Detay: {e}")
    print("Betiği OpenPCDet/3DTrans ana dizininden çalıştırmayı deneyin.")
    exit(1)

### SABİT PARAMETRELER ###
# Lütfen bu değerleri kendi yapılandırmanıza göre güncelleyin.
CFG_FILE_PATH = '/root/3DTrans/tools/cfgs/custom/pv_rcnn.yaml'  # Veri seti ve değerlendirme ayarları için yapılandırma dosyası
PREDICTIONS_PKL_FILE_PATH = '/root/3DTrans/output/cfgs/MDF/KNW/customnw_pvrcnn_feat_3_uni3d/default/eval/epoch_2/val/default/result.pkl'  # Önceden hesaplanmış tahminleri içeren .pkl dosyasının yolu
OUTPUT_DIR_PATH = 'analiz_sonuclari/esik_deger_analizi'  # Grafiklerin ve sonuçların kaydedileceği dizin
CONFIDENCE_THRESHOLDS_STR = '0.1,0.3,0.5,0.7'  # Virgülle ayrılmış güven eşik değerleri listesi
NMS_IOU_THRESHOLDS_STR = '0.1,0.3,0.5,0.7'  # Virgülle ayrılmış NMS IoU eşik değerleri listesi
##########################

def load_predictions_from_pkl(pkl_file_path):
    """
    Verilen .pkl dosyasından tahminleri yükler.
    OpenPCDet formatında (liste içinde sözlükler) olduğu varsayılır.
    """
    logger.info(f".pkl dosyasından tahminler yükleniyor: {pkl_file_path}")
    if not Path(pkl_file_path).exists():
        logger.error(f"PKL dosyası bulunamadı: {pkl_file_path}")
        raise FileNotFoundError(f"PKL dosyası bulunamadı: {pkl_file_path}")
    with open(pkl_file_path, 'rb') as f:
        try:
            predictions = pickle.load(f)
        except Exception as e:
            logger.error(f".pkl dosyası okunurken hata: {e}")
            raise
    logger.info(f"{len(predictions)} örnek için tahminler yüklendi.")
    return predictions

def run_evaluation_from_pkl(
    config_path,
    all_predictions_original, # Yüklenmiş orijinal tahmin listesi
    score_thresh,
    nms_iou_thresh,
    logger_func, # Ana logger nesnesi
    base_output_dir_for_eval_results
):
    """
    Yüklenmiş tahminlere belirtilen confidence ve NMS IoU eşiklerini uygulayarak değerlendirir.
    """
    cfg_from_yaml_file(config_path, cfg) # Global cfg'yi yükle/güncelle
    
    logger_func.info(f"Değerlendirme çalıştırılıyor (PKL'den): SCORE_THRESH={score_thresh}, NMS_IOU_THRESH={nms_iou_thresh}")

    # Tahminleri işle: confidence threshold ve NMS uygula
    processed_pred_dicts = []
    for original_preds_dict_sample in all_predictions_original:
        # Orijinal tahminleri değiştirmemek için derin kopya al
        current_preds = copy.deepcopy(original_preds_dict_sample)

        # .pkl dosyasındaki numpy array'lerini PyTorch tensor'lerine çevir (eğer henüz değilse)
        # ve GPU'ya taşı (NMS genellikle GPU'da çalışır)
        try:
            scores_np = current_preds['pred_scores']
            boxes_np = current_preds['pred_boxes']
            labels_np = current_preds['pred_labels'] # Genellikle 1-indeksli
        except KeyError as e:
            logger_func.error(f"PKL dosyasındaki örnekte eksik anahtar: {e}. Örnek: {current_preds.get('frame_id', 'ID YOK')}")
            # Bu örneği atla veya hata ver
            continue


        scores = torch.from_numpy(scores_np).float().cuda()
        boxes = torch.from_numpy(boxes_np).float().cuda()
        labels = torch.from_numpy(labels_np).long().cuda() # Sınıf etiketleri genellikle Long

        # 1. Güven eşiğini uygula
        conf_mask = scores >= float(score_thresh)
        
        scores_conf_filtered = scores[conf_mask]
        boxes_conf_filtered = boxes[conf_mask]
        labels_conf_filtered = labels[conf_mask]

        if len(scores_conf_filtered) == 0:
            # Güven eşiğini geçen kutu yoksa, boş tahmin olarak ekle
            processed_pred_dicts.append({
                'frame_id': current_preds.get('frame_id', current_preds.get('sample_idx', 'UNKNOWN_FRAME')),
                'pred_boxes': torch.empty((0, boxes.shape[1] if boxes.ndim > 1 else 7), device='cpu', dtype=torch.float),
                'pred_scores': torch.empty(0, device='cpu', dtype=torch.float),
                'pred_labels': torch.empty(0, device='cpu', dtype=torch.long),
            })
            continue

        # 2. NMS uygula
        # `multi_classes_nms` genellikle 1-indeksli etiketler bekler.
        # OpenPCDet'in standart çıktıları genellikle bu şekildedir.
        _nms_config = {
            'NMS_TYPE': cfg.MODEL.POST_PROCESSING.NMS_CONFIG.get('NMS_TYPE', 'nms_gpu'),
            'MULTI_CLASSES_NMS': cfg.MODEL.POST_PROCESSING.NMS_CONFIG.get('MULTI_CLASSES_NMS', False),
            'NMS_PRE_MAXSIZE': cfg.MODEL.POST_PROCESSING.NMS_CONFIG.get('NMS_PRE_MAXSIZE', len(scores_conf_filtered)),
            'NMS_POST_MAXSIZE': cfg.MODEL.POST_PROCESSING.NMS_CONFIG.get('NMS_POST_MAXSIZE', len(scores_conf_filtered)),
            'NMS_IOU_THRESH': float(nms_iou_thresh),
        }
        if _nms_config['MULTI_CLASSES_NMS']: # Eğer sınıfa özel NMS ise, her sınıf için aynı IoU'yu kullan
             _nms_config['IOU_THRESH_PER_CLASS'] = [float(nms_iou_thresh)] * len(cfg.CLASS_NAMES)


        selected_indices, selected_scores_after_nms = model_nms_utils.multi_classes_nms(
            box_scores=scores_conf_filtered,
            box_preds=boxes_conf_filtered,
            box_labels=labels_conf_filtered, # 1-indeksli olmalı
            nms_config=_nms_config,
            score_thresh=None # Güven eşiği zaten yukarıda uygulandı
        )
        
        boxes_nms = boxes_conf_filtered[selected_indices]
        # selected_scores_after_nms NMS sonrası skorları verir, orijinal skorları da kullanabiliriz: scores_conf_filtered[selected_indices]
        scores_nms = selected_scores_after_nms 
        labels_nms = labels_conf_filtered[selected_indices]
        
        # Sonuçları CPU'ya geri taşı (dataset.evaluation genellikle CPU tensor'leri bekler)
        processed_pred_dicts.append({
            'frame_id': current_preds.get('frame_id', current_preds.get('sample_idx', 'UNKNOWN_FRAME')),
            'pred_boxes': boxes_nms.cpu(),
            'pred_scores': scores_nms.cpu(),
            'pred_labels': labels_nms.cpu(),
        })

    # Veri seti nesnesini oluştur (değerlendirme fonksiyonu için)
    # build_dataloader, veri seti nesnesini de döndürür.
    # Değerlendirme için tüm veri setini yüklemeye gerek yok, sadece nesne lazım.
    class MinimalLogger: # build_dataloader için sessiz logger
        def info(self, msg): pass
        def warning(self, msg): pass
        def error(self, msg): pass
        def debug(self, msg): pass

    dataset_obj, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1, # Gerçekte veri yüklenmeyecek, sadece nesne için
        dist=False,
        workers=1, 
        logger=MinimalLogger(),
        training=False,
        see_more_information=False # 3DTrans'a özel olabilir, varsa ekleyin
    )

    # dataset.evaluation() fonksiyonunu işlemiş tahminlerle çağır
    eval_run_sub_dir = base_output_dir_for_eval_results / f"eval_score_{score_thresh:.2f}_nmsiou_{nms_iou_thresh:.2f}"
    eval_run_sub_dir.mkdir(parents=True, exist_ok=True)

    logger_func.info(f"`dataset.evaluation()` çağrılıyor. İşlenmiş {len(processed_pred_dicts)} tahmin mevcut.")
    
    # dataset.evaluation, ap_result_str ve ap_dict döndürür
    _, ap_dict = dataset_obj.evaluation(
        pred_dicts=processed_pred_dicts,
        class_names=cfg.CLASS_NAMES,
        output_dir=eval_run_sub_dir 
    )
    
    # Metrikleri çıkar (önceki betikteki gibi)
    metric_value = None
    main_metric_name = "Default_AP" # Varsayılan metrik ismi
    if cfg.DATA_CONFIG.DATASET == 'CustomDataset':
        # Örnek: KITTI için Car, Moderate, 3D IoU@0.7 R40 metriği
        # CLASS_NAMES yapılandırmadan alınır, genellikle ilki 'Car' olur.
        main_metric_name = f'{cfg.CLASS_NAMES[0]}_3d_moderate_R40' 
        if main_metric_name in ap_dict:
            metric_value = ap_dict[main_metric_name]
        else: # Genel bir AP bulmaya çalış
             for k, v_ap in ap_dict.items(): # v_ap ismini değiştirdim (v zaten döngüde)
                if 'R40' in k and 'moderate' in k and isinstance(v_ap, float): 
                    metric_value = v_ap
                    logger_func.info(f"KITTI için {main_metric_name} bulunamadı, alternatif olarak {k} kullanılıyor.")
                    main_metric_name = k
                    break
    elif cfg.DATA_CONFIG.DATASET == 'NuScenesDataset':
        main_metric_name = 'mAP'
        if main_metric_name in ap_dict:
            metric_value = ap_dict[main_metric_name]
    elif cfg.DATA_CONFIG.DATASET == 'WaymoDataset':
        main_class_name = cfg.CLASS_NAMES[0] if cfg.CLASS_NAMES else 'VEHICLE' 
        main_metric_name = f'{main_class_name.upper()}_3D_LEVEL_2_AP' # Genellikle Level 2
        metric_key_l1 = f'{main_class_name.upper()}_3D_LEVEL_1_AP'
        if main_metric_name in ap_dict:
             metric_value = ap_dict[main_metric_name]
        elif metric_key_l1 in ap_dict: # L1'e bak
             metric_value = ap_dict[metric_key_l1]
             main_metric_name = metric_key_l1
    
    # Genel fallback
    if metric_value is None:
        if 'mAP' in ap_dict: # Genel mAP
            metric_value = ap_dict['mAP']
            main_metric_name = 'mAP (fallback)'
        else: # Herhangi bir AP değeri
            for k, v_ap in ap_dict.items():
                if ('AP' in k.upper() or 'MAP' in k.upper()) and isinstance(v_ap, (float, np.float32, np.float64)):
                    metric_value = float(v_ap)
                    main_metric_name = k
                    logger_func.info(f"Özel metrik bulunamadı, genel AP/mAP metriği kullanılıyor: {k} = {metric_value}")
                    break
    
    if metric_value is None:
        logger_func.error(f"Uygun bir AP/mAP metriği `dataset.evaluation` sonuçlarında bulunamadı: {ap_dict.keys()}. Lütfen `run_evaluation_from_pkl` içindeki metrik çıkarma mantığını kontrol edin.")
        metric_value = 0.0 # Hata durumunda 0.0 döndür
        main_metric_name = "ERROR_NO_METRIC"

    logger_func.info(f"Değerlendirme sonucu [{main_metric_name}]: {metric_value:.4f}")
    return float(metric_value)

def main():
    cfg_file = CFG_FILE_PATH
    predictions_pkl_file = PREDICTIONS_PKL_FILE_PATH
    output_dir_str = OUTPUT_DIR_PATH
    conf_thresholds_str_list = CONFIDENCE_THRESHOLDS_STR
    nms_iou_thresholds_str_list = NMS_IOU_THRESHOLDS_STR

    conf_thresholds_list = [float(x.strip()) for x in conf_thresholds_str_list.split(',')]
    nms_iou_thresholds_list = [float(x.strip()) for x in nms_iou_thresholds_str_list.split(',')]

    output_path = Path(output_dir_str)
    output_path.mkdir(parents=True, exist_ok=True)
    
    global logger # `load_predictions_from_pkl` içinde kullanmak için global yap
    main_log_file_path = output_path / f'log_esik_analizi_pkl_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.txt'
    logger = common_utils.create_logger(main_log_file_path, rank=0) # global logger'ı ata
    logger.info("Eşik değeri analizi başlatılıyor (PKL dosyası ile, sabit parametrelerle)...")
    logger.info(f"Yapılandırma Dosyası (veri seti için): {cfg_file}")
    logger.info(f"Tahminlerin PKL Dosyası: {predictions_pkl_file}")
    logger.info(f"Çıktı Dizini: {output_path}")
    logger.info(f"Test edilecek Güven Eşikleri: {conf_thresholds_list}")
    logger.info(f"Test edilecek NMS IoU Eşikleri: {nms_iou_thresholds_list}")

    try:
        original_predictions = load_predictions_from_pkl(predictions_pkl_file)
    except Exception as e:
        logger.error(f"Ana tahmin PKL dosyası yüklenemedi, betik sonlandırılıyor. Hata: {e}")
        return

    all_evals_base_output_path = output_path / "gecici_degerlendirme_sonuclari"
    all_evals_base_output_path.mkdir(parents=True, exist_ok=True)

    results_list = []

    for conf_val in conf_thresholds_list:
        for nms_iou_val in nms_iou_thresholds_list:
            # `run_evaluation_from_pkl` logger olarak doğrudan logger nesnesini alır.
            try:
                metric = run_evaluation_from_pkl(
                    cfg_file, 
                    original_predictions, # Her döngüde aynı orijinal tahminleri kullan
                    conf_val, 
                    nms_iou_val, 
                    logger, # Ana logger'ı doğrudan ver
                    all_evals_base_output_path
                )
                results_list.append({
                    'confidence_threshold': conf_val,
                    'nms_iou_threshold': nms_iou_val,
                    'metric_value': metric 
                })
                # logger.info içinde zaten detaylı loglama yapılıyor (run_evaluation_from_pkl sonunda)
            except Exception as e:
                logger.error(f"PKL değerlendirme sırasında hata (Güven={conf_val}, NMS IoU={nms_iou_val}): {e}", exc_info=True)
                results_list.append({
                    'confidence_threshold': conf_val,
                    'nms_iou_threshold': nms_iou_val,
                    'metric_value': np.nan
                })

    results_dataframe = pd.DataFrame(results_list)
    csv_save_path = output_path / 'esik_analizi_pkl_metrikleri.csv'
    results_dataframe.to_csv(csv_save_path, index=False)
    logger.info(f"Sonuçlar CSV dosyasına kaydedildi: {csv_save_path}")

    # --- Grafik Çizimi (önceki betikle aynı) ---
    if not results_dataframe.empty:
        # Metrik vs. Güven Eşiği
        plt.figure(figsize=(12, 8))
        for nms_val_plot in nms_iou_thresholds_list: # nms_val -> nms_val_plot
            subset = results_dataframe[results_dataframe['nms_iou_threshold'] == nms_val_plot]
            if not subset.empty:
                plt.plot(subset['confidence_threshold'], subset['metric_value'], marker='o', label=f'NMS IoU = {nms_val_plot}')
        plt.xlabel('Güven Eşiği (SCORE_THRESH)')
        plt.ylabel('Performans Metriği (mAP/AP)')
        plt.title('Metrik vs. Güven Eşiği (Farklı NMS IoU Değerleri İçin) - PKL Verisi')
        plt.legend()
        plt.grid(True)
        plot_save_path_conf = output_path / 'metrik_vs_guven_esigi_pkl.png'
        plt.savefig(plot_save_path_conf)
        plt.close()
        logger.info(f"Grafik kaydedildi: {plot_save_path_conf}")

        # Metrik vs. NMS IoU Eşiği
        plt.figure(figsize=(12, 8))
        for conf_val_plot in conf_thresholds_list:
            subset = results_dataframe[results_dataframe['confidence_threshold'] == conf_val_plot]
            if not subset.empty:
                plt.plot(subset['nms_iou_threshold'], subset['metric_value'], marker='o', label=f'Güven = {conf_val_plot}')
        plt.xlabel('NMS IoU Eşiği')
        plt.ylabel('Performans Metriği (mAP/AP)')
        plt.title('Metrik vs. NMS IoU Eşiği (Farklı Güven Değerleri İçin) - PKL Verisi')
        plt.legend()
        plt.grid(True)
        plot_save_path_iou = output_path / 'metrik_vs_nms_iou_esigi_pkl.png'
        plt.savefig(plot_save_path_iou)
        plt.close()
        logger.info(f"Grafik kaydedildi: {plot_save_path_iou}")

        # Isı Haritası
        if len(conf_thresholds_list) > 1 and len(nms_iou_thresholds_list) > 1:
            try:
                pivot_data = results_dataframe.pivot(index='nms_iou_threshold', columns='confidence_threshold', values='metric_value')
                plt.figure(figsize=(10, 8))
                sns.heatmap(pivot_data, annot=True, fmt=".3f", cmap="viridis")
                plt.title('Performans Metriği Isı Haritası (mAP/AP) - PKL Verisi')
                plt.xlabel('Güven Eşiği (SCORE_THRESH)')
                plt.ylabel('NMS IoU Eşiği')
                heatmap_save_path = output_path / 'metrik_isi_haritasi_pkl.png'
                plt.savefig(heatmap_save_path)
                plt.close()
                logger.info(f"Isı haritası kaydedildi: {heatmap_save_path}")
            except Exception as e:
                logger.error(f"Isı haritası oluşturulamadı: {e}")
        else:
            logger.info("Isı haritası için yetersiz eşik değeri kombinasyonu (en az 2x2 gerekir).")
    else:
        logger.info("Çizilecek sonuç bulunamadı.")

    logger.info("PKL üzerinden eşik değeri analizi tamamlandı.")

if __name__ == '__main__':
    # Betiği çalıştırmadan önce SABİT PARAMETRELER bölümünü güncelleyin.
    main()