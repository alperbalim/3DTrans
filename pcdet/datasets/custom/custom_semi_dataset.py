import copy
import pickle
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm
from ...utils import box_utils, common_utils, object3d_custom
from ..semi_dataset import SemiDatasetTemplate


def split_custom_semi_data(dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, save_path=None):
    """
    Semi-KITTI formatına uygun olacak şekilde Custom veri setini eğitim, doğrulama ve test bölümlerine böler.
    
    Args:
        dataset: Veri setinin tam listesi veya dizini.
        train_ratio: Eğitim verisi oranı.
        val_ratio: Doğrulama verisi oranı.
        test_ratio: Test verisi oranı.
        save_path: Bölünmüş veri seti bilgilerini kaydetmek için kullanılacak dizin.
    """
    if not (train_ratio + val_ratio + test_ratio == 1.0):
        raise ValueError("train_ratio, val_ratio ve test_ratio toplamı 1.0 olmalıdır.")
    
    total_samples = len(dataset)
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    train_samples = dataset[:train_end]
    val_samples = dataset[train_end:val_end]
    test_samples = dataset[val_end:]

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
        # Eğitim, doğrulama ve test bölümleri için dosya listelerini kaydedin
        with open(os.path.join(save_path, 'train.txt'), 'w') as f:
            for item in train_samples:
                f.write(f"{item}\n")

        with open(os.path.join(save_path, 'val.txt'), 'w') as f:
            for item in val_samples:
                f.write(f"{item}\n")

        with open(os.path.join(save_path, 'test.txt'), 'w') as f:
            for item in test_samples:
                f.write(f"{item}\n")
    
    print(f"Toplam örnek sayısı: {total_samples}")
    print(f"Eğitim seti boyutu: {len(train_samples)}")
    print(f"Doğrulama seti boyutu: {len(val_samples)}")
    print(f"Test seti boyutu: {len(test_samples)}")

    return train_samples, val_samples, test_samples



class CustomSemiDataset(SemiDatasetTemplate):
    def __init__(self, dataset_cfg, class_names, infos=None, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None
        self.custom_infos = infos

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists()
        points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
        return points

    def get_label(self, idx):
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()
        return object3d_custom.get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return calibration_custom.Calibration(calib_file, False)

    def __len__(self):
        return len(self.custom_infos) if not self._merge_all_iters_to_one_epoch else len(self.custom_infos) * self.total_epochs

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.custom_infos)

        info = copy.deepcopy(self.custom_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        calib = self.get_calib(sample_idx)

        input_dict = {
            'db_flag': "custom",
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_custom_camera_to_lidar(gt_boxes_camera, calib)
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        points = self.get_lidar(sample_idx)
        if self.dataset_cfg.get('SHIFT_COOR', None):
            points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
        input_dict['points'] = points

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

class CustomPretrainDataset(CustomSemiDataset):
    def __init__(self, dataset_cfg, class_names, infos=None, training=True, root_path=None, logger=None):
        assert training is True
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, infos=infos, training=training, root_path=root_path, logger=logger)

class CustomLabeledDataset(CustomSemiDataset):
    def __init__(self, dataset_cfg, class_names, infos=None, training=True, root_path=None, logger=None):
        assert training is True
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, infos=infos, training=training, root_path=root_path, logger=logger)

class CustomUnlabeledDataset(CustomSemiDataset):
    def __init__(self, dataset_cfg, class_names, infos=None, training=True, root_path=None, logger=None):
        assert training is True
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, infos=infos, training=training, root_path=root_path, logger=logger)

    def __getitem__(self, index):
        data_dict = super().__getitem__(index)
        # Etiketsiz verilerde, etiket bilgilerini kaldırıyoruz
        data_dict.pop('gt_boxes', None)
        data_dict.pop('gt_names', None)
        return data_dict

class CustomTestDataset(CustomSemiDataset):
    def __init__(self, dataset_cfg, class_names, infos=None, training=False, root_path=None, logger=None):
        assert training is False
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, infos=infos, training=training, root_path=root_path, logger=logger)

    def __getitem__(self, index):
        data_dict = super().__getitem__(index)
        # Test setinde, etiket bilgilerini kaldırıyoruz
        data_dict.pop('gt_boxes', None)
        data_dict.pop('gt_names', None)
        return data_dict
