import os
import json

def check_lidar_top_files(meta_path, data_path):
    """
    NuScenes meta dosyalarını kullanarak LIDAR_TOP dosyalarının eksik olup olmadığını kontrol eder.
    
    Args:
        meta_path (str): Meta dosyalarının bulunduğu dizin.
        data_path (str): LIDAR_TOP dosyalarının bulunduğu kök dizin.
        
    Returns:
        eksik_dosyalar (list): Eksik LIDAR_TOP dosyalarının listesi.
    """
    sample_data_file = os.path.join(meta_path, 'sample_data.json')

    # sample_data.json dosyasını yükle
    with open(sample_data_file, 'r') as f:
        sample_data = json.load(f)

    eksik_dosyalar = []

    for item in sample_data:
        if item['fileformat'] == 'pcd' and item['filename'].split("/")[1] == 'LIDAR_TOP':
            lidar_top_filepath = os.path.join(data_path, item['filename'])
            if not os.path.exists(lidar_top_filepath):
                eksik_dosyalar.append(lidar_top_filepath)
                print(lidar_top_filepath)
                exit()

    return eksik_dosyalar

if __name__ == "__main__":
    # Meta dosyalarının bulunduğu dizin
    meta_path = '/root/3DTrans/data/nuscenes/v1.0-trainval/v1.0-trainval'
    
    # LIDAR_TOP dosyalarının bulunduğu kök dizin
    data_path = '/root/3DTrans/data/nuscenes/v1.0-trainval'
    
    # LIDAR_TOP dosyalarının eksik olup olmadığını kontrol etme
    eksik_dosyalar = check_lidar_top_files(meta_path, data_path)
    
    if eksik_dosyalar:
        print("Eksik LIDAR_TOP dosyaları:")
        for dosya in eksik_dosyalar:
            print(dosya)
    else:
        print("Tüm LIDAR_TOP dosyaları mevcut.")
