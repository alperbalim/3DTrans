import pickle
import numpy as np

# PKL dosyasını yüklemek
pkl_file = '/root/3DTrans/data/custom_kitti2/custom_infos_train.pkl'  # PKL dosyasının yolunu buraya girin

with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

# Car sınıfı için boyutları toplamak
car_lengths = []
car_widths = []
car_heights = []

# Veri yapısı, nesneler için bounding box bilgilerini içerir
for item in data:
    for label in item['annos']['name']:  # 'name' etiketi nesne sınıfını verir
        if label == 'Car':  # Sadece 'Car' sınıfına odaklanıyoruz
            index = list(item['annos']['name']).index(label)
            dimensions = item['annos']['dimensions'][index]  # Boyutlar (length, width, height)
            car_lengths.append(dimensions[0])
            car_widths.append(dimensions[1])
            car_heights.append(dimensions[2])

# Ortalamaları hesaplama
mean_length = np.mean(car_lengths)
mean_width = np.mean(car_widths)
mean_height = np.mean(car_heights)

print(f"Car Sınıfı için Ortalama Boyutlar (Uzunluk, Genişlik, Yükseklik):")
print(f"Uzunluk: {mean_length:.2f}, Genişlik: {mean_width:.2f}, Yükseklik: {mean_height:.2f}")
