import sys
sys.path.insert(0, '/root/3DTrans/')
sys.path.insert(0, '/root/3DTrans/tools')
import os
import re
from collections import defaultdict

def parse_inference_time(line):
    """Log satırından inference zamanını ayrıştırır."""
    match = re.search(r'sec_per_example: ([\d\.]+) second', line, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None

def calculate_average_inference_times(root_dir):
    """Log dosyalarını tarar ve modeller için ortalama inference sürelerini hesaplar."""
    inference_times = defaultdict(list)
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.txt'):  # Sadece .log uzantılı dosyaları incele
                log_path = os.path.join(dirpath, filename)
                with open(log_path, 'r', encoding='utf-8') as log_file:
                    current_model = None
                    for line in log_file:
                        if "P4000" in line.upper():
                            continue
                        # Modeli tespit et
                        if "PVRCNN" in line.upper():
                            current_model = "PVRCNN"
                            #print(current_model)
                        elif "VOXELRCNN" in line.upper():
                            current_model = "VoxelRCNN"
                            #print(current_model)
                        
                        # Inference zamanını tespit et
                        if current_model:
                            time = parse_inference_time(line)
                            if time is not None:
                                inference_times[current_model].append(time)
    
    # Ortalama süreleri hesapla
    average_times = {}
    for model, times in inference_times.items():
        if times:
            average_times[model] = sum(times) / len(times)
        else:
            average_times[model] = None
    
    return average_times

# Ana klasör yolunu buraya yazın
root_directory = '/root/3DTrans/output/'  # Burayı kendi ana klasör yolunuzla değiştirin.

# Ortalama inference sürelerini hesapla
average_times = calculate_average_inference_times(root_directory)

# Sonuçları yazdır
for model, avg_time in average_times.items():
    if avg_time is not None:
        print(f"{model} Modelinin Ortalama Inference Süresi: {avg_time:.2f} ms")
    else:
        print(f"{model} Modeli için inference zamanı bulunamadı.")
