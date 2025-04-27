import os
import re
from datetime import datetime, timedelta

def parse_log_time(line):
    """Log satırından zamanı ayrıştırır."""
    match = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', line)
    if match:
        return datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S,%f')
    return None

def calculate_log_duration(log_file_path):
    """Bir log dosyasındaki toplam süreyi hesaplar."""
    start_time = None
    end_time = None
    
    with open(log_file_path, 'r', encoding='utf-8') as log_file:
        for line in log_file:
            if "**********************Start training" in line:
                start_time = parse_log_time(line)
            if "**********************End training" in line:
                end_time = parse_log_time(line)
    
    if start_time and end_time:
        return end_time - start_time
    return timedelta(0)

def calculate_total_duration(root_dir):
    """Verilen klasör ve alt klasörlerdeki log dosyalarının toplam süresini hesaplar."""
    total_duration = timedelta(0)
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.txt'):  # Sadece .log uzantılı dosyaları incele
                log_path = os.path.join(dirpath, filename)
                duration = calculate_log_duration(log_path)
                total_duration += duration
                #print(f"{log_path}: {duration}")
    
    return total_duration

# Ana klasör yolunu buraya yazın
root_directory = '/root/3DTrans/output'  # Burayı kendi ana klasör yolunuzla değiştirin.

# Toplam süreyi hesapla
total_time = calculate_total_duration(root_directory)
print(f"Tüm Log Dosyalarının Toplam Çalışma Süresi: {total_time}")
