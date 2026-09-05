import cv2
import numpy as np
from PIL import Image

def calculate_blur_laplacian(image_path, box=None):
    """
    Menghitung tingkat ketajaman/blurriness menggunakan Variance of Laplacian.
    Nilai lebih tinggi = Lebih tajam. Nilai rendah = Blur.
    Args:
        image_path: Path ke file gambar
        box: Bounding box wajah (opsional) untuk menghitung blur hanya di area wajah
    """
    try:
        # Gunakan cv2 untuk baca gambar (grayscale)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
            
        if box is not None:
            # Crop berdasarkan box
            x1, y1, x2, y2 = [int(b) for b in box]
            
            # Bound checks
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                img = img[y1:y2, x1:x2]
                
        # Laplacian variance
        return cv2.Laplacian(img, cv2.CV_64F).var()
    except Exception:
        return 0.0

def calculate_face_resolution(box):
    """
    Menghitung ukuran wajah (luas area bounding box).
    """
    if box is None:
        return 0.0
    
    x1, y1, x2, y2 = box
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return width * height

def normalize_quality_features(confidence, laplacian_var, face_area):
    """
    Menggabungkan atau menormalisasi fitur kualitas menjadi satu nilai Q [0, 1].
    Secara empiris kita bisa menormalisasi masing-masing dan merata-ratakannya.
    (Pendekatan dasar, nantinya bisa dipelajari via model).
    """
    # 1. Confidence langsung di rentang [0, 1]
    norm_conf = max(0.0, min(1.0, confidence))
    
    # 2. Laplacian variance (empiris: < 10 sangat blur, > 100 tajam)
    norm_blur = min(1.0, laplacian_var / 200.0) 
    
    # 3. Face Area (empiris: < 1000 pixel kecil, > 10000 cukup besar)
    norm_area = min(1.0, face_area / 40000.0) # misal 200x200 pixel max
    
    # Kualitas gabungan = Rata-rata dari ketiganya
    q = (norm_conf + norm_blur + norm_area) / 3.0
    return q
