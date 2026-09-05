import numpy as np

def sample_frames_equidistant(frame_paths, num_frames):
    """
    Mengambil `num_frames` dari daftar path `frame_paths` secara ekuidistan
    (jarak waktu/temporal tersebar merata sepanjang video).
    
    Args:
        frame_paths (list of str): Daftar semua frame di suatu video.
        num_frames (int): Jumlah frame yang ingin diambil.
        
    Returns:
        list of str: Daftar frame yang sudah disampel.
    """
    total_frames = len(frame_paths)
    if total_frames == 0:
        return []
        
    if total_frames <= num_frames:
        return frame_paths
        
    # Menggunakan linspace untuk mendapatkan indeks yang tersebar rata
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    sampled_paths = [frame_paths[i] for i in indices]
    return sampled_paths
