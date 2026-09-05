import scipy.io as sio
import os

def explore_mat_file(file_path):
    print(f"\n--- Membaca file: {os.path.basename(file_path)} ---")
    try:
        data = sio.loadmat(file_path)
        keys = [k for k in data.keys() if not k.startswith('__')]
        print(f"Variabel yang tersedia: {keys}")
        
        # Tampilkan ukuran atau sebagian kecil data untuk setiap variabel
        for key in keys:
            val = data[key]
            # Mendapatkan tipe data atau ukuran
            if hasattr(val, 'shape'):
                print(f" - {key}: array berukuran {val.shape}")
            else:
                print(f" - {key}: {type(val)}")
                
    except Exception as e:
        print(f"Gagal membaca {file_path}: {e}")

if __name__ == "__main__":
    base_dir = r"g:\Penelitian S3\dataset ytf\meta_data"
    
    mat_files = [
        "meta_and_splits.mat",
        "LLC_and_MBGS_bases.align.mat",
        "LLC_and_MBGS_bases.not_align.mat"
    ]
    
    for filename in mat_files:
        full_path = os.path.join(base_dir, filename)
        explore_mat_file(full_path)
