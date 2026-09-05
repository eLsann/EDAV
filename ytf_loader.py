import os
import scipy.io as sio

class YTFDataLoader:
    def __init__(self, mat_file_path):
        """
        Inisialisasi loader untuk dataset YouTube Faces (YTF).
        
        Args:
            mat_file_path (str): Path lengkap menuju file 'meta_and_splits.mat'
        """
        if not os.path.exists(mat_file_path):
            raise FileNotFoundError(f"File tidak ditemukan: {mat_file_path}")
            
        print(f"Memuat metadata dari {mat_file_path}...")
        self.data = sio.loadmat(mat_file_path)
        
        # Mengekstrak nama-nama video dan menghapus list berlebih (karena format MATLAB cell array)
        # Struktur asli names biasanya berupa nested array seperti: [[['Name/0']], [['Name/1']]]
        raw_video_names = self.data['video_names']
        self.video_names = [name[0][0] for name in raw_video_names]
        
        # Mengekstrak splits (berukuran 500 x 3 x 10)
        # 500 baris, 3 kolom (index_video1, index_video2, label), 10 fold
        self.splits = self.data['Splits']
        
        print(f"Berhasil memuat! Total video: {len(self.video_names)}, Total Fold/Splits: {self.splits.shape[2]}")

    def get_video_name(self, index):
        """
        Mendapatkan nama video berdasarkan index MATLAB.
        (Index MATLAB dimulai dari 1, sedangkan Python mulai dari 0)
        """
        # Kurangi index dengan 1 karena index di MATLAB dimulai dari 1
        py_index = int(index) - 1
        return self.video_names[py_index]

    def get_fold(self, fold_idx):
        """
        Mendapatkan data pasangan (pairs) untuk fold tertentu.
        
        Args:
            fold_idx (int): Index fold (1 sampai 10)
            
        Returns:
            list of dict: Berisi pasangan video1, video2, dan label (1 untuk sama, 0 untuk beda)
        """
        if fold_idx < 1 or fold_idx > 10:
            raise ValueError("fold_idx harus antara 1 dan 10")
            
        # Mengambil data untuk fold tertentu (karena Python 0-based, kurangi fold_idx dengan 1)
        fold_data = self.splits[:, :, fold_idx - 1]
        
        pairs = []
        for row in fold_data:
            idx1 = row[0]
            idx2 = row[1]
            label = row[2]
            
            video1_name = self.get_video_name(idx1)
            video2_name = self.get_video_name(idx2)
            
            pairs.append({
                'video1': video1_name,
                'video2': video2_name,
                'label': label
            })
            
        return pairs

if __name__ == "__main__":
    # Contoh penggunaan
    mat_path = r"g:\Penelitian S3\dataset ytf\meta_data\meta_and_splits.mat"
    
    loader = YTFDataLoader(mat_path)
    
    # Ambil data untuk Split / Fold 1
    fold1_pairs = loader.get_fold(1)
    
    print("\nMenampilkan 5 pasangan pertama dari Fold 1:")
    for i in range(5):
        pair = fold1_pairs[i]
        status = "SAMA" if pair['label'] == 1 else "BEDA"
        print(f"{i+1}. {pair['video1']} & {pair['video2']} => Label: {status}")
