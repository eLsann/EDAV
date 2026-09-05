import torch
import torch.nn.functional as F

def compute_pairwise_cosine(emb1, emb2):
    """
    Menghitung cosine similarity untuk sepasang video.
    Args:
        emb1: Tensor embeddings dari video 1, shape (N, 512)
        emb2: Tensor embeddings dari video 2, shape (M, 512)
    Returns:
        float: Rata-rata dari semua skor similarity antar frame
    """
    if emb1 is None or emb2 is None or len(emb1) == 0 or len(emb2) == 0:
        return 0.0 # Atau nilai threshold terendah
        
    # Pastikan emb1 dan emb2 berada di device yang sama
    emb1 = emb1.to('cpu')
    emb2 = emb2.to('cpu')
    
    # Hitung similarity matrix berukuran (N, M)
    # Cosine similarity antara vektor x dan y (yg sdh dinormalisasi) adalah perkalian dot
    # Namun F.cosine_similarity lebih aman
    
    # Unsqueeze agar dapat di-broadcast
    # emb1 shape: (N, 1, 512)
    # emb2 shape: (1, M, 512)
    emb1_exp = emb1.unsqueeze(1)
    emb2_exp = emb2.unsqueeze(0)
    
    # Hitung similarity di dimensi fitur (dim=2)
    sim_matrix = F.cosine_similarity(emb1_exp, emb2_exp, dim=2)
    
    # Rata-rata dari seluruh pasangan N x M frame
    mean_sim = torch.mean(sim_matrix).item()
    return mean_sim
