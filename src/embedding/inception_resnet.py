import torch
from facenet_pytorch import InceptionResnetV1

class FaceEmbedder:
    def __init__(self, device=None, pretrained='vggface2'):
        """
        Wrapper untuk InceptionResnetV1 dari facenet_pytorch.
        Args:
            pretrained: 'vggface2' atau 'casia-webface'
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Model di set ke mode eval (evaluasi)
        self.model = InceptionResnetV1(pretrained=pretrained).to(self.device).eval()
        
    def preprocess_tensor(self, face_tensors):
        """
        Mengubah rentang nilai tensor dari [0, 255] menjadi [-1, 1] 
        standar input InceptionResnetV1 jika pre_process=False di MTCNN.
        """
        # Standarisasi nilai pixel: mean = 127.5, std = 128.0
        return (face_tensors.float() - 127.5) / 128.0

    @torch.no_grad()
    def get_embeddings(self, face_tensors):
        """
        Menghasilkan 512D embeddings.
        Args:
            face_tensors: Tensor of shape (B, C, H, W)
        Returns:
            embeddings: Tensor of shape (B, 512)
        """
        # Pindahkan tensor ke device yang tepat
        face_tensors = face_tensors.to(self.device)
        
        # Preprocessing (scale ke [-1, 1])
        processed_tensors = self.preprocess_tensor(face_tensors)
        
        # Forward pass
        embeddings = self.model(processed_tensors)
        
        # Normalisasi embedding L2
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings
