import torch
from facenet_pytorch import MTCNN
from PIL import Image

class FaceDetector:
    def __init__(self, device=None, image_size=160, margin=0):
        """
        Wrapper untuk facenet_pytorch MTCNN.
        Args:
            device: 'cuda' atau 'cpu'
            image_size: Ukuran output face crop (VGGFace2 butuh 160x160)
            margin: Pixel padding di sekitar wajah
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # MTCNN instance
        self.mtcnn = MTCNN(
            image_size=image_size, 
            margin=margin,
            keep_all=False, # YTF biasanya ada 1 orang utama
            device=self.device,
            post_process=False # Kita post-process manual atau biarkan model yang handle
        )
        
    def detect_and_crop(self, image_path):
        """
        Mendeteksi wajah dan mengembalikan PyTorch Tensor dari hasil crop, 
        serta probabilitas dan bounding box.
        Returns:
            face_tensor: Tensor wajah yang telah dicrop.
            prob: Probabilitas deteksi MTCNN.
            box: Bounding box [x1, y1, x2, y2].
        """
        try:
            img = Image.open(image_path).convert('RGB')
            # Gunakan mtcnn.detect untuk mendapatkan bounding boxes dan probabilities
            boxes, probs = self.mtcnn.detect(img)
            
            if boxes is None or len(boxes) == 0:
                return None, None, None
                
            # Kita asumsikan wajah pertama (paling dominan) karena YTF biasanya 1 orang
            box = boxes[0]
            prob = probs[0]
            
            # mtcnn callable () return tensor cropped jika keep_all=False
            face_tensor = self.mtcnn(img)
            
            # Jika mtcnn(img) mereturn None karena suatu alasan
            if face_tensor is None:
                return None, None, None
                
            return face_tensor, prob, box
        except Exception as e:
            return None, None, None
