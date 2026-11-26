# video_model.py
import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from pathlib import Path

class DeepGuardVideoModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Must match the Kaggle training architecture exactly
        self.cnn = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)

    def forward(self, clips):
        B, T, C, H, W = clips.shape
        flat = clips.reshape(B * T, C, H, W)
        # Process in chunks of 4 to prevent OOM during inference
        chunks = [self.cnn(flat[s:s+4]) for s in range(0, flat.size(0), 4)]
        logits = torch.cat(chunks, dim=0).view(B, T) 
        # Average logits across the 16 frames
        return logits.mean(dim=1, keepdim=True) 

class VideoPredictor:
    def __init__(self, checkpoint_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model = DeepGuardVideoModel()
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()
        
        # Load thresholds from the Kaggle export
        self.threshold = ckpt.get("validation_eer_threshold", 0.5)
        self.version = f"EfficientNet-B0 Video (Epoch {ckpt.get('epoch', '?')})"
        self.test_auc = ckpt.get("test_video_auc", 0.0)

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def predict_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 16:
            return {"error": "Video is too short (minimum 16 frames required)."}

        # Extract 16 frames from the middle (matching training logic)
        start_frame = max(0, (total_frames // 2) - 8)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        for _ in range(16):
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()

        if len(frames) != 16:
            return {"error": "Could not read enough frames from the video."}

        # Use the middle frame to find the face, then apply that crop to all 16 frames
        # This prevents jitter and matches how we handled the Kaggle extraction
        mid = frames[8]
        H, W = mid.shape[:2]
        
        # Detect face on middle frame
        gray = cv2.cvtColor(mid, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, 1.08, 4, minSize=(30, 30))
        
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            side = int(max(w, h) * 1.7)
            cx, cy = x + w // 2, y + h // 2
            x0 = max(0, cx - side // 2)
            y0 = max(0, cy - side // 2)
            side = min(side, W - x0, H - y0)
            crop_box = (x0, y0, side)
        else:
            # Fallback center crop
            side = min(H, W)
            crop_box = ((W - side) // 2, (H - side) // 2, side)

        x0, y0, side = crop_box
        clip = np.empty((16, 224, 224, 3), dtype=np.uint8)
        
        for i, f in enumerate(frames):
            crop = f[y0:y0+side, x0:x0+side]
            crop_rgb = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB)
            clip[i] = crop_rgb

        # Normalize
        tensor = torch.from_numpy(clip).permute(0, 3, 1, 2).float().div_(255.0)
        tensor = (tensor - self.mean) / self.std
        tensor = tensor.unsqueeze(0).to(self.device) # Shape: (1, 16, 3, 224, 224)

        with torch.no_grad():
            logits = self.model(tensor)
            score = torch.sigmoid(logits).item()

        return {
            "status": "ok",
            "score": round(score, 4),
            "threshold": self.threshold,
            "version": self.version,
            "test_auc": self.test_auc
        }