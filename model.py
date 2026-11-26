"""EfficientNet-B0 classifier. The architecture must match training (Cell 3) exactly."""
import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image

from preprocess import to_array


class DeepGuardImageModel(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=False)
        nf = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(nf, 256), nn.ReLU(),
            nn.Dropout(dropout * 0.5), nn.Linear(256, 1))

    def forward(self, x):
        return self.backbone(x)


class Predictor:
    def __init__(self, path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # weights_only=False runs code stored in the file: only load a checkpoint you trained yourself.
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.model = DeepGuardImageModel()
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.to(self.device).eval()
        self.version = f"EfficientNet-B0, epoch {ck.get('epoch', '?')}"
        self.val_auc = float(ck["val_auc"]) if "val_auc" in ck else None

    def predict(self, face_pil):
        """Returns (score for 'fake' from 0 to 1, attention overlay as a PIL image)."""
        x = torch.from_numpy(to_array(face_pil)).to(self.device)
        saved = {}
        hook = self.model.backbone.conv_head.register_forward_hook(
            lambda m, i, o: saved.__setitem__("a", o))
        try:
            logit = self.model(x)
            grad = torch.autograd.grad(logit.sum(), saved["a"])[0]
        finally:
            hook.remove()
        score = torch.sigmoid(logit).item()
        cam = torch.relu((grad.mean(dim=(2, 3), keepdim=True) * saved["a"]).sum(1))[0]
        cam = cam.detach().cpu().numpy().astype(np.float32)
        cam = cv2.resize(cam / (cam.max() + 1e-8), (224, 224))
        heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
        overlay = 0.4 * heat + 0.6 * np.asarray(face_pil, dtype=np.float32)
        return score, Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
