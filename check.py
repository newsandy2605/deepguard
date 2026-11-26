"""Score images from the command line:   python check.py photo1.jpg photo2.jpg"""
import sys
from pathlib import Path

import preprocess
from app import tier
from model import Predictor

p = Predictor(Path(__file__).parent / "checkpoints" / "deepguard_image_best.pth")
for name in sys.argv[1:]:
    face, no_face = preprocess.crop_face(preprocess.decode_image(Path(name).read_bytes()))
    if no_face:
        print(f"{name}: no face detected")
        continue
    score, _ = p.predict(preprocess.to_training_crop(face))
    print(f"{name}: score={score:.4f} tier={tier(score)}")
