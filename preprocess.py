"""Image preparation for serving. crop_face() is the same logic as the Kaggle Cell 2 used for training."""
import io
import cv2
import numpy as np
from PIL import Image

IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAX_PIXELS = 40_000_000

_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") \
    if hasattr(cv2, "data") else None


def cascade_ok():
    return _cascade is not None and not _cascade.empty()


def crop_face(bgr):
    H, W = bgr.shape[:2]
    scale = min(1.0, 960 / max(H, W))
    small = cv2.resize(bgr, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA) if scale < 1 else bgr
    faces = _cascade.detectMultiScale(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 1.05, 3, minSize=(30, 30))
    if len(faces) == 0:
        side = min(H, W)
        y0 = max(0, (H - side) // 2); x0 = max(0, (W - side) // 2)
        crop = bgr[y0:y0+side, x0:x0+side]
        return cv2.resize(crop, (IMG_SIZE, IMG_SIZE)), True
    x, y, w, h = [int(round(v / scale)) for v in max(faces, key=lambda f: f[2]*f[3])]
    side = min(int(max(w, h) * 1.6), H, W)
    cx, cy = x + w//2, y + h//2
    x0, y0 = int(np.clip(cx - side//2, 0, W - side)), int(np.clip(cy - side//2, 0, H - side))
    crop = bgr[y0:y0+side, x0:x0+side]
    interp = cv2.INTER_AREA if side > IMG_SIZE else cv2.INTER_CUBIC
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=interp), False


def decode_image(data: bytes):
    """Validate an upload and return a BGR array (read the same way as cv2.imread in training)."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in ("JPEG", "PNG", "WEBP"):
                raise ValueError("Please upload a JPG, PNG or WebP image.")
            if im.width * im.height > MAX_PIXELS:
                raise ValueError("Image dimensions are too large (limit 40 megapixels).")
    except ValueError:
        raise
    except Exception:
        raise ValueError("This file is not a readable image.")
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("This file is not a readable image.")
    return bgr


def to_training_crop(face_bgr):
    """Training saved each crop as JPEG quality 95, then ImageFolder reopened it with PIL. Do the same."""
    ok, buf = cv2.imencode(".jpg", face_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("Could not process this image.")
    return Image.open(io.BytesIO(buf.tobytes())).convert("RGB")


def to_array(pil):
    """Same result as ToTensor() + Normalize() for a 224x224 image. Shape: 1x3x224x224."""
    x = (np.asarray(pil, dtype=np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])
