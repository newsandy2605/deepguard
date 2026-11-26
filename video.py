"""EXPERIMENTAL: runs the IMAGE model on sampled frames. This is not a video model: it ignores motion
and sound, and it has not been evaluated on videos."""
import os
import statistics
import tempfile

import cv2
import numpy as np

import preprocess

SAMPLES = 16
MIN_FACE_FRAMES = 5
MAX_SECONDS = 120


def analyze_video(data, score_face):
    """score_face(pil_crop) -> score from 0 to 1."""
    fd, path = tempfile.mkstemp(suffix=".video")
    cap = None
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if not cap.isOpened() or fps <= 0 or total <= 0:
            raise ValueError("Could not read this video. Try an MP4 file.")
        if width * height > 3840 * 2160:
            raise ValueError("Video resolution is too high (maximum 4K).")
        if total / fps > MAX_SECONDS:
            raise ValueError(f"Video is too long (maximum {MAX_SECONDS} seconds).")
        frames, sampled = [], 0
        for index in sorted(set(np.linspace(0, total - 1, min(SAMPLES, total)).astype(int).tolist())):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                continue
            sampled += 1
            face, no_face = preprocess.crop_face(frame)
            if not no_face:
                score = score_face(preprocess.to_training_crop(face))
                frames.append({"time": round(index / fps, 2), "score": round(score, 4)})
        if len(frames) < MIN_FACE_FRAMES:
            return {"status": "not_enough_faces", "sampled": sampled, "face_frames": len(frames)}
        return {"status": "ok", "score": round(statistics.median(f["score"] for f in frames), 4),
                "sampled": sampled, "face_frames": len(frames), "frames": frames,
                "duration": round(total / fps, 1)}
    finally:
        if cap is not None:
            cap.release()
        os.unlink(path)
