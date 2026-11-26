"""DeepGuard server.   Run:  python app.py   then open http://localhost:8000"""
import base64
import io
import os
import re
import threading
import time
import tempfile
from collections import defaultdict, deque
from pathlib import Path

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

import accounts
import preprocess
from explain import explain
from video_model import VideoPredictor # NEW IMPORT

BASE = Path(__file__).parent


def load_env(path=BASE / ".env"):
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()
MAX_BYTES = 10 * 1024 * 1024
MAX_VIDEO_BYTES = 50 * 1024 * 1024
COOKIE = "dg_session"
accounts.init()
# Score cut-offs are chosen by hand. They are NOT calibrated probabilities.
LOW = float(os.getenv("LOW_THRESHOLD", "0.30"))
HIGH = float(os.getenv("HIGH_THRESHOLD", "0.70"))
predictor = None
video_predictor = None # NEW GLOBAL
model_lock = threading.Lock()
hits = defaultdict(deque)


def tier(score, custom_threshold=None):
    if custom_threshold is not None:
        if score < custom_threshold - 0.1:
            return "low"
        elif score > custom_threshold + 0.1:
            return "high"
        return "uncertain"
        
    if score < LOW:
        return "low"
    return "high" if score > HIGH else "uncertain"


def rate_limit(request, name, per_minute):
    if os.getenv("SPACE_ID"):
        ip = request.headers.get("x-forwarded-for", "").split(",")[-1].strip() or "proxy"
    else:
        ip = request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "?")
    q, now = hits[(name, ip)], time.time()
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= per_minute:
        raise HTTPException(429, "Too many requests. Please wait a minute.")
    q.append(now)


def global_limit(name, per_minute):
    q, now = hits[(name, "everyone")], time.time()
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= per_minute:
        raise HTTPException(429, "The service is busy. Please wait a minute.")
    q.append(now)


def require_user(request):
    user = accounts.user_for(request.cookies.get(COOKIE))
    if not user:
        raise HTTPException(401, "Please log in.")
    return user


def require_same_site(request):
    if request.headers.get("x-deepguard") != "1":
        raise HTTPException(403, "Request blocked.")


def guard(request):
    require_same_site(request)
    return require_user(request)


async def read_files(request, fields=("file",), limit=MAX_BYTES):
    too_big = f"File is too large (limit {limit // 1024 // 1024} MB)."
    try:
        length = int(request.headers["content-length"])
    except (KeyError, ValueError):
        raise HTTPException(411, "Missing upload size.")
    if length > limit * len(fields) + 100_000:
        raise HTTPException(413, too_big)
    out = []
    async with request.form(max_files=len(fields), max_fields=2) as form:
        for field in fields:
            f = form.get(field)
            if f is None or isinstance(f, str):
                raise HTTPException(400, "No file was uploaded.")
            data = await f.read()
            if len(data) > limit:
                raise HTTPException(413, too_big)
            out.append((f.filename or "upload", data))
    return out


async def decode(data):
    try:
        return await run_in_threadpool(preprocess.decode_image, data)
    except ValueError as e:
        raise HTTPException(422, str(e))


def run_model(bgr):
    face, no_face = preprocess.crop_face(bgr)
    if no_face:
        return {"status": "no_face"}
    crop = preprocess.to_training_crop(face)
    with model_lock:
        score, overlay = predictor.predict(crop)

    def b64(img):
        out = io.BytesIO()
        img.save(out, "PNG")
        return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()

    return {"status": "ok", "score": round(score, 4), "tier": tier(score),
            "face": b64(crop), "attention": b64(overlay), "model": predictor.version}


def record(user, kind, filename, result):
    if result["status"] == "ok":
        accounts.add_history(user["id"], kind, filename, result["tier"], result["score"], result["model"])


async def analyze(request):
    user = guard(request)
    rate_limit(request, "analyze", 20)
    [(name, data)] = await read_files(request)
    bgr = await decode(data)
    t0 = time.perf_counter()
    result = await run_in_threadpool(run_model, bgr)
    result["ms"] = int((time.perf_counter() - t0) * 1000)
    record(user, "image", name, result)
    return JSONResponse(result)


async def commentary(request):
    guard(request)
    rate_limit(request, "explain", 5)
    global_limit("explain", 20)
    [(_, data)] = await read_files(request)
    bgr = await decode(data)
    try:
        return JSONResponse(await run_in_threadpool(explain, bgr))
    except RuntimeError as e:
        raise HTTPException(502, str(e))


async def compare(request):
    user = guard(request)
    rate_limit(request, "compare", 10)
    files = await read_files(request, fields=("file_a", "file_b"))
    images = [await decode(data) for _, data in files]
    results = [await run_in_threadpool(run_model, bgr) for bgr in images]
    for (name, _), result in zip(files, results):
        record(user, "compare", name, result)
    both = all(r["status"] == "ok" for r in results)
    return JSONResponse({"a": results[0], "b": results[1], "identical_files": files[0][1] == files[1][1],
                         "difference": round(abs(results[0]["score"] - results[1]["score"]), 4) if both else None})


async def video(request):
    user = guard(request)
    rate_limit(request, "video", 4)
    [(name, data)] = await read_files(request, limit=MAX_VIDEO_BYTES)
    
    # Save uploaded video to a temporary file
    def process_video(data):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            return video_predictor.predict_video(tmp_path)
        finally:
            os.unlink(tmp_path)
            
    try:
        result = await run_in_threadpool(process_video, data)
    except Exception as e:
        raise HTTPException(500, f"Video processing error: {str(e)}")

    if result.get("status") != "ok":
        raise HTTPException(422, result.get("error", "Could not process video."))
        
    score = result["score"]
    threshold = result.get("threshold", 0.5)
    
    result["tier"] = tier(score, custom_threshold=threshold)
    result["model"] = result.get("version", "Video Model")
    result["face_frames"] = 16
    result["sampled"] = 16
    
    # Mock the frames array for the UI chart
    result["frames"] = [{"time": i, "score": score} for i in range(16)]
    
    record(user, "video", name, result)
    return JSONResponse(result)


async def read_json(request):
    try:
        if int(request.headers["content-length"]) > 4096:
            raise HTTPException(413, "Request too large.")
        body = await request.json()
        assert isinstance(body, dict)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "Bad request.")
    return {k: v for k, v in body.items() if isinstance(v, str)}


def signed_in(request, user_id):
    token = accounts.start_session(user_id)
    response = JSONResponse({"ok": True})
    https = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    response.set_cookie(COOKIE, token, max_age=accounts.SESSION_SECONDS, httponly=True,
                        samesite="strict", secure=https, path="/")
    return response


async def register(request):
    require_same_site(request)
    rate_limit(request, "register", 5)
    body = await read_json(request)
    name = body.get("name", "").strip()
    email = body.get("email", "").strip().lower()
    password = body.get("password", "")
    if not 2 <= len(name) <= 80:
        raise HTTPException(422, "Enter your name (2 to 80 characters).")
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email) or len(email) > 200:
        raise HTTPException(422, "Enter a valid email address.")
    if not 8 <= len(password) <= 128:
        raise HTTPException(422, "Password must be 8 to 128 characters.")
    user_id = await run_in_threadpool(accounts.create_user, name, email, password)
    if user_id is None:
        raise HTTPException(409, "An account with this email already exists.")
    return signed_in(request, user_id)


async def login(request):
    require_same_site(request)
    rate_limit(request, "login", 8)
    body = await read_json(request)
    user_id = await run_in_threadpool(accounts.check_login, body.get("email", "").strip().lower(),
                                      body.get("password", "")[:128])
    if user_id is None:
        raise HTTPException(401, "Incorrect email or password.")
    return signed_in(request, user_id)


async def logout(request):
    require_same_site(request)
    token = request.cookies.get(COOKIE)
    if token:
        accounts.end_session(token)
    response = JSONResponse({"ok": True})
    response.delete_cookie(COOKIE, path="/")
    return response


async def me(request):
    return JSONResponse(require_user(request))


async def history(request):
    return JSONResponse(accounts.get_history(require_user(request)["id"]))


async def history_delete(request):
    user = guard(request)
    if not accounts.delete_history(user["id"], request.path_params["item_id"]):
        raise HTTPException(404, "Record not found.")
    return JSONResponse({"ok": True})


async def health(request):
    return JSONResponse({"ready": predictor is not None,
                         "model": getattr(predictor, "version", None),
                         "device": str(getattr(predictor, "device", "")),
                         "commentary": bool(os.getenv("GROQ_API_KEY"))})


async def index(request):
    return FileResponse(BASE / "static" / "index.html")


async def on_error(request, exc):
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


app = Starlette(routes=[
    Route("/", index),
    Route("/api/health", health),
    Route("/api/analyze", analyze, methods=["POST"]),
    Route("/api/explain", commentary, methods=["POST"]),
    Route("/api/compare", compare, methods=["POST"]),
    Route("/api/video", video, methods=["POST"]),
    Route("/api/register", register, methods=["POST"]),
    Route("/api/login", login, methods=["POST"]),
    Route("/api/logout", logout, methods=["POST"]),
    Route("/api/me", me),
    Route("/api/history", history),
    Route("/api/history/{item_id:int}", history_delete, methods=["DELETE"]),
    Mount("/static", StaticFiles(directory=BASE / "static")),
], exception_handlers={HTTPException: on_error})


def main():
    global predictor, video_predictor
    import uvicorn
    from model import Predictor
    if not preprocess.cascade_ok():
        raise SystemExit("OpenCV face detector file is missing. Install the versions in requirements.txt.")
        
    # Load Image Model
    img_path = Path(os.getenv("MODEL_PATH", BASE / "checkpoints" / "deepguard_image_best.pth"))
    if not img_path.exists():
        raise SystemExit(f"Image model file not found: {img_path}")
    predictor = Predictor(img_path)
    print(f"Loaded {predictor.version} on {predictor.device} (checkpoint val AUC: {predictor.val_auc})")

    # Load Video Model
    vid_path = Path(BASE / "checkpoints" / "deepguard_video_deployment.pth")
    if not vid_path.exists():
        raise SystemExit(f"Video model file not found: {vid_path}")
    video_predictor = VideoPredictor(vid_path)
    print(f"Loaded {video_predictor.version} on {video_predictor.device} (Test AUC: {video_predictor.test_auc:.4f})")

    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()