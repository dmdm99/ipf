import io, os
from typing import Optional, Dict
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from ultralytics import YOLO
from PIL import Image
import numpy as np

app = FastAPI(title="Armament Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = str((ROOT / "runs" / "detect" / "bdipf_n_512_from640_last" / "weights" / "best.pt").resolve())
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
print(f"[startup] Loading model from: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at: {MODEL_PATH}")

DEFAULT_CONF  = float(os.getenv("CONF",  "0.25"))
DEFAULT_IOU   = float(os.getenv("IOU",   "0.50"))
DEFAULT_IMGSZ = int(os.getenv("IMGSZ",  "960"))

try:
    model = YOLO(MODEL_PATH)
    CLASS_NAMES = model.model.names if hasattr(model.model, "names") else model.names
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

@app.get("/")
def root():
    return {"message": "API up. Go to /health or /docs"}

@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.basename(MODEL_PATH), "classes": CLASS_NAMES}

class PredictResponse(BaseModel):
    has_armament: bool
    total_detections: int
    per_class: Dict[str, int]
    width: int
    height: int
    conf_used: float
    iou_used: float

def _read_image_from_upload(file: UploadFile) -> Image.Image:
    content = file.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        return Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

def _read_image_from_url(url: str) -> Image.Image:
    import requests
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: Optional[UploadFile] = File(None),
    url: Optional[HttpUrl] = Query(None),
    conf: float = Query(DEFAULT_CONF, ge=0.0, le=1.0),
    iou: float = Query(DEFAULT_IOU,  ge=0.0, le=1.0),
    imgsz: int = Query(DEFAULT_IMGSZ, ge=64, le=2048),
):
    if file is None and url is None:
        raise HTTPException(status_code=400, detail="Provide either file or url")

    img = _read_image_from_upload(file) if file else _read_image_from_url(str(url))
    width, height = img.size

    try:
        res = model.predict(
            source=np.array(img),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            verbose=False
        )[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    per_class: Dict[str, int] = {}
    if res.boxes is not None and len(res.boxes) > 0:
        clss = res.boxes.cls.cpu().numpy().astype(int)
        for k in clss:
            name = CLASS_NAMES.get(k, str(k)) if isinstance(CLASS_NAMES, dict) else (
                CLASS_NAMES[k] if 0 <= k < len(CLASS_NAMES) else str(k)
            )
            per_class[name] = per_class.get(name, 0) + 1

    has_armament = sum(per_class.values()) > 0

    return PredictResponse(
        has_armament=has_armament,
        total_detections=sum(per_class.values()),
        per_class=per_class,
        width=width,
        height=height,
        conf_used=conf,
        iou_used=iou
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8010")))
