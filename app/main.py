import io, os, base64
from typing import Optional, List, Dict
from pathlib import Path  # NEW: נשתמש כדי לבנות נתיבים מוחלטים

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel, Field, HttpUrl
from ultralytics import YOLO
from PIL import Image
import numpy as np

app = FastAPI(title="Armament Detection API", version="1.0.0")

# ===== הגדרות נתיב למשקולות (מוחלט) =====
# ROOT = תיקיית השורש של הפרויקט (שני צעדים מעל הקובץ הזה: app/main.py -> app -> ROOT)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = str((ROOT / "weights" / "ipf_yolo11n_v0.1.pt").resolve())

# אם קיים ENV בשם MODEL_PATH נשתמש בו, אחרת בברירת המחדל (המוחלטת)
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
print(f"[startup] Loading model from: {MODEL_PATH}")  # עוזר בדיבאג

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at: {MODEL_PATH}")

# פרמטרים ברירת מחדל (אפשר לשנות גם ב-Query)
DEFAULT_CONF = float(os.getenv("CONF", "0.25"))
DEFAULT_IOU  = float(os.getenv("IOU",  "0.50"))
DEFAULT_IMGSZ = int(os.getenv("IMGSZ", "960"))

# ===== טעינת המודל פעם אחת =====
try:
    model = YOLO(MODEL_PATH)  # טוען משקולות מהדיסק
    # שמות מחלקות (dict או list תלוי במודל)
    CLASS_NAMES = model.model.names if hasattr(model.model, "names") else model.names
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

@app.get("/")
def root():
    return {"message": "API up. Go to /health or /docs"}

@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.basename(MODEL_PATH), "classes": CLASS_NAMES}

# ===== מודלים ל-JSON (Pydantic) =====
class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    class_id: int = Field(..., alias="class_id")
    class_name: str

class PredictResponse(BaseModel):
    has_armament: bool
    total_detections: int
    per_class: Dict[str, int]
    boxes: Optional[List[Box]] = None
    image_base64: Optional[str] = None
    width: int
    height: int
    conf_used: float
    iou_used: float

# ===== עזר: קריאת תמונה =====
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

def _annotate_to_base64(result) -> str:
    plotted_bgr = result.plot()              # numpy BGR
    img_rgb = Image.fromarray(plotted_bgr[..., ::-1])
    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ===== נקודת חיזוי =====
@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: Optional[UploadFile] = File(None, description="Image file"),
    url: Optional[HttpUrl] = Query(None, description="Image URL (optional)"),
    conf: float = Query(DEFAULT_CONF, ge=0.0, le=1.0),
    iou: float = Query(DEFAULT_IOU,  ge=0.0, le=1.0),
    imgsz: int = Query(DEFAULT_IMGSZ, ge=64, le=2048),
    return_boxes: bool = Query(False),
    return_image: bool = Query(False),
):
    if file is None and url is None:
        raise HTTPException(status_code=400, detail="Provide either file or url")

    # 1) קריאת תמונה ל-PIL.Image
    img = _read_image_from_upload(file) if file else _read_image_from_url(str(url))
    width, height = img.size

    # 2) הרצת YOLO
    try:
        res = model.predict(
            source=np.array(img),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            verbose=False
        )[0]  # תוצאה אחת לתמונה אחת
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # 3) עיבוד תוצאות
    per_class: Dict[str, int] = {}
    boxes_out: List[Box] = []

    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss  = res.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
            name = CLASS_NAMES.get(k, str(k)) if isinstance(CLASS_NAMES, dict) else (
                CLASS_NAMES[k] if 0 <= k < len(CLASS_NAMES) else str(k)
            )
            per_class[name] = per_class.get(name, 0) + 1
            if return_boxes:
                boxes_out.append(Box(
                    x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                    conf=float(c), class_id=int(k), class_name=str(name)
                ))

    # 4) מסקנות ותוספות
    has_armament = sum(per_class.values()) > 0
    image_b64 = _annotate_to_base64(res) if return_image else None

    # 5) תשובה
    return PredictResponse(
        has_armament=has_armament,
        total_detections=sum(per_class.values()),
        per_class=per_class,
        boxes=boxes_out if return_boxes else None,
        image_base64=image_b64,
        width=width,
        height=height,
        conf_used=conf,
        iou_used=iou
    )
if __name__ == "__main__":
    import uvicorn, os
    port = int(os.getenv("PORT", "8010"))
    # כשמעבירים את אובייקט app ישירות, עדיף בלי reload (ה-reloader עובד הכי טוב עם מחרוזת import)
    uvicorn.run(app, host="0.0.0.0", port=port)
