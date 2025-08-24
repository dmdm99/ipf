<<<<<<< HEAD
# ipf
=======
# Armament Identification (YOLO)

## Setup (local)
1. Create venv and install:
pip install --upgrade pip
pip install ultralytics fastapi uvicorn[standard] opencv-python


2. Put dataset under `dataset/` (not tracked by git).
3. Edit `dataset/data.yaml` (5 classes: MK, Cluster, LGB, Durandal, MOAB).

## Train
yolo detect train model=yolo11n.pt data=dataset/data.yaml epochs=30 imgsz=640 batch=16


## Inference API (FastAPI)
uvicorn app:app --host 0.0.0.0 --port 8000


>>>>>>> 0cbdfb9 (Initial commit: project skeleton (.gitignore, README, requirements))
