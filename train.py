import torch
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8s.pt")   # ← n에서 s로 변경 (핵심)

    model.train(
        data="My-First-Project-5/data.yaml",   # ← 방금 받은 새 폴더명에 맞게
        epochs=50,
        imgsz=640,
        batch=16,                # GPU 부족하면 8, 여유 모르면 -1(자동)
        device=0 if torch.cuda.is_available() else "cpu",
        workers=4,
        name="train_s_chopsticks",   # 결과가 runs/detect/train_s_chopsticks/에 저장됨
    )