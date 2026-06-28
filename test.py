from ultralytics import YOLO
import glob

model = YOLO(r"C:\Hanieum\runs\detect\train-6\weights\best.pt")

files = glob.glob(r"C:\Hanieum\yolo_v8n\test_tray.*")
print("찾은 파일:", files)

results = model.predict(
    source=files[0],
    save=True,
    conf=0.10,      # ← 0.25 → 0.10으로 낮춤 (놓친 것까지 보기)
    imgsz=640       # ← 학습과 동일하게 640
)

for r in results:
    print(f"\n검출된 객체 {len(r.boxes)}개:")
    for box in r.boxes:
        name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        print(f"  {name}: {conf:.2f}")
    print("\n저장 위치:", r.save_dir)   # 결과가 어디 저장됐는지 출력