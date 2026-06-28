from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)

# 서버 시작 시 모델 한 번만 로드
model = YOLO(r"C:\Hanieum\runs\detect\train-5\weights\best.pt")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "no image"}), 400

    file = request.files["image" ]
    img = Image.open(io.BytesIO(file.read()))

    results = model.predict(source=img, conf=0.25, imgsz=640)

    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": model.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 3),
                "bbox": [round(x, 1) for x in box.xyxy[0].tolist()]
            })

    return jsonify({"count": len(detections), "detections": detections})

@app.route("/", methods=["GET"])
def home():
    return "YOLO 서버 작동 중"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)