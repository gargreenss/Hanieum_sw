"""2단계: 웹캠 YOLO 젓가락 탐지 + 멈춤 감지 + 크롭 + Haiku 판별"""
import cv2, time, sys, threading
from collections import deque
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from ultralytics import YOLO

BASE = Path(__file__).parent
load_dotenv(BASE / ".env")                            # ① 키 먼저 로드
sys.path.insert(0, str(BASE.parent / "haiku_test"))   # ② 옆 폴더 경로 추가
from food_recognizer import recognize_food            # ③ 그 다음 import

model = YOLO(str(BASE / "models" / "best.pt"))

# ── 튜닝 파라미터 ──
STOP_FRAMES = 10        # CPU 추론이라 실질 ~10fps → 약 1초 멈춤 기준
STOP_PIXELS = 30        # 이동량이 이 픽셀 이하면 "멈춤"
COOLDOWN = 4            # 트리거 후 재트리거 금지(초)
CROP_RATIO = 0.35       # 크롭 크기 = 프레임 짧은 변의 이 비율
MISS_TOLERANCE = 5      # 연속 이 프레임까지 인식 깜빡임 허용

recent = deque(maxlen=STOP_FRAMES)
last_trigger = 0
miss_count = 0

def get_tip(results):
    boxes = results[0].boxes
    if len(boxes) == 0:
        return None
    box = max(boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    return ((x1 + x2) // 2, (y1 + y2) // 2), (x1, y1, x2, y2)

def is_stopped():
    if len(recent) < STOP_FRAMES:
        return False
    xs = [p[0] for p in recent]; ys = [p[1] for p in recent]
    return (max(xs) - min(xs) < STOP_PIXELS) and (max(ys) - min(ys) < STOP_PIXELS)

def crop_around(frame, center):
    h, w = frame.shape[:2]
    size = int(min(h, w) * CROP_RATIO)
    cx, cy = center
    x1 = max(0, cx - size); y1 = max(0, cy - size)
    x2 = min(w, cx + size); y2 = min(h, cy + size)
    return frame[y1:y2, x1:x2]

def ask_haiku_async(crop_bgr):
    """크롭을 스레드로 Haiku에 보내 판별 (영상 루프 안 멈추게)"""
    def _work():
        img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        r = recognize_food(img)
        print(f"🍽 판별 결과: {r['food']}  (토큰 in {r['input_tokens']}/out {r['output_tokens']})")
    threading.Thread(target=_work, daemon=True).start()

cap = cv2.VideoCapture(1)
print("탐지 시작, q로 종료")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model(frame, conf=0.4, verbose=False)
    tip_info = get_tip(results)

    if tip_info:
        center, (x1, y1, x2, y2) = tip_info
        recent.append(center)
        miss_count = 0
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, center, 6, (0, 0, 255), -1)

        if is_stopped() and time.time() - last_trigger > COOLDOWN:
            last_trigger = time.time()
            crop = crop_around(frame, center)
            fname = f"crop_{int(last_trigger)}.jpg"
            cv2.imwrite(str(BASE / fname), crop)
            print(f"멈춤 감지. 크롭 저장: {fname}")
            ask_haiku_async(crop)
            cv2.putText(frame, "TRIGGER!", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    else:
        miss_count += 1
        if miss_count > MISS_TOLERANCE:
            recent.clear()   # 진짜 사라진 것으로 보고 리셋

    cv2.imshow("webcam_test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()