"""
AI 스마트 식사 보조 시스템 - 서버 (YOLO + 디버그 사진 저장)
=====================================================================
- banchan.pt(반찬+젓가락 학습 모델) 로드, 없으면 기본 yolov8n
- 젓가락(stick) 박스와 반찬 박스가 가까우면 "가리킨 반찬"으로 판정
- 영어 클래스명 → 한국어 음성 안내로 변환
- 인식 박스를 그려서 debug_frames/ 폴더에 사진 저장

실행:  python server.py
필요:  pip install ultralytics opencv-python websockets numpy
"""

import asyncio
import base64
import json
import os
import glob
import math
import cv2
import numpy as np
import websockets
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# 설정값
# ---------------------------------------------------------------------------
HOST = "0.0.0.0"
PORT = 8765
MODEL_PATH = "banchan.pt"
CONF_THRESH = 0.40
TEMP_WARN = 60.0
DIST_THRESH = 100            # 젓가락 박스~반찬 박스 거리가 이 픽셀 이내면 "가리킴" (겹치면 0)
UTENSIL_CLASSES = {"chopstick", "spoon", "fork", "knife", "stick", "젓가락", "숟가락"}

# 영어 클래스 이름 → 한국어 음성 안내용
NAME_MAP = {
    "Kimchi": "김치",
    "black_beans": "콩자반",
    "egg_roll": "계란말이",
    "rice": "밥",
    "soybean_paste_soup": "된장국",
}

def kor(name):
    return NAME_MAP.get(name, name)

# --- 디버그 사진 저장 설정 ---
SAVE_DEBUG = True
DEBUG_DIR = "debug_frames"
SAVE_EVERY = 3
MAX_DEBUG_FILES = 150

if SAVE_DEBUG:
    os.makedirs(DEBUG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 모델 로드
# ---------------------------------------------------------------------------
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
    print(f"[서버] 모델 로드 완료: {MODEL_PATH}")
else:
    model = YOLO("yolov8n.pt")
    print("[경고] banchan.pt 가 없어 기본 yolov8n 모델을 사용합니다.")


# ---------------------------------------------------------------------------
# 핵심 로직
# ---------------------------------------------------------------------------
def estimate_tip(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


def box_distance(b1, b2):
    """두 박스 사이의 최소 거리. 겹치면 0."""
    dx = max(b1[0] - b2[2], b2[0] - b1[2], 0)
    dy = max(b1[1] - b2[3], b2[1] - b1[3], 0)
    return math.hypot(dx, dy)


def analyze(frame_bgr, temp_c):
    yolo = model(frame_bgr, verbose=False)[0]
    foods, utensils = [], []
    for b in yolo.boxes:
        conf = float(b.conf[0])
        if conf < CONF_THRESH:
            continue
        x1, y1, x2, y2 = map(float, b.xyxy[0])
        box = [x1, y1, x2, y2]
        name = model.names[int(b.cls[0])]
        item = {"name": name, "conf": round(conf, 2), "box": box,
                "center": [(x1 + x2) / 2, (y1 + y2) / 2]}
        if name in UTENSIL_CLASSES:
            utensils.append(item)
        else:
            foods.append(item)

    tip = None
    pointed = None
    if utensils:
        utensil = max(utensils, key=lambda u: u["box"][3])
        tip = estimate_tip(utensil["box"])
        if foods:
            nearest = min(foods, key=lambda f: box_distance(utensil["box"], f["box"]))
            if box_distance(utensil["box"], nearest["box"]) <= DIST_THRESH:
                pointed = nearest

    temp_warning = (temp_c is not None) and (temp_c >= TEMP_WARN)

    result = {
        "pointing": pointed is not None,
        "food": kor(pointed["name"]) if pointed else None,
        "box": pointed["box"] if pointed else None,
        "temp_c": temp_c,
        "temp_warning": temp_warning,
        "all_foods": [kor(f["name"]) for f in foods],
        "utensils": [u["name"] for u in utensils],
    }
    return result, (foods, utensils, pointed, tip)


# ---------------------------------------------------------------------------
# 디버그 사진 저장
# ---------------------------------------------------------------------------
def draw_and_save(frame, foods, utensils, pointed, tip, frame_count):
    img = frame.copy()
    for f in foods:
        x1, y1, x2, y2 = map(int, f["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(img, f"{f['name']} {f['conf']}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
    for u in utensils:
        x1, y1, x2, y2 = map(int, u["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 128, 0), 2)
        cv2.putText(img, f"{u['name']} {u['conf']}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2)
    if tip is not None:
        cv2.circle(img, (int(tip[0]), int(tip[1])), 8, (0, 255, 255), -1)
    if pointed is not None:
        x1, y1, x2, y2 = map(int, pointed["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(img, f"-> {pointed['name']}", (x1, min(y2 + 22, img.shape[0] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    fname = os.path.join(DEBUG_DIR, f"frame_{frame_count:05d}.jpg")
    cv2.imwrite(fname, img)

    files = sorted(glob.glob(os.path.join(DEBUG_DIR, "*.jpg")), key=os.path.getmtime)
    while len(files) > MAX_DEBUG_FILES:
        try:
            os.remove(files.pop(0))
        except OSError:
            break


# ---------------------------------------------------------------------------
# WebSocket 핸들러
# ---------------------------------------------------------------------------
async def handler(ws):
    print("[서버] 라즈베리파이 연결됨")
    frame_count = 0
    try:
        async for message in ws:
            data = json.loads(message)
            jpg = base64.b64decode(data["image"])
            arr = np.frombuffer(jpg, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            result, dbg = await asyncio.to_thread(analyze, frame, data.get("temp_c"))
            result["frame_id"] = data.get("frame_id")

            frame_count += 1
            if frame_count % SAVE_EVERY == 0:
                print(f"[프레임 {frame_count}] "
                      f"음식:{result['all_foods']}  "
                      f"도구:{result['utensils']}  "
                      f"→ 가리킨 것:{result['food']}")
                if SAVE_DEBUG:
                    draw_and_save(frame, *dbg, frame_count)

            await ws.send(json.dumps(result, ensure_ascii=False))
    except websockets.ConnectionClosed:
        print("[서버] 연결 끊김")


async def main():
    async with websockets.serve(handler, HOST, PORT, max_size=None):
        print(f"[서버] 대기 중 → ws://{HOST}:{PORT}")
        if SAVE_DEBUG:
            print(f"[서버] 디버그 사진 저장 폴더: ./{DEBUG_DIR}/")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
