"""
AI 스마트 식사 보조 시스템 - WebSocket 서버 (v19 통합판 / 로컬·EC2 겸용)

[이번 수정]
1. 파란 원 마커 OFF — 실측(크롭 3/3 vs 마커 2/3)에서 크롭 방식 채택,
   현재 food_recognizer 프롬프트에 파란 원 문구 없음 → 마커는 음식만 가림.
   (프롬프트에 파란 원 문구를 다시 넣으면 MARK_BLUE_CIRCLE=True로 복원 가능)
2. 멈춤 판정 완화 — STOP_FRAMES 6→5, STOP_PIXELS 65→80
3. 끝점 신뢰도 보류 관문 추가 — TIP_MIN_CONF(0.5) 미만이면 Haiku 호출 보류
   → 흐림 검사 + 끝점 신뢰도 + (food_recognizer 내 판별 불확실 처리) 3중 관문
4. 색감 보정 추가 — Haiku로 보내는 크롭에만 그레이월드 화이트밸런스 + 감마
   + 노란색 약부스트 적용 (YOLO 입력에는 영향 없음)
   → 학습 데이터(밝고 쨍한 색감)와 실사용 카메라 색감 차이 완화 목적
   → ENABLE_COLOR_CORRECTION 플래그로 on/off A/B 테스트 가능
5. SHOW_YOLO_WINDOW 자동 감지 유지 — EC2(headless)에서는 자동으로 꺼짐

구조:
Raspberry Pi 5 → JPEG/WebSocket → 이 서버
  → YOLO(v19): 식기 '끝' 클래스 탐지 → 끝점 흔들림 보정
  → 멈춤 + 쿨다운 + 위치이동 + 끝점 신뢰도 조건 → 크롭 → 흐림 검사
  → 색감 보정 → Claude Haiku 판별 → JSON 반환 → 파이 TTS

필요(로컬): pip install ultralytics opencv-python websockets numpy anthropic pillow python-dotenv
필요(EC2) : pip install ultralytics opencv-python-headless websockets numpy anthropic pillow python-dotenv
실행: python server_m.py
"""
import asyncio
import base64
import json
import os
import platform
import time
import glob
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import websockets
from dotenv import load_dotenv
from PIL import Image
from ultralytics import YOLO

# ============================================================
# 경로 / API 키
# ============================================================
BASE = Path(__file__).parent
load_dotenv(BASE / ".env", override=True)

if not os.getenv("ANTHROPIC_API_KEY"):
    raise SystemExit("ANTHROPIC_API_KEY를 .env에서 찾지 못했습니다.")

from food_recognizer import recognize_food   # 같은 폴더의 food_recognizer.py

# ============================================================
# 서버 설정
# ============================================================
HOST = "0.0.0.0"
PORT = 8765

# ============================================================
# YOLO (v19: 끝 클래스 직접 탐지 모델)
# ============================================================
_candidates = [BASE / "models" / "best.pt", BASE / "best.pt"]
MODEL_PATH = next((p for p in _candidates if p.exists()), None)
if MODEL_PATH is None:
    raise SystemExit("YOLO 모델(best.pt)을 models/ 또는 현재 폴더에서 찾지 못했습니다.")

model = YOLO(str(MODEL_PATH))
print(f"[서버] YOLO 모델 로드 완료: {MODEL_PATH}")

TIP_CLASSES = {"top", "s_top", "f_top"}

# ============================================================
# 튜닝값 (v19 기준)
# ============================================================
YOLO_CONF = 0.30             # ★ 0.40 → 0.30 (추적 끊김 방지: conf가 문턱 근처에서 오가며 깜빡이는 것 완화)
TIP_MIN_CONF = 0.5           # ★ 끝점 신뢰도 관문: 미만이면 Haiku 호출 보류
STOP_FRAMES = 4              # ★ 5 → 4 (멈춤 판정 추가 완화: 깜빡임 사이에도 트리거 도달)
STOP_PIXELS = 80             # ★ 65 → 80
COOLDOWN = 4.0
CROP_RATIO = 0.35
MISS_TOLERANCE = 15          # ★ 5 → 15 (탐지 깜빡임 허용: 잠깐 놓쳐도 카운팅 초기화 안 함)
SMOOTH_ALPHA = 0.3
MOVE_RESET = 80
BLUR_THRESHOLD = 100
TIP_OFFSET = (-0.25, 0.25)   # 빨간 점 보정 (박스 크기 대비 x,y 비율)

# 파란 원 마커 — ★ OFF (크롭 방식 채택 + 현재 프롬프트에 파란 원 문구 없음)
MARK_BLUE_CIRCLE = False
BLUE_RADIUS = 14             # 원 반지름(px)
BLUE_THICKNESS = 3           # 선 두께

# ============================================================
# ★ 색감 보정 (Haiku 전송 크롭에만 적용, YOLO 입력엔 미적용)
# ============================================================
ENABLE_COLOR_CORRECTION = True   # A/B 테스트 시 여기만 토글
GAMMA = 1.2                      # 1.0=변화 없음, 클수록 밝아짐 (어두운 카메라 보정)
YELLOW_SAT_GAIN = 1.25           # 노란색 채도 배율 (계란말이 등) — 과하면 역효과
YELLOW_VAL_GAIN = 1.05           # 노란색 밝기 배율

_GAMMA_LUT = np.array(
    [((i / 255.0) ** (1.0 / GAMMA)) * 255 for i in range(256)]
).astype(np.uint8)


def gray_world_wb(frame):
    """그레이월드 화이트밸런스 — 조명으로 인한 전체 색조(누렇거나 푸른 톤) 제거"""
    b, g, r = cv2.split(frame.astype(np.float32))
    b_mean, g_mean, r_mean = b.mean(), g.mean(), r.mean()
    if min(b_mean, g_mean, r_mean) < 1e-6:
        return frame
    avg = (b_mean + g_mean + r_mean) / 3
    return cv2.merge([
        np.clip(b * avg / b_mean, 0, 255),
        np.clip(g * avg / g_mean, 0, 255),
        np.clip(r * avg / r_mean, 0, 255),
    ]).astype(np.uint8)


def boost_yellow(frame, sat_gain=YELLOW_SAT_GAIN, val_gain=YELLOW_VAL_GAIN):
    """노란색 계열(Hue 20~35)만 채도/밝기 약부스트 — 경계는 블러로 자연스럽게"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    mask = ((h >= 20) & (h <= 35)).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    s = np.clip(s * (1 + (sat_gain - 1) * mask), 0, 255)
    v = np.clip(v * (1 + (val_gain - 1) * mask), 0, 255)
    out = cv2.merge([h, s, v]).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_HSV2BGR)


def preprocess_for_haiku(crop_bgr):
    """순서 중요: 화이트밸런스 → 감마 → 노란색 부스트"""
    out = gray_world_wb(crop_bgr)
    out = cv2.LUT(out, _GAMMA_LUT)
    out = boost_yellow(out)
    return out


# ============================================================
# 디버그 설정
# ============================================================
SAVE_DEBUG = True
DEBUG_DIR = BASE / "debug_frames"
DEBUG_DIR.mkdir(exist_ok=True)
SAVE_YOLO_EVERY = 5
MAX_YOLO_DEBUG_FILES = 200
MAX_CROP_DEBUG_FILES = 100

# 디버그 창 자동 감지:
#   - 윈도우/맥(로컬) → 창 표시
#   - 리눅스인데 DISPLAY 없음(EC2 등 headless) → 자동으로 끔
SHOW_YOLO_WINDOW = True
if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
    SHOW_YOLO_WINDOW = False
    print("[서버] headless 환경 감지 → 디버그 창 비활성 (사진 저장은 유지)")


# ============================================================
# 클라이언트 상태
# ============================================================
class ClientState:
    def __init__(self):
        self.recent = deque(maxlen=STOP_FRAMES)
        self.last_trigger = 0.0
        self.last_pos = None
        self.prev_tip = None
        self.miss_count = 0
        self.frame_count = 0


def cleanup_debug_files(pattern, max_files):
    files = sorted(glob.glob(str(DEBUG_DIR / pattern)), key=os.path.getmtime)
    while len(files) > max_files:
        try:
            os.remove(files.pop(0))
        except OSError:
            break


# ============================================================
# v19: 끝 클래스에서 끝점 추출
# ============================================================
def get_tip(results):
    """top/s_top/f_top 박스 중 conf 최고의 중심 = 끝점"""
    tips = [b for b in results[0].boxes
            if model.names[int(b.cls[0])] in TIP_CLASSES]
    if not tips:
        return None
    box = max(tips, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cx += int((x2 - x1) * TIP_OFFSET[0])
    cy += int((y2 - y1) * TIP_OFFSET[1])
    return {
        "center": (cx, cy),
        "box": (x1, y1, x2, y2),
        "confidence": float(box.conf[0]),
        "class_name": model.names[int(box.cls[0])],
    }


def is_stopped(state):
    if len(state.recent) < STOP_FRAMES:
        return False
    xs = [p[0] for p in state.recent]
    ys = [p[1] for p in state.recent]
    return (max(xs) - min(xs) < STOP_PIXELS) and (max(ys) - min(ys) < STOP_PIXELS)


def is_sharp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score >= BLUR_THRESHOLD, score


def crop_around(frame, center):
    h, w = frame.shape[:2]
    size = int(min(h, w) * CROP_RATIO)
    cx, cy = center
    x1, y1 = max(0, cx - size), max(0, cy - size)
    x2, y2 = min(w, cx + size), min(h, cy + size)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def mark_tip_on_crop(crop, tip, crop_box):
    """(현재 미사용) Haiku로 보낼 크롭에 끝점을 파란 원으로 표시 (BGR: 파랑=(255,0,0))
    → 프롬프트에 파란 원 문구를 복원할 때만 MARK_BLUE_CIRCLE=True와 함께 사용"""
    marked = crop.copy()
    x1, y1, _, _ = crop_box
    px, py = tip[0] - x1, tip[1] - y1
    h, w = marked.shape[:2]
    px = max(0, min(w - 1, px))
    py = max(0, min(h - 1, py))
    cv2.circle(marked, (px, py), BLUE_RADIUS, (255, 0, 0), BLUE_THICKNESS)
    return marked


def ask_haiku(crop_bgr):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return recognize_food(Image.fromarray(rgb))


def make_debug_frame(frame, results, tip=None, tip_info=None, crop_box=None):
    debug = results[0].plot()
    if tip is not None and tip_info is not None:
        cv2.circle(debug, tip, 8, (0, 0, 255), -1)
        cv2.putText(debug, f"{tip_info['class_name']} {tip_info['confidence']:.2f}",
                    (tip[0] + 10, tip[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if crop_box:
        x1, y1, x2, y2 = crop_box
        cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.putText(debug, "HAIKU CROP", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    return debug


# ============================================================
# 프레임 1장 처리
# ============================================================
def process_frame(frame, state):
    results = model(frame, conf=YOLO_CONF, verbose=False)
    tip_info = get_tip(results)
    crop_box = None
    tip = None

    response = {
        "pointing": False,
        "triggered": False,
        "food": None,
        "tip_class": None,
        "yolo_confidence": None,
        "sharpness": None,
        "input_tokens": None,
        "output_tokens": None,
    }

    if tip_info:
        raw_tip = tip_info["center"]

        if state.prev_tip is None:
            tip = raw_tip
        else:
            tip = (int(SMOOTH_ALPHA * raw_tip[0] + (1 - SMOOTH_ALPHA) * state.prev_tip[0]),
                   int(SMOOTH_ALPHA * raw_tip[1] + (1 - SMOOTH_ALPHA) * state.prev_tip[1]))
        state.prev_tip = tip

        state.recent.append(tip)
        state.miss_count = 0

        response["pointing"] = True
        response["tip_class"] = tip_info["class_name"]
        response["yolo_confidence"] = round(tip_info["confidence"], 3)

        now = time.time()

        far_enough = (state.last_pos is None or
                      abs(tip[0] - state.last_pos[0]) > MOVE_RESET or
                      abs(tip[1] - state.last_pos[1]) > MOVE_RESET)

        if is_stopped(state) and (now - state.last_trigger > COOLDOWN) and far_enough:
            # ★ 끝점 신뢰도 관문: 낮으면 트리거 보류 (다음 프레임 재시도)
            if tip_info["confidence"] < TIP_MIN_CONF:
                print(f"[보류] 끝점 신뢰도 낮음({tip_info['confidence']:.2f} < {TIP_MIN_CONF}) → 재시도 대기")
            else:
                crop, crop_box = crop_around(frame, tip)

                if crop.size != 0:
                    sharp, score = is_sharp(crop)
                    response["sharpness"] = round(score, 1)

                    if not sharp:
                        print(f"[보류] 흐린 프레임(선명도 {score:.0f}) → 재시도 대기")
                    else:
                        state.last_trigger = now
                        state.last_pos = tip

                        send_img = mark_tip_on_crop(crop, tip, crop_box) if MARK_BLUE_CIRCLE else crop

                        # ★ 색감 보정: Haiku로 보내는 이미지에만 적용
                        if ENABLE_COLOR_CORRECTION:
                            send_img = preprocess_for_haiku(send_img)

                        # 디버그 저장은 보정 이후 → Haiku가 받는 것과 동일한 이미지가 저장됨
                        if SAVE_DEBUG:
                            cv2.imwrite(str(DEBUG_DIR / f"crop_{state.frame_count:06d}_{int(now)}.jpg"), send_img)
                            cleanup_debug_files("crop_*.jpg", MAX_CROP_DEBUG_FILES)

                        print(f"[트리거] {tip_info['class_name']} 멈춤(선명도 {score:.0f}) → Haiku 호출")
                        try:
                            r = ask_haiku(send_img)
                            response["triggered"] = True
                            response["food"] = r.get("food")
                            response["input_tokens"] = r.get("input_tokens")
                            response["output_tokens"] = r.get("output_tokens")
                            print(f"[Haiku] 음식={response['food']} "
                                  f"| in={response['input_tokens']} out={response['output_tokens']}")
                        except Exception as e:
                            print("[Haiku 오류]", repr(e))
    else:
        state.miss_count += 1
        if state.miss_count > MISS_TOLERANCE:
            state.recent.clear()
            state.prev_tip = None

    debug_frame = make_debug_frame(frame, results, tip, tip_info, crop_box)
    if SHOW_YOLO_WINDOW:
        cv2.imshow("YOLO DEBUG - Raspberry Pi Camera", debug_frame)
        cv2.waitKey(1)
    if SAVE_DEBUG and state.frame_count % SAVE_YOLO_EVERY == 0:
        cv2.imwrite(str(DEBUG_DIR / f"yolo_{state.frame_count:06d}.jpg"), debug_frame)
        cleanup_debug_files("yolo_*.jpg", MAX_YOLO_DEBUG_FILES)

    return response


# ============================================================
# WebSocket 핸들러
# ============================================================
async def handler(ws):
    print("[서버] Raspberry Pi 연결됨")
    state = ClientState()
    try:
        async for message in ws:
            data = json.loads(message)
            try:
                jpg = base64.b64decode(data["image"])
            except Exception as e:
                print("[서버] Base64 오류:", e)
                continue
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                print("[서버] JPEG 디코딩 실패")
                continue

            state.frame_count += 1
            response = await asyncio.to_thread(process_frame, frame, state)
            response["frame_id"] = data.get("frame_id")

            if state.frame_count % 10 == 0:
                if response["pointing"]:
                    print(f"[프레임 {state.frame_count}] {response['tip_class']} "
                          f"conf={response['yolo_confidence']}")
                else:
                    print(f"[프레임 {state.frame_count}] 식기 끝 없음")

            await ws.send(json.dumps(response, ensure_ascii=False))
    except websockets.ConnectionClosed:
        print("[서버] Raspberry Pi 연결 끊김")
    except Exception as e:
        print("[서버 오류]", repr(e))
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


async def main():
    async with websockets.serve(handler, HOST, PORT, max_size=None):
        print()
        print("=" * 65)
        print("AI 스마트 식사 보조 서버 (v19 / 로컬·EC2 겸용)")
        print("=" * 65)
        print(f"WebSocket : ws://{HOST}:{PORT}")
        print(f"YOLO      : {MODEL_PATH}")
        print(f"끝 클래스 : {TIP_CLASSES}")
        print("Claude    : Haiku 4.5")
        print(f"정지 감지 : {STOP_FRAMES} frames / {STOP_PIXELS}px")
        print(f"끝점 관문 : TIP_MIN_CONF {TIP_MIN_CONF}")
        print(f"Cooldown  : {COOLDOWN}s / 이동 리셋 {MOVE_RESET}px / 블러 {BLUR_THRESHOLD}")
        print(f"색감 보정 : {'ON' if ENABLE_COLOR_CORRECTION else 'OFF'} "
              f"(감마 {GAMMA} / 노랑 채도 x{YELLOW_SAT_GAIN})")
        print(f"파란 원   : {'ON' if MARK_BLUE_CIRCLE else 'OFF'} / 디버그 창: {'ON' if SHOW_YOLO_WINDOW else 'OFF(headless)'}")
        print(f"디버그    : {DEBUG_DIR}")
        print("=" * 65)
        print()
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[서버] 종료")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass