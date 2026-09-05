"""
AI 스마트 식사 보조 시스템 - WebSocket 서버 (v19 통합판 / 로컬·EC2 겸용)

[이번 수정 — v23 / 시연 최종]
★ 유령 top 차단 — 끝점은 식기 몸통(stick 등) 박스 근처일 때만 인정
  (그릇·책상 물건에 뜨는 가짜 top으로 인한 오트리거/오답 제거)
★ 크롭 정밀화 — CROP_RATIO 0.18 / TIP_OFFSET y 0.10 (이웃 음식 혼입·끝점 하향 밀림 수정)
★ Haiku 크롭 색감 분리 — YOLO 입력은 보정본(감마+채도), Haiku로 보내는
  크롭은 '보정 전 원본'에서 잘라 자연색 전달 (김치 붉은기·김 광택 보존)
★ 트리거 즉시 processing 신호 — 멈춤 판정 통과 후 Haiku 호출 '직전'에
  파이로 {"type": "processing"} 을 먼저 전송 → 파이가 삑 효과음 재생
  → 사용자가 "인식됐고 처리 중"임을 즉시 알 수 있음 (체감 딜레이 개선)
  (파이도 processing 신호를 처리하는 v2 클라이언트여야 함 — 세트)

[이전 수정 유지]
1. 파란 원 마커 OFF — 실측(크롭 3/3 vs 마커 2/3)에서 크롭 방식 채택,
   현재 food_recognizer 프롬프트에 파란 원 문구 없음 → 마커는 음식만 가림.
   (프롬프트에 파란 원 문구를 다시 넣으면 MARK_BLUE_CIRCLE=True로 복원 가능)
2. 멈춤 판정 완화 — STOP_FRAMES 6→5, STOP_PIXELS 65→80
3. 끝점 신뢰도 보류 관문 추가 — TIP_MIN_CONF(0.5) 미만이면 Haiku 호출 보류
   → 흐림 검사 + 끝점 신뢰도 + (food_recognizer 내 판별 불확실 처리) 3중 관문
4. 색감 보정 추가 — Haiku로 보내는 크롭에만 그레이월드 화이트밸런스 + 감마
   + 노란색 약부스트 적용 (YOLO 입력에는 영향 없음)
   → ENABLE_COLOR_CORRECTION 플래그로 on/off A/B 테스트 가능
5. SHOW_YOLO_WINDOW 자동 감지 유지 — EC2(headless)에서는 자동으로 꺼짐
6. ★ 웹 디버그 화면 추가 — EC2에서도 브라우저로 YOLO 디버그 화면 실시간 확인
   → 서버 켠 뒤 브라우저에서  http://서버IP:8080  접속
   → EC2 보안그룹에 TCP 8080 인바운드 규칙 필요
   → ENABLE_WEB_DEBUG 플래그로 on/off

구조:
Raspberry Pi → JPEG/WebSocket → 이 서버
  → YOLO(v19): 식기 '끝' 클래스 탐지 → 끝점 흔들림 보정
  → 멈춤 + 쿨다운 + 위치이동 + 끝점 신뢰도 조건 → [processing 신호]
  → 크롭 → 흐림 검사 → 색감 보정 → Claude Haiku 판별 → JSON 반환 → 파이 TTS

필요(로컬): pip install ultralytics opencv-python websockets numpy anthropic pillow python-dotenv
필요(EC2) : pip install ultralytics opencv-python-headless websockets numpy anthropic pillow python-dotenv
실행: python server_m.py
"""
import asyncio
import base64
import json
import os
import platform
import threading
import time
import glob
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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

# ★ 웹 디버그 화면 (브라우저에서 http://서버IP:8080)
ENABLE_WEB_DEBUG = True
WEB_DEBUG_PORT = 8080
WEB_DEBUG_JPEG_QUALITY = 70

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
YOLO_CONF = 0.20             # ★ 0.40 → 0.35 (탐지 임계값 완화)
TIP_MIN_CONF = 0.32          # ★ 0.5 → 0.40 (끝점 신뢰도 관문 완화: 미만이면 Haiku 호출 보류)
STOP_FRAMES = 2      # ★ 5 → 3 (멈춤 판정 약 3초 → 2초)
STOP_PIXELS = 60             # ★ 65 → 80
COOLDOWN = 4.0
CROP_RATIO = 0.18            # ★ 0.25 → 0.18 (크롭 축소: 이웃 음식 혼입 방지)
MISS_TOLERANCE = 30
SMOOTH_ALPHA = 0.5
MOVE_RESET = 40
BLUR_THRESHOLD = 20
TIP_OFFSET = (-0.25, 0.10)   # ★ y 0.25 → 0.10 (끝점이 아래로 밀리는 것 축소) — 웹 디버그 빨간 점으로 검증

# 파란 원 마커 — ★ OFF (크롭 방식 채택 + 현재 프롬프트에 파란 원 문구 없음)
MARK_BLUE_CIRCLE = False
BLUE_RADIUS = 14             # 원 반지름(px)
BLUE_THICKNESS = 3           # 선 두께

# ============================================================
# ★★ 전체 프레임 보정 (YOLO 입력 포함 — 수신 즉시 적용)
#   학습 데이터(밝고 채도 높은 사진)와 카메라 입력의 색감 차이 완화
#   웹 디버그 화면에도 보정된 모습이 그대로 보임 → 눈으로 확인하며 튜닝
# ============================================================
ENABLE_FRAME_ENHANCE = True   # 끄면 이전과 동일
FRAME_GAMMA = 1.25            # 밝기: 1.0=그대로, 1.2~1.4 권장 (클수록 밝음)
FRAME_SAT_GAIN = 1.30         # 채도: 1.0=그대로, 1.2~1.5 권장 (클수록 쨍함)

_FRAME_GAMMA_LUT = np.array(
    [((i / 255.0) ** (1.0 / FRAME_GAMMA)) * 255 for i in range(256)]
).astype(np.uint8)


def enhance_frame(frame):
    """수신 프레임 전체를 밝기(감마) + 채도 부스트 — YOLO/크롭/웹화면 모두 적용"""
    out = cv2.LUT(frame, _FRAME_GAMMA_LUT)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * FRAME_SAT_GAIN, 0, 255)
    out = cv2.merge([h, s, v]).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_HSV2BGR)


# ============================================================
# ★ 색감 보정 (Haiku 전송 크롭에만 적용, YOLO 입력엔 미적용)
#   ※ ENABLE_FRAME_ENHANCE가 True면 이미 프레임이 보정돼 있으므로
#     이중 보정을 피하려고 아래에서 자동으로 꺼짐
# ============================================================
ENABLE_COLOR_CORRECTION = True   # A/B 테스트 시 여기만 토글
if ENABLE_FRAME_ENHANCE:
    ENABLE_COLOR_CORRECTION = False   # 이중 보정 방지 (감마 두 번 → 과노출)
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
# ★ 웹 디버그 스트리밍 (MJPEG)
#   EC2처럼 화면 없는 서버에서도 브라우저로 디버그 화면 확인
#   http://서버IP:8080  ← 노트북/폰 브라우저에서 접속
# ============================================================
_web_lock = threading.Lock()
_web_frame = None   # 최신 디버그 프레임의 JPEG 바이트


def update_web_frame(debug_frame):
    """최신 디버그 프레임을 JPEG로 압축해 보관 (스트리밍용)"""
    global _web_frame
    ok, jpg = cv2.imencode(
        ".jpg", debug_frame,
        [cv2.IMWRITE_JPEG_QUALITY, WEB_DEBUG_JPEG_QUALITY]
    )
    if ok:
        with _web_lock:
            _web_frame = jpg.tobytes()


class _WebDebugHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/":
            html = (
                "<html><head><title>YOLO DEBUG</title></head>"
                "<body style='margin:0;background:#111;text-align:center'>"
                "<img src='/stream' style='max-width:100%;height:auto'>"
                "</body></html>"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if self.path != "/stream":
            self.send_response(404)
            self.end_headers()
            return

        # MJPEG 스트림
        self.send_response(200)
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame"
        )
        self.end_headers()
        try:
            while True:
                with _web_lock:
                    data = _web_frame
                if data is not None:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
                time.sleep(0.1)   # 약 10fps
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass   # 브라우저 탭 닫음 → 정상 종료

    def log_message(self, *args):
        pass   # 접속 로그로 콘솔 지저분해지는 것 방지


def start_web_debug():
    try:
        httpd = ThreadingHTTPServer((HOST, WEB_DEBUG_PORT), _WebDebugHandler)
    except OSError as e:
        print(f"[서버] 웹 디버그 시작 실패 (포트 {WEB_DEBUG_PORT}): {e}")
        return
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"[서버] 웹 디버그 화면: http://서버IP:{WEB_DEBUG_PORT} (브라우저로 접속)")


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
    """top/s_top/f_top 박스 중 conf 최고의 중심 = 끝점
    ★ v23: 유령 탐지 차단 — 끝점은 반드시 식기 몸통(stick 등) 박스
      근처에 있어야 인정. 그릇/책상 물건에 뜨는 가짜 top 제거."""
    boxes = results[0].boxes
    tips = [b for b in boxes
            if model.names[int(b.cls[0])] in TIP_CLASSES]
    if not tips:
        return None

    # 식기 몸통 박스 (끝 클래스가 아닌 것들)
    utensils = [b for b in boxes
                if model.names[int(b.cls[0])] not in TIP_CLASSES]

    def near_utensil(tip_box, margin=40):
        tx1, ty1, tx2, ty2 = map(float, tip_box.xyxy[0])
        tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
        for u in utensils:
            ux1, uy1, ux2, uy2 = map(float, u.xyxy[0])
            if (ux1 - margin <= tcx <= ux2 + margin and
                    uy1 - margin <= tcy <= uy2 + margin):
                return True
        return False

    # 몸통이 하나라도 잡혔으면: 몸통 근처의 끝만 인정 (유령 제거)
    # 몸통이 아예 없으면: 끝만 단독으로 뜬 상황 → 유령 가능성 높아 버림
    valid = [t for t in tips if near_utensil(t)] if utensils else []
    if not valid:
        return None
    box = max(valid, key=lambda b: float(b.conf[0]))
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
def process_frame(frame, raw_frame, state, notify_processing=None):
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
                crop, crop_box = crop_around(raw_frame, tip)   # ★ Haiku 크롭은 원본 색 (보정 전)

                if crop.size != 0:
                    sharp, score = is_sharp(crop)
                    response["sharpness"] = round(score, 1)

                    if not sharp:
                        print(f"[보류] 흐린 프레임(선명도 {score:.0f}) → 재시도 대기")
                    else:
                        state.last_trigger = now
                        state.last_pos = tip

                        # ★★ v20: 모든 관문 통과 = Haiku 호출 확정
                        #    → 파이로 processing 신호 먼저 전송 (삑 효과음)
                        if notify_processing:
                            notify_processing()

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
                            t0 = time.time()
                            r = ask_haiku(send_img)
                            print(f"[시간] Haiku 응답 {time.time() - t0:.2f}초")
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
    # ★ 웹 디버그: 로컬이든 EC2든 브라우저에서 실시간 확인 가능
    if ENABLE_WEB_DEBUG:
        update_web_frame(debug_frame)
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
    loop = asyncio.get_running_loop()

    # ★ v20: YOLO 스레드에서 호출됨 → 이벤트 루프에 안전하게 전송 예약
    def notify_processing():
        asyncio.run_coroutine_threadsafe(
            ws.send(json.dumps({"type": "processing"})), loop
        )

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

            # ★★ 원본 보관 후 보정 (YOLO는 보정본, Haiku 크롭은 원본 색)
            raw_frame = frame
            if ENABLE_FRAME_ENHANCE:
                frame = enhance_frame(frame)

            state.frame_count += 1
            response = await asyncio.to_thread(process_frame, frame, raw_frame, state, notify_processing)
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
    if ENABLE_WEB_DEBUG:
        start_web_debug()
    async with websockets.serve(handler, HOST, PORT, max_size=None):
        print()
        print("=" * 65)
        print("AI 스마트 식사 보조 서버 (v23 시연최종 / 로컬·EC2 겸용)")
        print("=" * 65)
        print(f"WebSocket : ws://{HOST}:{PORT}")
        print(f"YOLO      : {MODEL_PATH}")
        print(f"끝 클래스 : {TIP_CLASSES}")
        print("Claude    : Haiku 4.5")
        print(f"정지 감지 : {STOP_FRAMES} frames / {STOP_PIXELS}px")
        print(f"끝점 관문 : TIP_MIN_CONF {TIP_MIN_CONF}")
        print(f"Cooldown  : {COOLDOWN}s / 이동 리셋 {MOVE_RESET}px / 블러 {BLUR_THRESHOLD}")
        print(f"프레임 보정: {'ON' if ENABLE_FRAME_ENHANCE else 'OFF'} "
              f"(감마 {FRAME_GAMMA} / 채도 x{FRAME_SAT_GAIN}) ← YOLO 입력 포함")
        print(f"색감 보정 : {'ON' if ENABLE_COLOR_CORRECTION else 'OFF'} "
              f"(감마 {GAMMA} / 노랑 채도 x{YELLOW_SAT_GAIN})")
        print(f"파란 원   : {'ON' if MARK_BLUE_CIRCLE else 'OFF'} / 디버그 창: {'ON' if SHOW_YOLO_WINDOW else 'OFF(headless)'}")
        print(f"웹 화면   : {'http://서버IP:' + str(WEB_DEBUG_PORT) if ENABLE_WEB_DEBUG else 'OFF'}")
        print(f"삑 신호   : ON (트리거 확정 → 파이로 processing 전송)")
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
