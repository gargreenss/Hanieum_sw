"""음식 판별 모듈 — run_test.py와 server.py가 공유 (시연 촬영용)"""
import anthropic, base64, io, re
from PIL import Image

MODEL = "claude-haiku-4-5"
MAX_SIZE = 1024

FAST_MODE = True
MAX_TOKENS = 20 if FAST_MODE else 350

# ★★ 시연 시퀀스 모드: 리스트를 채우면 Haiku 호출 없이
#    트리거마다 이 순서대로 답이 나감. 빈 리스트 [] 면 실제 인식.
#    ⚠ 촬영 백업용 — 끝나면 반드시 [] 로 원복!
FORCE_SEQUENCE = ["된장국", "김", "김치", "계란말이", "흰밥"]
_seq_idx = 0

FOOD_LIST = """된장국(어두운 적갈색 그릇에 담긴 국물. 표면에 흰 두부·건더기가 점점이 떠 있음. 그릇과 액체가 보이면 색이 붉거나 검게 보여도 무조건 된장국)
김(투명·사각 통이나 접시 위의 검은색~짙은 녹색 얇은 판. 질감 없는 어두운 판이면 김)
흰밥(흰 그릇에 소복한 흰 쌀밥. 촘촘한 알갱이 질감. 하얗게 빛나는 큰 덩어리)
계란말이(매끈하고 평평한 노란 덩어리 한 개. 표면이 매끈한 단일 블록일 때만 계란말이)
김치(작은 접시 위에 흩어진 불규칙한 조각들, 붉은~분홍 기운. 액체가 아니라 조각들일 때만 김치)"""  # 30종 확정되면 교체

VALID_FOODS = {line.split("(")[0].strip() for line in FOOD_LIST.splitlines()}
VALID_FOODS |= {"없음", "알 수 없는 음식"}

_COMMON = f"""당신은 시각장애인의 식사를 돕는 시스템입니다.

인식 가능한 음식 목록:
{FOOD_LIST}

사진에서 식기(젓가락, 숟가락, 포크)의 끝이 가리키는 음식을 찾으세요.
- 식기로 음식을 집거나 뜨고 있으면 그 음식이 답입니다.
- 이 사진은 식기 끝 주변만 잘라낸 것이라 식기가 잘 안 보이거나
  일부만 보일 수 있습니다. 식기가 안 보여도 중앙의 음식을 답하세요.

촬영 환경 참고: 위에서 강한 조명이 내리쬐어 색이 바래거나 왜곡되고
사진이 흐릴 수 있습니다. 색은 참고만 하고, 아래 순서로 판정하세요.

판정 순서 (위에서부터 차례로 확인, 먼저 해당되면 그것이 답):
1. 중앙에 그릇에 담긴 액체(국물)가 있는가? → 된장국
   (그릇이 붉은색이어도, 국물이 검붉게 보여도 된장국)
2. 질감 없는 어둡고 얇은 판인가? → 김
3. 촘촘한 알갱이 질감의 흰 덩어리인가? → 흰밥
4. 매끈한 노란 단일 블록인가? → 계란말이
5. 접시 위 흩어진 조각들에 붉은 기운이 있는가? → 김치

주의: 음식의 재료(두부, 계란, 버터 등)가 아니라 위 목록의 음식 이름으로 답하세요.
예: 국물 속 두부를 가리키면 답은 "된장국"입니다.

중요: 사진 중앙 부근에 있는 음식이 답입니다. 가장자리에 다른 음식이
더 크고 선명하게 보여도 그것은 답이 아닙니다."""

if FAST_MODE:
    PROMPT = _COMMON + """

반드시 다음 5개 중 하나만 답하세요: 김치, 계란말이, 된장국, 흰밥, 김
"없음", "알 수 없는 음식", 다른 어떤 답도 금지입니다.
확실하지 않아도 가장 가능성 높은 것 하나를 반드시 고르세요.
다른 말 없이 답 하나만. 설명, 서식, 문장 금지."""
else:
    PROMPT = _COMMON + """

1단계: 식기 끝이 사진의 어느 위치에 있는지 한 문장으로 설명
2단계: 그 위치의 음식을 판정 순서에 따라 선택

굵게 표시(**) 등 어떤 서식도 사용하지 마세요.
마지막 줄은 반드시 "최종답: OOO" 형식으로, 음식 이름만 쓰세요."""

client = anthropic.Anthropic()  # ANTHROPIC_API_KEY 환경변수 사용


def encode_image(img: Image.Image) -> str:
    img = img.convert("RGB")
    img.thumbnail((MAX_SIZE, MAX_SIZE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode()


def recognize_food(img: Image.Image) -> dict:
    """이미지 1장 → 음식명 판별. 서버에서도 이 함수만 호출하면 됨."""
    # ★ 시연 시퀀스 모드 (FORCE_SEQUENCE 비어있으면 실제 인식)
    global _seq_idx
    if FORCE_SEQUENCE:
        food = FORCE_SEQUENCE[_seq_idx % len(FORCE_SEQUENCE)]
        _seq_idx += 1
      
        return {"food": food, "raw": "(시연 시퀀스)",
                "input_tokens": 0, "output_tokens": 0}

    msg = client.messages.create(
        model=MODEL, max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
             "media_type": "image/jpeg", "data": encode_image(img)}},
            {"type": "text", "text": PROMPT},
        ]}],
    )
    full = msg.content[0].text
    m = re.search(r"최종답\s*[:：]\s*(.+)", full)
    food = m.group(1) if m else full
    food = food.replace("*", "").strip().rstrip(".")

    if food not in VALID_FOODS:
        print(f"[판별] 목록 밖 응답: '{food}' → 계란말이로 대체")
        food = "계란말이"

    return {
        "food": food,
        "raw": full,
        "input_tokens": 0 if not msg else msg.usage.input_tokens,
        "output_tokens": 0 if not msg else msg.usage.output_tokens,
    }
