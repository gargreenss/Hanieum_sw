"""음식 판별 모듈 — run_test.py와 server.py가 공유 (쌍별 구분 기준 v3)"""
import anthropic, base64, io, re
from PIL import Image

MODEL = "claude-haiku-4-5"
MAX_SIZE = 1024

FAST_MODE = True
MAX_TOKENS = 20 if FAST_MODE else 350

FOOD_LIST = """김치(얕은 접시 위 붉은~분홍 기운의 불규칙한 조각들)
계란말이(각지고 매끈한 블록 모양 — 조명에 희게 바래 보여도 모양이 각진 블록이면 계란말이)
된장국(깊은 그릇에 담긴 국물 — 액체 표면 광택, 잠긴 건더기)
흰밥(흰 그릇에 소복한 쌀밥 — 촘촘한 알갱이 질감)
김(마른 검은색~짙은 녹색의 얇은 판/시트)"""  # 30종 확정되면 교체

VALID_FOODS = {line.split("(")[0].strip() for line in FOOD_LIST.splitlines()}
VALID_FOODS |= {"없음", "알 수 없는 음식"}

_COMMON = f"""당신은 시각장애인의 식사를 돕는 시스템입니다.

인식 가능한 음식 목록:
{FOOD_LIST}

사진에서 식기(젓가락, 숟가락, 포크)의 끝이 가리키는 음식을 찾으세요.
- 식기로 음식을 집거나 뜨고 있으면 그 음식이 답입니다.
- 이 사진은 식기 끝 주변만 잘라낸 것이라 식기가 잘 안 보일 수 있습니다.
  식기가 안 보여도 사진 중앙의 음식을 답하세요.
- "없음"은 음식이 전혀 보이지 않을 때만(빈 식탁, 빈 그릇 바닥) 답하세요.

촬영 환경: 위에서 강한 조명이 내리쬐어 색이 바래고 사진이 흐릴 수 있습니다.
색보다 모양·질감·그릇 형태를 우선하세요.

★ 헷갈리기 쉬운 구분 (반드시 이 기준으로 판단):

[김치 vs 된장국]
- 깊은 그릇 + 액체가 사진의 절반 이상 = 된장국 (그릇이 붉어도 된장국)
- 얕은 접시 + 조각들 위주 = 김치 (접시에 김칫국물이 고여 있어도 김치)

[계란말이 vs 흰밥]
- 각진 블록 모양 + 표면 매끈 = 계란말이 (희게 바래 보여도 계란말이)
- 둥글게 소복 + 알갱이 질감 = 흰밥
- 흰 것 같아도 모양이 각진 블록이면 무조건 계란말이

[김 vs 된장국/그림자]
- 마른 판/시트 모양 = 김
- 광택 있는 액체 = 된장국
- 어두워도 액체면 김이 아님

주의: 음식의 재료(두부, 계란, 버터 등)가 아니라 위 목록의 음식 이름으로 답하세요.
예: 국물 속 두부를 가리키면 답은 "된장국"입니다.

중요: 사진 중앙 부근에 있는 음식이 답입니다. 가장자리에 다른 음식이
더 크고 선명하게 보여도 그것은 답이 아닙니다."""

if FAST_MODE:
    PROMPT = _COMMON + """

반드시 위 목록에 있는 이름 그대로, 또는 "없음"/"알 수 없는 음식" 중 하나만 답하세요.
목록에 없는 이름(예: 두부, 버터, 반찬 재료명)은 절대 답하지 마세요.
다른 말 없이 답 하나만. 설명, 서식, 문장 금지."""
else:
    PROMPT = _COMMON + """

1단계: 식기 끝이 사진의 어느 위치에 있는지 한 문장으로 설명
2단계: 그 위치의 음식을 구분 기준에 따라 선택

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
        print(f"[판별] 목록 밖 응답 차단: '{food}' → 알 수 없는 음식")
        food = "알 수 없는 음식"

    return {
        "food": food,
        "raw": full,
        "input_tokens": msg.usage.input_tokens,
        "output_tokens": msg.usage.output_tokens,
    }
