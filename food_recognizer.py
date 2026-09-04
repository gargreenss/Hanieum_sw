"""음식 판별 모듈 — run_test.py와 server.py가 공유 (시연 최종)"""
import anthropic, base64, io, re
from PIL import Image

MODEL = "claude-haiku-4-5"
MAX_SIZE = 1024

# ★ 속도 모드: True면 음식명만 즉답 (응답 ~1초)
#   False면 기존 방식 (위치 설명 후 답 — 정확도 비교용)
FAST_MODE = True

MAX_TOKENS = 20 if FAST_MODE else 350

FOOD_LIST = """김치(배추김치 — 흰 배추 줄기에 붉은 양념이 묻은 반찬. 양념이 옅어 전체적으로 희끗해 보여도, 붉거나 주황빛이 조금이라도 돌면 김치)
계란말이(매끈한 노란색 계란 요리, 네모나게 썰린 단면. 버터처럼 보이는 네모난 노란 덩어리도 계란말이. 붉은 양념이나 고춧가루가 전혀 없음)
된장국(갈색 국물에 두부·야채가 든 국. 국물이 보이면 건더기가 아니라 된장국)
흰밥(흰 쌀밥 알갱이)
김(검은색 얇은 사각형 조각, 표면에 광택. 짙은 녹색~검정의 얇은 판 모양이면 김)"""  # 30종 확정되면 교체

# ★ 목록에서 이름만 추출한 화이트리스트 — 목록 밖 답 차단용
VALID_FOODS = {line.split("(")[0].strip() for line in FOOD_LIST.splitlines()}
VALID_FOODS |= {"없음", "알 수 없는 음식"}

_COMMON = f"""당신은 시각장애인의 식사를 돕는 시스템입니다.

인식 가능한 음식 목록:
{FOOD_LIST}

사진에서 식기(젓가락, 숟가락, 포크)의 끝이 가리키는 음식을 찾으세요.
식기의 끝은 손에서 먼 쪽입니다. 손이나 손잡이 근처의 음식이 아니라,
손에서 먼 끝이 향하거나 닿아 있는 음식을 고르세요.
- 식기로 음식을 집거나 뜨고 있으면 그 음식이 답입니다.
- 식기가 안 보이면 답은 "없음"입니다.
- 목록에 없는 음식이면 가장 비슷한 것을 고르되, 확실히 다르면 "알 수 없는 음식".

주의: 식기의 몸통이나 손이 어떤 음식 위를 지나가더라도 그 음식은 답이 아닙니다.
오직 뾰족한 끝이 닿아 있거나 가장 가까운 음식만 답하세요.
주의: 음식의 재료(두부, 계란, 버터 등)가 아니라 위 목록의 음식 이름으로 답하세요.
예: 국물 속 두부를 가리키면 답은 "된장국"입니다.

중요: 이 사진은 식기 끝을 중심으로 잘라낸 것입니다.
사진 중앙 부근에 있는 음식이 답입니다. 가장자리에 다른 음식이
더 크고 선명하게 보여도 그것은 답이 아닙니다."""

if FAST_MODE:
    PROMPT = _COMMON + """

반드시 위 목록에 있는 이름 그대로, 또는 "없음"/"알 수 없는 음식" 중 하나만 답하세요.
목록에 없는 이름(예: 두부, 버터, 반찬 재료명)은 절대 답하지 마세요.
다른 말 없이 답 하나만. 설명, 서식, 문장 금지."""
else:
    PROMPT = _COMMON + """

1단계: 식기 끝(손에서 먼 쪽)이 사진의 어느 위치에 있는지 한 문장으로 설명
2단계: 그 위치에 있거나 끝이 닿아 있는 음식을 위 목록에서 선택

굵게 표시(**) 등 어떤 서식도 사용하지 마세요.
마지막 줄은 반드시 "최종답: OOO" 형식으로, 음식 이름만 쓰세요."""

client = anthropic.Anthropic()  # ANTHROPIC_API_KEY 환경변수 사용


def encode_image(img: Image.Image) -> str:
    """PIL 이미지 → 리사이즈 → JPEG → base64"""
    img = img.convert("RGB")
    img.thumbnail((MAX_SIZE, MAX_SIZE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode()


def recognize_food(img: Image.Image) -> dict:
    """이미지 1장 → 음식명 판별. 서버에서도 이 함수만 호출하면 됨."""
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
    food = m.group(1) if m else full   # FAST_MODE: 응답 전체가 곧 음식명
    food = food.replace("*", "").strip().rstrip(".")

    # ★ 화이트리스트 검증 — 목록 밖 답(두부, 버터 등)은 차단
    if food not in VALID_FOODS:
        print(f"[판별] 목록 밖 응답 차단: '{food}' → 알 수 없는 음식")
        food = "알 수 없는 음식"

    return {
        "food": food,
        "raw": full,
        "input_tokens": msg.usage.input_tokens,
        "output_tokens": msg.usage.output_tokens,
    }
