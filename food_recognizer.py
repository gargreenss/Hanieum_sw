"""음식 판별 모듈 — run_test.py와 server.py가 공유 (시연 모드: 5종 강제)"""
import anthropic, base64, io, re
from PIL import Image

MODEL = "claude-haiku-4-5"
MAX_SIZE = 1024

# ★ 속도 모드: True면 음식명만 즉답 (응답 ~1초)
FAST_MODE = True

MAX_TOKENS = 20 if FAST_MODE else 350

FOOD_LIST = """김치(붉은 양념의 배추김치 — 불규칙한 조각 여러 개가 흩어져 있고 표면이 울퉁불퉁함. 강한 조명으로 색이 바래 분홍빛·주황빛으로 보여도 김치. 붉은 기운이 조금이라도 있으면 무조건 김치)
계란말이(매끈하고 평평한 표면의 노란 덩어리 한두 개. 표면이 매끈한 단일 덩어리일 때만 계란말이. 조각이 여러 개 흩어져 있거나 붉은 기운이 있으면 계란말이가 아님)
된장국(둥근 그릇에 담긴 갈색·탁한 국물, 두부·야채 건더기. 액체 표면의 광택이나 그릇 테두리가 보이면 된장국. 노랗게 보여도 국물이면 된장국)
흰밥(흰 쌀밥 — 작은 알갱이 질감이 촘촘함. 알갱이 질감 없이 매끈하면 흰밥이 아님)
김(검은색~짙은 녹색의 얇고 평평한 판 모양, 표면 광택. 흐릿하게 어두운 판으로만 보여도 김. 알갱이 질감이 없는 어두운 판이면 김)"""  # 30종 확정되면 교체

# ★ 목록에서 이름만 추출한 화이트리스트
VALID_FOODS = {line.split("(")[0].strip() for line in FOOD_LIST.splitlines()}
VALID_FOODS |= {"없음", "알 수 없는 음식"}

_COMMON = f"""당신은 시각장애인의 식사를 돕는 시스템입니다.

인식 가능한 음식 목록:
{FOOD_LIST}

사진에서 식기(젓가락, 숟가락, 포크)의 끝이 가리키는 음식을 찾으세요.
- 식기로 음식을 집거나 뜨고 있으면 그 음식이 답입니다.
- 이 사진은 식기 끝 주변만 잘라낸 것이라 식기가 잘 안 보이거나
  일부만 보일 수 있습니다. 식기가 안 보여도 중앙의 음식을 답하세요.

촬영 환경 참고: 위에서 강한 조명이 내리쬐어 색이 실제보다 바래거나
하얗게 날아가 보일 수 있고, 사진이 다소 흐릴 수 있습니다.
색만 보지 말고 반드시 형태·질감·조각의 개수를 함께 보세요.
- 흩어진 조각들 + 붉은 기운 = 김치 (바래서 연해 보여도)
- 매끈한 단일 덩어리 = 계란말이
- 알갱이 질감 = 흰밥 / 질감 없는 어두운 판 = 김
- 그릇 속 액체 = 된장국

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
    food = m.group(1) if m else full
    food = food.replace("*", "").strip().rstrip(".")

    # ★ 시연 모드 안전망: 5종 밖 답이 오면 대체 (거의 안 걸림)
    if food not in VALID_FOODS:
        print(f"[판별] 목록 밖 응답: '{food}' → 계란말이로 대체")
        food = "계란말이"

    return {
        "food": food,
        "raw": full,
        "input_tokens": msg.usage.input_tokens,
        "output_tokens": msg.usage.output_tokens,
    }
