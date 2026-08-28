"""음식 판별 모듈 — run_test.py와 server.py가 공유"""
import anthropic, base64, io, re
from PIL import Image

MODEL = "claude-haiku-4-5"
MAX_SIZE = 1024
MAX_TOKENS = 350

FOOD_LIST = """김치(붉은 양념이 묻은 배추, 통잎 또는 썬 조각)
계란말이(노란색 계란 요리, 네모난 단면이 층층이 보임)
된장국(갈색 국물에 두부·야채가 든 국)
흰밥(흰 쌀밥)
콩자반(검은콩 조림, 윤기 있는 검은 알갱이)
김밥(김으로 만 원형 단면, 속재료가 보임)
초밥(밥 위에 생선 등이 올라간 형태)"""  # 30종 확정되면 교체

PROMPT = f"""당신은 시각장애인의 식사를 돕는 시스템입니다.

인식 가능한 음식 목록:
{FOOD_LIST}

사진에서 식기(젓가락, 숟가락, 포크)의 끝이 가리키는 음식을 찾으세요.
식기의 끝은 손에서 먼 쪽입니다. 손이나 손잡이 근처의 음식이 아니라,
손에서 먼 끝이 향하거나 닿아 있는 음식을 고르세요.

1단계: 식기 끝(손에서 먼 쪽)이 사진의 어느 위치에 있는지 한 문장으로 설명
2단계: 그 위치에 있거나 끝이 닿아 있는 음식을 위 목록에서 선택
- 식기로 음식을 집거나 뜨고 있으면 그 음식이 답입니다.
- 식기가 안 보이면 최종답은 "없음"입니다.
- 목록에 없는 음식이면 가장 비슷한 것을 고르되, 확실히 다르면 "알 수 없는 음식".

주의: 식기의 몸통이나 손이 어떤 음식 위를 지나가더라도 그 음식은 답이 아닙니다.
오직 뾰족한 끝이 닿아 있거나 가장 가까운 음식만 답하세요.

- 사진에 파란색 원이 표시되어 있으면, 그 원이 가리키는 위치의 음식이 답입니다.

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
    food = m.group(1) if m else "알 수 없는 음식"
    food = food.replace("*", "").strip().rstrip(".")
    return {
        "food": food,
        "raw": full,
        "input_tokens": msg.usage.input_tokens,
        "output_tokens": msg.usage.output_tokens,
    }
