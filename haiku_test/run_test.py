import anthropic, base64, csv, glob, io, os, re, time
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image

BASE = Path(__file__).parent
load_dotenv(BASE / ".env")

if not os.getenv("ANTHROPIC_API_KEY"):
    raise SystemExit("ANTHROPIC_API_KEY를 .env에서 못 읽었어. .env 파일 확인!")

client = anthropic.Anthropic()

# ── 설정 ──────────────────────────────
MODEL = "claude-haiku-4-5"
MAX_SIZE = 1024  # 긴 변 기준 리사이즈 (px)

FOOD_LIST = """김치, 계란말이, 된장국, 흰밥, 콩자반, 김밥, 유부초밥,
초밥, 샐러드, 김치찌개, 돈까스, 미소된장국"""  # ← 나중에 30개로 확장

PROMPT = f"""당신은 시각장애인의 식사를 돕는 시스템입니다.

인식 가능한 음식 목록:
{FOOD_LIST}

사진에서 젓가락 끝이 가리키는 음식을 찾으세요.
1단계: 젓가락 끝(뾰족한 쪽)이 사진의 어느 위치에 있는지 설명
2단계: 그 위치에 있거나 끝이 닿아 있는 음식을 위 목록에서 선택
- 젓가락으로 음식을 집고 있으면 그 음식이 답입니다.
- 젓가락이 안 보이면 최종답은 "없음"입니다.
- 목록에 없는 음식이면 가장 비슷한 것을 고르되, 확실히 다르면 "알 수 없는 음식".

마지막 줄은 반드시 "최종답: OOO" 형식으로 쓰세요."""

# ── 이미지 → 리사이즈 → base64 (전부 JPEG로 통일) ──
def load_image(path):
    img = Image.open(path).convert("RGB")
    img.thumbnail((MAX_SIZE, MAX_SIZE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode()

# ── 정답표 읽기 ──
answers = {}
with open(BASE / "answers.csv", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        answers[row["file"].strip()] = row["answer"].strip()

# ── 테스트 루프 ──
results, correct = [], 0
total_in = total_out = total_time = 0

files = sorted(glob.glob(str(BASE / "test_images" / "*")))
if not files:
    raise SystemExit("test_images 폴더에 사진이 없어!")

for path in files:
    name = os.path.basename(path)
    try:
        img = load_image(path)
    except Exception as e:
        print(f"{name:35s} 이미지 읽기 실패 → 건너뜀 ({e})")
        continue

    t0 = time.time()
    msg = client.messages.create(
        model=MODEL, max_tokens=200,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
             "media_type": "image/jpeg", "data": img}},
            {"type": "text", "text": PROMPT},
        ]}],
    )
    latency = time.time() - t0

    full = msg.content[0].text
    m = re.search(r"최종답\s*[:：]\s*(.+)", full)
    pred = m.group(1).strip() if m else full.strip().splitlines()[-1]

    truth = answers.get(name, "?")
    ok = truth != "?" and (truth in pred or pred in truth)
    correct += ok

    total_in += msg.usage.input_tokens
    total_out += msg.usage.output_tokens
    total_time += latency

    results.append([name, truth, pred, "O" if ok else "X",
                    f"{latency:.2f}", msg.usage.input_tokens, msg.usage.output_tokens])
    print(f"{name:35s} 정답:{truth:8s} 예측:{pred:10s} {'O' if ok else 'X'}  {latency:.2f}s")

# ── 결과 요약 ──
n = len(results)
cost = total_in / 1e6 * 1.0 + total_out / 1e6 * 5.0
print("\n" + "=" * 55)
print(f"정확도: {correct}/{n} = {correct/n*100:.1f}%")
print(f"평균 응답시간: {total_time/n:.2f}초")
print(f"총 토큰: 입력 {total_in} / 출력 {total_out}")
print(f"총 비용: ${cost:.4f} (약 {cost*1400:.0f}원) | 요청당 약 {cost/n*1400:.2f}원")
print(f"5만원 예산(≈$36)으로 약 {36/(cost/n):,.0f}회 요청 가능")

with open(BASE / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["file", "truth", "pred", "correct", "latency_s", "in_tok", "out_tok"])
    w.writerows(results)
print("\nresults.csv 저장 완료")