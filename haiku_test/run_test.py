import csv, glob, os, time
from pathlib import Path
from dotenv import load_dotenv

BASE = Path(__file__).parent
load_dotenv(BASE / ".env")
if not os.getenv("ANTHROPIC_API_KEY"):
    raise SystemExit("ANTHROPIC_API_KEY를 .env에서 못 읽었어!")

from PIL import Image
from food_recognizer import recognize_food

answers = {}
with open(BASE / "answers.csv", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        answers[row["file"].strip()] = row["answer"].strip()

results, correct = [], 0
total_in = total_out = total_time = 0
files = sorted(glob.glob(str(BASE / "test_images" / "*")))

for path in files:
    name = os.path.basename(path)
    try:
        img = Image.open(path)
    except Exception as e:
        print(f"{name:35s} 읽기 실패 → 건너뜀 ({e})")
        continue

    t0 = time.time()
    r = recognize_food(img)
    latency = time.time() - t0

    truth = answers.get(name, "?")
    ok = truth != "?" and (truth in r["food"] or r["food"] in truth)
    correct += ok
    total_in += r["input_tokens"]; total_out += r["output_tokens"]; total_time += latency

    results.append([name, truth, r["food"], "O" if ok else "X",
                    f"{latency:.2f}", r["input_tokens"], r["output_tokens"]])
    print(f"{name:35s} 정답:{truth:8s} 예측:{r['food']:12s} {'O' if ok else 'X'}  {latency:.2f}s")

n = len(results)
cost = total_in/1e6*1.0 + total_out/1e6*5.0
print("\n" + "="*55)
print(f"정확도: {correct}/{n} = {correct/n*100:.1f}%")
print(f"평균 응답시간: {total_time/n:.2f}초")
print(f"총 토큰: 입력 {total_in} / 출력 {total_out}")
print(f"총 비용: ${cost:.4f} (약 {cost*1400:.0f}원) | 요청당 약 {cost/n*1400:.2f}원")

with open(BASE / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["file","truth","pred","correct","latency_s","in_tok","out_tok"])
    w.writerows(results)
print("results.csv 저장 완료")