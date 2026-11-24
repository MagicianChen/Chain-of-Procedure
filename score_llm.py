import argparse
import asyncio
import csv
import json
import os
import re
from pathlib import Path

from openai import AsyncOpenAI, AuthenticationError

ROOT = Path(__file__).resolve().parent
JUDGES = ["gpt-4.1-mini", "claude-haiku-5-5", "deepseek-v4.1-flash"]
PROMPT = """
## Correct Next Step:
{LABEL}

## Predict Next Step:
{PREDICT}

##Instructions:
- The 'Correcct Next Step' is the ground truth of the next step.
- The 'Predict Next Step' is the model output of the next step.
- Score the predicted next step (0-10) based on how concisely and accurately it covers all essential operations from the correct next step, without omissions or extra information.
- The score should be between 0 to 10.
- Please output the score directly without any introduction or explanation.
""".strip()


def load_gold(path):
    gold = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            gold[str(item["id"])] = f"{item['label']}\n{item['next_step']}"
    return gold


def load_predictions(path):
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row:
                rows[str(row[0])] = row[-1]
    return rows


def load_done(path):
    done = set()
    if not path.exists() or path.stat().st_size == 0:
        return done
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row:
                done.add(str(row[0]))
    return done


def parse_score(text):
    match = re.search(r"[-+]?\d*\.?\d+", text or "")
    if not match:
        return None
    score = float(match.group())
    if 0 <= score <= 10:
        return score
    return None


async def judge_once(client, model, prompt):
    delay = 2
    last_error = None
    for _ in range(8):
        try:
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 128,
            }
            if model == "deepseek-v4.1-flash":
                kwargs["extra_body"] = {"enable_thinking": False}
            response = await client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            text = message.content or getattr(message, "reasoning_content", None) or ""
            score = parse_score(text)
            if score is None:
                raise ValueError(f"unparsed score from {model}: {text!r}")
            return score
        except AuthenticationError:
            raise
        except Exception as exc:
            last_error = exc
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30)
    raise last_error


async def main():
    parser = argparse.ArgumentParser(description="LLM score, average of three judges, 0-10")
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "cases.jsonl")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args()

    output = args.output or args.pred.with_name(args.pred.stem + ".llm.csv")
    gold = load_gold(args.data)
    predictions = load_predictions(args.pred)
    done = load_done(output)
    pending = [item_id for item_id in predictions if item_id not in done and item_id in gold]
    print(f"pending={len(pending)} -> {output}", flush=True)
    if not pending:
        return

    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=args.base_url or os.environ.get("OPENAI_BASE_URL") or None,
        timeout=120,
        max_retries=0,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.workers)
    lock = asyncio.Lock()
    scores = []

    async def handle(item_id):
        prompt = PROMPT.format(LABEL=gold[item_id], PREDICT=predictions[item_id])
        async with sem:
            try:
                judged = await asyncio.gather(*(judge_once(client, name, prompt) for name in JUDGES))
            except AuthenticationError:
                raise
            except Exception as exc:
                print(f"id={item_id} failed: {exc}", flush=True)
                return
        value = sum(judged) / len(judged)
        async with lock:
            with open(output, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([item_id, f"{value:.4f}"])
            scores.append(value)
            if len(scores) % 20 == 0 or len(scores) == len(pending):
                print(f"{len(scores)}/{len(pending)} mean={sum(scores) / len(scores):.3f}", flush=True)

    await asyncio.gather(*(handle(item_id) for item_id in pending))
    if scores:
        print(f"done n={len(scores)} mean={sum(scores) / len(scores):.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
