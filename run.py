import argparse
import asyncio
import csv
import json
import os
from pathlib import Path

from openai import AuthenticationError

from cop.llm import LLMClient
from cop.metrics import exact_match, token_scores
from cop.predict import PREDICTORS

ROOT = Path(__file__).resolve().parent


def load_samples(path):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_finished_ids(path):
    if not path.exists() or path.stat().st_size == 0:
        return set()
    finished = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row:
                finished.add(str(row[0]))
    return finished


def parse_args():
    parser = argparse.ArgumentParser(description="ProcedureVQA next-step prediction")
    parser.add_argument("--method", choices=["baseline", "cot", "ours", "all"], default="ours")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "cases.jsonl")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    return parser.parse_args()


async def run_method(method, samples, args, llm):
    predict = PREDICTORS[method]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{method}.csv"
    finished = load_finished_ids(output_path)
    pending = [sample for sample in samples if str(sample["id"]) not in finished]
    if args.limit is not None:
        pending = pending[: args.limit]

    print(
        f"\n[{method}] model={args.model} pending={len(pending)} workers={args.workers} "
        f"thinking={args.enable_thinking} effort={args.reasoning_effort} max_tokens={args.max_tokens} "
        f"-> {output_path}",
        flush=True,
    )
    if not pending:
        return

    write_lock = asyncio.Lock()
    sem = asyncio.Semaphore(args.workers)
    scores = []
    failures = []
    error_path = output_path.with_suffix(".errors.log")

    async def handle(sample):
        image_path = args.data_root / sample["img_path"]
        gold = sample["next_step"]
        async with sem:
            try:
                prediction = await predict(llm, sample, str(image_path))
            except AuthenticationError:
                raise
            except Exception as exc:
                async with write_lock:
                    failures.append(sample["id"])
                    with open(error_path, "a", encoding="utf-8") as log:
                        log.write(f"{sample['id']}\t{exc}\n")
                    print(f"[{method}] id={sample['id']} failed: {exc}", flush=True)
                return
        em = exact_match(prediction, gold)
        precision, recall, f1 = token_scores(prediction, gold)
        async with write_lock:
            writer.writerow([
                sample["id"],
                f"{em:.1f}",
                f"{precision:.4f}",
                f"{recall:.4f}",
                f"{f1:.4f}",
                prediction,
            ])
            output_file.flush()
            scores.append((em, f1))
            done = len(scores)
            if done % 20 == 0 or done == len(pending):
                mean_em = sum(item[0] for item in scores) / done
                mean_f1 = sum(item[1] for item in scores) / done
                print(
                    f"[{method}] {done}/{len(pending)} EM={mean_em:.3f} F1={mean_f1:.3f} fail={len(failures)}",
                    flush=True,
                )

    with open(output_path, "a", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        await asyncio.gather(*(handle(sample) for sample in pending))

    if scores:
        mean_em = sum(item[0] for item in scores) / len(scores)
        mean_f1 = sum(item[1] for item in scores) / len(scores)
        print(
            f"[{method}] done n={len(scores)} fail={len(failures)} EM={mean_em:.3f} F1={mean_f1:.3f}",
            flush=True,
        )


async def main():
    args = parse_args()
    if args.output_dir is None:
        safe_model = args.model.replace("/", "_")
        args.output_dir = ROOT / "outputs" / safe_model
    samples = load_samples(args.data)
    llm = LLMClient(
        model=args.model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        enable_thinking=True if args.enable_thinking else None,
        timeout=args.timeout,
    )
    methods = ["baseline", "cot", "ours"] if args.method == "all" else [args.method]
    for method in methods:
        await run_method(method, samples, args, llm)


if __name__ == "__main__":
    os.chdir(ROOT)
    asyncio.run(main())
