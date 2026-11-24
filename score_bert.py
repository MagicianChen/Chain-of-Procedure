import argparse
import csv
import json
import re
from pathlib import Path

import torch
from bert_score import BERTScorer

ROOT = Path(__file__).resolve().parent
STEP_PREFIX = re.compile(r"^\s*step[_\s-]*\d+\s*[:.\-]?\s*", re.IGNORECASE)


def load_gold(path):
    gold = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            gold[str(item["id"])] = item["next_step"]
    return gold


def load_predictions(path):
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            text = STEP_PREFIX.sub("", row[-1])
            rows[str(row[0])] = text
    return rows


def main():
    parser = argparse.ArgumentParser(description="BERTScore against next_step")
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "cases.jsonl")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    gold = load_gold(args.data)
    predictions = load_predictions(args.pred)
    ids = [item_id for item_id in predictions if item_id in gold]
    if not ids:
        raise SystemExit("No overlapping ids between predictions and data.")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"loading BERTScore on {device}", flush=True)
    scorer = BERTScorer(lang="en", device=device, batch_size=args.batch_size, rescale_with_baseline=True)
    candidates = [predictions[item_id] or " " for item_id in ids]
    references = [gold[item_id] or " " for item_id in ids]
    _, _, f1 = scorer.score(candidates, references)
    values = [float(item) for item in f1]

    output = args.output or args.pred.with_name(args.pred.stem + ".bert.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for item_id, value in zip(ids, values):
            writer.writerow([item_id, f"{value:.6f}"])
    print(f"n={len(values)} BERTScore={sum(values) / len(values):.4f} -> {output}", flush=True)


if __name__ == "__main__":
    main()
