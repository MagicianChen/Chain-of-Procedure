import re
import string
from collections import Counter


def normalize_answer(text):
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.split())


def exact_match(prediction, ground_truth):
    pred = normalize_answer(prediction)
    gold = normalize_answer(ground_truth)
    if not gold:
        return 0.0
    return 1.0 if gold in pred else 0.0


def token_scores(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0

    special = {"yes", "no", "noanswer"}
    pred_norm = " ".join(pred_tokens)
    gold_norm = " ".join(gold_tokens)
    if pred_norm in special and pred_norm != gold_norm:
        return 0.0, 0.0, 0.0
    if gold_norm in special and pred_norm != gold_norm:
        return 0.0, 0.0, 0.0

    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0, 0.0, 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def token_f1(prediction, ground_truth):
    return token_scores(prediction, ground_truth)[2]
