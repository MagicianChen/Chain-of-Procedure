import re

from cop.prompts import (
    BASELINE_PROMPT,
    COT_PROMPT,
    LOCATE_STEP_PROMPT,
    SELECT_MANUAL_PROMPT,
    SPLIT_STEPS_PROMPT,
    format_manuals,
    format_one_manual,
)


def last_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def unwrap_code_block(text):
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_manual_id(text):
    match = re.search(r"instruction manual\s*:?\s*(\d+)", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    stripped = text.strip()
    if re.fullmatch(r"\d+", stripped):
        return int(stripped)
    return None


def extract_step_number(text):
    line = last_nonempty_line(text)
    match = re.search(r"step_(\d+)", line, flags=re.IGNORECASE)
    if match is None:
        matches = list(re.finditer(r"step_(\d+)", text, flags=re.IGNORECASE))
        if not matches:
            return None
        match = matches[-1]
    return int(match.group(1))


def next_line_after_step(manual_text, step_number):
    lines = [line for line in manual_text.splitlines() if line.strip()]
    if not lines:
        return ""
    index = min(step_number, len(lines) - 1)
    return lines[index]


async def predict_baseline(llm, sample, image_path):
    prompt = BASELINE_PROMPT.format(steps=format_manuals(sample["steps"]))
    response = await llm.complete(prompt, image_path=image_path, temperature=0)
    return last_nonempty_line(response)


async def predict_cot(llm, sample, image_path):
    prompt = COT_PROMPT.format(steps=format_manuals(sample["steps"]))
    response = await llm.complete(prompt, image_path=image_path, temperature=0)
    return last_nonempty_line(response)


async def predict_ours(llm, sample, image_path):
    manuals = format_manuals(sample["steps"])
    selected = await llm.complete(
        SELECT_MANUAL_PROMPT.format(steps=manuals),
        image_path=image_path,
        temperature=0,
    )
    manual_id = extract_manual_id(unwrap_code_block(selected))
    if manual_id is None or manual_id < 1 or manual_id > len(sample["steps"]):
        split_input = manuals
    else:
        split_input = format_one_manual(sample["steps"][manual_id - 1])

    split_response = await llm.complete(
        SPLIT_STEPS_PROMPT.format(steps=split_input),
        temperature=0,
    )
    split_manual = unwrap_code_block(split_response)

    located = await llm.complete(
        LOCATE_STEP_PROMPT.format(steps=split_manual),
        image_path=image_path,
        temperature=0,
    )
    step_number = extract_step_number(located)
    if step_number is None:
        raise ValueError(f"could not parse current step from: {located!r}")
    prediction = next_line_after_step(split_manual, step_number)
    if not prediction:
        raise ValueError("split manual is empty")
    return prediction


PREDICTORS = {
    "baseline": predict_baseline,
    "cot": predict_cot,
    "ours": predict_ours,
}
