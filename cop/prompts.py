BASELINE_PROMPT = """
## Task: Identify the next step in the instruction manual.

## Instruction Manuals:
{steps}

## Instructions:
- Three instruction manuals are provided, each containing steps in sequential order (with no shuffling).
- Some steps may combine multiple actions into a single step.
- Based on the given image, identify the corresponding instruction manual and determine the next step.
- Ensure that your response includes only the next step, even if the current and next steps are combined in the same instruction.
- Output your result in the following format:
    [content of the next step]
- Do not include any explanations, reasoning, or additional information in your response.
""".strip()


COT_PROMPT = """
## Task: Identify the next step in the instruction manual.

## Instruction Manuals:
{steps}

## Instructions:
- Three instruction manuals are provided, each containing steps in sequential order (with no shuffling).
- Some steps may combine multiple actions into a single step.
- Based on the given image, identify the corresponding instruction manual and determine the next step.
- Think step by step. The last line of the output must use this format:
    step_X: [content of the next step]
""".strip()


SELECT_MANUAL_PROMPT = """
## Task: Identify which instruction manual the image belongs to.

## Instruction Manuals:
{steps}

## Instructions:
- Three instruction manuals are provided (start with ###), each containing steps in sequential order (with no shuffling).
- Based on the given image, identify the corresponding instruction manual among these three provided candidate instruction manuals.
- Ensure that your response includes only the manual and its id.
- Output your result in the following format:
    Instruction Manual id
- Do not include any explanations, reasoning, or additional information in your response.
""".strip()


SPLIT_STEPS_PROMPT = """
## Task: Identify and split those steps contain distinct parts of actions in the instruction manual.

## Instruction Manual:
{steps}

## Instructions:
- The instruction manual contains steps in strict sequential order (no shuffling or reordering).
- Some steps may contain two distinct parts of actions that can be separated into two steps.
- **Only split if BOTH conditions are met:**
  1. The step contains clearly separate actions (unrelated tools, completely different targets, or independent operations)
  2. **The actions are NOT causally connected or serving the same immediate goal**
- **Do NOT split if:**
  - Actions are sequential steps toward one goal (e.g., "do A to enable B, then do B")
  - One action is preparatory for the next action
  - Actions use connecting words like "then", "and then", "so that"
- Output the modified instruction manual with only the necessary splits applied, keeping all other steps and the step description unchanged.
- If there is no any step need to be splitted, please output the same instruction manual as input unchanged.
- Output your result in the following format:
    step_1: [content of step_1]
    step_2: [content of step_2]
    ...
    step_X: [content of step_X]
- Do not include any explanations, reasoning, or additional information in your response.
""".strip()


LOCATE_STEP_PROMPT = """
## Task: Identify which step in the instruction manual the input image belongs to.

## Instruction Manual:
{steps}

## Instructions:
- The instruction manual contains steps in sequential order (with no shuffling).
- Based on the given image, identify which step this image belongs to.
- Ensure that your response includes only the step of the image.
- Output your result in the following format:
    step_X: [content of the current step]
- Do not include any explanations, reasoning, or additional information in your response.
""".strip()


def format_manuals(steps):
    blocks = []
    for idx, manual in enumerate(steps, start=1):
        lines = "\n".join("- " + step.replace("\n", ": ") for step in manual["context"])
        blocks.append(f"### Instruction Manual {idx}: \n{lines}")
    return "\n".join(blocks)


def format_one_manual(manual):
    return "\n".join("- " + step.replace("\n", ": ") for step in manual["context"])
