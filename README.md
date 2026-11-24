# Chain-of-Procedure: Hierarchical Visual-Language Reasoning for Procedural QA

[中文说明](README_zh.md) | [Paper](https://aclanthology.org/2026.findings-acl.850/) | [arXiv](https://arxiv.org/abs/2605.14928)

ProcedureVQA is the benchmark from *Chain-of-Procedure: Hierarchical Visual-Language Reasoning for Procedural QA*. A model receives an image of a task caught mid-way and a handful of written procedures, and has to say what should happen next. The point is not a caption or a one-off visual question. The model has to stay oriented inside a multi-step activity: pick the procedure the image actually belongs to, tell how far that procedure has gone, and name the following action when the text and the photograph are not sliced at the same grain.

It probes four abilities:

- **Procedure retrieval.** Tell the matching procedure from others that share tools, scenes, or goals.
- **State grounding.** Tie the image to the step that has just been reached, rather than to a nearby or similar one.
- **Next-step prediction.** State the action that follows, not a paraphrase of the step already shown.
- **Granularity alignment.** Recover a usable next action when one written step bundles several actions, or when the image is finer than the text.

This repository ships the evaluation source and a small slice of the data.

## Dashboard

| Rank | Model | Setting | F1 | BERTScore | LLM |
|---:|---|---|---:|---:|---:|
| 1 | gpt-6.1-sol | effort max | 0.637 | 0.648 | 75.22 |
| 2 | glm-5.3-flash | effort max | 0.627 | 0.623 | 74.67 |
| 3 | deepseek-v4.1-flash | effort high | 0.646 | 0.625 | 71.56 |
| 4 | gemini-4-argon | effort high | 0.695 | 0.644 | 69.86 |
| 5 | qwen3.8-flash | effort max | 0.678 | 0.664 | 69.30 |
| 6 | qwen3.5-27b | effort max | 0.673 | 0.656 | 68.69 |
| 7 | qwen3.5-122b-a10b | effort max | 0.666 | 0.653 | 67.22 |
| 8 | claude-opus-5-5 | effort max | 0.614 | 0.593 | 63.85 |

## Experiments results

| Model | Method | Setting | F1 | BERTScore | LLM |
|---|---|---|---:|---:|---:|
| qwen3.5-27b | base | effort max | 0.673 | 0.656 | 74.92 |
| qwen3.5-27b | cop | effort max | **0.696** | **0.688** | **77.44** |
| gpt-4.1 | base | no extra reasoning | 0.639 | 0.616 | 74.81 |
| gpt-4.1 | cop | no extra reasoning | **0.686** | **0.678** | **77.56** |
| claude-opus-5-5 | base | effort medium | 0.601 | 0.581 | 69.17 |
| claude-opus-5-5 | cop | effort medium | **0.678** | **0.668** | **71.94** |

## Run

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key
export OPENAI_BASE_URL=https://api.openai.com/v1

python run.py --method base --model gpt-4.1 --workers 8
python run.py --method cot --model gpt-4.1 --workers 8
python run.py --method cop --model gpt-4.1 --workers 8

python score_bert.py --pred outputs/gpt-4.1/base.csv
python score_llm.py --pred outputs/gpt-4.1/base.csv --workers 8
```

These commands run only the five samples shipped with the repository. The full set has 2,844 examples. For the full data, or any questions, email nlp2ct.guanhua@gmail.com. Once you have the file, pass it with `--data`.

## Citation

```bibtex
@inproceedings{chen-etal-2026-chain,
    title = "Chain-of-Procedure: Hierarchical Visual-Language Reasoning for Procedural {QA}",
    author = "Chen, Guanhua  and
      Yao, Yutong  and
      Sun, Shenghe  and
      Gao, Ci-jun  and
      Liu, Shudong  and
      Chao, Lidia S.  and
      Wan, Feng  and
      Wong, Derek F.",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2026",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-acl.850/",
    doi = "10.18653/v1/2026.findings-acl.850",
    pages = "17207--17224",
    ISBN = "979-8-89176-395-1"
}
```
