# Chain-of-Procedure: Hierarchical Visual-Language Reasoning for Procedural QA

[English](README.md) | [论文](https://aclanthology.org/2026.findings-acl.850/) | [arXiv](https://arxiv.org/abs/2605.14928)

ProcedureVQA 是论文 *Chain-of-Procedure: Hierarchical Visual-Language Reasoning for Procedural QA* 的评测基准。模型看到任务做到一半时的图像，以及若干份文字流程，需要说出接下来该做的动作。它不测看图说话，也不测单轮视觉问答。模型要在一条多步流程里保持位置：认出图像属于哪一份流程，判断流程已经走到哪一步，并在文字步骤和画面切分并不对齐时给出下一步。

评测落在四个能力上：

- **流程检索。** 从工具、场景或目标相近的干扰流程里，找出图像对应的那一份。
- **状态定位。** 把图像对到刚刚完成的那一步，而不是相邻或看起来相似的步骤。
- **下一步预测。** 给出接下来的动作，而不是把已经出现的步骤再说一遍。
- **粒度对齐。** 当一步文字里叠了多个动作，或画面比文字更细时，仍能抽出可用的下一步。

本仓库包含评测源码和一小部分数据。

## Dashboard

| 排名 | 模型 | 设置 | F1 | BERTScore | LLM |
|---:|---|---|---:|---:|---:|
| 1 | gpt-6.1-sol | effort max | 0.637 | 0.648 | 75.22 |
| 2 | glm-5.3-flash | effort max | 0.627 | 0.623 | 74.67 |
| 3 | deepseek-v4.1-flash | effort high | 0.646 | 0.625 | 71.56 |
| 4 | gemini-4-argon | effort high | 0.695 | 0.644 | 69.86 |
| 5 | qwen3.5-27b | effort max | 0.673 | 0.656 | 68.69 |
| 6 | claude-opus-5-5 | effort max | 0.614 | 0.593 | 63.85 |

## Experiments results

| 模型 | 方法 | 设置 | F1 | BERTScore | LLM |
|---|---|---|---:|---:|---:|
| qwen3.5-27b | base | effort max | 0.673 | 0.656 | 74.92 |
| qwen3.5-27b | cop | effort max | **0.696** | **0.688** | **77.44** |
| gpt-4.1 | base | 不额外开推理 | 0.639 | 0.616 | 74.81 |
| gpt-4.1 | cop | 不额外开推理 | **0.686** | **0.678** | **77.56** |
| claude-opus-5-5 | base | effort medium | 0.601 | 0.581 | 69.17 |
| claude-opus-5-5 | cop | effort medium | **0.678** | **0.668** | **71.94** |

## 启动

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

上面的命令只跑仓库里附带的 5 条样例。完整评测集有 2844 条。要完整数据或有任何问题，请联系 nlp2ct.guanhua@gmail.com。拿到文件后，用 `--data` 传入。

## 引用

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
