# -*- coding: utf-8 -*-
"""离线评估工具。

该模块的核心功能是提供一个 `evaluate_model` 函数，用于对训练好的模型
进行性能评估。它通过在验证集上运行模型生成答案，然后使用奖励模型
（`reward.py`）对生成的答案进行打分，最终将问题、参考答案、生成答案
和奖励分数汇总到一个 `pandas.DataFrame` 中，便于进行后续的分析和可视化。
"""

# from __future__ import annotations: 同样是为了支持延迟解析类型注解。
from __future__ import annotations

from typing import Optional

# pandas 是 Python 中用于数据分析和处理的强大库。
import pandas as pd

from .config import ProjectConfig
from .data import build_sft_datasets
from .modeling import generate_answers, load_base_model
from .prompts import format_inference_prompt
from .reward import batch_reward


def evaluate_model(
    project: ProjectConfig,
    *, # 强制后续参数为关键字参数
    model_path: Optional[str] = None,
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """对模型进行离线评估，并以 pandas DataFrame 的形式返回详细结果。

    执行流程:
    1.  加载模型和分词器。`model_path` 可以指定一个特定的模型路径（例如，
        一个合并后的模型或一个 LoRA 适配器目录），如果未提供，则会根据
        `project.training` 配置加载 SFT 阶段训练好的模型。
    2.  使用 `build_sft_datasets` 构建 SFT 的验证集。这个验证集是评估的基础。
    3.  根据 `sample_size` 或配置中的默认值，从验证集中选取一个子集进行评估，
        以控制评估时间和成本。
    4.  从数据集中提取问题（`question`），并使用 `format_inference_prompt`
        将其格式化为推理时所需的提示词格式。
    5.  调用 `generate_answers` 函数，让模型为所有提示词批量生成答案。
    6.  从数据集中提取参考答案（`final_answer`）和元数据（`metadata`）。
    7.  调用 `batch_reward` 函数，使用奖励模型为每一对（生成答案, 参考答案）
        计算奖励分数。
    8.  将所有信息（问题、参考答案、生成答案、奖励分数）组织成一个
        `pandas.DataFrame` 并返回。
    """
    training_cfg = project.training
    evaluation_cfg = project.evaluation

    # 加载模型和分词器
    model, tokenizer = load_base_model(
        training_cfg,
        model_path=str(model_path) if model_path else None,
    )

    # 构建并选择评估数据集
    _, eval_dataset = build_sft_datasets(training_cfg, tokenizer)
    limit = sample_size or evaluation_cfg.sample_size
    # `select` 方法可以高效地创建一个子集视图，而无需复制数据。
    dataset = eval_dataset.select(range(min(len(eval_dataset), limit)))

    # `dataset[:]` 是一种获取数据集中所有记录的简洁方式。
    batch = dataset[:]
    questions = batch["question"]

    # 批量准备推理提示
    prompts = [
        format_inference_prompt(
            q,
            system_prompt=evaluation_cfg.system_prompt,
        )
        for q in questions
    ]

    # 批量生成答案
    generations = generate_answers(
        model,
        tokenizer,
        prompts,
        max_new_tokens=evaluation_cfg.max_new_tokens,
    )

    # 准备奖励计算的输入
    references = batch["final_answer"]
    metadatas = batch.get("metadata") or [{} for _ in references]

    # 批量计算奖励分数
    rewards = batch_reward(generations, references, metadatas)

    # 将结果组织成 DataFrame
    df = pd.DataFrame(
        {
            "question": questions,
            "reference": references,
            "generation": generations,
            "reward": rewards,
        }
    )
    return df
