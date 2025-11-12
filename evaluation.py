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

# pandas 是 Python 中用于数据分析和处理的强大库，DataFrame 是其核心数据结构。
import pandas as pd

from .config import ProjectConfig
from .data import build_sft_datasets
from .modeling import generate_answers, load_base_model
from .prompts import format_inference_prompt
from .reward import batch_reward


def evaluate_model(
    project: ProjectConfig,
    *, # Python 语法：星号 `*` 强制后续参数必须以关键字形式传递（e.g., `evaluate_model(..., model_path="...")`）。
       # 这可以提高代码的可读性，避免因参数位置混淆导致的错误。
    model_path: Optional[str] = None,
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """对模型进行离线评估，并以 pandas DataFrame 的形式返回详细结果。

    此函数是模型性能验证的关键环节，它模拟了真实的推理场景，并使用量化指标（奖励分数）
    来评估模型的表现。

    执行流程:
    1.  **加载模型和分词器**:
        - 调用 `load_base_model` 加载基础模型和对应的分词器。
        - `model_path` 参数允许用户指定一个特定的模型路径进行评估（例如，一个已经合并好的模型或一个 LoRA 适配器目录）。
        - 如果 `model_path` 未提供，则会根据 `project.training` 配置加载 SFT 阶段训练好的模型，这通常是评估默认训练流程效果的标准做法。

    2.  **构建评估数据集**:
        - 使用 `build_sft_datasets` 构建 SFT 的训练集和验证集，这里我们只关心验证集 (`eval_dataset`)。
        - 这个验证集是在训练过程中未被模型见过的数据，能够客观地反映模型的泛化能力。

    3.  **采样评估数据**:
        - 根据 `sample_size` 参数或配置中的默认值 (`evaluation_cfg.sample_size`)，从验证集中选取一个子集进行评估。
        - 这样做是为了在保证评估代表性的同时，控制评估所需的时间和计算资源。

    4.  **准备推理输入**:
        - 从数据集中提取问题（`question`），并使用 `format_inference_prompt` 将其格式化为推理时所需的提示词格式（例如，ChatML 格式）。

    5.  **批量生成答案**:
        - 调用 `generate_answers` 函数，让加载的模型为所有提示词批量生成答案。批量处理能显著提高 GPU 的利用率和评估效率。

    6.  **准备奖励计算输入**:
        - 从数据集中提取参考答案（`final_answer`）和元数据（`metadata`）。参考答案是评估生成答案质量的“黄金标准”。

    7.  **批量计算奖励**:
        - 调用 `batch_reward` 函数，使用奖励模型为每一对（生成答案, 参考答案）计算奖励分数。奖励分数是一个量化指标，反映了生成答案的正确性、完整性等。

    8.  **组织并返回结果**:
        - 将所有信息（问题、参考答案、生成答案、奖励分数）组织成一个 `pandas.DataFrame`。
        - DataFrame 格式非常便于进行后续的数据分析，例如计算平均奖励、按问题类型分组统计、或者将结果保存为 CSV/Parquet 文件。
    """
    training_cfg = project.training
    evaluation_cfg = project.evaluation

    # 1. 加载模型和分词器
    model, tokenizer = load_base_model(
        training_cfg,
        model_path=str(model_path) if model_path else None,
    )

    # 2. & 3. 构建并选择评估数据集
    _, eval_dataset = build_sft_datasets(training_cfg, tokenizer)
    limit = sample_size or evaluation_cfg.sample_size
    # `select` 方法可以高效地创建一个子集视图，而无需复制数据，节省内存。
    dataset = eval_dataset.select(range(min(len(eval_dataset), limit)))

    # `dataset[:]` 是一种获取数据集中所有记录的简洁方式，返回一个字典，键是列名，值是包含所有数据的列表。
    batch = dataset[:]
    questions = batch["question"]

    # 4. 批量准备推理提示
    prompts = [
        format_inference_prompt(
            q,
            system_prompt=evaluation_cfg.system_prompt,
        )
        for q in questions
    ]

    # 5. 批量生成答案
    generations = generate_answers(
        model,
        tokenizer,
        prompts,
        max_new_tokens=evaluation_cfg.max_new_tokens,
    )

    # 6. 准备奖励计算的输入
    references = batch["final_answer"]
    # `batch.get(...) or [...]` 是一种健壮的写法，确保即使 "metadata" 列不存在，也能创建一个正确长度的空元数据列表。
    metadatas = batch.get("metadata") or [{} for _ in references]

    # 7. 批量计算奖励分数
    rewards = batch_reward(generations, references, metadatas)

    # 8. 将结果组织成 DataFrame
    df = pd.DataFrame(
        {
            "question": questions,
            "reference": references,
            "generation": generations,
            "reward": rewards,
        }
    )
    return df
