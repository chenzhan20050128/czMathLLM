# -*- coding: utf-8 -*-
"""数据加载与标准化模块。

核心目标是兼容来自 Hugging Face Hub 或本地磁盘的多种格式的数据集，
并将它们统一清洗、转换为模型训练所需的标准格式，即包含 `question`、
`reasoning` 和 `final_answer` 等字段。

本模块深度整合了 `datasets` 库，利用其强大的功能（如 `load_dataset`、
`interleave_datasets`、`map`）来高效地处理数据。同时，广泛使用
`typing` 模块进行类型提示，以增强代码的可读性和健壮性。
"""

# from __future__ import annotations: 同样是为了支持延迟解析类型注解。
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
# `datasets` 是 Hugging Face 生态中用于数据处理的核心库。
from datasets import Dataset, interleave_datasets, load_dataset

from .config import DatasetSource, TrainingConfig
from .prompts import format_inference_prompt, format_sft_example

# --- 字段名标准化 ---
# 不同的开源数据集对相同的概念（如“问题”、“答案”）可能有不同的命名。
# 这里定义了一系列可能的字段名（keys），在处理数据时，会按顺序查找这些键，
# 使用找到的第一个非空值，从而实现对多种数据源的兼容。
QUESTION_KEYS = (
    "question", "prompt", "instruction", "problem", "input",
)
ANSWER_KEYS = (
    "answer", "response", "output", "completion", "target", "label", "solution", "generated_solution",
)
REASONING_KEYS = (
    "reasoning", "rationale", "chain_of_thought", "chain_of_thought_output", "cot", "explanation", "generated_solution",
)
FINAL_ANSWER_KEYS = (
    "final_answer", "final", "answer_box", "boxed_answer", "expected_answer", "solution",
)


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    """从 JSON 文件中读取记录，并确保返回一个字典列表。

    该函数设计用于处理两种常见的 JSON 格式：
    1.  文件内容本身就是一个 JSON 数组（`[...]`）。
    2.  文件内容是一个 JSON 对象，数据存储在 "data" 键下（`{"data": [...]}`）。
    如果文件格式不符合预期，会抛出异常。
    """
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, dict):
        records = data.get("data")
        if isinstance(records, list):
            return list(records)
        raise ValueError(f"JSON 文件应包含列表或 {{'data': [...]}} 结构: {path}")

    if not isinstance(data, list):
        raise TypeError(f"不支持的 JSON 结构: {path}")

    return list(data)


def _is_hidden(path: Path) -> bool:
    """判断路径是否为隐藏文件或目录（以 `.` 开头）。

    这在递归扫描目录时很有用，可以避免读取如 `.ipynb_checkpoints` 等无关文件。
    """
    return path.name.startswith(".")


def _load_local_dataset(path: Path) -> Dataset:
    """根据文件后缀或目录结构，动态地从本地加载数据集。

    这个函数是本地数据加载的核心，它能智能地处理多种情况：
    - 如果 `path` 是一个目录，它会递归地查找 `*.parquet`, `*.jsonl`, `*.json` 文件。
    - 如果 `path` 是一个文件，它会根据文件后缀（`.parquet`, `.jsonl`, `.json`）选择合适的加载方法。
    - `Path.rglob` 用于递归地搜索文件，非常方便。
    - `cast(Dataset, ...)`: 这是一个类型提示技巧。`Dataset.from_parquet` 等函数的返回值
      类型在静态分析时可能不明确，`cast` 告诉类型检查器（如 mypy）：“相信我，我知道
      这里的返回值一定是 `Dataset` 类型”，从而避免不必要的类型警告。
    """
    if path.is_dir():
        # 优先查找 Parquet 文件
        parquet_files = tuple(str(p) for p in path.rglob("*.parquet") if not _is_hidden(p) and p.is_file())
        if parquet_files:
            return cast(Dataset, Dataset.from_parquet(list(parquet_files)))

        # 其次查找 JSONL 文件
        jsonl_files = tuple(
            str(p)
            for pattern in ("*.jsonl", "*.jsonl.gz")
            for p in path.rglob(pattern)
            if not _is_hidden(p) and p.is_file()
        )
        if jsonl_files:
            return cast(Dataset, Dataset.from_json(list(jsonl_files)))

        # 最后查找 JSON 文件
        json_files = sorted(p for p in path.rglob("*.json") if not _is_hidden(p) and p.is_file())
        if json_files:
            records: list[dict[str, Any]] = []
            for file_path in json_files:
                records.extend(_load_json_records(file_path))
            return Dataset.from_list(records)

        raise ValueError(f"在 {path} 目录下未找到支持的数据文件")

    # 如果 path 是单个文件
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return cast(Dataset, Dataset.from_parquet(str(path)))
    if suffix in {".jsonl", ".jsonl.gz"}:
        return cast(Dataset, Dataset.from_json(str(path)))
    if suffix == ".json":
        records = _load_json_records(path)
        return Dataset.from_list(records)

    raise ValueError(f"不支持的文件格式: {path}")


def _first_non_empty(record: dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    """在记录中按顺序查找 `keys` 列表，返回第一个找到的非空字符串值。

    这是实现字段名标准化的关键辅助函数。它还处理了值为列表的情况，
    将其中的非空字符串用换行符连接起来。
    """
    for key in keys:
        value = record.get(key)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
        if isinstance(value, list):
            pieces: list[str] = []
            for item in value:
                text = str(item).strip()
                if text:
                    pieces.append(text)
            if pieces:
                return "\n".join(pieces)
    return None


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """将单个数据记录（字典）标准化。

    主要工作：
    1.  使用 `_first_non_empty` 和预定义的 `*_KEYS` 来提取 `question` 和 `answer`。
    2.  如果 `reasoning` 字段缺失，则用 `answer` 字段填充。
    3.  如果 `final_answer` 字段缺失，则调用 `_infer_final_answer` 从 `answer` 中推断。
    4.  返回一个包含 `question`, `answer`, `reasoning`, `final_answer` 四个标准字段的字典。
    """
    question = _first_non_empty(record, QUESTION_KEYS)
    answer = _first_non_empty(record, ANSWER_KEYS)
    reasoning = _first_non_empty(record, REASONING_KEYS)
    final_answer = _first_non_empty(record, FINAL_ANSWER_KEYS)

    if not question or not answer:
        raise ValueError("记录缺少 'question' 或 'answer' 字段")

    # 如果没有明确的推理过程，就假设整个答案都是推理过程。
    if not reasoning:
        reasoning = answer
    # 如果没有明确的最终答案，就尝试从答案文本中推断。
    if not final_answer:
        final_answer = _infer_final_answer(answer)

    return {
        "question": question,
        "answer": answer,
        "reasoning": reasoning,
        "final_answer": final_answer,
    }


def _infer_final_answer(answer: str) -> str:
    """从答案文本中智能推断最终答案。

    这是一个基于规则的启发式函数，按以下优先级尝试提取最终答案：
    1.  查找 `\\boxed{...}`: 这是数学题解中标记最终答案的常见格式。
    2.  查找 `Answer:`: 寻找 "Answer:" 标记并提取其后的内容。
    3.  查找 `最终答案`: 中文场景下的标记。
    4.  使用最后一行: 如果以上都不匹配，则假设答案的最后非空行是最终答案。
    5.  回退: 如果所有规则都失败，则返回整个答案文本。
    """
    marker = "\\boxed{"
    marker_idx = answer.rfind(marker)
    if marker_idx != -1:
        closing = answer.find("}", marker_idx + len(marker))
        if closing != -1:
            start = marker_idx + len(marker)
            return answer[start:closing]
    if "Answer:" in answer:
        return answer.split("Answer:")[-1].strip()
    if "最终答案" in answer:
        return answer.split("最终答案")[-1].strip().strip("。")
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if lines:
        return lines[-1]
    return answer.strip()


def _derive_metadata(normalized: dict[str, Any]) -> dict[str, Any]:
    """根据标准化后的记录，派生出一些元数据（metadata）。

    这些元数据（如问题长度、难度、标签）可以用于数据分析，或者在更高级的
    训练策略（如奖励建模）中作为输入特征。
    """
    question = normalized["question"]
    reasoning = normalized["reasoning"]
    final_answer = normalized["final_answer"]

    def _length(text: str) -> int:
        return len(text.split()) # 简单地以空格分割的单词数作为长度

    # 基于关键词为问题打标签
    tags = []
    if any(keyword in question.lower() for keyword in ("prove", "show", "证明")):
        tags.append("proof")
    if any(sym in question for sym in ("∫", "integral", "积分")):
        tags.append("calculus")
    if any(sym in question for sym in ("√", "square", "平方根")):
        tags.append("algebra")
    if any(sym in question for sym in ("triangle", "三角形", "angle", "角")):
        tags.append("geometry")

    # 基于推理过程的长度简单地估计难度
    difficulty = "medium"
    if _length(reasoning) > 220:
        difficulty = "hard"
    elif _length(reasoning) < 80:
        difficulty = "easy"

    return {
        "question_length": _length(question),
        "reasoning_length": _length(reasoning),
        "tags": tags,
        "difficulty": difficulty,
        "target_value": final_answer, # 最终答案也作为元数据的一部分
    }


def _normalize_dataset(dataset: Dataset) -> Dataset:
    """对整个 `datasets.Dataset` 对象进行标准化。

    它使用 `dataset.map` 方法将 `_normalize_record` 函数应用到数据集的每一行。
    `remove_columns` 参数会移除所有非标准化的原始列，只保留 `question`, `answer`,
    `reasoning`, `final_answer`，保持数据集的整洁。
    """
    keep = {"question", "answer", "reasoning", "final_answer"}
    columns_to_remove = [col for col in dataset.column_names if col not in keep]
    return dataset.map(_normalize_record, remove_columns=columns_to_remove)


def load_dataset_source(
    source: DatasetSource, *, cache_dir: Optional[Path] = None
) -> Dataset:
    """根据 `DatasetSource` 配置对象加载单个数据集。

    这是数据加载的统一入口，它会根据 `source` 的属性决定加载策略：
    - 如果 `source.path` 存在，则调用 `_load_local_dataset` 从本地加载。
    - 否则，使用 `datasets.load_dataset` 从 Hugging Face Hub 下载。
    - 加载后，会调用 `_normalize_dataset` 进行标准化。
    - 如果 `source.max_samples` 被设置，还会对数据集进行截断，方便快速调试。
    """
    if source.path:
        dataset = _load_local_dataset(Path(source.path))
        dataset = _normalize_dataset(dataset)
        if source.max_samples is not None:
            limit = min(len(dataset), source.max_samples)
            dataset = dataset.select(range(limit))
        return dataset

    if not source.name:
        raise ValueError("必须提供数据集名称或路径")

    # 兼容 source.name 实际上是一个本地路径的情况
    potential_path = Path(source.name)
    if potential_path.exists():
        dataset = _load_local_dataset(potential_path)
        dataset = _normalize_dataset(dataset)
        if source.max_samples is not None:
            limit = min(len(dataset), source.max_samples)
            dataset = dataset.select(range(limit))
        return dataset

    # 从 Hugging Face Hub 加载
    dataset = load_dataset(
        source.name,
        source.subset,
        split=source.split,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    dataset = cast(Dataset, dataset)
    dataset = _normalize_dataset(dataset)
    if source.max_samples is not None:
        limit = min(len(dataset), source.max_samples)
        dataset = dataset.select(range(limit))
    return dataset


def _attach_metadata(dataset: Dataset) -> Dataset:
    """为数据集的每一行添加一个 `metadata` 列。"""
    def _add_metadata(example: dict[str, Any]) -> dict[str, Any]:
        return {"metadata": _derive_metadata(example)}

    return dataset.map(_add_metadata, keep_in_memory=False)


def build_sft_datasets(config: TrainingConfig, tokenizer) -> tuple[Dataset, Dataset]:
    """构建用于监督微调（SFT）的训练集和验证集。

    这是 SFT 数据准备的顶层函数，执行以下步骤：
    1.  遍历 `config.dataset_mix` 中的每个数据源，使用 `load_dataset_source` 加载并标准化。
    2.  为每个数据集附加元数据。
    3.  使用 `format_sft_example` 将每个样本格式化为模型需要的 ChatML 格式文本。
        - 根据数据源的 `reasoning` 标志，决定是使用 `reasoning` 字段还是 `answer` 字段作为训练目标。
    4.  如果配置了多个数据集，使用 `interleave_datasets` 根据权重将它们混合成一个大数据集。
        这是一种流式混合方法，可以避免将所有数据一次性加载到内存中。
    5.  将混合后的数据集随机打乱。
    6.  根据 `config.eval_split_ratio` 将数据集分割为训练集和验证集。
    """
    datasets = []
    weights = []
    for ds in config.dataset_mix:
        dataset = load_dataset_source(ds, cache_dir=config.cache_dir)
        dataset = _attach_metadata(dataset)
        eos_token = tokenizer.eos_token or tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id)

        # 根据数据源是否为“推理”类型，选择不同的训练目标
        target_field = "reasoning" if ds.reasoning else "answer"

        def _format(example: dict[str, Any]) -> dict[str, Any]:
            return {
                "text": format_sft_example(
                    example["question"],
                    example[target_field],
                    eos_token=eos_token,
                )
            }

        dataset = dataset.map(_format)
        datasets.append(dataset)
        weights.append(ds.weight)

    if not datasets:
        raise RuntimeError("没有可用于训练的数据集")

    if len(datasets) == 1:
        merged = datasets[0]
    else:
        # 计算采样概率
        probs = np.array(weights, dtype=np.float64)
        probs = probs / probs.sum()
        # 混合数据集
        merged = interleave_datasets(
            datasets,
            probabilities=probs.tolist(),
            seed=config.random_seed,
        )

    merged = merged.shuffle(seed=config.random_seed)

    # 分割训练/验证集
    if config.eval_split_ratio > 0:
        split = merged.train_test_split(
            test_size=config.eval_split_ratio,
            seed=config.random_seed,
        )
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        # 如果不分割，则整个数据集都是训练集，并取一小部分作为形式上的验证集
        train_dataset = merged
        eval_dataset = merged.select(range(min(256, len(merged))))

    return train_dataset, eval_dataset


def build_grpo_dataset(
    base_dataset: Dataset, *, max_samples: Optional[int] = None
) -> Dataset:
    """将 SFT 数据集转换为 GRPO (强化学习) 阶段所需的格式。

    GRPO 训练需要的数据格式与 SFT 不同，它通常需要：
    - `prompt`: 提供给模型用于生成回答的提示。
    - `reference`: 用于计算奖励的参考答案（通常是 `final_answer`）。
    - `metadata`: 可能影响奖励计算的元数据。

    此函数负责完成这种格式转换。
    """
    dataset = base_dataset
    if "metadata" not in dataset.column_names:
        dataset = _attach_metadata(dataset)
    if max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), max_samples)))

    def _to_prompt(example: dict[str, Any]) -> dict[str, Any]:
        return {
            "prompt": format_inference_prompt(example["question"]),
            "reference": example["final_answer"],
            "metadata": example.get("metadata", {}),
        }

    # 只保留 GRPO 需要的列
    keep = {"question", "final_answer", "metadata"}
    columns_to_drop = [col for col in dataset.column_names if col not in keep]

    return dataset.map(_to_prompt, remove_columns=columns_to_drop)
