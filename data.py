"""数据加载与标准化模块。

核心目标：兼容本地/云端多种格式的数据集，并统一清洗成模型训练所需的
``question``、``reasoning``、``final_answer`` 等字段。使用 `datasets`
库可以让我们无缝地在 Hugging Face Hub 与本地文件之间切换。

本文件还大量使用了 `typing` 模块中的 ``Optional``、``Iterable`` 等类型提示，
以提升静态检查友好度；`cast` 则用于告诉类型检查器在运行时一定会返回
特定类型。"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
from datasets import Dataset, interleave_datasets, load_dataset

from .config import DatasetSource, TrainingConfig
from .prompts import format_inference_prompt, format_sft_example

# 针对不同开源数据集中字段命名不一致的问题，列出可能的候选键，
# 通过“遇到第一个非空字段就返回”的策略统一接口。
QUESTION_KEYS = (
    "question",
    "prompt",
    "instruction",
    "problem",
    "input",
)
ANSWER_KEYS = (
    "answer",
    "response",
    "output",
    "completion",
    "target",
    "label",
    "solution",
    "generated_solution",
)
REASONING_KEYS = (
    "reasoning",
    "rationale",
    "chain_of_thought",
    "chain_of_thought_output",
    "cot",
    "explanation",
    "generated_solution",
)
FINAL_ANSWER_KEYS = (
    "final_answer",
    "final",
    "answer_box",
    "boxed_answer",
    "expected_answer",
    "solution",
)


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    """从 JSON/JSONL 文件中读取记录并确保返回列表。

    该函数兼容两种常见格式：顶层即为 ``list``，或使用 ``{"data": [...]}``
    包裹。若结构不符合预期则抛出异常，提醒用户检查数据文件。"""

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, dict):
        records = data.get("data")
        if isinstance(records, list):
            return list(records)
        raise ValueError("JSON 文件应包含列表或 {'data': [...]} 结构: " f"{path}")

    if not isinstance(data, list):
        raise TypeError(f"Unsupported JSON structure in {path}")

    return list(data)


def _is_hidden(path: Path) -> bool:
    """判定文件是否为隐藏文件，避免误读 ``.ipynb_checkpoints`` 等噪声。"""
    return path.name.startswith(".")


def _load_local_dataset(path: Path) -> Dataset:
    """根据后缀或目录结构动态加载本地数据集。

    通过 `Path.rglob` 遍历子目录，可以兼容 Parquet、JSONL、JSON 等格式。
    若未找到合适文件，会抛出错误提示。"""
    if path.is_dir():
        parquet_files = tuple(
            str(p) for p in path.rglob("*.parquet") if not _is_hidden(p) and p.is_file()
        )
        if parquet_files:
            return cast(
                Dataset,
                Dataset.from_parquet(list(parquet_files)),
            )

        jsonl_files = tuple(
            str(p)
            for pattern in ("*.jsonl", "*.jsonl.gz")
            for p in path.rglob(pattern)
            if not _is_hidden(p) and p.is_file()
        )
        if jsonl_files:
            return cast(
                Dataset,
                Dataset.from_json(list(jsonl_files)),  # type: ignore[arg-type]
            )

        json_files = sorted(
            p for p in path.rglob("*.json") if not _is_hidden(p) and p.is_file()
        )
        if json_files:
            records: list[dict[str, Any]] = []
            for file_path in json_files:
                records.extend(_load_json_records(file_path))
            return Dataset.from_list(records)

        raise ValueError(f"No supported data files found under {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return cast(Dataset, Dataset.from_parquet(str(path)))
    if suffix in {".jsonl", ".jsonl.gz"}:
        return cast(Dataset, Dataset.from_json(str(path)))
    if suffix == ".json":
        records = _load_json_records(path)
        return Dataset.from_list(records)

    raise ValueError(f"Unsupported file format: {path}")


def _first_non_empty(record: dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    """返回记录中第一个非空字段，用于字段名映射。"""
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
    """统一字段名称，并补充缺失的推理/最终答案。"""
    question = _first_non_empty(record, QUESTION_KEYS)
    answer = _first_non_empty(record, ANSWER_KEYS)
    reasoning = _first_non_empty(record, REASONING_KEYS)
    final_answer = _first_non_empty(record, FINAL_ANSWER_KEYS)

    if not question or not answer:
        raise ValueError("Record missing question or answer fields")

    if not reasoning:
        reasoning = answer
    if not final_answer:
        final_answer = _infer_final_answer(answer)

    return {
        "question": question,
        "answer": answer,
        "reasoning": reasoning,
        "final_answer": final_answer,
    }


def _infer_final_answer(answer: str) -> str:
    """尝试从答案文本中提取带 ``\boxed`` 的最终答案。

    若未找到盒装答案，则使用启发式规则（例如 ``Answer:`` 分隔符或末行
    文本）作为最终答案。"""
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
    """根据题目文本估计难度、标签等辅助信息。"""
    question = normalized["question"]
    reasoning = normalized["reasoning"]
    final_answer = normalized["final_answer"]

    def _length(text: str) -> int:
        return len(text.split())

    tags = []
    if any(keyword in question.lower() for keyword in ("prove", "show", "证明")):
        tags.append("proof")
    if any(sym in question for sym in ("∫", "integral", "积分")):
        tags.append("calculus")
    if any(sym in question for sym in ("√", "square", "平方根")):
        tags.append("algebra")
    if any(sym in question for sym in ("triangle", "三角形", "angle", "角")):
        tags.append("geometry")

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
        "target_value": final_answer,
    }


def _normalize_dataset(dataset: Dataset) -> Dataset:
    """批量调用 ``_normalize_record`` 清洗数据集，并移除多余字段。"""
    keep = {"question", "answer", "reasoning", "final_answer"}
    columns_to_remove = [col for col in dataset.column_names if col not in keep]
    return dataset.map(_normalize_record, remove_columns=columns_to_remove)


def load_dataset_source(
    source: DatasetSource, *, cache_dir: Optional[Path] = None
) -> Dataset:
    """根据 ``DatasetSource`` 描述加载数据。

    支持三种路径：直接传入文件路径、本地目录、HF 数据仓库名。
    ``cache_dir`` 参数会传递给 `datasets.load_dataset` 用于缓存。"""
    if source.path:
        dataset = _load_local_dataset(Path(source.path))
        dataset = _normalize_dataset(dataset)
        if source.max_samples is not None:
            limit = min(len(dataset), source.max_samples)
            dataset = dataset.select(range(limit))
        return dataset

    if not source.name:
        raise ValueError("Either dataset name or path must be provided")

    potential_path = Path(source.name)
    if potential_path.exists():
        dataset = _load_local_dataset(potential_path)
        dataset = _normalize_dataset(dataset)
        if source.max_samples is not None:
            limit = min(len(dataset), source.max_samples)
            dataset = dataset.select(range(limit))
        return dataset

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
    """为数据集追加 ``metadata`` 列，后续奖励模型会参考这些信息。"""

    def _add_metadata(example: dict[str, Any]) -> dict[str, Any]:
        return {"metadata": _derive_metadata(example)}

    return dataset.map(_add_metadata, keep_in_memory=False)


def build_sft_datasets(config: TrainingConfig, tokenizer) -> tuple[Dataset, Dataset]:
    """构建 SFT 训练/验证集。

    - 根据权重将多个数据源按概率混合；
    - 通过 ``format_sft_example`` 生成 ChatML 格式文本；
    - 按照 ``eval_split_ratio`` 拆分训练/验证集合。

    由于 `datasets` 的惰性映射特性，`dataset.map` 返回的新数据集共享底层
    存储，大幅减少额外内存。"""
    datasets = []
    weights = []
    for ds in config.dataset_mix:
        dataset = load_dataset_source(ds, cache_dir=config.cache_dir)
        dataset = _attach_metadata(dataset)
        eos_token = tokenizer.eos_token
        if not eos_token:
            eos_token = tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id)

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
        raise RuntimeError("No datasets available for training")

    if len(datasets) == 1:
        merged = datasets[0]
    else:
        probs = np.array(weights, dtype=np.float64)
        probs = probs / probs.sum()
        merged = interleave_datasets(
            datasets,
            probabilities=probs.tolist(),
            seed=config.random_seed,
        )

    merged = merged.shuffle(seed=config.random_seed)

    if config.eval_split_ratio > 0:
        split = merged.train_test_split(
            test_size=config.eval_split_ratio,
            seed=config.random_seed,
        )
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = merged
        eval_dataset = merged.select(range(min(256, len(merged))))

    return train_dataset, eval_dataset


def build_grpo_dataset(
    base_dataset: Dataset, *, max_samples: Optional[int] = None
) -> Dataset:
    """将源数据整理为 GRPO 奖励模型使用的格式。"""
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

    keep = {"question", "final_answer", "metadata"}
    columns_to_drop = [col for col in dataset.column_names if col not in keep]

    return dataset.map(_to_prompt, remove_columns=columns_to_drop)
