# -*- coding: utf-8 -*-
"""奖励函数（Reward Function）相关的工具。

该模块的核心任务是定义一个 `score_math_answer` 函数，用于评估模型
生成的答案（prediction）与标准答案（reference）的匹配程度，并给出一个
数值奖励分数。这个分数在强化学习（如 GRPO）阶段至关重要，因为它指导着
模型的优化方向，即让模型学会生成能够获得更高奖励分数的答案。

这里的奖励机制是专门为数学问题设计的，综合考虑了多种情况，如数值匹配、
字符串相似度以及解题过程的复杂性。
"""

# from __future__ import annotations: 同样是为了支持延迟解析类型注解。
from __future__ import annotations

import logging
import math
import os
import re
from typing import Iterable, Sequence

try:
    import requests
except ImportError:  # pragma: no cover - 作为可选依赖处理
    requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

PRM_BASE_URL = os.environ.get("PRM_API_BASE_URL", "http://127.0.0.2:8025")
PRM_SCORE_ENDPOINT = "/score"
PRM_TIMEOUT_SECONDS = float(os.environ.get("PRM_API_TIMEOUT", "30"))

MAX_REASONING_LENGTH = int(os.environ.get("PRM_MAX_REASONING_LENGTH", "4096"))

RESULT_WEIGHT = 0.4
PROCESS_WEIGHT = 0.4
LENGTH_WEIGHT = 0.2

# --- 正则表达式定义 ---
# 使用 `re.compile` 预编译正则表达式可以提升重复使用时的性能，因为编译过程只需要执行一次。

# 匹配 LaTeX 的 `\boxed{...}` 命令，用于提取最终答案。
# - `r"..."`: 原始字符串（raw string），可以避免在字符串中对反斜杠 `\` 进行转义。
# - `\\boxed\{`: 匹配字面量 `\boxed{`。`\` 在正则表达式中是特殊字符，所以需要用 `\\` 来匹配它本身。
# - `([^}]*)`: 这是一个捕获组。`[...]` 定义了一个字符集，`^` 在字符集开头表示“非”，所以 `[^}]` 匹配任何不是 `}` 的字符。
#   `*` 表示匹配前面的元素零次或多次。捕获组 `(...)` 会将匹配到的内容保存下来，以便后续通过 `match.group(1)` 获取。
# - `\}`: 匹配字面量 `}`。
BOX_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")

# 匹配通用的数值格式，包括整数、小数和科学记数法。
# - `[-+]?`: 可选的正负号。`?` 表示前面的元素出现零次或一次。
# - `\d*\.?\d+`: 匹配数字。`\d*` 匹配零个或多个数字，`\.?` 匹配零个或一个小数点，`\d+` 匹配一个或多个数字。
#   这个组合可以匹配 `123`, `.5`, `1.23` 等。
# - `(?:[eE][-+]?\d+)?`: 一个可选的非捕获组 `(?:...)`，用于匹配科学记数法部分。
#   非捕获组意味着它会进行匹配，但不会像 `(...)` 那样将匹配内容保存为单独的组。
#   这可以匹配 `e-5`, `E+10` 等。
NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def extract_boxed(text: str) -> str | None:
    """从文本中提取第一个 `\\boxed{...}` 命令中包含的内容。"""
    match = BOX_PATTERN.search(text)
    if match:
        # `match.group(1)` 返回第一个捕获组的内容，即 `([^}]*)` 匹配到的部分。
        return match.group(1).strip()
    return None


def extract_numeric_candidates(text: str) -> list[str]:
    """从文本中提取所有符合数值格式的字符串候选项。"""
    # `finditer` 返回一个迭代器，包含所有不重叠的匹配项。
    # 列表推导式 `[... for ... in ...]` 提供了一种简洁的方式来构建列表。
    return [match.group(0) for match in NUMBER_PATTERN.finditer(text)]


def normalize_expression(expr: str) -> str:
    """对字符串进行轻量级归一化，主要用于非数值答案的比较。

    例如，移除多余的空格和 LaTeX 命令中的反斜杠，使得 "x = 1" 和 "x=1"
    以及 "\\pi" 和 "pi" 在比较时能够被视为更接近。
    """
    cleaned = expr.strip()
    cleaned = cleaned.replace("\\", "")
    return cleaned


def numerical_reward(pred: str, target: str) -> float:
    """计算基于数值匹配的奖励。

    - 尝试将预测值和目标值都转换为浮点数。如果任一转换失败（说明不是有效数值），则奖励为 0。
    - 如果两个数值非常接近（使用 `math.isclose` 判断），则给予满分奖励 1.0。
    - 如果不接近，则根据差异大小给予一个指数衰减的奖励。差异越大，奖励越低。
      这种平滑的奖励函数比简单的 0/1 奖励能提供更丰富的梯度信号，有助于模型学习。
    """
    try:
        pred_val = float(pred)
        target_val = float(target)
    except (ValueError, TypeError):
        return 0.0

    # `rel_tol` 是相对容差，`abs_tol` 是绝对容差。这对于处理浮点数比较非常重要。
    if math.isclose(pred_val, target_val, rel_tol=1e-4, abs_tol=1e-4):
        return 1.0

    diff = abs(pred_val - target_val)
    # 使用目标值的大小进行缩放，避免大数值（如 10000 vs 10001）产生过大的相对差异，而小数值（0.1 vs 0.2）的差异被放大。
    scale = max(abs(target_val), 1.0)
    # 指数衰减函数，当 diff/scale 增大时，奖励迅速下降。
    return max(0.0, math.exp(-diff / scale))


def string_reward(pred: str, target: str) -> float:
    """计算基于字符串相似度的奖励，适用于非数值或符号表达式的答案。

    - 首先对预测和目标字符串进行归一化，去除格式上的细微差异。
    - 如果归一化后完全相同（忽略大小写），则给予满分奖励 1.0。
    - 否则，计算两个字符串按空格分割后的单词集合的交集大小，并将其
      与目标字符串的单词数之比作为奖励。这是一种基于 Jaccard 相似系数的变体，
      用于衡量内容的重叠程度。
    """
    pred_norm = normalize_expression(pred)
    target_norm = normalize_expression(target)
    if not pred_norm or not target_norm:
        return 0.0
    if pred_norm.lower() == target_norm.lower():
        return 1.0

    pred_words = set(pred_norm.split())
    target_words = set(target_norm.split())
    # `&` 操作符计算集合的交集，即两个集合中共同存在的元素。
    overlap = len(pred_words & target_words)
    denom = max(len(target_words), 1)
    return overlap / denom


def _compute_result_reward(
    prediction: str,
    reference: str,
    metadata: dict,
) -> float:
    """沿用历史逻辑计算结果奖励。"""
    target = extract_boxed(reference) or reference.strip()
    cleaned_prediction = prediction.strip()

    pred_box = extract_boxed(cleaned_prediction)
    if pred_box:
        pred_candidates = [pred_box]
    else:
        pred_candidates = extract_numeric_candidates(cleaned_prediction)
        if not pred_candidates and cleaned_prediction:
            pred_candidates = [cleaned_prediction.splitlines()[-1]]

    best = 0.0
    if not pred_candidates:
        return 0.0

    for candidate in pred_candidates:
        score = max(
            numerical_reward(candidate, target),
            string_reward(candidate, target),
        )
        best = max(best, score)

    difficulty = metadata.get("difficulty", "medium")
    reasoning_len = len(cleaned_prediction.split())
    if best > 0.9:
        if difficulty == "hard" and reasoning_len > 160:
            best += 0.1
        elif difficulty == "easy" and reasoning_len < 60:
            best += 0.05

    return float(max(0.0, min(best, 1.2)))


class ProcessRewardError(RuntimeError):
    """过程奖励评估异常。"""


def _split_into_steps(text: str) -> list[str]:
    """将一个长文本智能拆分成步骤列表。"""

    if not isinstance(text, str):
        return []

    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []

    # 优先使用空行切分（常用于列条步骤）。
    candidates = [
        part.strip() for part in re.split(r"\n{2,}", normalized) if part.strip()
    ]
    if len(candidates) > 1:
        return candidates

    # 其次按单行切分。
    candidates = [part.strip() for part in normalized.split("\n") if part.strip()]
    if len(candidates) > 1:
        return candidates

    # 最后按句号/分号等符号切分，保留原文本。
    candidates = [
        part.strip()
        for part in re.split(r"(?<=[。！？!?；;])\s+", normalized)
        if part.strip()
    ]
    if len(candidates) > 1:
        return candidates

    return [normalized]


def _infer_truncation(metadata: dict, reasoning_text: str | None) -> bool:
    """根据 metadata 与文本长度推断是否被截断。"""

    truncation_flags = (
        "truncated",
        "was_truncated",
        "hit_max_length",
        "cutoff",
        "length_truncated",
    )
    for key in truncation_flags:
        if metadata.get(key):
            return True

    finish_reason = str(metadata.get("finish_reason", "")).lower()
    if finish_reason in {"length", "max_tokens", "max_length"}:
        return True

    if reasoning_text and len(reasoning_text) > MAX_REASONING_LENGTH:
        return True

    return False


def _gather_process_context(
    prediction: str,
    metadata: dict,
) -> tuple[str | None, list[str], str, bool]:
    """收集生成过程相关信息，用于过程与长度奖励。"""

    question = (
        metadata.get("question") or metadata.get("prompt") or metadata.get("input")
    )

    provided_steps = metadata.get("steps") or metadata.get("process_steps")
    steps: list[str] = []
    if isinstance(provided_steps, Sequence) and not isinstance(
        provided_steps, (str, bytes)
    ):
        for item in provided_steps:
            text = str(item).strip()
            if text:
                steps.append(text)

    reasoning_text_parts: list[str] = []
    for key in ("thinking", "reasoning", "chain_of_thought", "scratchpad"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            reasoning_text_parts.append(value.strip())

    answer_text = metadata.get("answer")
    if not isinstance(answer_text, str) or not answer_text.strip():
        answer_text = prediction
    else:
        answer_text = answer_text.strip()

    if not steps:
        for part in reasoning_text_parts:
            steps.extend(_split_into_steps(part))
        if isinstance(answer_text, str):
            steps.extend(_split_into_steps(answer_text))

    cleaned_steps: list[str] = []
    previous = None
    for item in steps:
        text = item.strip()
        if text and text != previous:
            cleaned_steps.append(text)
            previous = text

    reasoning_text = "\n\n".join(reasoning_text_parts)
    if not reasoning_text:
        reasoning_text = answer_text if isinstance(answer_text, str) else prediction

    truncated = _infer_truncation(metadata, reasoning_text)
    return question, cleaned_steps, reasoning_text or "", truncated


def _fetch_process_reward(question: str, steps: Sequence[str]) -> dict:
    """调用过程奖励 API。"""

    if requests is None:
        raise ProcessRewardError("requests 库未安装，无法调用过程奖励 API。")

    url = PRM_BASE_URL.rstrip("/") + PRM_SCORE_ENDPOINT
    payload = {"question": question, "steps": list(steps)}

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=PRM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except Exception as exc:  # pragma: no cover - 网络异常路径
        raise ProcessRewardError(f"调用 {url} 失败: {exc}") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise ProcessRewardError("过程奖励 API 返回非 JSON 数据") from exc

    if not isinstance(data, dict):
        raise ProcessRewardError("过程奖励 API 返回的 JSON 不是对象类型")

    return data


def _reduce_process_scores(response_json: dict) -> float:
    """根据 API 返回值提取过程奖励分数，默认几何平均。"""

    scores = response_json.get("scores")
    values: list[float] = []
    if isinstance(scores, Sequence) and not isinstance(scores, (str, bytes)):
        for item in scores:
            try:
                values.append(float(item))
            except (TypeError, ValueError):
                continue
    if values:
        return sum(values) / len(values)

    for key in ("geometric_mean", "average", "mean", "last", "maximum"):
        value = response_json.get(key)
        if isinstance(value, (int, float)):
            return float(value)

    return 0.0


def _compute_process_reward(
    question: str | None,
    steps: Sequence[str],
) -> float:
    """计算过程奖励，若信息不足返回 0。"""

    if not question or not steps:
        return 0.0

    try:
        response = _fetch_process_reward(question, steps)
    except ProcessRewardError as exc:
        logger.warning("过程奖励 API 调用失败，默认 0 分：%s", exc)
        return 0.0

    score = _reduce_process_scores(response)
    return float(max(0.0, min(score, 1.0)))


def _compute_length_reward(reasoning_text: str, truncated: bool) -> float:
    """计算长度奖励。"""

    if truncated:
        return -0.5

    normalized = reasoning_text.strip()
    if not normalized:
        return 0.0

    length = len(normalized)
    if length > MAX_REASONING_LENGTH:
        return -0.5

    return min(length / MAX_REASONING_LENGTH, 1.0)


def score_math_answer(
    prediction: str,
    reference: str,
    *,
    metadata: dict | None = None,
) -> float:
    """综合奖励函数，用于给出一个最终的奖励分数。

    新的奖励函数包含三部分：

    1. **结果奖励（40%）**：沿用原有的答案匹配逻辑，对预测最终答案进行评分。
    2. **过程奖励（40%）**：将 `thinking`/`answer` 文本拆分为步骤列表，调用过程奖励模型 API，并对返回的分数取算术平均。
    3. **长度奖励（20%）**：鼓励更长的推理过程；若推理被截断（>4096 字符或显式标记），给予 -0.5 惩罚。

    如果缺少某一部分所需信息，则该部分奖励默认降为 0 分（长度奖励截断仍可能为负分），并继续组合其它部分，确保整体鲁棒。
    """
    metadata = metadata or {}

    result_reward = min(
        _compute_result_reward(prediction, reference, metadata),
        1.0,
    )
    question, steps, reasoning_text, truncated = _gather_process_context(
        prediction,
        metadata,
    )
    process_reward = _compute_process_reward(question, steps)
    length_reward = _compute_length_reward(reasoning_text, truncated)

    final_reward = (
        RESULT_WEIGHT * result_reward
        + PROCESS_WEIGHT * process_reward
        + LENGTH_WEIGHT * length_reward
    )
    return float(max(-1.0, min(final_reward, 1.0)))


def batch_reward(
    predictions: Iterable[str],
    references: Iterable[str],
    metadatas: Iterable[dict],
) -> list[float]:
    """批量计算奖励分数。

    这是一个便利的封装函数，它接收一批预测、参考答案和元数据，
    然后为每一项调用 `score_math_answer`，并返回一个包含所有分数的列表。
    这通常是强化学习训练器（如 `trl` 的 `PPOTrainer`）所期望的接口格式，
    因为它需要为 mini-batch 中的每个样本计算奖励。
    """
    # 使用 zip 将三个可迭代对象打包在一起，然后用列表推导式进行处理。
    return [
        score_math_answer(pred, ref, metadata=meta)
        for pred, ref, meta in zip(predictions, references, metadatas)
    ]
