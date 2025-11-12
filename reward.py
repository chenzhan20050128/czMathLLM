"""奖励函数相关工具。

该模块实现了从模型输出中提取最终答案、并与参考答案比对的逻辑。
奖励信号综合考虑数值匹配、字符串相似度以及题目难度等因素，
用于 GRPO 阶段的策略更新。"""

from __future__ import annotations

import math
import re
from typing import Iterable


# 使用原始字符串（前缀 r）避免大量反斜杠转义。
BOX_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")
NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def extract_boxed(text: str) -> str | None:
    """从文本中提取 ``\boxed{...}`` 包含的内容。"""
    match = BOX_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return None


def extract_numeric_candidates(text: str) -> list[str]:
    """使用正则匹配所有数值候选项。"""
    return [match.group(0) for match in NUMBER_PATTERN.finditer(text)]


def normalize_expression(expr: str) -> str:
    """轻量化归一化表达式，去掉空白与多余反斜杠。"""
    cleaned = expr.strip()
    cleaned = cleaned.replace("\\", "")
    return cleaned


def numerical_reward(pred: str, target: str) -> float:
    """基于浮点数比较的奖励。

    先尝试将字符串解析为浮点数；解析失败直接返回 0。
    若两数足够接近则奖励 1，否则按指数衰减。"""
    try:
        pred_val = float(pred)
        target_val = float(target)
    except Exception:
        return 0.0
    if math.isclose(pred_val, target_val, rel_tol=1e-4, abs_tol=1e-4):
        return 1.0
    diff = abs(pred_val - target_val)
    scale = max(abs(target_val), 1.0)
    return max(0.0, math.exp(-diff / scale))


def string_reward(pred: str, target: str) -> float:
    """基于字符串集合交集的奖励，适用于非纯数字答案。"""
    pred_norm = normalize_expression(pred)
    target_norm = normalize_expression(target)
    if not pred_norm or not target_norm:
        return 0.0
    if pred_norm.lower() == target_norm.lower():
        return 1.0
    overlap = len(set(pred_norm.split()) & set(target_norm.split()))
    denom = max(len(set(target_norm.split())), 1)
    return overlap / denom


def score_math_answer(
    prediction: str,
    reference: str,
    *,
    metadata: dict | None = None,
) -> float:
    """综合得分函数，融合数值/字符串匹配与额外元信息。"""
    metadata = metadata or {}
    target = extract_boxed(reference) or reference.strip()
    prediction = prediction.strip()

    pred_box = extract_boxed(prediction)
    if pred_box:
        pred_candidates = [pred_box]
    else:
        pred_candidates = extract_numeric_candidates(prediction)
        if not pred_candidates:
            pred_candidates = [prediction.splitlines()[-1]]

    best = 0.0
    for candidate in pred_candidates:
        score = max(
            numerical_reward(candidate, target),
            string_reward(candidate, target),
        )
        best = max(best, score)

    # 根据题目难度调节奖励，鼓励高难题生成更完整的推理链。
    difficulty = metadata.get("difficulty", "medium")
    reasoning_len = metadata.get("reasoning_length", 120)
    if difficulty == "hard" and reasoning_len > 160:
        best += 0.1
    elif difficulty == "easy" and reasoning_len < 60:
        best += 0.05

    return float(max(0.0, min(best, 1.2)))


def batch_reward(
    predictions: Iterable[str],
    references: Iterable[str],
    metadatas: Iterable[dict],
) -> list[float]:
    """批量计算奖励，经常被 RL Trainer 调用。"""

    return [
        score_math_answer(pred, ref, metadata=meta)
        for pred, ref, meta in zip(predictions, references, metadatas)
    ]
