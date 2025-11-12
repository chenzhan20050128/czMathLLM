# -*- coding: utf-8 -*-
"""奖励函数（Reward Function）相关的工具。

该模块的核心任务是定义一个 `score_math_answer` 函数，用于评估模型
生成的答案（prediction）与标准答案（reference）的匹配程度，并给出一个
数值奖励分数。这个分数在强化学习（如 GRPO）阶段至关重要，因为它指导着
模型的优化方向，即让模型学会生成能够获得更高奖励分数的答案。

这里的奖励机制是专门为数学问题设计的，综合考虑了多种情况。
"""

# from __future__ import annotations: 同样是为了支持延迟解析类型注解。
from __future__ import annotations

import math
import re
from typing import Iterable

# --- 正则表达式定义 ---
# 使用 `re.compile` 预编译正则表达式可以提升重复使用时的性能。

# 匹配 LaTeX 的 `\boxed{...}` 命令，用于提取最终答案。
# - `r"..."`: 原始字符串（raw string），可以避免反斜杠 `\` 的转义问题。
# - `\\boxed\{`: 匹配字面量 `\boxed{`。
# - `([^}]*)`: 这是一个捕获组。`[^}]*` 匹配任意数量（`*`）的非 `}` 字符。
#   捕获组 `(...)` 会将匹配到的内容保存下来，以便后续通过 `match.group(1)` 获取。
# - `\}`: 匹配字面量 `}`。
BOX_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")

# 匹配通用的数值格式，包括整数、小数和科学记数法。
# - `[-+]?`: 可选的正负号。
# - `\d*\.?\d+`: 匹配数字，可以有小数点。例如 `123`, `.5`, `1.23`。
# - `(?:[eE][-+]?\d+)?`: 一个可选的非捕获组 `(?:...)`，用于匹配科学记数法部分，
#   例如 `e-5`, `E+10`。
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
    return [match.group(0) for match in NUMBER_PATTERN.finditer(text)]


def normalize_expression(expr: str) -> str:
    """对字符串进行轻量级归一化，主要用于非数值答案的比较。

    例如，移除多余的空格和 LaTeX 命令中的反斜杠。
    """
    cleaned = expr.strip()
    cleaned = cleaned.replace("\\", "")
    return cleaned


def numerical_reward(pred: str, target: str) -> float:
    """计算基于数值匹配的奖励。

    - 尝试将预测值和目标值都转换为浮点数。如果任一转换失败，则奖励为 0。
    - 如果两个数值非常接近（使用 `math.isclose` 判断），则给予满分奖励 1.0。
    - 如果不接近，则根据差异大小给予一个指数衰减的奖励。差异越大，奖励越低。
      这种平滑的奖励函数比简单的 0/1 奖励能提供更丰富的梯度信号。
    """
    try:
        pred_val = float(pred)
        target_val = float(target)
    except (ValueError, TypeError):
        return 0.0

    # `rel_tol` 是相对容差，`abs_tol` 是绝对容差。
    if math.isclose(pred_val, target_val, rel_tol=1e-4, abs_tol=1e-4):
        return 1.0

    diff = abs(pred_val - target_val)
    scale = max(abs(target_val), 1.0) # 使用目标值的大小进行缩放，避免大数值产生过大的相对差异。
    return max(0.0, math.exp(-diff / scale))


def string_reward(pred: str, target: str) -> float:
    """计算基于字符串相似度的奖励，适用于非数值或符号表达式的答案。

    - 首先对预测和目标字符串进行归一化。
    - 如果归一化后完全相同（忽略大小写），则给予满分奖励 1.0。
    - 否则，计算两个字符串按空格分割后的单词集合的交集大小，并将其
      与目标字符串的单词数之比作为奖励。这是一种基于 Jaccard 相似系数的变体。
    """
    pred_norm = normalize_expression(pred)
    target_norm = normalize_expression(target)
    if not pred_norm or not target_norm:
        return 0.0
    if pred_norm.lower() == target_norm.lower():
        return 1.0

    pred_words = set(pred_norm.split())
    target_words = set(target_norm.split())
    overlap = len(pred_words & target_words) # `&` 操作符计算集合的交集
    denom = max(len(target_words), 1)
    return overlap / denom


def score_math_answer(
    prediction: str,
    reference: str,
    *,
    metadata: dict | None = None,
) -> float:
    """综合奖励函数，用于给出一个最终的奖励分数。

    这是本模块的核心，它整合了多种策略来给出一个尽可能准确的评估：
    1.  **提取目标答案**: 首先从参考答案 `reference` 中提取 `\boxed` 内容，
        如果找不到，则使用整个参考答案作为目标 `target`。
    2.  **提取预测候选**: 从模型生成的 `prediction` 中提取候选答案。
        - 优先使用 `\boxed` 内容。
        - 如果没有，则提取所有数值。
        - 如果连数值都没有，则使用生成文本的最后一行作为候选。
    3.  **计算最佳分数**: 遍历所有预测候选，分别计算它们与目标的 `numerical_reward`
        和 `string_reward`，取其中的最大值作为该候选的分数。最终，所有候选的
        最高分即为 `best` 分数。
    4.  **元数据加成**: 根据 `metadata` 中的信息（如题目难度）对分数进行微调。
        例如，如果模型为一个“困难”问题生成了很长的推理过程，即使最终答案
        略有偏差，也给予一定的额外奖励，以鼓励模型进行更详细的思考。
    5.  **分数裁剪**: 确保最终分数在 [0.0, 1.2] 的合理范围内。
    """
    metadata = metadata or {}

    # 1. 确定目标答案
    target = extract_boxed(reference) or reference.strip()
    prediction = prediction.strip()

    # 2. 提取预测候选
    pred_box = extract_boxed(prediction)
    if pred_box:
        pred_candidates = [pred_box]
    else:
        pred_candidates = extract_numeric_candidates(prediction)
        if not pred_candidates and prediction:
            # 如果没有数值候选，则取最后一行作为候选
            pred_candidates = [prediction.splitlines()[-1]]

    # 3. 计算最佳分数
    best = 0.0
    for candidate in pred_candidates:
        score = max(
            numerical_reward(candidate, target),
            string_reward(candidate, target),
        )
        best = max(best, score)

    # 4. 根据元数据进行奖励调整
    difficulty = metadata.get("difficulty", "medium")
    reasoning_len = len(prediction.split()) # 使用生成答案的长度作为推理长度的近似
    if difficulty == "hard" and reasoning_len > 160:
        best += 0.1 # 对难题的详细解答给予奖励
    elif difficulty == "easy" and reasoning_len < 60:
        best += 0.05 # 对简单问题的简洁解答给予奖励

    # 5. 裁剪分数
    return float(max(0.0, min(best, 1.2)))


def batch_reward(
    predictions: Iterable[str],
    references: Iterable[str],
    metadatas: Iterable[dict],
) -> list[float]:
    """批量计算奖励分数。

    这是一个便利的封装函数，它接收一批预测、参考答案和元数据，
    然后为每一项调用 `score_math_answer`，并返回一个包含所有分数的列表。
    这通常是强化学习训练器（如 `trl` 的 `PPOTrainer`）所期望的接口格式。
    """
    return [
        score_math_answer(pred, ref, metadata=meta)
        for pred, ref, meta in zip(predictions, references, metadatas)
    ]
