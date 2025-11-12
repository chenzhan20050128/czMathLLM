"""提示词模板工具。

定义了 ChatML 风格的系统提示词、训练样本格式化函数等，
便于在 SFT/推理阶段保持一致的输入格式。"""

from __future__ import annotations

from textwrap import dedent
from typing import Iterable

DEFAULT_SYSTEM_PROMPT = dedent(
    """
    You are an experienced mathematics tutor who explains concepts clearly,
    validates intermediate steps, and always concludes with a boxed final
    answer when appropriate. Encourage learners, highlight important formulas,
    and use LaTeX for math expressions.
    """
).strip()


def build_chat_prompt(question: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """构造符合 Qwen ChatML 规范的对话模板。"""
    return (
        "<|im_start|>system\n"
        f"{system_prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{question.strip()}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def format_sft_example(question: str, answer: str, *, eos_token: str) -> str:
    """将问答拼接为训练集样本。

    - 末尾补上 ``eos_token``，确保模型知道何时停止；
    - 若答案中出现 ``<think>`` 却没有 ``</think>``，则自动补齐闭合标签。
    """
    prompt = build_chat_prompt(question)
    cleaned_answer = answer.rstrip()
    if not cleaned_answer.endswith("</think>") and "<think>" in cleaned_answer:
        cleaned_answer += "</think>"
    return f"{prompt}{cleaned_answer}{eos_token}"


def format_inference_prompt(
    question: str, *, system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> str:
    """推理专用的 prompt 构造函数，主要是保持接口统一。"""

    return f"{build_chat_prompt(question, system_prompt=system_prompt)}"


def batched_prompts(
    questions: Iterable[str], *, system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> list[str]:
    """批量生成 prompt，供推理或评估阶段直接使用。"""

    return [format_inference_prompt(q, system_prompt=system_prompt) for q in questions]
