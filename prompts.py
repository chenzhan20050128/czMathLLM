from __future__ import annotations

from textwrap import dedent
from typing import Iterable

DEFAULT_SYSTEM_PROMPT = dedent(
    """
    You are an experienced mathematics tutor who explains concepts clearly, validates intermediate steps, and always concludes with a boxed final answer when appropriate. Encourage learners, highlight important formulas, and use LaTeX for math expressions.
    """
).strip()


def build_chat_prompt(question: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
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
    prompt = build_chat_prompt(question)
    cleaned_answer = answer.rstrip()
    if not cleaned_answer.endswith("</think>") and "<think>" in cleaned_answer:
        cleaned_answer += "</think>"
    return f"{prompt}{cleaned_answer}{eos_token}"


def format_inference_prompt(question: str, *, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f"{build_chat_prompt(question, system_prompt=system_prompt)}"


def batched_prompts(questions: Iterable[str], *, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> list[str]:
    return [format_inference_prompt(q, system_prompt=system_prompt) for q in questions]
