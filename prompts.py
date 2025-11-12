# -*- coding: utf-8 -*-
"""提示词（Prompt）工程相关的模板和工具函数。

该模块集中管理了项目中使用的所有提示词格式，确保在训练、评估和推理等
不同阶段，模型接收到的输入格式保持一致。这对于模型的性能和稳定性至关重要。
这里主要采用了 Qwen 系列模型推荐的 ChatML 格式。
"""

# from __future__ import annotations: 同样是为了支持延迟解析类型注解。
from __future__ import annotations

# `textwrap.dedent` 是一个非常实用的小工具，用于去除多行字符串中每一行
# 开头的公共空白。这使得我们可以在代码中以整洁、缩进的方式定义多行字符串，
# 而最终得到的字符串不会包含这些多余的缩进，让代码更美观。
from textwrap import dedent
from typing import Iterable

# --- 默认系统提示 ---
# 定义一个默认的系统提示（System Prompt）。系统提示是在对话开始前给模型的
# 一个高级指令，用于设定模型的角色、行为准则和输出风格。一个好的系统提示
# 能极大地影响模型的输出质量。
DEFAULT_SYSTEM_PROMPT = dedent(
    """
    你是一位经验丰富的数学导师，能够清晰地解释概念，验证中间步骤，
    并在适当时总是用 `\\boxed{}` 框出最终答案。鼓励学习者，
    突出重要的公式，并使用 LaTeX 来表示数学表达式。
    """
).strip() # .strip() 用于移除字符串开头和结尾的空白（包括 dedent 后的换行符）。


def build_chat_prompt(question: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """构造符合 Qwen 系列模型 ChatML 规范的对话模板。

    ChatML 是一种用特殊标记来区分不同角色（如 system, user, assistant）
    的对话格式。一个典型的 ChatML 示例如下：
    <|im_start|>system
    You are a helpful assistant.
    <|im_end|>
    <|im_start|>user
    Hello!
    <|im_end|>
    <|im_start|>assistant

    这个函数就是将系统提示和用户问题（question）组装成这种格式，
    并以 `<|im_start|>assistant\n` 结尾，引导模型开始生成助手的回答。
    遵循模型预训练时使用的格式对于获得最佳性能至关重要。
    """
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
    """将一个（问题，答案）对格式化为一条监督微调（SFT）的训练样本。

    SFT 的训练数据通常是一整个完整的对话回合，即 "提示 + 回答"。
    这个函数将 `question` 和 `answer` 拼接成一个完整的 ChatML 格式字符串，
    形成一个可供模型学习的完整示例。

    关键操作：
    - 使用 `build_chat_prompt` 生成对话的 "system" 和 "user" 部分。
    - 将模型的期望输出 `answer` 附加在后面。
    - 在最末尾添加 `eos_token` (End-Of-Sequence token)，这是一个特殊标记，
      它告诉模型一句话或一个段落到这里就结束了。在训练时，这有助于模型
      学会适时地停止生成，避免产生冗长或不完整的输出。
    - 包含一个小的健壮性处理：如果答案中包含了 `<think>` 标签但没有闭合，
      则自动补全 `</think>`，防止因数据格式错误影响训练。
    """
    prompt = build_chat_prompt(question)
    cleaned_answer = answer.rstrip() # 移除答案末尾的空白
    # 这是一个小的容错处理，确保 <think> 标签总是成对出现。
    if not cleaned_answer.endswith("</think>") and "<think>" in cleaned_answer:
        cleaned_answer += "</think>"
    return f"{prompt}{cleaned_answer}{eos_token}"


def format_inference_prompt(
    question: str, *, system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> str:
    """为推理（Inference）阶段构造提示词。

    在推理时，我们只需要提供对话的上下文（system 和 user 的部分），
    然后让模型来生成 `assistant` 的部分。因此，这个函数本质上只是
    `build_chat_prompt` 的一个简单封装，以保持 API 接口的统一性和清晰性。
    它明确地表示这个函数是用于生成推理时输入给模型的文本。
    """
    return f"{build_chat_prompt(question, system_prompt=system_prompt)}"


def batched_prompts(
    questions: Iterable[str], *, system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> list[str]:
    """为一批问题批量生成推理用的提示词。

    这是一个便利函数，它接收一个问题列表（或任何可迭代对象），然后对其中的每个问题调用
    `format_inference_prompt`，最后返回一个提示词字符串的列表。
    这在批量评估或批量推理的场景下非常有用，可以一次性准备好所有输入。
    """
    # 使用列表推导式（list comprehension）高效地完成批量转换。
    # 这是 Python 中处理序列转换的惯用且高效的方式。
    return [format_inference_prompt(q, system_prompt=system_prompt) for q in questions]
