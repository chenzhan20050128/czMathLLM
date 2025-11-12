"""数学大模型微调工具集。

封装了监督微调（SFT）、GRPO 强化学习、离线评估等完整流程，
依赖 Unsloth 与 Hugging Face TRL 实现高效的 LoRA 训练。"""

import unsloth  # noqa: F401  # ensure Unsloth patches are applied early

from typing import Any

from .config import (
    DatasetSource,
    EvaluationConfig,
    GRPOConfig,
    ProjectConfig,
    TrainingConfig,
)


def build_parser(*args: Any, **kwargs: Any):
    """Late import wrapper to avoid eager loading of CLI module."""

    from .cli_core import build_parser as _build_parser

    return _build_parser(*args, **kwargs)


def main(*args: Any, **kwargs: Any) -> Any:
    """Late import wrapper to avoid eager loading of CLI module."""

    from .cli_core import main as _main

    return _main(*args, **kwargs)


__all__ = [
    "DatasetSource",
    "EvaluationConfig",
    "GRPOConfig",
    "ProjectConfig",
    "TrainingConfig",
    "build_parser",
    "main",
]
