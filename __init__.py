"""Math fine-tuning toolkit built on top of Unsloth and TRL."""

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
