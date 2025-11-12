"""通用辅助函数集合。"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """同步设置 Python、NumPy、PyTorch 的随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dump_dataclass(obj: Any, path: Path) -> None:
    """将数据类序列化为 JSON，常用于保存配置快照。"""
    if not is_dataclass(obj):
        raise TypeError("Only dataclass instances can be dumped")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            asdict(obj),
            f,
            indent=2,
            ensure_ascii=False,
        )  # type: ignore[arg-type]
