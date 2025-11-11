from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dump_dataclass(obj: Any, path: Path) -> None:
    if not is_dataclass(obj):
        raise TypeError("Only dataclass instances can be dumped")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(obj), f, indent=2, ensure_ascii=False)  # type: ignore[arg-type]
