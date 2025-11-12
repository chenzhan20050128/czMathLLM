# -*- coding: utf-8 -*-
"""通用辅助函数集合。

这个模块包含一些在项目中多个地方都可能用到的小工具函数，
例如设置全局随机种子、序列化数据类等。
将这些通用功能集中在这里，可以提高代码的复用性并保持其他模块的简洁。
"""

# from __future__ import annotations: 同样是为了支持延迟解析类型注解。
from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """为所有相关的库（Python 内置 random, NumPy, PyTorch）设置全局随机种子。

    在机器学习实验中，设置固定的随机种子是保证实验结果可复现的关键步骤。
    如果不固定种子，每次运行代码时，像权重初始化、数据打乱、Dropout 等
    随机过程都会产生不同的结果，导致难以比较不同实验的效果。

    这个函数确保了项目中的主要随机性来源都被统一控制，从而增强了实验的
    科学性和可靠性。
    """
    # 设置 Python 内置的 random 模块的种子
    random.seed(seed)
    # 设置 NumPy 的随机种子
    np.random.seed(seed)
    # 设置 PyTorch 在 CPU 上的随机种子
    torch.manual_seed(seed)
    # 如果 CUDA 可用，则为所有 GPU 设置随机种子。
    # `torch.cuda.manual_seed_all` 是一个幂等操作，如果 CUDA 不可用，它不会做任何事。
    torch.cuda.manual_seed_all(seed)


def dump_dataclass(obj: Any, path: Path) -> None:
    """将一个数据类（dataclass）实例序列化为格式化的 JSON 文件。

    这在保存实验配置时非常有用。通过将 `ProjectConfig` 这样的配置对象
    保存为 JSON，我们可以精确地记录下某次实验运行时的所有超参数，
    便于日后查阅、分享和复现。

    - `is_dataclass(obj)`: 检查传入的对象是否是一个数据类实例。
    - `asdict(obj)`: `dataclasses` 模块提供的函数，可以将一个数据类实例
      递归地转换为一个字典，这对于 JSON 序列化是必需的。
    - `json.dump(...)`: 将字典写入 JSON 文件。
      - `indent=2`: 生成带 2 个空格缩进的、人类可读的 JSON 格式。
      - `ensure_ascii=False`: 允许直接写入非 ASCII 字符（如中文），
        而不是将它们转义为 `\\uXXXX` 格式，提高了可读性。
      - `default=str`: 这是一个回退函数。当 `json.dump` 遇到它不认识的
        数据类型时（例如 `pathlib.Path` 对象），它会调用 `str()` 将该对象
        转换为字符串，从而避免序列化错误。
    """
    if not is_dataclass(obj):
        raise TypeError("只能转储数据类（dataclass）的实例")

    # 确保目标目录存在，如果不存在则创建。
    path.parent.mkdir(parents=True, exist_ok=True)

    # 将 dataclass 实例转换为字典
    payload: dict[str, Any] = asdict(obj)

    # 使用 with 语句确保文件被正确关闭
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
            default=str, # 处理 Path 等非原生 JSON 类型
        )
