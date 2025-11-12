# -*- coding: utf-8 -*-
"""数学大模型微调工具集 `czMathLLM` 的主入口。

这个 `__init__.py` 文件是 `czMathLLM` 包的门面，它定义了
该包的公共 API，并使用了一些技巧来优化导入性能。

主要功能：
1.  **确保 Unsloth 补丁优先应用**: 在文件顶部立即导入 `unsloth`，
    确保其对 `transformers` 等库的运行时补丁能够尽早生效。
2.  **提升核心配置类**: 从 `config` 模块中导入所有的数据类配置
    （如 `ProjectConfig`, `TrainingConfig` 等），使得用户可以直接
    `from czMathLLM import ProjectConfig` 来使Late Import）用它们。
3.  **延迟加载 CLI**: 使用“延迟导入”。这意味着，只有当这两个函数
    被实际调用时，才会真正去导入重量级的 `cli_core` 模块。
    这对于希望以库（library）的形式使用 `czMathLLM` 的用户非常友好，
    因为他们可以只导入配置类而无需承担导入 `argparse` 等 CLI
    相关模块的开销。
4.  **定义公共 API**: 通过 `__all__` 列表，清晰地声明了哪些对象
    是包的公共接口，供外部使用。
"""

# 1. 确保 Unsloth 补丁优先应用
# 再次强调，这个导入虽然看起来未使用，但其副作用是关键。
import unsloth  # noqa: F401
# ↑ 这一行代码执行时，unsloth 会：
# 1. 修改 transformers 库的行为，优化训练性能
# 2. 可能修改 torch 的某些操作
# 3. 注册自定义的模型实现等
# 后续代码可以享受 unsloth 带来的优化，但不需要直接调用它
# 此时使用的 transformers 已经是经过 unsloth 优化的版本
from typing import Any

# 2. 提升核心配置类，使其成为包的顶层 API 的一部分。
from .config import (
    DatasetSource,
    EvaluationConfig,
    GRPOConfig,
    ProjectConfig,
    TrainingConfig,
)


# 3. 延迟加载 CLI 模块
def build_parser(*args: Any, **kwargs: Any):
    """延迟导入 `build_parser` 函数。

    这是一个包装函数（wrapper）。当外部代码调用 `czMathLLM.build_parser()` 时，
    此函数才会被执行。在函数内部，它才真正地从 `.cli_core` 模块导入
    `_build_parser` 函数并执行它。
    """
    from .cli_core import build_parser as _build_parser

    return _build_parser(*args, **kwargs)


def main(*args: Any, **kwargs: Any) -> Any:
    """延迟导入 `main` 函数。

    与 `build_parser` 类似，这个包装器推迟了对 `cli_core.main` 的导入，
    直到 `czMathLLM.main()` 被实际调用。
    """
    from .cli_core import main as _main

    return _main(*args, **kwargs)


# 4. 定义公共 API
# `__all__` 列表告诉 Python，当执行 `from czMathLLM import *` 时，
# 应该导入哪些名称。
__all__ = [
    "DatasetSource",
    "EvaluationConfig",
    "GRPOConfig",
    "ProjectConfig",
    "TrainingConfig",
    "build_parser",
    "main",
]
