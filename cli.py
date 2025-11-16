"""命令行入口的兼容性封装。

这个文件提供了一个简洁的顶层入口，使得用户可以直接通过 `python -m czMathLLM.cli`
或类似的命令来运行命令行工具。

它的唯一作用就是从 `cli_core` 模块中导入并重新导出（re-export）`main` 函数。
这种分离关注点的设计模式是良好的实践：
"""

# from __future__ import annotations: 同样是为了支持延迟解析类型注解。
from __future__ import annotations

from .cli_core import main  # 正确导入实际入口函数，避免递归自引

# __all__ 明确定义了当其他模块使用 `from .cli import *` 时可以导入的公共 API。
# 在这里，我们只希望 `main` 函数被视为公共接口。
__all__ = ["main"]
