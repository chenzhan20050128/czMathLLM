"""命令行入口的兼容性封装。

这个文件提供了一个简洁的顶层入口，使得用户可以直接通过 `python -m czMathLLM.cli`
或类似的命令来运行命令行工具。

它的唯一作用就是从 `cli_core` 模块中导入并重新导出（re-export）`main` 函数。
这种分离关注点的设计模式是良好的实践：
- `cli_core.py`: 包含所有复杂的命令行参数解析和逻辑处理。
- `cli.py`: 提供一个干净、稳定的入口点，隐藏了内部实现细节。
"""

# from __future__ import annotations: 同样是为了支持延迟解析类型注解。
from __future__ import annotations

# 从 .cli_core 模块导入 main 函数。
# 这里的 "re-export" 注释意味着这个导入的目的是为了将 main 函数暴露给
# 任何导入本模块（cli.py）的代码。
# 这使得我们可以通过 `python -m czMathLLM.cli train ...` 的方式来执行命令，
# Python 会找到这个文件并执行它，进而调用 cli_core.py 中的 main 函数。
from .cli_core import main  # re-export for `python cli.py`

# __all__ 明确定义了当其他模块使用 `from .cli import *` 时可以导入的公共 API。
# 在这里，我们只希望 `main` 函数被视为公共接口。
__all__ = ["main"]
