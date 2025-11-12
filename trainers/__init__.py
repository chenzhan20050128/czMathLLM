# -*- coding: utf-8 -*-
"""`trainers` 包的初始化文件。

这个 `__init__.py` 文件有两个主要作用：

1.  **提升（Promote）包内成员**: 它从子模块 `sft` 和 `grpo` 中导入核心的
    训练函数 `run_sft_training` 和 `run_grpo_training`。这样做之后，
    项目中的其他部分就可以直接通过 `from trainers import run_sft_training`
    来导入这些函数，而不需要关心它们具体定义在哪个子模块中。这是一种
    常见的封装技巧，可以简化包的外部接口。

2.  **定义 `__all__`**: `__all__` 是一个列表，它明确指定了当其他模块使用
    `from trainers import *` 这种通配符导入时，哪些名称应该被导入。
    这是一种良好的编程实践，可以避免无意中导入过多不需要的名称，
    保持命名空间的清洁。
"""

# 从同级目录下的 sft.py 文件中导入 run_sft_training 函数
from .sft import run_sft_training
# 从同级目录下的 grpo.py 文件中导入 run_grpo_training 函数
from .grpo import run_grpo_training

# 定义当 `from . import *` 时要导出的公共 API
__all__ = ["run_sft_training", "run_grpo_training"]
