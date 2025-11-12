# -*- coding: utf-8 -*-
# from __future__ import annotations 是一种延迟注解评估的特性，
# 它允许在定义类之前使用该类的类型提示，这对于复杂的类型依赖关系特别有用。
# 在Python 3.10+中，这已成为默认行为，但在旧版本中需要显式导入。
from __future__ import annotations

"""项目配置模块。

该文件集中定义了训练、强化学习（GRPO）与评估阶段用到的所有配置
数据结构。通过 `@dataclass` 装饰器（Python 3.7+ 引入的简化数据类语法）
我们可以快速声明仅包含属性的类，避免手写 `__init__` 和 `__repr__`
等样板代码。`slots=True` 则利用 CPython 的 `__slots__` 机制约束可用属性，
既能减少内存占用，也可以在访问不存在的字段时更早报错。
"""

# `dataclasses` 模块提供了 `dataclass` 装饰器，用于自动生成特殊方法（如 __init__, __repr__）。
# `field` 用于为 `dataclass` 字段提供额外配置。
from dataclasses import dataclass, field

# `pathlib` 模块提供了面向对象的接口来处理文件系统路径。
from pathlib import Path

# `typing` 模块为Python代码提供类型提示支持。
from typing import Optional, Sequence


# --- 路径配置 ---
# `Path(__file__)` 获取当前脚本的路径。
# `.resolve()` 将路径转换为绝对路径，解析任何符号链接。
# `.parent` 获取父目录。这里连续两次 `.parent` 是为了从 `czMathLLM/czMathLLM` 目录上溯到项目根目录 `czMathLLM`。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 定义一个候选的本地基础模型路径。如果这个模型存在于本地，程序将优先使用它，以节省下载时间。
_CANDIDATE_BASE_MODEL_PATH = _PROJECT_ROOT / "models" / ("Qwen3-4B-Thinking-2507")
# 检查候选路径是否存在。如果存在，则将其解析为绝对路径字符串；否则，设为 None。
# 这是一个三元条件表达式：`value_if_true if condition else value_if_false`。
DEFAULT_BASE_MODEL_PATH = (
    str(_CANDIDATE_BASE_MODEL_PATH.resolve())
    if _CANDIDATE_BASE_MODEL_PATH.exists()
    else None
)

# 定义本地数据集的根目录。
_LOCAL_DATA_DIR = _PROJECT_ROOT / "data"
# 定义 OpenMathReasoning 数据集的本地路径。
_OPEN_MATH_LOCAL = _LOCAL_DATA_DIR / "OpenMathReasoning" / "data"
# 定义 DAPO-Math 数据集的根目录。
_DAPO_ROOT = _LOCAL_DATA_DIR / "DAPO-Math-17k-Processed"
# 定义 DAPO-Math 数据集的具体数据文件路径。
_DAPO_LOCAL = _DAPO_ROOT / "all"


# --- 默认数据集配置函数 ---


def _default_dataset_mix() -> tuple[DatasetSource, ...]:
    """构造默认的监督微调（SFT）数据配比。

    这个函数体现了“本地优先，云端回退”的策略：
    1.  检查 `_OPEN_MATH_LOCAL` 路径是否存在。
    2.  如果存在，则使用本地的 OpenMath 数据集。
    3.  如果不存在，则从 Hugging Face Hub 下载 `unsloth/OpenMathReasoning-mini` 数据集。
    4.  对 DAPO 数据集也采用相同的逻辑。
    5.  函数返回一个包含 `DatasetSource` 对象的元组（tuple）。元组是不可变序列，适合用作 `dataclass` 字段的默认值。
    """
    # 配置推理数据集 (reasoning-focused dataset)
    if _OPEN_MATH_LOCAL.exists():
        # 如果本地 OpenMath 数据存在，则创建一个指向本地路径的 DatasetSource 对象。
        reasoning_source = DatasetSource(
            path=str(_OPEN_MATH_LOCAL),  # 数据集路径
            reasoning=True,  # 标记为推理数据
            weight=0.75,  # 在混合数据时占 75% 的权重
        )
    else:
        # 如果本地数据不存在，则配置从 Hugging Face Hub 下载。
        reasoning_source = DatasetSource(
            name="unsloth/OpenMathReasoning-mini",  # Hugging Face 数据集名称
            split="cot",  # 使用 'cot' (Chain-of-Thought) 分割
            reasoning=True,
            weight=0.75,
        )

    # 配置指令微调数据集 (instruction-following dataset)
    if _DAPO_LOCAL.exists():
        # 如果本地 DAPO 数据存在，则使用本地路径。
        instruction_source = DatasetSource(
            path=str(_DAPO_LOCAL),
            reasoning=False,  # 标记为非推理数据（通用指令）
            weight=0.25,  # 占 25% 的权重
        )
    else:
        # 如果本地数据不存在，则从 Hugging Face Hub 下载。
        instruction_source = DatasetSource(
            name="mlabonne/FineTome-100k",  # 一个通用的指令微调数据集
            split="train",
            reasoning=False,
            weight=0.25,
        )

    # 返回一个包含两个数据源配置的元组。
    return (reasoning_source, instruction_source)


def _default_grpo_dataset() -> DatasetSource:
    """为 GRPO (Group-wise Reward Policy Optimization) 阶段提供默认的数据来源配置。

    同样采用“本地优先，云端回退”的策略。
    """
    if _DAPO_ROOT.exists():
        # 如果本地 DAPO 数据集根目录存在，则使用它。
        return DatasetSource(path=str(_DAPO_ROOT), reasoning=True)
    # 否则，从 Hugging Face Hub 下载。
    return DatasetSource(
        name="open-r1/DAPO-Math-17k-Processed",  # 数据集名称
        subset="en",  # 使用 'en' (英文) 子集
        split="train",  # 使用 'train' 分割
        reasoning=True,  # 标记为推理数据
    )


# --- 数据类定义 ---


# `@dataclass(slots=True)`:
#   - `@dataclass`: 自动为类生成 `__init__`, `__repr__`, `__eq__` 等方法。
#   - `slots=True`: 使用 `__slots__` 优化内存。这会创建一个固定大小的属性数组，
#     而不是为每个对象实例创建一个 `__dict__`。优点是内存占用更少，属性访问更快；
#     缺点是不能在运行时动态添加新属性。
@dataclass(slots=True)
class DatasetSource:
    """描述一个可从 Hugging Face Hub 或本地磁盘加载的数据源。

    这个类封装了加载数据集所需的所有信息，无论是来自网络还是本地文件系统。

    关键属性说明：
    - `name`/`subset`: 对应 Hugging Face 数据集仓库名和子集名。
    - `path`: 指向本地文件或目录的路径。
    - `weight`: 在混合多个数据集进行训练时，这个权重决定了从该数据源采样的频率。
    - `reasoning`: 一个布尔标记，用于区分数据集的类型。
                   `True` 表示数据集包含详细的解题步骤（链式思维），
                   `False` 表示它可能只包含问题和最终答案。
                   这个标记在数据预处理阶段非常重要，因为它决定了如何构造训练样本的目标字段。
    - `max_samples`: 可选，用于限制从该数据源加载的最大样本数，便于快速调试。
    """

    name: Optional[str] = (
        None  # HF Hub 数据集名称, e.g., "unsloth/OpenMathReasoning-mini"
    )
    subset: Optional[str] = None  # 数据集子集, e.g., "en"
    split: str = "train"  # 数据集分割, e.g., "train", "test"
    path: Optional[str] = None  # 本地数据集路径
    weight: float = 1.0  # 数据集采样权重
    reasoning: bool = True  # 是否为推理数据集
    max_samples: Optional[int] = None  # 最大样本数限制

    def display_name(self) -> str:
        """生成一个易于阅读的数据源名称，用于日志或调试输出。"""
        if self.path:
            # 如果是本地路径，只返回目录名。
            return Path(self.path).name
        if self.subset:
            # 如果有子集，格式化为 "name:subset"。
            return f"{self.name}:{self.subset}"
        # 否则，直接返回数据集名称。
        return str(self.name)


@dataclass(slots=True)
class TrainingConfig:
    """监督微调（SFT）阶段的超参数与资源配置。

    这个类集中管理了模型训练所需的所有参数，包括模型选择、量化、LoRA 配置、
    优化器参数、数据处理细节以及输出路径等。
    """

    # --- 模型加载配置 ---
    base_model_id: str = (
        "Qwen/Qwen3-4B-Thinking-2507"  # Hugging Face Hub 上的基础模型ID
    )
    base_model_path: Optional[str] = (
        DEFAULT_BASE_MODEL_PATH or "models/Qwen3-4B-Thinking-2507"
    )
    tokenizer_id: Optional[str] = (
        None  # 分词器ID，如果为 None，则使用与 base_model_id 相同的ID
    )
    max_seq_length: int = 4096  # 模型支持的最大序列长度
    dtype: Optional[str] = (
        None  # 数据类型 (e.g., "float16", "bfloat16")，None 表示自动推断
    )
    load_in_4bit: bool = True  # 是否以4位量化加载模型，可大幅减少显存占用
    load_in_8bit: bool = False  # 是否以8位量化加载模型
    full_finetuning: bool = (
        False  # 是否进行全参数微调。False 表示使用 LoRA 等参数高效微调方法
    )
    gradient_checkpointing: bool = (
        True  # 是否启用梯度检查点。这是一种用计算时间换取显存的技术
    )

    # --- LoRA (Low-Rank Adaptation) 配置 ---
    # LoRA 是一种参数高效的微调技术，它通过训练小型的“适配器”矩阵来修改模型行为，而无需改动原始的大量权重。
    lora_rank: int = 64  # LoRA 矩阵的秩。越高的秩意味着更强的表达能力，但参数也更多。
    lora_alpha: int = 64  # LoRA 缩放因子。通常设置为与 lora_rank 相同或两倍。
    lora_dropout: float = 0.05  # LoRA 层的 Dropout 概率，用于防止过拟合。
    use_rslora: bool = False  # 是否使用 Rank-Stabilized LoRA，一种改进的 LoRA 算法。

    # --- 训练过程配置 ---
    batch_size: int = 1  # 每个设备上的批次大小
    micro_batch_size: int = 2  # 实际通过模型前向传播的最小批次大小
    gradient_accumulation_steps: int = (
        1  # 梯度累积步数（有效批次 = batch_size * num_gpus）
    )
    learning_rate: float = 2e-5  # 学习率
    weight_decay: float = 0.01  # 权重衰减，一种正则化技术
    warmup_steps: int = 50  # 学习率预热步数，在训练初期逐步增加学习率以稳定训练
    num_train_epochs: float = 1.0  # 总训练轮数
    max_steps: int = -1  # 最大训练步数。如果不是-1，则覆盖 num_train_epochs
    logging_steps: int = 10  # 每隔多少步记录一次日志
    eval_steps: int = 50  # 每隔多少步进行一次评估
    save_steps: int = 200  # 每隔多少步保存一次模型检查点
    save_total_limit: int = 2  # 最多保存多少个检查点
    random_seed: int = 3407  # 随机种子，用于保证实验可复现

    # --- 路径和实验管理 ---
    output_dir: Path = Path("outputs/local_sft_big")  # 输出目录的根路径
    experiment_name: str = "qwen_math_tutor"  # 实验名称，用于构成具体的输出子目录
    dataset_mix: Sequence[DatasetSource] = field(
        default_factory=_default_dataset_mix
    )  # 训练数据集混合配置
    eval_split_ratio: float = 0.02  # 从训练集中划分出用于评估的比例
    dataset_num_proc: int = 1  # 数据预处理时使用的进程数
    cache_dir: Optional[Path] = None  # Hugging Face datasets 的缓存目录

    # --- 模型合并与保存 ---
    save_merged_model: bool = True  # 训练结束后是否将 LoRA 适配器与基础模型合并并保存
    merge_dtype: str = "fp16"  # 合并后模型的数据类型 ("fp16", "bf16", "float32")

    def base_model_local_path(self) -> str:
        """优先返回本地权重路径，若无则回退至远端模型标识。"""
        return self.base_model_path or self.base_model_id

    # `@property` 装饰器:
    #   - 将一个方法变成一个只读属性。
    #   - 调用时无需加括号，例如 `config.project_root` 而不是 `config.project_root()`。
    #   - 这使得代码更简洁，访问方式更像访问一个普通的字段。
    @property
    def project_root(self) -> Path:
        """根据当前工作目录推断项目根路径。"""
        return Path.cwd()

    @property
    def finetuned_model_dir(self) -> Path:
        """返回保存 LoRA 适配器（未合并）的目录。"""
        return self.output_dir / f"{self.experiment_name}_lora"

    @property
    def merged_model_dir(self) -> Path:
        """返回合并后（LoRA + 基座）的完整模型目录。"""
        return self.output_dir / f"{self.experiment_name}_merged"

    @property
    def checkpoints_dir(self) -> Path:
        """返回用于保存中间训练检查点的目录。"""
        return self.output_dir / "checkpoints"


@dataclass(slots=True)
class GRPOConfig:
    """基于已训练好的 LoRA 权重继续进行 GRPO 强化学习阶段的配置。

    GRPO 是一种对齐（Alignment）算法，旨在让模型的输出更符合人类偏好。
    它通过比较模型生成的多个回答，并根据奖励信号来更新策略模型，
    使其倾向于生成得分更高的回答。
    """

    enable: bool = True
    steps: int = 500
    learning_rate: float = 8e-6
    beta: float = 0.2
    clip_range: float = 0.2
    kl_coef: float = 0.06
    value_loss_coef: float = 0.01
    mini_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_generations_per_prompt: int = 2
    max_prompt_len: int = 768
    max_completion_len: int = 2048
    reward_temperature: float = 1.0
    reference_free: bool = False
    mixed_precision: Optional[str] = "bf16"
    save_steps: int = 20
    unsloth_num_chunks: int = 1
    dataset: Optional[DatasetSource] = field(default_factory=_default_grpo_dataset)
    max_tokens_per_step: Optional[int] = None

    def describe_workload(self, training: TrainingConfig) -> dict[str, int]:
        """估算单步 GRPO 训练的 token 与样本开销。"""

        effective_batch = max(
            1,
            self.mini_batch_size * max(1, self.gradient_accumulation_steps),
        )
        prompt_len = max(1, min(self.max_prompt_len, training.max_seq_length))

        # 计算可分配给 completion 的最大长度，确保不超过模型支持的序列长度。
        completion_capacity = max(1, training.max_seq_length - prompt_len)
        completion_len = max(
            1,
            min(self.max_completion_len, completion_capacity),
        )

        num_generations = max(1, self.num_generations_per_prompt)
        completions_per_step = effective_batch * num_generations

        prompt_tokens = effective_batch * prompt_len
        completion_tokens = completions_per_step * completion_len
        total_tokens = prompt_tokens + completion_tokens

        return {
            "effective_batch": effective_batch,
            "prompt_len": prompt_len,
            "completion_len": completion_len,
            "num_generations": num_generations,
            "completions_per_step": completions_per_step,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tokens_per_step": total_tokens,
        }


@dataclass(slots=True)
class EvaluationConfig:
    """离线评估阶段的推理配置。

    这个类定义了在评估模型性能时，文本生成过程所使用的参数。
    """

    sample_size: int = 100  # 用于评估的样本数量
    max_new_tokens: int = 512  # 生成文本的最大长度
    system_prompt: str = (  # 在生成时提供给模型的系统级指令
        "You are an expert math tutor. Provide concise step-by-step reasoning "
        "and highlight the final answer using \\boxed{} when appropriate."
    )


@dataclass(slots=True)
class ProjectConfig:
    """聚合项目所有阶段的配置，便于通过单一入口进行管理。

    这个顶层配置类将 `TrainingConfig`, `GRPOConfig`, 和 `EvaluationConfig` 组合在一起，
    使得在命令行接口（CLI）或主脚本中可以方便地访问和修改所有配置。
    """

    training: TrainingConfig = field(default_factory=TrainingConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def ensure_directories(self) -> None:
        """确保所有在配置中定义的输出目录都存在。

        在训练开始前调用此方法可以避免因目录不存在而导致的错误。
        `path.mkdir(parents=True, exist_ok=True)`:
          - `parents=True`: 自动创建所有必需的父目录。
          - `exist_ok=True`: 如果目录已经存在，则不引发错误。
        """
        self.training.output_dir.mkdir(parents=True, exist_ok=True)
        self.training.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.training.finetuned_model_dir.mkdir(parents=True, exist_ok=True)
        if self.training.save_merged_model:
            self.training.merged_model_dir.mkdir(parents=True, exist_ok=True)
