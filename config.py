# -*- coding: utf-8 -*-
from __future__ import annotations

"""项目配置模块。

该文件集中定义了训练、强化学习（GRPO）与评估阶段用到的所有配置
数据结构。通过 `@dataclass` 装饰器（Python 3.7+ 引入的简化数据类语法）
我们可以快速声明仅包含属性的类，避免手写 `__init__` 和 `__repr__`
等样板代码。`slots=True` 则利用 CPython 的 `__slots__` 机制约束可用属性，
既能减少内存占用，也可以在访问不存在的字段时更早报错。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

# ===========================
# 集中常量：单一默认值来源 (Single Source of Truth)
# ===========================
# 路径 / 模型
BASE_MODEL_HF_ID_DEFAULT = "Qwen/Qwen3-4B-Thinking-2507"
BASE_MODEL_LOCAL_FALLBACK = "models/Qwen3-4B-Thinking-2507"
TOKENIZER_ID_DEFAULT: Optional[str] = None
MAX_SEQ_LENGTH_DEFAULT = 4096
DTYPE_DEFAULT: Optional[str] = None
LOAD_IN_4BIT_DEFAULT = True
LOAD_IN_8BIT_DEFAULT = False
FULL_FINETUNING_DEFAULT = False
GRADIENT_CHECKPOINTING_DEFAULT = True

# LoRA
LORA_RANK_DEFAULT = 64
LORA_ALPHA_DEFAULT = 64
LORA_DROPOUT_DEFAULT = 0.05
USE_RSLORA_DEFAULT = False

# 训练流程
BATCH_SIZE_DEFAULT = 1
MICRO_BATCH_SIZE_DEFAULT = 2
GRADIENT_ACC_STEPS_DEFAULT = 1
LEARNING_RATE_DEFAULT = 2e-5
WEIGHT_DECAY_DEFAULT = 0.01
WARMUP_STEPS_DEFAULT = 50
NUM_TRAIN_EPOCHS_DEFAULT = 1.0
MAX_STEPS_DEFAULT = -1
LOGGING_STEPS_DEFAULT = 10
EVAL_STEPS_DEFAULT = 50
SAVE_STEPS_DEFAULT = 200
SAVE_TOTAL_LIMIT_DEFAULT = 2
RANDOM_SEED_DEFAULT = 3407
OUTPUT_DIR_DEFAULT = Path("outputs/local_sft_big")
EXPERIMENT_NAME_DEFAULT = "qwen_math_tutor"
EVAL_SPLIT_RATIO_DEFAULT = 0.02
DATASET_NUM_PROC_DEFAULT = 1
CACHE_DIR_DEFAULT: Optional[Path] = None
SAVE_MERGED_MODEL_DEFAULT = True
MERGE_DTYPE_DEFAULT = "fp16"

# 数据集混合权重
REASONING_DATASET_DEFAULT_WEIGHT = 0.75
INSTRUCTION_DATASET_DEFAULT_WEIGHT = 0.25

# GRPO 默认（供 CLI 参考，不在此重复覆盖 dataclass 内已有定义）
GRPO_ENABLE_DEFAULT = True
GRPO_STEPS_DEFAULT = 500
GRPO_LR_DEFAULT = 8e-6
GRPO_BETA_DEFAULT = 0.0
GRPO_KL_COEF_DEFAULT = 0.06
GRPO_REFERENCE_FREE_DEFAULT = True
GRPO_EPSILON_DEFAULT = 0.2
GRPO_MINI_BATCH_DEFAULT = 8
GRPO_GENERATION_BATCH_DEFAULT = 4
GRPO_GRAD_ACC_DEFAULT = 4
GRPO_NUM_GENERATIONS_DEFAULT = 8
GRPO_MAX_PROMPT_LEN_DEFAULT = 768
GRPO_MAX_COMPLETION_LEN_DEFAULT = 2048
GRPO_LOGGING_STEPS_DEFAULT = 10
GRPO_SAVE_STEPS_DEFAULT = 20
GRPO_TORCH_COMPILE_DEFAULT = True
GRPO_OPTIM_DEFAULT = "adamw_8bit"
GRPO_UNSLOTH_NUM_CHUNKS_DEFAULT = 1
GRPO_UNSLOTH_GPU_MEM_UTIL_DEFAULT = 0.95

# ===========================
# 路径派生（保持在常量之后）
# ===========================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CANDIDATE_BASE_MODEL_PATH = _PROJECT_ROOT / "models" / ("Qwen3-4B-Thinking-2507")
DEFAULT_BASE_MODEL_PATH = (
    str(_CANDIDATE_BASE_MODEL_PATH.resolve())
    if _CANDIDATE_BASE_MODEL_PATH.exists()
    else None
)
_LOCAL_DATA_DIR = _PROJECT_ROOT / "data"
_OPEN_MATH_LOCAL = _LOCAL_DATA_DIR / "OpenMathReasoning" / "data"
_DAPO_ROOT = _LOCAL_DATA_DIR / "DAPO-Math-17k-Processed"
_DAPO_LOCAL = _DAPO_ROOT / "all"

# ===========================
# 默认数据集构造函数
# ===========================

def _default_dataset_mix() -> tuple[DatasetSource, ...]:
    """构造默认的监督微调（SFT）数据配比。

    这个函数体现了“本地优先，云端回退”的策略，这是一种非常实用的设计模式：
    1.  检查 `_OPEN_MATH_LOCAL` 路径是否存在。
    2.  如果存在，则使用本地的 OpenMath 数据集，避免不必要的下载。
    3.  如果不存在，则从 Hugging Face Hub 下载 `unsloth/OpenMathReasoning-mini` 数据集作为替代。
    4.  对 DAPO 数据集也采用相同的逻辑。
    5.  函数返回一个包含 `DatasetSource` 对象的元组（tuple）。元组是不可变序列，适合用作 `dataclass` 字段的默认值，
        可以防止在多个实例之间意外共享和修改默认值。
    """
    # 配置推理数据集 (reasoning-focused dataset)
    if _OPEN_MATH_LOCAL.exists():
        # 如果本地 OpenMath 数据存在，则创建一个指向本地路径的 DatasetSource 对象。
        reasoning_source = DatasetSource(
            path=str(_OPEN_MATH_LOCAL),  # 数据集路径
            reasoning=True,  # 标记为推理数据，包含详细的解题步骤
            weight=REASONING_DATASET_DEFAULT_WEIGHT,  # 在混合数据时占 75% 的权重
        )
    else:
        # 如果本地数据不存在，则配置从 Hugging Face Hub 下载。
        reasoning_source = DatasetSource(
            name="unsloth/OpenMathReasoning-mini",  # Hugging Face 数据集名称
            split="cot",  # 使用 'cot' (Chain-of-Thought) 分割
            reasoning=True,
            weight=REASONING_DATASET_DEFAULT_WEIGHT,
        )

    # 配置指令微调数据集 (instruction-following dataset)
    if _DAPO_LOCAL.exists():
        # 如果本地 DAPO 数据存在，则使用本地路径。
        instruction_source = DatasetSource(
            path=str(_DAPO_LOCAL),
            reasoning=False,  # 标记为非推理数据（通用指令）
            weight=INSTRUCTION_DATASET_DEFAULT_WEIGHT,  # 占 25% 的权重
        )
    else:
        # 如果本地数据不存在，则从 Hugging Face Hub 下载一个通用的指令微调数据集。
        instruction_source = DatasetSource(
            name="mlabonne/FineTome-100k",
            split="train",
            reasoning=False,
            weight=INSTRUCTION_DATASET_DEFAULT_WEIGHT,
        )

    # 返回一个包含两个数据源配置的元组。
    return (reasoning_source, instruction_source)


def _default_grpo_dataset() -> DatasetSource:
    """为 GRPO (Group-wise Reward Policy Optimization) 阶段提供默认的数据来源配置。

    同样采用“本地优先，云端回退”的策略。GRPO 通常需要高质量的 prompt 来生成对比样本。
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

# ===========================
# 数据类定义
# ===========================
@dataclass(slots=True)
class DatasetSource:
    """描述一个可从 Hugging Face Hub 或本地磁盘加载的数据源。

    这个类封装了加载数据集所需的所有信息，无论是来自网络还是本地文件系统，
    提供了一个统一的接口来描述数据。

    关键属性说明：
    - `name`/`subset`: 对应 Hugging Face 数据集仓库名和子集名。
    - `path`: 指向本地文件或目录的路径。`name` 和 `path` 通常是互斥的。
    - `weight`: 在混合多个数据集进行训练时，这个权重决定了从该数据源采样的频率。
    - `reasoning`: 一个布尔标记，用于区分数据集的类型。
                   `True` 表示数据集包含详细的解题步骤（链式思维），
                   `False` 表示它可能只包含问题和最终答案。
                   这个标记在数据预处理阶段非常重要，因为它决定了如何构造训练样本的目标字段（即模型需要学习预测的内容）。
    - `max_samples`: 可选，用于限制从该数据源加载的最大样本数，便于在完整数据集上进行快速调试或实验。
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
            # 如果是本地路径，只返回目录名，更简洁。
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
    优化器参数、数据处理细节以及输出路径等。将其组织在一个类中，使得配置的管理和传递更加清晰。
    """

    # --- 模型加载配置 ---
    base_model_id: str = BASE_MODEL_HF_ID_DEFAULT
    base_model_path: Optional[str] = (DEFAULT_BASE_MODEL_PATH or BASE_MODEL_LOCAL_FALLBACK)
    tokenizer_id: Optional[str] = TOKENIZER_ID_DEFAULT
    max_seq_length: int = MAX_SEQ_LENGTH_DEFAULT
    dtype: Optional[str] = DTYPE_DEFAULT
    load_in_4bit: bool = LOAD_IN_4BIT_DEFAULT
    load_in_8bit: bool = LOAD_IN_8BIT_DEFAULT
    full_finetuning: bool = FULL_FINETUNING_DEFAULT
    gradient_checkpointing: bool = GRADIENT_CHECKPOINTING_DEFAULT

    # --- LoRA (Low-Rank Adaptation) 配置 ---
    # LoRA 是一种参数高效的微调技术，它通过训练小型的“适配器”矩阵来修改模型行为，而无需改动原始的大量权重。
    lora_rank: int = LORA_RANK_DEFAULT
    lora_alpha: int = LORA_ALPHA_DEFAULT
    lora_dropout: float = LORA_DROPOUT_DEFAULT
    use_rslora: bool = USE_RSLORA_DEFAULT

    # --- 训练过程配置 ---
    batch_size: int = BATCH_SIZE_DEFAULT
    micro_batch_size: int = MICRO_BATCH_SIZE_DEFAULT
    gradient_accumulation_steps: int = GRADIENT_ACC_STEPS_DEFAULT
    learning_rate: float = LEARNING_RATE_DEFAULT
    weight_decay: float = WEIGHT_DECAY_DEFAULT
    warmup_steps: int = WARMUP_STEPS_DEFAULT
    num_train_epochs: float = NUM_TRAIN_EPOCHS_DEFAULT
    max_steps: int = MAX_STEPS_DEFAULT
    logging_steps: int = LOGGING_STEPS_DEFAULT
    eval_steps: int = EVAL_STEPS_DEFAULT
    save_steps: int = SAVE_STEPS_DEFAULT
    save_total_limit: int = SAVE_TOTAL_LIMIT_DEFAULT
    random_seed: int = RANDOM_SEED_DEFAULT

    # --- 路径和实验管理 ---
    output_dir: Path = OUTPUT_DIR_DEFAULT
    experiment_name: str = EXPERIMENT_NAME_DEFAULT
    # `field(default_factory=...)` 用于为可变类型的字段（如列表、字典）提供默认值。
    # 它确保每次创建 `TrainingConfig` 实例时，都会调用 `_default_dataset_mix()` 来生成一个新的元组，
    # 避免了所有实例共享同一个可变默认值的问题。
    dataset_mix: Sequence[DatasetSource] = field(
        default_factory=_default_dataset_mix
    )
    eval_split_ratio: float = EVAL_SPLIT_RATIO_DEFAULT
    dataset_num_proc: int = DATASET_NUM_PROC_DEFAULT
    cache_dir: Optional[Path] = CACHE_DIR_DEFAULT
    save_merged_model: bool = SAVE_MERGED_MODEL_DEFAULT
    merge_dtype: str = MERGE_DTYPE_DEFAULT

    def base_model_local_path(self) -> str:
        """优先返回本地权重路径，若无则回退至远端模型标识。"""
        return self.base_model_path or self.base_model_id

    # `@property` 装饰器:
    #   - 将一个方法变成一个只读属性。调用时无需加括号，例如 `config.project_root` 而不是 `config.project_root()`。
    #   - 这使得代码更简洁，访问方式更像访问一个普通的字段，同时保留了动态计算的能力。
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

    GRPO (Group-wise Reward Policy Optimization) 是一种对齐（Alignment）算法，旨在让模型的输出更符合人类偏好。
    它通过比较模型生成的多个回答（例如，一个好的回答和一个坏的回答），并根据奖励信号（由奖励模型给出或基于规则判断）
    来更新策略模型（即我们正在训练的模型），使其倾向于生成得分更高的回答。
    """

    enable: bool = GRPO_ENABLE_DEFAULT
    steps: int = GRPO_STEPS_DEFAULT
    learning_rate: float = GRPO_LR_DEFAULT
    # KL 系数与参考模型控制
    beta: float = GRPO_BETA_DEFAULT
    kl_coef: float = GRPO_KL_COEF_DEFAULT
    reference_free: bool = GRPO_REFERENCE_FREE_DEFAULT

    # 比率裁剪与算法细节
    epsilon: float = GRPO_EPSILON_DEFAULT
    delta: Optional[float] = None
    epsilon_high: Optional[float] = None
    clip_range: float = 0.2  # 向后兼容某些实现的 clip_range

    # 重要性采样与损失缩放
    importance_sampling_level: str = "sequence"  # "token" | "sequence"
    reward_weights: Optional[Sequence[float]] = None  # 每个奖励的权重
    scale_rewards: Optional[str | bool] = False  # "group" | "batch" | False/"none"
    loss_type: str = "dapo"  # "dapo" | "grpo" | "dr_grpo" | "bnpo"
    mask_truncated_completions: bool = False

    # vLLM / 离策略校正
    vllm_importance_sampling_correction: bool = True
    vllm_importance_sampling_cap: float = 2.0

    # 生成相关参数
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: Optional[int] = 50
    min_p: Optional[float] = 0.05
    repetition_penalty: float = 1.1

    # 批次与吞吐
    mini_batch_size: int = GRPO_MINI_BATCH_DEFAULT  # per_device_train_batch_size
    generation_batch_size: Optional[int] = GRPO_GENERATION_BATCH_DEFAULT
    gradient_accumulation_steps: int = GRPO_GRAD_ACC_DEFAULT
    num_generations_per_prompt: int = GRPO_NUM_GENERATIONS_DEFAULT  # 必须 > 2
    steps_per_generation: Optional[int] = None
    num_iterations: int = 1  # 每批次的 PPO/GRPO 周期数（μ）

    # 长度与预算
    max_prompt_len: int = GRPO_MAX_PROMPT_LEN_DEFAULT
    max_completion_len: int = GRPO_MAX_COMPLETION_LEN_DEFAULT
    max_tokens_per_step: Optional[int] = None  # 每步优化的最大 token 数预算

    # 其他优化
    reward_temperature: float = 1.0
    mixed_precision: Optional[str] = "bf16"
    save_steps: int = GRPO_SAVE_STEPS_DEFAULT
    logging_steps: int = GRPO_LOGGING_STEPS_DEFAULT
    torch_compile: bool = GRPO_TORCH_COMPILE_DEFAULT
    optim: Optional[str] = GRPO_OPTIM_DEFAULT
    unsloth_num_chunks: int = GRPO_UNSLOTH_NUM_CHUNKS_DEFAULT
    unsloth_gpu_memory_utilization: Optional[float] = GRPO_UNSLOTH_GPU_MEM_UTIL_DEFAULT

    # 数据集
    dataset: Optional[DatasetSource] = field(default_factory=_default_grpo_dataset)

    def describe_workload(self, training: TrainingConfig) -> dict[str, int]:
        """估算单步 GRPO 训练的 token 与样本开销，用于资源规划和调试。"""

        effective_batch = max(
            1,
            self.mini_batch_size * max(1, self.gradient_accumulation_steps),
        )
        prompt_len = max(1, min(self.max_prompt_len, training.max_seq_length))

        # 计算可分配给 completion 的最大长度，确保 prompt + completion 不超过模型支持的序列长度。
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

    这个类定义了在评估模型性能时，文本生成过程所使用的参数（即采样参数）。
    """


    sample_size: int = 100  # 用于评估的样本数量
    max_new_tokens: int = 512  # 生成文本的最大长度
    system_prompt: str = (  # 在生成时提供给模型的系统级指令，引导模型扮演特定角色
        "You are an expert math tutor. Provide concise step-by-step reasoning "
        "and highlight the final answer using \\boxed{} when appropriate."
    )


@dataclass(slots=True)
class ProjectConfig:
    """聚合项目所有阶段的配置，便于通过单一入口进行管理。

    这个顶层配置类将 `TrainingConfig`, `GRPOConfig`, 和 `EvaluationConfig` 组合在一起，
    形成一个单一的配置对象。这使得在命令行接口（CLI）或主脚本中可以方便地访问和修改所有配置，
    而不需要传递多个独立的配置对象。
    """

    training: TrainingConfig = field(default_factory=TrainingConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def ensure_directories(self) -> None:
        """确保所有在配置中定义的输出目录都存在。

        在训练开始前调用此方法可以避免因目录不存在而导致的 `FileNotFoundError`。
        `path.mkdir(parents=True, exist_ok=True)` 是一个非常有用的调用：
          - `parents=True`: 自动创建所有必需的父目录。例如，如果 `outputs/` 不存在，它会一并创建。
          - `exist_ok=True`: 如果目录已经存在，则不引发 `FileExistsError` 错误。
        """
        self.training.output_dir.mkdir(parents=True, exist_ok=True)
        self.training.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.training.finetuned_model_dir.mkdir(parents=True, exist_ok=True)
        if self.training.save_merged_model:
            self.training.merged_model_dir.mkdir(parents=True, exist_ok=True)
