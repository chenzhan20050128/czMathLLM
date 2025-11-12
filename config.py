from __future__ import annotations

"""项目配置模块。

该文件集中定义了训练、强化学习（GRPO）与评估阶段用到的所有配置
数据结构。通过 `@dataclass` 装饰器（Python 3.7+ 引入的简化数据类语法）
我们可以快速声明仅包含属性的类，避免手写 `__init__` 和 `__repr__`
等样板代码。`slots=True` 则利用 CPython 的 `__slots__` 机制约束可用属性，
既能减少内存占用，也可以在访问不存在的字段时更早报错。"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence


# 通过 `Path(__file__).resolve()` 获取当前文件的绝对路径，再向上两级定位至项目根目录。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 提前约定一个默认的本地基座模型目录，若存在则优先使用本地文件以避免频繁下载。
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


def _default_dataset_mix() -> tuple[DatasetSource, ...]:
    """构造默认的监督微调数据配比。

    这里演示了“优先本地、回退云端”的策略：先检测本地缓存，若不存在再回退
    到 Hugging Face 数据集。函数返回一个不可变的 `tuple`，便于作为
    `dataclass` 字段的默认值。
    """
    if _OPEN_MATH_LOCAL.exists():
        reasoning_source = DatasetSource(
            path=str(_OPEN_MATH_LOCAL),
            reasoning=True,
            weight=0.75,
        )
    else:
        reasoning_source = DatasetSource(
            name="unsloth/OpenMathReasoning-mini",
            split="cot",
            reasoning=True,
            weight=0.75,
        )

    if _DAPO_LOCAL.exists():
        instruction_source = DatasetSource(
            path=str(_DAPO_LOCAL),
            reasoning=False,
            weight=0.25,
        )
    else:
        instruction_source = DatasetSource(
            name="mlabonne/FineTome-100k",
            split="train",
            reasoning=False,
            weight=0.25,
        )

    return (reasoning_source, instruction_source)


def _default_grpo_dataset() -> DatasetSource:
    """为 GRPO 阶段提供默认的数据来源配置。"""
    if _DAPO_ROOT.exists():
        return DatasetSource(path=str(_DAPO_ROOT), reasoning=True)
    return DatasetSource(
        name="open-r1/DAPO-Math-17k-Processed",
        subset="en",
        split="train",
        reasoning=True,
    )


@dataclass(slots=True)
class DatasetSource:
    """描述一个可从 Hugging Face Hub 或本地磁盘加载的数据源。

    关键属性说明：
    - ``name``/``subset`` 对应 HF 数据集仓库与子集；
    - ``path`` 指向本地文件或目录；
    - ``weight`` 会在数据混合时转换为采样概率；
    - ``reasoning`` 标记该数据是否包含链式思维推理，以便决定训练目标字段。
    """

    name: Optional[str] = None
    subset: Optional[str] = None
    split: str = "train"
    path: Optional[str] = None
    weight: float = 1.0
    reasoning: bool = True
    max_samples: Optional[int] = None

    def display_name(self) -> str:
        """生成易读的数据源名称，调试或可视化时非常方便。"""
        if self.path:
            return Path(self.path).name
        if self.subset:
            return f"{self.name}:{self.subset}"
        return str(self.name)


@dataclass(slots=True)
class TrainingConfig:
    """监督微调（SFT）阶段的超参数与资源配置。

    这里包含模型加载、LoRA 低秩适配、优化器参数、数据处理等信息。"""

    base_model_id: str = "Qwen/Qwen3-4B-Thinking-2507"
    base_model_path: Optional[str] = DEFAULT_BASE_MODEL_PATH
    tokenizer_id: Optional[str] = None
    max_seq_length: int = 4096
    dtype: Optional[str] = None
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    full_finetuning: bool = False
    gradient_checkpointing: bool = True

    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    use_rslora: bool = False

    batch_size: int = 1
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 50
    num_train_epochs: float = 1.0
    max_steps: int = -1
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 200
    save_total_limit: int = 2
    random_seed: int = 3407

    output_dir: Path = Path("outputs")
    experiment_name: str = "qwen_math_tutor"
    dataset_mix: Sequence[DatasetSource] = field(default_factory=_default_dataset_mix)
    eval_split_ratio: float = 0.02
    dataset_num_proc: int = 1
    cache_dir: Optional[Path] = None

    save_merged_model: bool = True
    merge_dtype: str = "fp16"

    def base_model_local_path(self) -> str:
        """优先返回本地权重路径，若无则回退至远端模型标识。"""
        return self.base_model_path or self.base_model_id

    @property
    def project_root(self) -> Path:
        """根据当前工作目录推断项目根路径。

        利用 `@property` 装饰器，将方法伪装成属性使用，读取时更自然。"""
        return Path.cwd()

    @property
    def finetuned_model_dir(self) -> Path:
        """返回保存 LoRA 适配器的目录。"""
        return self.output_dir / f"{self.experiment_name}_lora"

    @property
    def merged_model_dir(self) -> Path:
        """返回合并后（LoRA + 基座）的完整模型目录。"""
        return self.output_dir / f"{self.experiment_name}_merged"

    @property
    def checkpoints_dir(self) -> Path:
        """返回用于保存中间检查点的目录。"""
        return self.output_dir / "checkpoints"


@dataclass(slots=True)
class GRPOConfig:
    """基于已训练好的 LoRA 权重继续进行 GRPO 强化学习阶段的配置。"""

    enable: bool = True
    steps: int = 500
    learning_rate: float = 8e-6
    beta: float = 0.2
    clip_range: float = 0.2
    kl_coef: float = 0.06
    value_loss_coef: float = 0.01
    mini_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_generations_per_prompt: int = 2
    max_prompt_len: int = 1536
    max_completion_len: int = 512  # 更短的生成长度以提升迭代速度
    reward_temperature: float = 1.0
    reference_free: bool = True
    mixed_precision: Optional[str] = "bf16"
    save_steps: int = 100
    dataset: Optional[DatasetSource] = field(default_factory=_default_grpo_dataset)


@dataclass(slots=True)
class EvaluationConfig:
    """离线评估阶段的推理配置，例如采样数量与系统提示词。"""

    sample_size: int = 100
    max_new_tokens: int = 512
    system_prompt: str = (
        "You are an expert math tutor. Provide concise step-by-step reasoning "
        "and highlight the final answer using \\boxed{} when appropriate."
    )


@dataclass(slots=True)
class ProjectConfig:
    """聚合项目全局配置，便于 CLI 或脚本一次性管理所有阶段。"""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def ensure_directories(self) -> None:
        self.training.output_dir.mkdir(parents=True, exist_ok=True)
        self.training.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.training.finetuned_model_dir.mkdir(parents=True, exist_ok=True)
        if self.training.save_merged_model:
            self.training.merged_model_dir.mkdir(parents=True, exist_ok=True)
