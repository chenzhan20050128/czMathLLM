from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CANDIDATE_BASE_MODEL_PATH = _PROJECT_ROOT / "models" / "Qwen3-4B-Thinking-2507"
DEFAULT_BASE_MODEL_PATH = (
    str(_CANDIDATE_BASE_MODEL_PATH.resolve())
    if _CANDIDATE_BASE_MODEL_PATH.exists()
    else None
)

_LOCAL_DATA_DIR = _PROJECT_ROOT / "data"
_OPEN_MATH_LOCAL = _LOCAL_DATA_DIR / "OpenMathReasoning" / "data"
_DAPO_LOCAL = _LOCAL_DATA_DIR / "DAPO-Math-17k-Processed" / "all"


def _default_dataset_mix() -> tuple[DatasetSource, ...]:
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
    if _OPEN_MATH_LOCAL.exists():
        return DatasetSource(path=str(_OPEN_MATH_LOCAL), reasoning=True)
    return DatasetSource(
        name="open-r1/DAPO-Math-17k-Processed",
        subset="en",
        split="train",
        reasoning=True,
    )


@dataclass(slots=True)
class DatasetSource:
    """Describe a dataset source that can be loaded from HF or local files."""

    name: Optional[str] = None
    subset: Optional[str] = None
    split: str = "train"
    path: Optional[str] = None
    weight: float = 1.0
    reasoning: bool = True
    max_samples: Optional[int] = None

    def display_name(self) -> str:
        if self.path:
            return Path(self.path).name
        if self.subset:
            return f"{self.name}:{self.subset}"
        return str(self.name)


@dataclass(slots=True)
class TrainingConfig:
    """Hyper-parameters and resources for supervised fine-tuning."""

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
        return self.base_model_path or self.base_model_id

    @property
    def project_root(self) -> Path:
        return Path.cwd()

    @property
    def finetuned_model_dir(self) -> Path:
        return self.output_dir / f"{self.experiment_name}_lora"

    @property
    def merged_model_dir(self) -> Path:
        return self.output_dir / f"{self.experiment_name}_merged"

    @property
    def checkpoints_dir(self) -> Path:
        return self.output_dir / "checkpoints"


@dataclass(slots=True)
class GRPOConfig:
    """Configuration for GRPO fine-tuning on top of SFT LoRA weights."""

    enable: bool = True
    steps: int = 500
    learning_rate: float = 5e-6
    beta: float = 0.2
    clip_range: float = 0.2
    kl_coef: float = 0.02
    value_loss_coef: float = 0.01
    mini_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_generations_per_prompt: int = 1
    max_prompt_len: int = 2048
    max_completion_len: int = 512
    reward_temperature: float = 0.9
    reference_free: bool = False
    mixed_precision: Optional[str] = "bf16"
    save_steps: int = 100
    dataset: Optional[DatasetSource] = field(default_factory=_default_grpo_dataset)


@dataclass(slots=True)
class EvaluationConfig:
    """Settings for offline evaluation."""

    sample_size: int = 100
    max_new_tokens: int = 512
    system_prompt: str = (
        "You are an expert math tutor. Provide concise step-by-step reasoning "
        "and highlight the final answer using \\boxed{} when appropriate."
    )


@dataclass(slots=True)
class ProjectConfig:
    """Aggregate configuration for the complete project."""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def ensure_directories(self) -> None:
        self.training.output_dir.mkdir(parents=True, exist_ok=True)
        self.training.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.training.finetuned_model_dir.mkdir(parents=True, exist_ok=True)
        if self.training.save_merged_model:
            self.training.merged_model_dir.mkdir(parents=True, exist_ok=True)
