# flake8: noqa
"""命令行入口的核心实现。

该模块使用 Python 内置的 `argparse` 库来构建一个强大的命令行接口（CLI）。
它定义了不同的子命令（如 `train`, `evaluate`, `predict`），并为每个子命令
注册了一系列参数。核心功能是将解析出的命令行参数转换为项目内部使用的
`ProjectConfig` 配置对象，然后调用相应的业务逻辑函数。

为了优化启动性能，真正的入口函数 `main` 被设计为可以从其他地方（如 `__init__.py`）
调用，从而实现延迟加载，避免在不必要时导入重量级的深度学习库。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from peft import PeftModel

# 引入所有需要的配置与常量（单一默认值来源）
from .config import (
    DatasetSource,
    ProjectConfig,
    MAX_SEQ_LENGTH_DEFAULT,
    REASONING_DATASET_DEFAULT_WEIGHT,
    INSTRUCTION_DATASET_DEFAULT_WEIGHT,
)
from .evaluation import evaluate_model
from .modeling import generate_answers, load_base_model, prepare_for_inference
from .prompts import batched_prompts
from .trainers import run_grpo_training, run_sft_training


def _load_dataset_mix(
    args: argparse.Namespace, default_mix: Iterable[DatasetSource]
) -> tuple[DatasetSource, ...]:
    """根据命令行参数生成数据源组合。

    此函数实现了灵活的数据集配置加载，优先级如下：
    1.  如果提供了 `--dataset-config` 参数，它会读取一个 JSON 文件。
        该文件定义了要使用的所有数据集及其属性（名称、权重、是否为推理任务等）。
        这是最灵活和推荐的方式，便于管理复杂的数据集组合。
    2.  否则，它会尝试从 `--reasoning-source` 和 `--instruction-source`
        参数中单独解析每个数据源。这适用于简单的一或两个数据集的场景。
    3.  如果命令行没有提供任何数据源信息，则使用 `default_mix` 中定义的默认配置。
        这确保了即使不提供任何参数，程序也能以一个合理的默认设置运行。
    """
    if args.dataset_config:
        config_path = Path(args.dataset_config)
        # 使用 with 语句确保文件被正确关闭
        with config_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        # 使用列表推导和星号解包（`**entry`）将 JSON 对象列表转换为 `DatasetSource` 对象元组。
        # 这是一个非常 Pythonic 的写法，简洁地将字典数据映射到 Pydantic 模型实例。
        return tuple(DatasetSource(**entry) for entry in data)

    reasoning_source = _parse_source_arg(getattr(args, "reasoning_source", None))
    instruction_source = _parse_source_arg(getattr(args, "instruction_source", None))

    mix: list[DatasetSource] = []
    if reasoning_source:
        mix.append(
            DatasetSource(
                **reasoning_source,
                weight=(
                    args.reasoning_weight
                    if getattr(args, "reasoning_weight", None) is not None
                    else REASONING_DATASET_DEFAULT_WEIGHT
                ),
                reasoning=True,
            )
        )
    if instruction_source:
        mix.append(
            DatasetSource(
                **instruction_source,
                weight=(
                    args.instruction_weight
                    if getattr(args, "instruction_weight", None) is not None
                    else INSTRUCTION_DATASET_DEFAULT_WEIGHT
                ),
                reasoning=False,
            )
        )
    if not mix:
        mix = list(default_mix)
    return tuple(mix)


def _parse_source_arg(arg: Optional[str]) -> Optional[dict]:
    """解析单个数据源字符串，将其转换为 `DatasetSource` 构造函数所需的字典。

    这个辅助函数非常灵活，可以接受多种格式的输入，体现了良好的用户体验设计：
    - "none" 或空字符串: 明确表示不使用该数据源。
    - 本地路径: 如果字符串对应一个存在的本地文件或目录，则将其识别为路径。
    - Hugging Face 仓库名: 例如 "mlabonne/FineTome-100k"。
    - "仓库名:子集名": 例如 "open-r1/DAPO-Math-17k-Processed:en"，用于加载包含多个子集的数据集。
    - 启发式路径判断：即使文件当前不存在，如果它以常见的数据文件扩展名结尾，也假定它是一个路径。

    返回一个字典，其键值对可以直接用于创建 `DatasetSource` 对象。
    """
    if arg is None:
        return None
    if arg.lower() in {"none", ""}:
        return None
    path_candidate = Path(arg)
    if path_candidate.exists():
        return {"path": str(path_candidate)}
    if any(arg.endswith(ext) for ext in (".json", ".jsonl", ".jsonl.gz", ".parquet")):
        return {"path": arg}
    if ":" in arg:
        name, subset = arg.split(":", 1)
        return {"name": name, "subset": subset}
    return {"name": arg}


def _maybe_parse_csv_floats(value: Optional[str]) -> Optional[list[float]]:
    if value is None:
        return None
    parts = [p.strip() for p in str(value).split(",") if str(p).strip()]
    try:
        return [float(p) for p in parts]
    except Exception:
        return None


def _apply_common_overrides(project: ProjectConfig, args: argparse.Namespace) -> None:
    """将命令行传入的参数覆盖到 `ProjectConfig` 对象上。

    这个函数是连接命令行和内部配置的桥梁。它遍历所有可能的命令行参数，
    如果用户在命令行中指定了某个参数（即其值不是 None），就用该参数的值更新 `ProjectConfig` 对象中
    对应的字段。

    使用 `getattr(args, "param_name", default_value)` 是一种非常优雅和健壮的编程模式：
    - 它尝试从 `args` 对象（由 argparse 解析命令行参数生成）中获取名为 "param_name" 的属性。
    - 如果该属性不存在（例如，因为这个参数属于另一个子命令，在当前调用中未定义），它不会抛出 `AttributeError`，
      而是返回一个 `default_value`（这里通常是 `None`）。
    - 这使得我们可以编写一个通用的覆盖函数来处理所有子命令的共享参数，而无需担心哪些参数在当前上下文中是可用的。
      代码变得更加简洁和可维护。
    """
    training = project.training

    # --- 模型相关覆盖 ---
    if getattr(args, "base_model_id", None) is not None:
        training.base_model_id = args.base_model_id
    if getattr(args, "base_model_path", None) is not None:
        training.base_model_path = args.base_model_path
    if getattr(args, "max_seq_length", None) is not None:
        training.max_seq_length = args.max_seq_length
    if getattr(args, "load_in_4bit", None) is not None:
        training.load_in_4bit = args.load_in_4bit
    if getattr(args, "load_in_8bit", None) is not None:
        training.load_in_8bit = args.load_in_8bit
    if getattr(args, "full_finetuning", None) is not None:
        training.full_finetuning = args.full_finetuning

    # --- LoRA 相关覆盖 ---
    if getattr(args, "lora_rank", None) is not None:
        training.lora_rank = args.lora_rank
    if getattr(args, "lora_alpha", None) is not None:
        training.lora_alpha = args.lora_alpha
    if getattr(args, "lora_dropout", None) is not None:
        training.lora_dropout = args.lora_dropout

    # --- 训练过程相关覆盖 ---
    for field_name in [
        "gradient_accumulation_steps",
        "micro_batch_size",
        "learning_rate",
        "weight_decay",
        "warmup_steps",
        "num_train_epochs",
        "max_steps",
        "logging_steps",
        "eval_steps",
        "save_steps",
        "save_total_limit",
        "random_seed",
    ]:
        val = getattr(args, field_name, None)
        if val is not None:
            setattr(training, field_name, val)

    if getattr(args, "output_dir", None) is not None:
        training.output_dir = Path(args.output_dir)
    if getattr(args, "experiment_name", None) is not None:
        training.experiment_name = args.experiment_name

    # --- 数据集相关覆盖 ---
    if any(
        hasattr(args, nm)
        for nm in ["dataset_config", "reasoning_source", "instruction_source"]
    ):
        training.dataset_mix = _load_dataset_mix(args, training.dataset_mix)
    if getattr(args, "eval_split_ratio", None) is not None:
        training.eval_split_ratio = args.eval_split_ratio
    if getattr(args, "dataset_num_proc", None) is not None:
        training.dataset_num_proc = args.dataset_num_proc

    # --- 模型保存相关覆盖 ---
    if getattr(args, "save_merged_model", None) is not None:
        training.save_merged_model = args.save_merged_model
    if getattr(args, "merge_dtype", None) is not None:
        training.merge_dtype = args.merge_dtype

    # --- GRPO 相关覆盖 ---
    if getattr(args, "with_grpo", None):
        project.grpo.enable = True

    grpo_map = {
        "grpo_steps": "steps",
        "grpo_learning_rate": "learning_rate",
        "grpo_beta": "beta",
        "grpo_epsilon": "epsilon",
        "grpo_delta": "delta",
        "grpo_epsilon_high": "epsilon_high",
        "grpo_kl": "kl_coef",
        "grpo_importance_level": "importance_sampling_level",
        "grpo_mini_batch": "mini_batch_size",
        "grpo_generation_batch_size": "generation_batch_size",
        "grpo_gradient_accumulation": "gradient_accumulation_steps",
        "grpo_num_generations": "num_generations_per_prompt",
        "grpo_steps_per_generation": "steps_per_generation",
        "grpo_num_iterations": "num_iterations",
        "grpo_max_prompt_len": "max_prompt_len",
        "grpo_max_completion_len": "max_completion_len",
        "grpo_max_tokens_per_step": "max_tokens_per_step",
        "grpo_temperature": "temperature",
        "grpo_top_p": "top_p",
        "grpo_top_k": "top_k",
        "grpo_min_p": "min_p",
        "grpo_repetition_penalty": "repetition_penalty",
        "grpo_loss_type": "loss_type",
        "grpo_mask_truncated_completions": "mask_truncated_completions",
        "grpo_vllm_importance_sampling_correction": "vllm_importance_sampling_correction",
        "grpo_vllm_importance_sampling_cap": "vllm_importance_sampling_cap",
        "grpo_gpu_memory_utilization": "unsloth_gpu_memory_utilization",
        "grpo_reference_free": "reference_free",
        "grpo_mixed_precision": "mixed_precision",
        "grpo_logging_steps": "logging_steps",
        "grpo_save_steps": "save_steps",
        "grpo_torch_compile": "torch_compile",
        "grpo_optim": "optim",
        "grpo_unsloth_num_chunks": "unsloth_num_chunks",
    }
    for arg_name, attr_name in grpo_map.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            # 特殊处理 max_tokens_per_step 允许 <=0 视为 None
            if attr_name == "max_tokens_per_step" and isinstance(val, int) and val <= 0:
                val = None
            setattr(project.grpo, attr_name, val)

    if getattr(args, "grpo_scale_rewards", None) is not None:
        val = args.grpo_scale_rewards
        if isinstance(val, str) and val.lower() in {"true", "false"}:
            val = val.lower() == "true"
        project.grpo.scale_rewards = val

    if getattr(args, "grpo_reward_weights", None) is not None:
        parsed = _maybe_parse_csv_floats(args.grpo_reward_weights)
        if parsed is not None:
            project.grpo.reward_weights = parsed

    grpo_dataset_arg = getattr(args, "grpo_dataset", None)
    if grpo_dataset_arg is not None:
        parsed = _parse_source_arg(grpo_dataset_arg)
        if parsed:
            dataset_kwargs = dict(parsed)
            grpo_split = getattr(args, "grpo_dataset_split", None)
            if grpo_split is not None:
                dataset_kwargs["split"] = grpo_split
            grpo_max_samples = getattr(args, "grpo_dataset_max_samples", None)
            if grpo_max_samples is not None:
                dataset_kwargs["max_samples"] = grpo_max_samples
            project.grpo.dataset = DatasetSource(**dataset_kwargs)


# ---------------- 参数注册 ----------------

def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    """注册与数据源相关的命令行参数."""
    parser.add_argument("--dataset-config", type=str, default=None, help="定义数据集混合的JSON文件路径")
    parser.add_argument("--reasoning-source", type=str, default=None, help="主要的推理数据集 (HF 仓库或本地文件)")
    parser.add_argument("--instruction-source", type=str, default=None, help="辅助的指令数据集")
    parser.add_argument(
        "--reasoning-weight",
        type=float,
        default=None,
        help=f"推理数据集的权重 (默认 {REASONING_DATASET_DEFAULT_WEIGHT})",
    )
    parser.add_argument(
        "--instruction-weight",
        type=float,
        default=None,
        help=f"指令数据集的权重 (默认 {INSTRUCTION_DATASET_DEFAULT_WEIGHT})",
    )


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    """注册模型加载相关的参数."""
    parser.add_argument("--base-model-id", type=str, default=None, help="Hugging Face 上的基础模型 ID")
    parser.add_argument(
        "--base-model-path",
        type=str,
        default=None,
        help="本地基础模型的路径 (未提供则使用 config 默认)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help=f"最大序列长度 (默认 {MAX_SEQ_LENGTH_DEFAULT})",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=None,
        help="启用4bit量化 (默认 config 中设置)",
    )
    parser.add_argument(
        "--no-4bit",
        dest="load_in_4bit",
        action="store_false",
        help="禁用4bit量化",
    )
    parser.add_argument(
        "--load-in-8bit", action="store_true", default=None, help="启用8bit量化"
    )
    parser.add_argument(
        "--full-finetuning", action="store_true", default=None, help="进行全参数微调 (默认使用 LoRA)"
    )


def _add_lora_args(parser: argparse.ArgumentParser) -> None:
    """注册 LoRA (低秩适配) 相关的参数."""
    parser.add_argument("--lora-rank", type=int, default=None, help="LoRA 秩 (默认 config)")
    parser.add_argument("--lora-alpha", type=int, default=None, help="LoRA alpha (默认 config)")
    parser.add_argument("--lora-dropout", type=float, default=None, help="LoRA dropout (默认 config)")


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    """注册训练阶段的超参数选项."""
    parser.add_argument("--micro-batch-size", type=int, default=None, help="每个设备上的微批次大小")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None, help="梯度累积步数")
    parser.add_argument("--learning-rate", type=float, default=None, help="学习率 (默认 config)")
    parser.add_argument("--weight-decay", type=float, default=None, help="权重衰减")
    parser.add_argument("--warmup-steps", type=int, default=None, help="学习率预热步数")
    parser.add_argument("--num-train-epochs", type=float, default=None, help="总训练轮数")
    parser.add_argument("--max-steps", type=int, default=None, help="最大训练步数 (若非-1，则覆盖 epochs)")
    parser.add_argument("--logging-steps", type=int, default=None, help="日志记录步数")
    parser.add_argument("--eval-steps", type=int, default=None, help="评估步数")
    parser.add_argument("--save-steps", type=int, default=None, help="模型保存步数")
    parser.add_argument("--save-total-limit", type=int, default=None, help="最多保存的检查点数量")
    parser.add_argument("--random-seed", type=int, default=None, help="随机种子")
    parser.add_argument(
        "--output-dir", type=str, default=None, help="训练输出和检查点的根目录"
    )
    parser.add_argument(
        "--experiment-name", type=str, default=None, help="实验名称，用于区分不同的运行"
    )
    parser.add_argument("--eval-split-ratio", type=float, default=None, help="验证集比例")
    parser.add_argument("--dataset-num-proc", type=int, default=None, help="数据预处理进程数")
    parser.add_argument(
        "--save-merged-model",
        action="store_true",
        default=None,
        help="训练后合并并保存模型 (默认参见 config)",
    )
    parser.add_argument(
        "--no-save-merged-model", dest="save_merged_model", action="store_false", help="不合并保存模型"
    )
    parser.add_argument(
        "--merge-dtype", type=str, default=None, help="合并模型使用的数据类型 (fp16/bf16/float)"
    )


def _add_grpo_args(parser: argparse.ArgumentParser) -> None:
    """注册 GRPO 阶段的特有参数。"""
    parser.add_argument("--with-grpo", action="store_true", help="在 SFT 后启用 GRPO 训练")
    parser.add_argument("--grpo-steps", type=int, default=None, help="GRPO 训练的总步数")
    parser.add_argument("--grpo-learning-rate", type=float, default=None, help="GRPO 阶段的学习率")
    parser.add_argument("--grpo-beta", type=float, default=None, help="KL 系数；0.0 则参考模型不加载")
    parser.add_argument("--grpo-kl", type=float, default=None, help="KL 散度系数")
    parser.add_argument("--grpo-epsilon", type=float, default=None, help="令牌级对数概率比率裁剪值 ε")
    parser.add_argument("--grpo-delta", type=float, default=None, help="双面 GRPO 上裁剪边界 δ")
    parser.add_argument("--grpo-epsilon-high", type=float, default=None, help="上界 ε_high；未设则为 ε")
    parser.add_argument("--grpo-mini-batch", type=int, default=None, help="per-device train batch size")
    parser.add_argument("--grpo-generation-batch-size", type=int, default=None, help="生成批次大小")
    parser.add_argument("--grpo-gradient-accumulation", type=int, default=None, help="梯度累积步数")
    parser.add_argument("--grpo-num-generations", type=int, default=None, help="每个 prompt 生成的数量 (需>2)")
    parser.add_argument("--grpo-steps-per-generation", type=int, default=None, help="每次生成的步数")
    parser.add_argument("--grpo-num-iterations", type=int, default=None, help="每批次的 GRPO 周期数 (μ)")
    parser.add_argument("--grpo-max-prompt-len", type=int, default=None, help="最大 prompt token 数")
    parser.add_argument("--grpo-max-completion-len", type=int, default=None, help="最大 completion token 数")
    parser.add_argument("--grpo-max-tokens-per-step", type=int, default=None, help="每步 token 预算软上限")
    parser.add_argument("--grpo-temperature", type=float, default=None, help="采样温度")
    parser.add_argument("--grpo-top-p", type=float, default=None, help="Top-p 阈值")
    parser.add_argument("--grpo-top-k", type=int, default=None, help="Top-k 上限")
    parser.add_argument("--grpo-min-p", type=float, default=None, help="Min-p 阈值")
    parser.add_argument("--grpo-repetition-penalty", type=float, default=None, help="重复惩罚系数")
    parser.add_argument(
        "--grpo-scale-rewards",
        type=str,
        default=None,
        help='奖励缩放方式："group" | "batch" | "none" | true | false',
    )
    parser.add_argument(
        "--grpo-loss-type",
        type=str,
        choices=["dapo", "grpo", "dr_grpo", "bnpo"],
        default=None,
        help="损失类型",
    )
    parser.add_argument(
        "--grpo-mask-truncated-completions",
        action="store_true",
        help="将被截断的完成从损失中排除",
    )
    parser.add_argument(
        "--no-grpo-mask-truncated-completions",
        dest="grpo_mask_truncated_completions",
        action="store_false",
        help="不排除被截断的完成 (默认)",
    )
    parser.set_defaults(grpo_mask_truncated_completions=None)
    parser.add_argument(
        "--grpo-importance-level",
        type=str,
        choices=["token", "sequence"],
        default=None,
        help="重要性采样级别",
    )
    parser.add_argument(
        "--grpo-reward-weights",
        type=str,
        default=None,
        help="逗号分隔的奖励权重，例如: 1.0,0.5",
    )
    parser.add_argument(
        "--grpo-vllm-importance-sampling-correction",
        dest="grpo_vllm_importance_sampling_correction",
        action="store_true",
        help="启用截断重要性采样校正",
    )
    parser.add_argument(
        "--no-grpo-vllm-importance-sampling-correction",
        dest="grpo_vllm_importance_sampling_correction",
        action="store_false",
        help="禁用截断重要性采样校正",
    )
    parser.set_defaults(grpo_vllm_importance_sampling_correction=None)
    parser.add_argument(
        "--grpo-vllm-importance-sampling-cap", type=float, default=None, help="TIS 截断参数 C"
    )
    parser.add_argument(
        "--grpo-gpu-memory-utilization", type=float, default=None, help="Unsloth GPU memory 利用率"
    )
    parser.add_argument(
        "--grpo-reference-free", dest="grpo_reference_free", action="store_true", help="使用无参考模型 GRPO"
    )
    parser.add_argument(
        "--no-grpo-reference-free", dest="grpo_reference_free", action="store_false", help="使用参考模型"
    )
    parser.set_defaults(grpo_reference_free=None)
    parser.add_argument("--grpo-mixed-precision", type=str, default=None, help="混合精度 (fp16/bf16)")
    parser.add_argument("--grpo-logging-steps", type=int, default=None, help="GRPO 日志步数")
    parser.add_argument("--grpo-save-steps", type=int, default=None, help="GRPO 保存步数")
    parser.add_argument(
        "--grpo-torch-compile", action="store_true", help="启用 torch.compile"
    )
    parser.add_argument(
        "--no-grpo-torch-compile", dest="grpo_torch_compile", action="store_false", help="禁用 torch.compile"
    )
    parser.set_defaults(grpo_torch_compile=None)
    parser.add_argument("--grpo-optim", type=str, default=None, help="优化器 (如 adamw_8bit)")
    parser.add_argument("--grpo-unsloth-num-chunks", type=int, default=None, help="Unsloth num chunks")
    parser.add_argument(
        "--grpo-dataset", type=str, default=None, help="用于 GRPO 的数据集 (HF 仓库或本地文件)"
    )
    parser.add_argument(
        "--grpo-dataset-split", type=str, default=None, help="GRPO 数据集使用的数据切分"
    )
    parser.add_argument(
        "--grpo-dataset-max-samples", type=int, default=None, help="限制 GRPO 数据集的最大样本数"
    )


# --- 子命令处理函数 ---


def _handle_train(args: argparse.Namespace) -> None:
    """处理 `train` 子命令，执行 SFT，并按需衔接 GRPO。"""
    # 1. 创建一个默认的配置对象
    project = ProjectConfig()
    # 2. 将命令行的参数覆盖到配置对象上
    _apply_common_overrides(project, args)
    # 3. 运行监督微调 (SFT)
    metrics = run_sft_training(
        project, resume_from_checkpoint=args.resume_from_checkpoint
    )
    print("SFT 训练指标:", json.dumps(metrics, indent=2, default=str))

    # 4. 如果配置中启用了 GRPO，则在 SFT 之后继续运行 GRPO
    if project.grpo.enable:
        grpo_metrics = run_grpo_training(project)
        print("GRPO 训练信息:", json.dumps(grpo_metrics, indent=2, default=str))


def _handle_grpo(args: argparse.Namespace) -> None:
    """处理 `grpo` 子命令，仅运行强化学习阶段。"""
    project = ProjectConfig()
    _apply_common_overrides(project, args)
    project.grpo.enable = True  # 强制启用 GRPO
    grpo_metrics = run_grpo_training(
        project, resume_from_checkpoint=args.resume_from_checkpoint
    )
    print("GRPO 训练信息:", json.dumps(grpo_metrics, indent=2, default=str))


def _handle_evaluate(args: argparse.Namespace) -> None:
    """处理 `evaluate` 子命令，对模型进行评估并输出统计信息。"""
    project = ProjectConfig()
    _apply_common_overrides(project, args)
    # 调用评估函数
    df = evaluate_model(
        project, model_path=args.model_path, sample_size=args.sample_size
    )
    print("评估结果描述性统计:")
    # 使用 pandas 的 describe 方法快速获取统计摘要
    print(df.describe(include="all"))
    # 如果指定了保存路径，则将结果保存为 Parquet 文件
    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path)
        print(f"评估结果已保存至: {save_path}")


def _handle_predict(args: argparse.Namespace) -> None:
    """处理 `predict` 子命令，对给定的问题进行批量解答。"""
    project = ProjectConfig()
    _apply_common_overrides(project, args)

    # 加载基础模型和分词器
    model, tokenizer = load_base_model(project.training, model_path=args.model_path)

    # 如果提供了适配器路径，则加载 LoRA 权重并应用到模型上
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=False)

    # 准备推理用的提示词
    prompts = batched_prompts(
        args.question, system_prompt=project.evaluation.system_prompt
    )

    # 准备模型以进行推理（例如，合并 LoRA 权重，设置为评估模式等）
    prepare_for_inference(model)

    # 调用核心的生成函数
    outputs = generate_answers(
        model, tokenizer, prompts, max_new_tokens=args.max_new_tokens
    )

    # 格式化并打印结果
    for question, output in zip(args.question, outputs):
        print("\n=== 问题 ===\n", question)
        print("\n=== 回答 ===\n", output)


def build_parser() -> argparse.ArgumentParser:
    """创建顶级的命令行解析器并注册所有子命令。"""
    parser = argparse.ArgumentParser(description="数学大模型微调命令行工具")
    # `add_subparsers` 用于创建子命令。`dest` 参数会将选中的子命令名称存储在 `command` 属性中。
    # `required=True` 确保用户必须提供一个子命令。
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- 定义 'train' 子命令 ---
    train_parser = subparsers.add_parser("train", help="运行监督微调 (SFT)，可选择性地后跟 GRPO")
    _add_dataset_args(train_parser)
    _add_model_args(train_parser)
    _add_lora_args(train_parser)
    _add_train_args(train_parser)
    _add_grpo_args(train_parser)
    train_parser.add_argument(
        "--resume-from-checkpoint", type=str, default=None, help="从检查点恢复训练"
    )
    # `set_defaults(func=...)` 是一个关键技巧：它将一个函数与该子命令关联起来。
    # 当 `train` 命令被选中时，`argparse` 会在返回的 `Namespace` 对象中添加一个 `func` 属性，其值为 `_handle_train`。
    # 这样，主函数 `main` 就可以通过 `args.func(args)` 来调用正确的处理函数。
    train_parser.set_defaults(func=_handle_train)

    # --- 定义 'grpo' 子命令 ---
    grpo_parser = subparsers.add_parser("grpo", help="仅运行 GRPO 强化学习阶段")
    # 复用大部分与 train 命令相同的参数，因为 GRPO 也需要模型、数据等配置
    _add_dataset_args(grpo_parser)
    _add_model_args(grpo_parser)
    _add_lora_args(grpo_parser)
    _add_train_args(grpo_parser)
    _add_grpo_args(grpo_parser)
    grpo_parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default="outputs/local_sft_big/checkpoints/grpo/checkpoint-20",
        help="从 GRPO 检查点恢复训练",
    )
    grpo_parser.set_defaults(func=_handle_grpo)

    # --- 定义 'evaluate' 子命令 ---
    eval_parser = subparsers.add_parser("evaluate", help="对模型进行离线评估")
    _add_dataset_args(eval_parser)
    _add_model_args(eval_parser)
    eval_parser.add_argument(
        "--sample-size", type=int, default=None, help="评估样本大小 (默认使用全部)"
    )
    eval_parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="要评估的模型的路径 (可以是合并后的模型或基础模型)",
    )
    eval_parser.add_argument(
        "--save-path", type=str, default=None, help="保存评估结果的 Parquet 文件路径"
    )
    eval_parser.set_defaults(func=_handle_evaluate)

    # --- 定义 'predict' 子命令 ---
    predict_parser = subparsers.add_parser("predict", help="为自定义问题生成答案")
    _add_model_args(predict_parser)
    predict_parser.add_argument(
        "--model-path", type=str, default=None, help="基础模型的路径"
    )
    predict_parser.add_argument(
        "--adapter-path", type=str, default=None, help="LoRA 适配器的路径 (可选)"
    )
    predict_parser.add_argument(
        "--max-new-tokens", type=int, default=512, help="生成的最大 token 数"
    )
    predict_parser.add_argument(
        "--question", action="append", required=True, help="要提问的问题 (可多次指定)"
    )
    predict_parser.set_defaults(func=_handle_predict)

    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    """项目 CLI 的主入口函数。"""
    parser = build_parser()
    # 解析命令行参数。如果 `argv` 为 None，则 `argparse` 会自动使用 `sys.argv[1:]`。
    # 传入 `argv` 的能力使得这个函数易于在测试中调用。
    args = parser.parse_args(list(argv) if argv is not None else None)
    # 调用与所选子命令关联的函数（通过 `set_defaults` 设置）。
    # 这是 `argparse` 实现分发逻辑的推荐方式，避免了大量的 if/elif/else 判断。
    args.func(args)
