# flake8: noqa
"""命令行入口的核心实现。

该模块使用 Python 内置的 `argparse` 库来构建一个强大的命令行接口（CLI）。
它定义了不同的子命令（如 `train`, `evaluate`, `predict`），并为每个子命令
注册了一系列参数。核心功能是将解析出的命令行参数转换为项目内部使用的
`ProjectConfig` 配置对象，然后调用相应的业务逻辑函数。

为了优化启动性能，真正的入口函数 `main` 被设计为可以从其他地方（如 `__init__.py`）
调用，从而实现延迟加载，避免在不必要时导入重量级的深度学习库。
"""

# from __future__ import annotations: 同样是为了支持延迟解析类型注解。
# 在 Python 3.10+ 中，类型注解可以在定义前使用，此导入是为了向前兼容。
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

# PeftModel 用于加载和应用 PEFT（Parameter-Efficient Fine-Tuning）适配器，如 LoRA。
from peft import PeftModel

# 导入项目内部的配置类、评估函数、模型加载函数、提示词处理工具和训练器。
from .config import DatasetSource, ProjectConfig
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

    # 解析推理和指令数据集的参数。
    reasoning_source = _parse_source_arg(args.reasoning_source)
    instruction_source = _parse_source_arg(args.instruction_source)

    mix = []
    if reasoning_source:
        mix.append(
            DatasetSource(
                **reasoning_source,
                weight=args.reasoning_weight,
                reasoning=True,  # 标记为推理数据集
            )
        )
    if instruction_source:
        mix.append(
            DatasetSource(
                **instruction_source,
                weight=args.instruction_weight,
                reasoning=False,  # 标记为指令数据集
            )
        )

    # 如果 mix 列表为空（即命令行未指定任何数据源），则使用默认值。
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

    # 启发式地判断是否为文件路径（即使文件当前不存在），这在某些部署场景下很有用。
    if any(arg.endswith(ext) for ext in (".json", ".jsonl", ".jsonl.gz", ".parquet")):
        return {"path": arg}

    # 如果包含冒号，则解析为 "仓库名:子集名" 的格式
    if ":" in arg:
        name, subset = arg.split(":", 1)
        return {"name": name, "subset": subset}

    # 默认情况下，将其视为一个 Hugging Face 仓库名
    return {"name": arg}


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
    base_model_id = getattr(args, "base_model_id", None)
    if base_model_id:
        training.base_model_id = base_model_id
    # 如果命令行参数存在，则使用它，否则保持 training 对象中的原值。
    training.base_model_path = getattr(
        args, "base_model_path", training.base_model_path
    )
    training.max_seq_length = getattr(args, "max_seq_length", training.max_seq_length)
    training.load_in_4bit = getattr(args, "load_in_4bit", training.load_in_4bit)
    training.load_in_8bit = getattr(args, "load_in_8bit", training.load_in_8bit)
    training.full_finetuning = getattr(
        args, "full_finetuning", training.full_finetuning
    )

    # --- LoRA 相关覆盖 ---
    training.lora_rank = getattr(args, "lora_rank", training.lora_rank)
    training.lora_alpha = getattr(args, "lora_alpha", training.lora_alpha)
    training.lora_dropout = getattr(args, "lora_dropout", training.lora_dropout)

    # --- 训练过程相关覆盖 ---
    training.gradient_accumulation_steps = getattr(
        args, "gradient_accumulation_steps", training.gradient_accumulation_steps
    )
    training.micro_batch_size = getattr(
        args, "micro_batch_size", training.micro_batch_size
    )
    training.learning_rate = getattr(args, "learning_rate", training.learning_rate)
    training.weight_decay = getattr(args, "weight_decay", training.weight_decay)
    training.warmup_steps = getattr(args, "warmup_steps", training.warmup_steps)
    training.num_train_epochs = getattr(
        args, "num_train_epochs", training.num_train_epochs
    )
    training.max_steps = getattr(args, "max_steps", training.max_steps)
    training.logging_steps = getattr(args, "logging_steps", training.logging_steps)
    training.eval_steps = getattr(args, "eval_steps", training.eval_steps)
    training.save_steps = getattr(args, "save_steps", training.save_steps)
    training.save_total_limit = getattr(
        args, "save_total_limit", training.save_total_limit
    )
    training.random_seed = getattr(args, "random_seed", training.random_seed)
    training.output_dir = Path(getattr(args, "output_dir", training.output_dir))
    training.experiment_name = getattr(
        args, "experiment_name", training.experiment_name
    )

    # --- 数据集相关覆盖 ---
    # 检查是否存在与数据集相关的参数，如果存在，则调用 _load_dataset_mix 更新配置
    if hasattr(args, "dataset_config") or hasattr(args, "reasoning_source"):
        training.dataset_mix = _load_dataset_mix(args, training.dataset_mix)
    training.eval_split_ratio = getattr(
        args, "eval_split_ratio", training.eval_split_ratio
    )
    training.dataset_num_proc = getattr(
        args, "dataset_num_proc", training.dataset_num_proc
    )

    # --- 模型保存相关覆盖 ---
    training.save_merged_model = getattr(
        args, "save_merged_model", training.save_merged_model
    )
    training.merge_dtype = getattr(args, "merge_dtype", training.merge_dtype)

    # --- GRPO 相关覆盖 ---
    # 检查 with_grpo 标志，如果为 True，则启用 GRPO
    if hasattr(args, "with_grpo"):
        project.grpo.enable = args.with_grpo or project.grpo.enable
    # 对于 GRPO 的参数，使用 `or` 链接默认值，确保即使命令行传入 0 或空字符串也能正确处理
    project.grpo.steps = (
        getattr(args, "grpo_steps", project.grpo.steps) or project.grpo.steps
    )
    project.grpo.learning_rate = (
        getattr(args, "grpo_learning_rate", project.grpo.learning_rate)
        or project.grpo.learning_rate
    )
    project.grpo.beta = (
        getattr(args, "grpo_beta", project.grpo.beta) or project.grpo.beta
    )
    project.grpo.kl_coef = (
        getattr(args, "grpo_kl", project.grpo.kl_coef) or project.grpo.kl_coef
    )
    project.grpo.mini_batch_size = (
        getattr(args, "grpo_mini_batch", project.grpo.mini_batch_size)
        or project.grpo.mini_batch_size
    )
    project.grpo.gradient_accumulation_steps = (
        getattr(
            args,
            "grpo_gradient_accumulation",
            project.grpo.gradient_accumulation_steps,
        )
        or project.grpo.gradient_accumulation_steps
    )
    grpo_save_steps = getattr(args, "grpo_save_steps", None)
    if grpo_save_steps is not None:
        project.grpo.save_steps = grpo_save_steps
    grpo_reference_free = getattr(args, "grpo_reference_free", None)
    if grpo_reference_free is not None:
        project.grpo.reference_free = grpo_reference_free
    max_prompt_len = getattr(args, "grpo_max_prompt_len", None)
    if max_prompt_len is not None:
        project.grpo.max_prompt_len = max_prompt_len
    max_completion_len = getattr(args, "grpo_max_completion_len", None)
    if max_completion_len is not None:
        project.grpo.max_completion_len = max_completion_len
    num_generations = getattr(args, "grpo_num_generations", None)
    if num_generations is not None:
        project.grpo.num_generations_per_prompt = num_generations
    max_tokens_per_step = getattr(args, "grpo_max_tokens_per_step", None)
    if max_tokens_per_step is not None:
        project.grpo.max_tokens_per_step = (
            max_tokens_per_step if max_tokens_per_step > 0 else None
        )
    grpo_dataset_arg = getattr(args, "grpo_dataset", None)
    if grpo_dataset_arg is not None:
        parsed = _parse_source_arg(grpo_dataset_arg)
        if parsed is None:
            project.grpo.dataset = None
        else:
            dataset_kwargs = dict(parsed)
            grpo_split = getattr(args, "grpo_dataset_split", None)
            if grpo_split:
                dataset_kwargs["split"] = grpo_split
            grpo_max_samples = getattr(args, "grpo_dataset_max_samples", None)
            if grpo_max_samples is not None:
                dataset_kwargs["max_samples"] = grpo_max_samples
            project.grpo.dataset = DatasetSource(reasoning=True, **dataset_kwargs)


# --- 参数注册函数 ---
# 将相关参数的定义封装在独立的函数中，使 `build_parser` 函数更清晰、更模块化。
# 这种做法提高了代码的可读性和可维护性。

def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    """注册与数据源相关的命令行参数."""
    parser.add_argument(
        "--dataset-config", type=str, help="定义数据集混合的JSON文件路径"
    )
    parser.add_argument(
        "--reasoning-source", type=str, help="主要的推理数据集 (HF 仓库或本地文件)"
    )
    parser.add_argument("--instruction-source", type=str, help="辅助的指令数据集")
    parser.add_argument(
        "--reasoning-weight", type=float, default=0.75, help="推理数据集的权重"
    )
    parser.add_argument(
        "--instruction-weight", type=float, default=0.25, help="指令数据集的权重"
    )


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    """注册模型加载相关的参数."""
    parser.add_argument(
        "--base-model-id", type=str, default=None, help="Hugging Face 上的基础模型 ID"
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="models/Qwen3-4B-Thinking-2507",
        help="本地基础模型的路径",
    )
    parser.add_argument("--max-seq-length", type=int, default=4096, help="最大序列长度")
    # `action="store_true"`: 这是一个布尔标志。当命令行出现 `--load-in-4bit` 标志时，
    # argparse 会将 `load_in_4bit` 属性设为 True。如果标志不存在，则为 False。
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=True,
        help="以4位模式加载模型 (默认启用)",
    )
    # `dest` 和 `action="store_false"`: 这是一种创建“否定”标志的技巧。
    # 当命令行出现 `--no-4bit` 标志时，argparse 会将 `load_in_4bit` (`dest`指定的目标) 设为 False。
    # 这比让用户输入 `--load-in-4bit False` 更直观。
    parser.add_argument(
        "--no-4bit", dest="load_in_4bit", action="store_false", help="禁用4位加载模式"
    )
    parser.add_argument("--load-in-8bit", action="store_true", help="以8位模式加载模型")
    parser.add_argument(
        "--full-finetuning", action="store_true", help="进行全参数微调 (而不是 LoRA)"
    )


def _add_lora_args(parser: argparse.ArgumentParser) -> None:
    """注册 LoRA (低秩适配) 相关的参数."""
    parser.add_argument("--lora-rank", type=int, default=64, help="LoRA 秩 (rank)")
    parser.add_argument(
        "--lora-alpha", type=int, default=64, help="LoRA alpha 缩放因子"
    )
    parser.add_argument(
        "--lora-dropout", type=float, default=0.05, help="LoRA dropout 概率"
    )


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    """注册训练阶段的超参数选项."""
    parser.add_argument(
        "--micro-batch-size", type=int, default=2, help="每个设备上的微批次大小"
    )
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=1, help="梯度累积步数"
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--warmup-steps", type=int, default=100, help="学习率预热步数")
    parser.add_argument(
        "--num-train-epochs", type=float, default=1.0, help="总训练轮数"
    )
    parser.add_argument(
        "--max-steps", type=int, default=-1, help="最大训练步数 (若非-1，则覆盖 epochs)"
    )
    parser.add_argument("--logging-steps", type=int, default=10, help="日志记录步数")
    parser.add_argument("--eval-steps", type=int, default=50, help="评估步数")
    parser.add_argument("--save-steps", type=int, default=200, help="模型保存步数")
    parser.add_argument(
        "--save-total-limit", type=int, default=3, help="最多保存的检查点数量"
    )
    parser.add_argument("--random-seed", type=int, default=3407, help="随机种子")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/local_sft_big",
        help="训练输出和检查点的根目录",
    )
    parser.add_argument(
        "--experiment-name", type=str, default="qwen_math_tutor", help="实验名称，用于区分不同的运行"
    )
    parser.add_argument(
        "--eval-split-ratio", type=float, default=0.02, help="从训练集中划分出的验证集比例"
    )
    parser.add_argument(
        "--dataset-num-proc", type=int, default=1, help="数据预处理时使用的进程数"
    )
    parser.add_argument(
        "--save-merged-model",
        action="store_true",
        default=True,
        help="训练后自动将 LoRA 适配器与基础模型合并并保存 (默认启用)",
    )
    parser.add_argument(
        "--no-save-merged-model",
        dest="save_merged_model",
        action="store_false",
        help="不保存合并后的模型",
    )
    parser.add_argument(
        "--merge-dtype",
        type=str,
        default="fp16",
        help="合并模型时使用的数据类型 (fp16, bf16, float)",
    )


def _add_grpo_args(parser: argparse.ArgumentParser) -> None:
    """注册 GRPO 阶段的特有参数。"""
    parser.add_argument("--with-grpo", action="store_true", help="在 SFT 后启用 GRPO 训练")
    parser.add_argument("--grpo-steps", type=int, default=500, help="GRPO 训练的总步数")
    parser.add_argument("--grpo-learning-rate", type=float, default=8e-6, help="GRPO 阶段的学习率")
    parser.add_argument("--grpo-beta", type=float, default=0.2, help="GRPO 损失中的 beta 参数 (KL 散度权重)")
    parser.add_argument("--grpo-kl", type=float, default=0.06, help="GRPO 的 KL 散度系数")
    parser.add_argument("--grpo-mini-batch", type=int, default=16, help="GRPO 的 mini-batch 大小")
    parser.add_argument("--grpo-gradient-accumulation", type=int, default=1, help="GRPO 的梯度累积步数")
    parser.add_argument(
        "--grpo-reference-free",
        dest="grpo_reference_free",
        action="store_true",
        help="使用无参考模型的 GRPO 变体",
    )
    parser.add_argument(
        "--no-grpo-reference-free",
        dest="grpo_reference_free",
        action="store_false",
        help="使用带参考模型的标准 GRPO",
    )
    # `set_defaults` 用于为可能不出现的参数设置一个默认值，以避免 AttributeError。
    parser.set_defaults(grpo_reference_free=None)
    parser.add_argument(
        "--grpo-dataset",
        type=str,
        default=None,
        help="用于 GRPO 的特定数据集 (HF 仓库或本地文件)",
    )
    parser.add_argument(
        "--grpo-dataset-split", type=str, default=None, help="GRPO 数据集使用的数据切分 (如 'train')"
    )
    parser.add_argument(
        "--grpo-dataset-max-samples",
        type=int,
        default=None,
        help="可选，限制 GRPO 数据集的最大样本数",
    )
    parser.add_argument(
        "--grpo-max-prompt-len",
        type=int,
        default=768,
        help="覆盖 GRPO 生成阶段的最大 prompt token 数",
    )
    parser.add_argument(
        "--grpo-max-completion-len",
        type=int,
        default=2048,
        help="覆盖 GRPO 生成阶段的最大 completion token 数",
    )
    parser.add_argument(
        "--grpo-num-generations",
        type=int,
        default=2,
        help="在 GRPO 期间，每个 prompt 生成的响应数量",
    )
    parser.add_argument(
        "--grpo-save-steps",
        type=int,
        default=None,
        help="每 N 个训练步数保存一次 GRPO 检查点",
    )
    parser.add_argument(
        "--grpo-max-tokens-per-step",
        type=int,
        default=None,
        help=(
            "每个优化步骤的估计 token 数的软上限；设置后，"
            "训练器将裁剪 prompt/completion 长度以遵守此预算。"
        ),
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
    # `action="append"`: 允许多次使用 `--question` 参数，每次的值都会被追加到一个列表中。
    # e.g., `... predict --question "1+1=?" --question "2+2=?"`
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
