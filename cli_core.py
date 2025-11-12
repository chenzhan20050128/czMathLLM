# flake8: noqa
"""命令行入口的核心实现。

将 ``argparse`` 解析出的参数转换为项目配置对象，并根据子命令执行
训练、评估或推理。为了降低首次导入开销，真正的入口函数放在
``__init__.py`` 中进行延迟加载。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from peft import PeftModel

from .config import DatasetSource, ProjectConfig
from .evaluation import evaluate_model
from .modeling import generate_answers, load_base_model, prepare_for_inference
from .prompts import batched_prompts
from .trainers import run_grpo_training, run_sft_training


def _load_dataset_mix(
    args, default_mix: Iterable[DatasetSource]
) -> tuple[DatasetSource, ...]:
    """根据命令行参数生成数据源组合。"""
    if args.dataset_config:
        config_path = Path(args.dataset_config)
        with config_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return tuple(DatasetSource(**entry) for entry in data)

    reasoning_source = _parse_source_arg(args.reasoning_source)
    instruction_source = _parse_source_arg(args.instruction_source)

    mix = []
    if reasoning_source:
        mix.append(
            DatasetSource(
                **reasoning_source,
                weight=args.reasoning_weight,
                reasoning=True,
            )
        )
    if instruction_source:
        mix.append(
            DatasetSource(
                **instruction_source,
                weight=args.instruction_weight,
                reasoning=False,
            )
        )
    if not mix:
        mix = list(default_mix)
    return tuple(mix)


def _parse_source_arg(arg: Optional[str]) -> Optional[dict]:
    """解析数据源字符串，可接受路径、``repo:subset`` 或仓库名。"""
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


def _apply_common_overrides(project: ProjectConfig, args) -> None:
    """将命令行覆盖项映射到项目配置对象上。"""
    training = project.training
    base_model_id = getattr(args, "base_model_id", None)
    if base_model_id:
        training.base_model_id = base_model_id
    training.base_model_path = getattr(
        args, "base_model_path", training.base_model_path
    )
    training.max_seq_length = getattr(args, "max_seq_length", training.max_seq_length)
    training.load_in_4bit = getattr(args, "load_in_4bit", training.load_in_4bit)
    training.load_in_8bit = getattr(args, "load_in_8bit", training.load_in_8bit)
    training.full_finetuning = getattr(
        args, "full_finetuning", training.full_finetuning
    )
    training.lora_rank = getattr(args, "lora_rank", training.lora_rank)
    training.lora_alpha = getattr(args, "lora_alpha", training.lora_alpha)
    training.lora_dropout = getattr(args, "lora_dropout", training.lora_dropout)
    training.gradient_accumulation_steps = getattr(
        args,
        "gradient_accumulation_steps",
        training.gradient_accumulation_steps,
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
    if hasattr(args, "dataset_config") or hasattr(args, "reasoning_source"):
        training.dataset_mix = _load_dataset_mix(args, training.dataset_mix)
    training.eval_split_ratio = getattr(
        args, "eval_split_ratio", training.eval_split_ratio
    )
    training.dataset_num_proc = getattr(
        args, "dataset_num_proc", training.dataset_num_proc
    )
    training.save_merged_model = getattr(
        args, "save_merged_model", training.save_merged_model
    )
    training.merge_dtype = getattr(args, "merge_dtype", training.merge_dtype)

    if hasattr(args, "with_grpo"):
        project.grpo.enable = args.with_grpo or project.grpo.enable
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


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    """注册与数据源相关的命令行参数。"""
    parser.add_argument(
        "--dataset-config", type=str, help="Path to JSON defining dataset mix"
    )
    parser.add_argument(
        "--reasoning-source",
        type=str,
        help="Primary reasoning dataset (HF repo or local file)",
    )
    parser.add_argument(
        "--instruction-source", type=str, help="Complementary instruction dataset"
    )
    parser.add_argument("--reasoning-weight", type=float, default=0.75)
    parser.add_argument("--instruction-weight", type=float, default=0.25)


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    """注册模型加载相关参数。"""
    parser.add_argument("--base-model-id", type=str, default=None)
    parser.add_argument("--base-model-path", type=str, default=None)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--full-finetuning", action="store_true")


def _add_lora_args(parser: argparse.ArgumentParser) -> None:
    """注册 LoRA 低秩适配相关参数。"""
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    """注册训练阶段的超参数选项。"""
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=3407)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--experiment-name", type=str, default="qwen_math_tutor")
    parser.add_argument("--eval-split-ratio", type=float, default=0.02)
    parser.add_argument("--dataset-num-proc", type=int, default=1)
    parser.add_argument("--save-merged-model", action="store_true", default=True)
    parser.add_argument(
        "--no-save-merged-model", dest="save_merged_model", action="store_false"
    )
    parser.add_argument("--merge-dtype", type=str, default="fp16")


def _add_grpo_args(parser: argparse.ArgumentParser) -> None:
    """注册 GRPO 阶段的特有参数。"""
    parser.add_argument("--with-grpo", action="store_true")
    parser.add_argument("--grpo-steps", type=int, default=None)
    parser.add_argument("--grpo-learning-rate", type=float, default=None)
    parser.add_argument("--grpo-beta", type=float, default=None)
    parser.add_argument("--grpo-kl", type=float, default=None)
    parser.add_argument("--grpo-mini-batch", type=int, default=None)
    parser.add_argument("--grpo-gradient-accumulation", type=int, default=None)
    parser.add_argument(
        "--grpo-reference-free",
        dest="grpo_reference_free",
        action="store_true",
    )
    parser.add_argument(
        "--no-grpo-reference-free",
        dest="grpo_reference_free",
        action="store_false",
    )
    parser.set_defaults(grpo_reference_free=None)
    parser.add_argument(
        "--grpo-dataset",
        type=str,
        default=None,
        help="Dataset for GRPO (HF repo or local file)",
    )
    parser.add_argument(
        "--grpo-dataset-split", type=str, default=None, help="Split for GRPO dataset"
    )
    parser.add_argument(
        "--grpo-dataset-max-samples",
        type=int,
        default=None,
        help="Optional cap on GRPO samples",
    )
    parser.add_argument(
        "--grpo-max-prompt-len",
        type=int,
        default=None,
        help="Override max prompt tokens for GRPO generation",
    )
    parser.add_argument(
        "--grpo-max-completion-len",
        type=int,
        default=None,
        help="Override max completion tokens for GRPO generation",
    )
    parser.add_argument(
        "--grpo-num-generations",
        type=int,
        default=None,
        help="Override number of generations per prompt during GRPO",
    )


def _handle_train(args: argparse.Namespace) -> None:
    """处理 ``train`` 子命令，执行 SFT，并按需衔接 GRPO。"""
    project = ProjectConfig()
    _apply_common_overrides(project, args)
    metrics = run_sft_training(
        project, resume_from_checkpoint=args.resume_from_checkpoint
    )
    print("SFT training metrics:", json.dumps(metrics, indent=2, default=str))

    if args.with_grpo:
        grpo_metrics = run_grpo_training(project)
        print("GRPO training info:", json.dumps(grpo_metrics, indent=2, default=str))


def _handle_grpo(args: argparse.Namespace) -> None:
    """处理 ``grpo`` 子命令，仅运行强化学习阶段。"""
    project = ProjectConfig()
    _apply_common_overrides(project, args)
    project.grpo.enable = True
    grpo_metrics = run_grpo_training(
        project,
        resume_from_checkpoint=args.resume_from_checkpoint,
        resume_trainer_state=args.resume_trainer_state,
    )
    print("GRPO training info:", json.dumps(grpo_metrics, indent=2, default=str))


def _handle_evaluate(args: argparse.Namespace) -> None:
    """处理 ``evaluate`` 子命令，输出统计信息并可选择保存结果。"""
    project = ProjectConfig()
    _apply_common_overrides(project, args)
    df = evaluate_model(
        project, model_path=args.model_path, sample_size=args.sample_size
    )
    print(df.describe(include="all"))
    if args.save_path:
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.save_path)


def _handle_predict(args: argparse.Namespace) -> None:
    """处理 ``predict`` 子命令，实现批量问题解答。"""
    project = ProjectConfig()
    _apply_common_overrides(project, args)
    model, tokenizer = load_base_model(project.training, model_path=args.model_path)
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=False)
    prompts = batched_prompts(
        args.question, system_prompt=project.evaluation.system_prompt
    )
    prepare_for_inference(model)
    outputs = generate_answers(
        model, tokenizer, prompts, max_new_tokens=args.max_new_tokens
    )
    for question, output in zip(args.question, outputs):
        print("\n=== Question ===\n", question)
        print("\n=== Answer ===\n", output)


def build_parser() -> argparse.ArgumentParser:
    """创建顶级命令行解析器并注册各子命令。"""
    parser = argparse.ArgumentParser(description="Math fine-tuning CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run supervised fine-tuning")
    _add_dataset_args(train_parser)
    _add_model_args(train_parser)
    _add_lora_args(train_parser)
    _add_train_args(train_parser)
    _add_grpo_args(train_parser)
    train_parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    train_parser.set_defaults(func=_handle_train)

    grpo_parser = subparsers.add_parser("grpo", help="Run GRPO stage only")
    _add_dataset_args(grpo_parser)
    _add_model_args(grpo_parser)
    _add_lora_args(grpo_parser)
    _add_train_args(grpo_parser)
    _add_grpo_args(grpo_parser)
    grpo_parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    grpo_parser.add_argument(
        "--resume-trainer-state",
        type=str,
        default=None,
        help="Resume GRPO trainer state from this directory (optional).",
    )
    grpo_parser.set_defaults(func=_handle_grpo)

    eval_parser = subparsers.add_parser("evaluate", help="Offline evaluation")
    _add_dataset_args(eval_parser)
    _add_model_args(eval_parser)
    eval_parser.add_argument("--sample-size", type=int, default=None)
    eval_parser.add_argument("--model-path", type=str, default=None)
    eval_parser.add_argument("--save-path", type=str, default=None)
    eval_parser.set_defaults(func=_handle_evaluate)

    predict_parser = subparsers.add_parser(
        "predict", help="Generate answers for custom prompts"
    )
    _add_model_args(predict_parser)
    predict_parser.add_argument("--model-path", type=str, default=None)
    predict_parser.add_argument("--adapter-path", type=str, default=None)
    predict_parser.add_argument("--max-new-tokens", type=int, default=512)
    predict_parser.add_argument("--question", action="append", required=True)
    predict_parser.set_defaults(func=_handle_predict)

    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    """项目 CLI 主入口。"""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.func(args)
