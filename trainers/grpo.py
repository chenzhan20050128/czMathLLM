# -*- coding: utf-8 -*-
"""基于 TRL 的 GRPO（Group Relative Policy Optimization）训练脚本。

GRPO 是 DPO (Direct Preference Optimization) 的一种泛化，它允许在策略优化
过程中，对同一个提示（prompt）生成的多个响应（completions）进行比较，
而不仅仅是比较一对“赢家”和“输家”。这使得它能更充分地利用偏好数据。

该模块的核心是 `run_grpo_training` 函数，它负责：
1.  加载 SFT 阶段训练好的 LoRA 适配器作为初始策略模型。
2.  （可选）加载一个参考模型（reference model），用于计算 KL 散度惩罚，
    防止策略模型偏离初始状态太远。
3.  准备 GRPO 训练所需的数据集，其格式与 SFT 不同。
4.  将项目自定义的 `reward.py` 中的奖励函数包装成 `trl` 库期望的格式。
5.  动态地构建 `GRPOConfig` 和 `GRPOTrainer` 的参数，以兼容不同版本的 `trl` 库。
6.  执行训练并保存最终的模型。
"""

# from __future__ import annotations: 同样是为了支持延迟解析类型注解。
from __future__ import annotations

import inspect
from dataclasses import fields
from pathlib import Path
from typing import Optional

from peft import PeftModel

# 从 trl 库导入 GRPO 专用的配置类和训练器类。
from trl.trainer.grpo_config import GRPOConfig as HFGRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer

from ..config import GRPOConfig, ProjectConfig, TrainingConfig
from ..data import build_grpo_dataset, build_sft_datasets, load_dataset_source
from ..modeling import load_base_model, merge_and_save
from ..reward import batch_reward
from ..utils import set_global_seed, dump_dataclass


TOKEN_WARNING_THRESHOLD = 160_000
BASE_THROUGHPUT_TOK_PER_S = 320.0


def _format_int(value: int) -> str:
    return f"{value:,}"


def _apply_token_budget_once(
    training_cfg: TrainingConfig,
    grpo_cfg: GRPOConfig,
    workload: dict[str, int],
    budget: int,
) -> tuple[dict[str, int], bool]:
    """Apply a single round of token budget clipping."""

    updated = False
    effective_batch = workload["effective_batch"]
    completions_per_step = workload["completions_per_step"]
    prompt_tokens = workload["prompt_tokens"]
    available_for_completions = budget - prompt_tokens

    if available_for_completions > 0 and completions_per_step > 0:
        raw_limit = available_for_completions // max(completions_per_step, 1)
        if raw_limit < 32:
            print(
                "[GRPO] token预算过低，无法仅通过裁剪 max_completion_len 满足开销。"
                " 可尝试减小 --grpo-num-generations、--grpo-mini-batch 或"
                " --grpo-gradient-accumulation。"
            )
        elif raw_limit < grpo_cfg.max_completion_len:
            print(
                "[GRPO] 自动裁剪 max_completion_len: "
                f"{grpo_cfg.max_completion_len} -> {raw_limit} "
                f"（预算 {_format_int(budget)}）。"
            )
            grpo_cfg.max_completion_len = int(raw_limit)
            updated = True
    else:
        prompt_limit = budget // max(effective_batch, 1)
        if prompt_limit < 64:
            print(
                "[GRPO] token预算过低，当前批次规模下无法满足。"
                " 请考虑减小 --grpo-mini-batch 或 --grpo-gradient-accumulation。"
            )
        elif prompt_limit < grpo_cfg.max_prompt_len:
            print(
                "[GRPO] 自动裁剪 max_prompt_len: "
                f"{grpo_cfg.max_prompt_len} -> {prompt_limit} "
                f"（预算 {_format_int(budget)}）。"
            )
            grpo_cfg.max_prompt_len = int(prompt_limit)
            updated = True

    if (
        grpo_cfg.max_prompt_len + grpo_cfg.max_completion_len
        > training_cfg.max_seq_length
    ):
        cap = max(training_cfg.max_seq_length - grpo_cfg.max_prompt_len, 64)
        if cap < grpo_cfg.max_completion_len:
            print(
                "[GRPO] 自动限制 max_completion_len 以符合模型最大序列长度: "
                f"{grpo_cfg.max_completion_len} -> {cap}。"
            )
            grpo_cfg.max_completion_len = cap
            updated = True

    if updated:
        return grpo_cfg.describe_workload(training_cfg), True
    return workload, False


def _apply_token_budget(
    training_cfg: TrainingConfig,
    grpo_cfg: GRPOConfig,
    workload: dict[str, int],
) -> dict[str, int]:
    budget = grpo_cfg.max_tokens_per_step
    if not budget or budget <= 0:
        return workload
    while workload["tokens_per_step"] > budget:
        workload, updated = _apply_token_budget_once(
            training_cfg,
            grpo_cfg,
            workload,
            budget,
        )
        if not updated:
            break
    return workload


def _log_workload(grpo_cfg: GRPOConfig, workload: dict[str, int]) -> None:
    effective_batch = workload["effective_batch"]
    completions_per_step = workload["completions_per_step"]
    tokens_per_step = workload["tokens_per_step"]
    prompt_len = workload["prompt_len"]
    completion_len = workload["completion_len"]

    batch_info = _format_int(effective_batch)
    mini_batch = grpo_cfg.mini_batch_size
    grad_accum = grpo_cfg.gradient_accumulation_steps
    print(
        "[GRPO] 有效prompt批次 = "
        f"{batch_info} (mini_batch={mini_batch}, "
        f"grad_accum={grad_accum})"
    )
    print(
        "        每step生成 "
        f"{_format_int(completions_per_step)} 条completion "
        f"(每prompt {grpo_cfg.num_generations_per_prompt} 条)。"
    )
    print(
        "[GRPO] 估算单step token 开销 ≈ "
        f"{_format_int(tokens_per_step)} (prompt_len={prompt_len}, "
        f"completion_len={completion_len})。"
    )
    if not grpo_cfg.reference_free:
        print("[GRPO] 当前启用了参考模型，logprob 计算将额外增加一次完整的前向传播。")

    if grpo_cfg.max_tokens_per_step:
        print(
            "[GRPO] token预算设定为 "
            f"{_format_int(grpo_cfg.max_tokens_per_step)}，"
            f"估算开销 {_format_int(tokens_per_step)}。"
        )

    if (
        grpo_cfg.max_tokens_per_step in (None, 0)
        and tokens_per_step > TOKEN_WARNING_THRESHOLD
    ):
        approx_minutes = tokens_per_step / BASE_THROUGHPUT_TOK_PER_S / 60.0
        print(
            "[GRPO][提示] 该配置预计每step耗时约 "
            f"{approx_minutes:.1f} 分钟（按 "
            f"{int(BASE_THROUGHPUT_TOK_PER_S)} tok/s 估算）。"
        )
        print(
            "        若需更快迭代，可减小 --grpo-num-generations、"
            "--grpo-max-completion-len、--grpo-mini-batch 或"
            " --grpo-gradient-accumulation。"
        )


def _reward_function(
    samples: list[str],
    *,  # 强制后续参数为关键字参数
    references: list[str],
    metadatas: list[dict],
    **_,
) -> list[float]:
    """一个简单的包装函数，将项目内部的 `batch_reward` 函数连接到 TRL 训练器。

    TRL 的训练器在计算奖励时，会调用一个签名为 `reward_fn(samples, **kwargs)` 的函数。
    这个包装器确保了 `batch_reward` 所需的 `references` 和 `metadatas` 参数
    能够被正确地从 `kwargs` 中提取并传递过去。
    """
    return batch_reward(samples, references, metadatas)


def run_grpo_training(
    project: ProjectConfig,
    *,  # 强制后续参数为关键字参数
    resume_from_checkpoint: Optional[str] = None,
) -> dict:
    """执行 GRPO 强化学习阶段，并返回训练统计信息。

    执行流程:
    1.  **初始化**: 检查 GRPO 是否启用，并设置随机种子。
    2.  **模型加载**:
        - 加载基础模型。
        - 从 SFT 阶段保存的目录 (`finetuned_model_dir`) 加载 LoRA 适配器，
          并将其应用到基础模型上，得到 `peft_model`。这个模型是我们要优化的策略模型。
        - （可选）如果不是 `reference_free` 模式，则额外加载一个同样的模型作为
          `ref_model`，但其权重是冻结的，仅用于计算 KL 散度。
    3.  **数据准备**:
        - 加载用于 GRPO 的源数据集。
        - 使用 `build_grpo_dataset` 将其转换为包含 `prompt`, `reference`, `metadata` 的格式。
        - 创建一个 `prompt_lookup` 字典，用于在奖励计算时根据 `prompt` 快速查找其对应的
          `reference` 和 `metadata`。
    4.  **奖励函数准备**: 定义一个闭包 `reward_fn`。这个函数在被 TRL 调用时，
        能够捕获外部作用域的 `prompt_lookup` 字典，从而为每个生成的样本找到
        正确的参考答案和元数据，然后调用 `_reward_function` 计算奖励。
    5.  **配置参数兼容性处理**:
        - `trl` 库的 API 变化较快。为了让代码能在不同版本的 `trl` 上运行，这里
          使用了 Python 的 `inspect` 和 `dataclasses.fields` 模块来动态检查
          `HFGRPOConfig` 和 `GRPOTrainer` 的构造函数需要哪些参数。
        - 然后，根据检查结果，动态地构建一个 `config_kwargs` 和 `trainer_kwargs` 字典，
          只填充当前版本 `trl` 支持的参数。这是一种非常健壮的编程技巧。
    6.  **初始化并执行训练**: 创建 `GRPOTrainer` 实例并调用其 `train` 方法。
    7.  **保存模型**: 训练结束后，保存更新后的 LoRA 适配器，并（可选地）合并保存
        完整的模型。
    """
    training_cfg = project.training
    grpo_cfg = project.grpo

    # 1. 初始化
    if not grpo_cfg.enable:
        return {}

    dump_dataclass(project, training_cfg.output_dir / "project_config_grpo.json")
    set_global_seed(training_cfg.random_seed)

    workload = grpo_cfg.describe_workload(training_cfg)
    workload = _apply_token_budget(training_cfg, grpo_cfg, workload)
    _log_workload(grpo_cfg, workload)

    def _resolve_peft_dir() -> Path:
        candidates: list[Path] = []
        finetuned_dir = Path(training_cfg.finetuned_model_dir)
        candidates.append(finetuned_dir)
        if resume_from_checkpoint:
            candidates.append(Path(resume_from_checkpoint))
        checkpoints_dir = Path(training_cfg.checkpoints_dir)
        if checkpoints_dir.exists():
            checkpoint_dirs = sorted(
                (p for p in checkpoints_dir.iterdir() if p.is_dir()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            candidates.extend(checkpoint_dirs)
        seen: set[Path] = set()
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / "adapter_config.json").exists():
                return candidate
        raise FileNotFoundError(
            "未在输出目录或检查点中找到 adapter_config.json，"
            "请确保已完成 SFT 或指定有效的 --resume-from-checkpoint。"
        )

    peft_source_dir = _resolve_peft_dir()
    base_model, tokenizer = load_base_model(training_cfg)
    # 加载 SFT 训练好的 LoRA 权重，并设置为可训练状态
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(peft_source_dir),
        is_trainable=True,
    )
    # 某些 PEFT 版本需要显式调用此方法来确保梯度能够流向 LoRA 权重。
    if hasattr(peft_model, "enable_input_require_grads"):
        peft_model.enable_input_require_grads()
    peft_model.train()  # 切换到训练模式

    # 加载参考模型（如果需要）
    ref_model = None
    if not grpo_cfg.reference_free:
        ref_base, _ = load_base_model(training_cfg)
        ref_model = PeftModel.from_pretrained(
            ref_base,
            str(peft_source_dir),
            is_trainable=False,  # 参考模型不需要训练
        )

    # 3. 数据准备
    if grpo_cfg.dataset is None:
        # 如果没有专门指定 GRPO 数据集，则复用 SFT 的训练集
        train_dataset, _ = build_sft_datasets(training_cfg, tokenizer)
        rl_source_dataset = train_dataset
    else:
        rl_source_dataset = load_dataset_source(
            grpo_cfg.dataset,
            cache_dir=training_cfg.cache_dir,
        )

    rl_dataset = build_grpo_dataset(
        rl_source_dataset,
        max_samples=grpo_cfg.steps,  # 限制样本数以匹配训练步数
    )

    # 创建一个查找表，用于在奖励计算时快速获取参考答案和元数据
    prompt_lookup = {
        example["prompt"]: (example["reference"], example.get("metadata", {}))
        for example in rl_dataset
    }

    # 4. 奖励函数准备
    def reward_fn(*args, **_kwargs) -> list[float]:
        samples = _kwargs.pop("samples", None)
        prompts = _kwargs.pop("prompts", None)
        completions = _kwargs.pop("completions", None)

        remaining_args = list(args)
        if samples is None and remaining_args:
            samples = remaining_args.pop(0)
        if prompts is None and remaining_args:
            prompts = remaining_args.pop(0)

        if samples is None:
            samples = completions
        if prompts is None and _kwargs.get("original_prompts"):
            prompts = _kwargs["original_prompts"]

        if samples is None or prompts is None:
            raise ValueError("reward_fn missing samples or prompts")

        refs, metas = [], []
        for prompt in prompts:
            reference, metadata = prompt_lookup.get(prompt, ("", {}))
            refs.append(reference)
            metas.append(metadata)
        return _reward_function(samples, references=refs, metadatas=metas)

    # `dataclasses.fields` 可动态枚举 HF 配置的字段，便于跨版本兼容。
    grpo_output_dir = Path(training_cfg.checkpoints_dir) / "grpo"
    grpo_output_dir.mkdir(parents=True, exist_ok=True)
    config_kwargs = {}
    available_fields = {field.name for field in fields(HFGRPOConfig)}
    if "learning_rate" in available_fields:
        config_kwargs["learning_rate"] = grpo_cfg.learning_rate
    if "beta" in available_fields:
        config_kwargs["beta"] = grpo_cfg.beta
    if "per_device_train_batch_size" in available_fields:
        config_kwargs["per_device_train_batch_size"] = grpo_cfg.mini_batch_size
    if "gradient_accumulation_steps" in available_fields:
        config_kwargs["gradient_accumulation_steps"] = (
            grpo_cfg.gradient_accumulation_steps
        )
    if "total_episodes" in available_fields:
        config_kwargs["total_episodes"] = grpo_cfg.steps
    if "steps" in available_fields:
        config_kwargs["steps"] = grpo_cfg.steps
    if "max_prompt_length" in available_fields:
        config_kwargs["max_prompt_length"] = grpo_cfg.max_prompt_len
    if "max_completion_length" in available_fields:
        config_kwargs["max_completion_length"] = grpo_cfg.max_completion_len
    if "kl_coef" in available_fields:
        config_kwargs["kl_coef"] = grpo_cfg.kl_coef
    if "kl_alpha" in available_fields:
        config_kwargs["kl_alpha"] = grpo_cfg.kl_coef
    if "mixed_precision" in available_fields:
        config_kwargs["mixed_precision"] = grpo_cfg.mixed_precision
    if "save_steps" in available_fields:
        config_kwargs["save_steps"] = grpo_cfg.save_steps
    if "output_dir" in available_fields:
        config_kwargs["output_dir"] = str(grpo_output_dir)
    if "generation_batch_size" in available_fields:
        config_kwargs["generation_batch_size"] = grpo_cfg.mini_batch_size
    if "num_generations" in available_fields:
        config_kwargs["num_generations"] = grpo_cfg.num_generations_per_prompt
    if "unsloth_num_chunks" in available_fields:
        config_kwargs["unsloth_num_chunks"] = grpo_cfg.unsloth_num_chunks

    hf_config = HFGRPOConfig(**config_kwargs)
    # 确保 max_steps 被正确设置
    if hasattr(hf_config, "max_steps"):
        setattr(hf_config, "max_steps", grpo_cfg.steps)
    if not hasattr(hf_config, "unsloth_num_chunks"):
        setattr(hf_config, "unsloth_num_chunks", grpo_cfg.unsloth_num_chunks)

    # 动态构建 GRPOTrainer 的初始化参数
    trainer_kwargs = {}
    trainer_sig = inspect.signature(GRPOTrainer.__init__)
    if "model" in trainer_sig.parameters:
        trainer_kwargs["model"] = peft_model
    if "ref_model" in trainer_sig.parameters:
        trainer_kwargs["ref_model"] = ref_model
    if "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    expects_reward_fn = "reward_fn" in trainer_sig.parameters
    expects_reward_function = "reward_function" in trainer_sig.parameters
    expects_reward_funcs = "reward_funcs" in trainer_sig.parameters

    if expects_reward_fn:
        trainer_kwargs["reward_fn"] = reward_fn
    if expects_reward_function and not expects_reward_fn:
        trainer_kwargs["reward_function"] = reward_fn
    if expects_reward_funcs:
        trainer_kwargs["reward_funcs"] = [reward_fn]
    if "args" in trainer_sig.parameters:
        trainer_kwargs["args"] = hf_config
    if "train_dataset" in trainer_sig.parameters:
        trainer_kwargs["train_dataset"] = rl_dataset

    # 6. 初始化并执行训练
    trainer = GRPOTrainer(**trainer_kwargs)

    train_kwargs = {}
    train_sig = inspect.signature(trainer.train)
    if "resume_from_checkpoint" in train_sig.parameters:
        train_kwargs["resume_from_checkpoint"] = resume_from_checkpoint

    trainer.train(**train_kwargs)

    # 7. 保存模型
    # GRPOTrainer 内部可能修改了模型，所以我们用 trainer.model
    final_model = trainer.model
    final_model.save_pretrained(str(training_cfg.finetuned_model_dir))
    tokenizer.save_pretrained(str(training_cfg.finetuned_model_dir))
    merge_and_save(final_model, tokenizer, training_cfg)

    return {"grpo_steps": grpo_cfg.steps}
