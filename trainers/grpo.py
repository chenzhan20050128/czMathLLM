# -*- coding: utf-8 -*-
"""基于 TRL 的 GRPO（Group Relative Policy Optimization）训练脚本。

GRPO 是 DPO (Direct Preference Optimization) 的一种泛化，它允许在策略优化
过程中，对同一个提示（prompt）生成的多个响应（completions）进行比较，
而不仅仅是比较一对“赢家”和“输家”。这使得它能更充分地利用偏好数据。

该模块的核心是 `run_grpo_training` 函数，它负责：
1.  加载 SFT 阶段训练好的 LoRA 适配器作为初始策略模型。
2.  （可选）加载一个参考模型（reference model），用于计算 KL 散度惩罚，
    防止策略模型偏离初始状态太远，从而保持生成质量。
3.  准备 GRPO 训练所需的数据集，其格式与 SFT 不同，通常包含 `prompt` 和 `reference`。
4.  将项目自定义的 `reward.py` 中的奖励函数包装成 `trl` 库期望的格式。
5.  动态地构建 `GRPOConfig` 和 `GRPOTrainer` 的参数，以兼容不同版本的 `trl` 库，
    这是一个非常关键的健壮性设计。
6.  执行训练并保存最终的模型。
"""

# from __future__ import annotations: 同样是为了支持延迟解析类型注解。
from __future__ import annotations

import inspect
from dataclasses import fields
from pathlib import Path
from typing import Any, Optional

from peft import PeftModel

# 从 trl 库导入 GRPO 专用的配置类和训练器类。
# TRL (Transformer Reinforcement Learning) 是 Hugging Face 提供的用于强化学习微调的库。
from trl.trainer.grpo_config import GRPOConfig as HFGRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer
from transformers.trainer_callback import TrainerCallback

from ..config import GRPOConfig, ProjectConfig, TrainingConfig
from ..data import build_grpo_dataset, build_sft_datasets, load_dataset_source
from ..modeling import load_base_model, merge_and_save
from ..reward import batch_reward
from ..utils import set_global_seed, dump_dataclass


# --- 常量定义 ---
TOKEN_WARNING_THRESHOLD = 160_000  # 当单步 token 开销超过此阈值时发出警告
BASE_THROUGHPUT_TOK_PER_S = 320.0  # 用于估算训练时间的基准吞吐量 (tokens/sec)
MAX_LOGGED_COMPLETION_CHARS = 4096  # 日志中打印的最长 completion 字符数，防止日志过长
# 预留给系统/模板/特殊 token 的冗余，避免 prompt+completion 正好贴边导致运行期再截断
RESERVED_SPECIAL_TOKENS = 128


class _RewardBuffer:
    """一个用于在训练过程中缓冲和聚合奖励信息的辅助类."""

    def __init__(self, max_chars: int = MAX_LOGGED_COMPLETION_CHARS) -> None:
        self._records: list[tuple[str, str, float]] = []
        self._max_chars = max_chars

    def record(
        self,
        prompts: list[str],
        samples: list[str],
        rewards: list[float],
    ) -> None:
        """记录一批奖励信息."""
        if not prompts or not samples or not rewards:
            return
        for prompt, sample, reward in zip(prompts, samples, rewards):
            try:
                reward_value = float(reward)
            except (TypeError, ValueError):
                continue
            self._records.append((prompt, sample, reward_value))

    def flush(self, step: int) -> None:
        """计算并打印缓冲的奖励统计信息，然后清空缓冲区."""
        if not self._records:
            return

        total_reward = sum(record[2] for record in self._records)
        count = len(self._records)
        avg_reward = total_reward / max(count, 1)
        # 找到奖励最高的记录，用于展示
        best_prompt, best_sample, best_reward = max(
            self._records, key=lambda record: record[2]
        )

        message = (
            "[GRPO][step {step}] reward mean = {avg:.4f}, " "max = {best:.4f}"
        ).format(
            step=step,
            avg=avg_reward,
            best=best_reward,
        )
        print(message, flush=True)

        # 对过长的样本进行截断，避免日志爆炸
        display_sample = best_sample
        truncated = False
        if len(display_sample) > self._max_chars:
            display_sample = display_sample[: self._max_chars] + "\n... [truncated]"
            truncated = True

        print(
            f"[GRPO][step {step}] prompt (best reward):\n{best_prompt}",
            flush=True,
        )
        print(
            f"[GRPO][step {step}] completion (best reward):\n{display_sample}",
            flush=True,
        )
        if truncated:
            print(
                f"[GRPO][step {step}] 注意：输出已截断为 {self._max_chars} 字符，避免日志过长。",
                flush=True,
            )

        self._records.clear()


class _RewardLoggingCallback(TrainerCallback):
    """一个自定义的 `transformers.TrainerCallback`，用于在训练的特定阶段打印奖励信息."""

    def __init__(self, buffer: _RewardBuffer) -> None:
        self._buffer = buffer

    def on_step_end(self, args, state, control, **kwargs):
        """在每个训练步骤结束时被调用."""
        self._buffer.flush(state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        """在整个训练过程结束时被调用."""
        # 确保训练结束时残留的记录被输出。
        self._buffer.flush(state.global_step)


def _format_int(value: int) -> str:
    """将整数格式化为带千位分隔符的字符串，提高可读性."""
    return f"{value:,}"


def _apply_token_budget_once(
    training_cfg: TrainingConfig,
    grpo_cfg: GRPOConfig,
    workload: dict[str, int],
    budget: int,
) -> tuple[dict[str, int], bool]:
    """尝试通过裁剪 `max_completion_len` 或 `max_prompt_len` 来满足单步 token 预算."""

    updated = False
    effective_batch = workload["effective_batch"]
    completions_per_step = workload["completions_per_step"]
    prompt_tokens = workload["prompt_tokens"]
    available_for_completions = budget - prompt_tokens

    if available_for_completions > 0 and completions_per_step > 0:
        # 如果还有预算给 completion，则尝试裁剪 completion 长度
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
        # 如果连 prompt 的预算都不够，则尝试裁剪 prompt 长度
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

    # 确保 prompt + completion 不超过模型的最大序列长度（留出一定冗余给模板/特殊token）
    if (
        grpo_cfg.max_prompt_len + grpo_cfg.max_completion_len
        > training_cfg.max_seq_length - RESERVED_SPECIAL_TOKENS
    ):
        effective_max_seq = max(
            256, training_cfg.max_seq_length - RESERVED_SPECIAL_TOKENS
        )
        cap = max(effective_max_seq - grpo_cfg.max_prompt_len, 64)
        if cap < grpo_cfg.max_completion_len:
            print(
                "[GRPO] 自动限制 max_completion_len 以符合模型最大序列长度: "
                f"{grpo_cfg.max_completion_len} -> {cap}。"
            )
            grpo_cfg.max_completion_len = cap
            updated = True

    if updated:
        # 如果配置被更新，重新计算工作负载
        return grpo_cfg.describe_workload(training_cfg), True
    return workload, False


def _apply_token_budget(
    training_cfg: TrainingConfig,
    grpo_cfg: GRPOConfig,
    workload: dict[str, int],
) -> dict[str, int]:
    """循环应用 token 预算限制，直到满足预算或无法再裁剪."""
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
    """打印 GRPO 训练的工作负载估算信息."""
    effective_batch = workload["effective_batch"]
    completions_per_step = workload["completions_per_step"]
    tokens_per_step = workload["tokens_per_step"]
    prompt_len = workload["prompt_len"]
    completion_len = workload["completion_len"]

    batch_info = _format_int(effective_batch)
    mini_batch = grpo_cfg.mini_batch_size
    grad_accum = grpo_cfg.gradient_accumulation_steps
    print(
        (
            f"[GRPO] 有效prompt批次 = {batch_info} "
            f"(mini_batch={mini_batch}, grad_accum={grad_accum})"
        )
    )
    print(
        (
            f"        每step生成 {_format_int(completions_per_step)} 条completion "
            f"(每prompt {grpo_cfg.num_generations_per_prompt} 条)。"
        )
    )
    print(
        (
            f"[GRPO] 估算单step token 开销 ≈ {_format_int(tokens_per_step)} "
            f"(prompt_len={prompt_len}, completion_len={completion_len})。"
        )
    )
    if not grpo_cfg.reference_free:
        print("[GRPO] 当前启用了参考模型，logprob 计算将额外增加一次完整的前向传播。")

    if grpo_cfg.max_tokens_per_step:
        print(
            (
                f"[GRPO] token预算设定为 "
                f"{_format_int(grpo_cfg.max_tokens_per_step)}，"
                f"估算开销 {_format_int(tokens_per_step)}。"
            )
        )

    if (
        grpo_cfg.max_tokens_per_step in (None, 0)
        and tokens_per_step > TOKEN_WARNING_THRESHOLD
    ):
        approx_minutes = tokens_per_step / BASE_THROUGHPUT_TOK_PER_S / 60.0
        print(
            (
                f"[GRPO][提示] 该配置预计每step耗时约 {approx_minutes:.1f} 分钟"
                f"（按 {int(BASE_THROUGHPUT_TOK_PER_S)} tok/s 估算）。"
            )
        )
        print(
            (
                "        若需更快迭代，可减小 --grpo-num-generations、"
                "--grpo-max-completion-len、--grpo-mini-batch 或 "
                "--grpo-gradient-accumulation。"
            )
        )


def _reward_function(
    samples: list[str],
    *,  # 强制后续参数为关键字参数
    references: list[str],
    metadatas: list[dict],
    **_: Any,  # 使用 `**_` 忽略其他所有未使用的关键字参数，增加函数的健壮性
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
    1.  **初始化**: 检查 GRPO 是否启用，保存配置，设置随机种子，并估算和打印工作负载。
    2.  **模型加载**:
        - 加载基础模型和分词器。
        - 从 SFT 阶段保存的目录 (`finetuned_model_dir`) 或指定的检查点加载 LoRA 适配器，
          并将其应用到基础模型上，得到 `peft_model`。这个模型是我们要优化的策略模型。
        - （可选）如果不是 `reference_free` 模式，则额外加载一个同样的模型作为
          `ref_model`，但其权重是冻结的，仅用于计算 KL 散度。
    3.  **数据准备**:
        - 加载用于 GRPO 的源数据集。如果未指定，则复用 SFT 的训练集。
                - 使用 `build_grpo_dataset` 将其转换为包含 `prompt`, `reference`,
                    `metadata` 的格式。
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
          只填充当前版本 `trl` 支持的参数。这是一种非常健壮的、面向未来的编程技巧。
    6.  **初始化并执行训练**: 创建 `GRPOTrainer` 实例并调用其 `train` 方法。
    7.  **保存模型**: 训练结束后，保存更新后的 LoRA 适配器，并（可选地）合并保存
        完整的模型。
    """
    training_cfg = project.training
    grpo_cfg = project.grpo

    # 1. 初始化
    if not grpo_cfg.enable:
        return {}

    dump_dataclass(
        project,
        training_cfg.output_dir / "project_config_grpo.json",
    )
    set_global_seed(training_cfg.random_seed)

    workload = grpo_cfg.describe_workload(training_cfg)
    workload = _apply_token_budget(training_cfg, grpo_cfg, workload)
    _log_workload(grpo_cfg, workload)

    reward_buffer = _RewardBuffer()

    def _resolve_peft_dir() -> Path:
        """智能地查找最新的、有效的 PEFT (LoRA) 适配器目录."""
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
                print(f"[GRPO] 找到并使用 PEFT 适配器于: {candidate}")
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
        """闭包，捕获 prompt_lookup 并将其传递给实际的奖励计算函数."""
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
        # 优先使用 original_prompts（通常与 samples 等长，已重复展开）
        if prompts is None and _kwargs.get("original_prompts"):
            prompts = _kwargs["original_prompts"]

        if samples is None or prompts is None:
            raise ValueError("reward_fn missing samples or prompts")

        # 确保 prompts 与 samples 数量对齐：
        # - 若长度相同，直接对齐；
        # - 若 samples 是 prompts 的整数倍（每个 prompt 生成多个 completions），按倍数展开 prompts；
        # - 否则，回退为按最短长度对齐并给出提示（极少出现）。
        try:
            num_samples = len(samples)  # type: ignore[arg-type]
            num_prompts = len(prompts)  # type: ignore[arg-type]
        except Exception:
            num_samples = num_prompts = 0

        if num_samples and num_prompts and num_samples != num_prompts:
            if num_samples % max(num_prompts, 1) == 0:
                repeat = num_samples // max(num_prompts, 1)
                prompts = [
                    p for p in prompts for _ in range(repeat)
                ]  # type: ignore[list-item]
            else:
                print(
                    f"[GRPO][warn] prompts 与 samples 数量不一致，"
                    f"samples={num_samples}, prompts={num_prompts}。"
                    "将按最短长度对齐，可能影响奖励质量。"
                )

        # 构建与 samples 等长的 refs / metas
        refs, metas = [], []
        for prompt in list(prompts)[: len(samples)]:  # type: ignore[index]
            reference, metadata = prompt_lookup.get(prompt, ("", {}))
            meta = dict(metadata or {})
            meta.setdefault("question", prompt)
            meta.setdefault("prompt", prompt)
            meta.setdefault("original_prompt", prompt)
            refs.append(reference)
            metas.append(meta)

        rewards = _reward_function(samples, references=refs, metadatas=metas)
        reward_buffer.record(prompts, samples, rewards)
        return rewards

    # 5. 动态构建 HFGRPOConfig 参数
    grpo_output_dir = Path(training_cfg.checkpoints_dir) / "grpo"
    grpo_output_dir.mkdir(parents=True, exist_ok=True)
    config_kwargs = {}
    available_fields = {field.name for field in fields(HFGRPOConfig)}

    # 这是一个非常健壮的设计：只填充当前 TRL 版本支持的参数
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
        gen_batch = grpo_cfg.generation_batch_size or grpo_cfg.mini_batch_size
        num_gen = max(1, grpo_cfg.num_generations_per_prompt)
        if gen_batch % num_gen != 0:
            adjusted = ((gen_batch + num_gen - 1) // num_gen) * num_gen
            print(
                (
                    "[GRPO] 自动调整 generation_batch_size: "
                    f"{gen_batch} -> {adjusted} "
                    f"(num_generations={num_gen})"
                )
            )
            gen_batch = adjusted
        config_kwargs["generation_batch_size"] = gen_batch
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

    # 一些环境（如从 checkpoint 恢复时）可能会让外部库根据 trainer_state.json 打印旧的 batch 参数。
    # 这里显式再次覆盖，确保当前 CLI 传入的配置具有最高优先级。
    try:
        hf_config.per_device_train_batch_size = grpo_cfg.mini_batch_size
        hf_config.gradient_accumulation_steps = grpo_cfg.gradient_accumulation_steps
    except Exception:
        pass

    # 调试输出，帮助确认最终生效到 HF 配置中的关键训练参数
    hf_bs = getattr(hf_config, "per_device_train_batch_size", "n/a")
    hf_gas = getattr(hf_config, "gradient_accumulation_steps", "n/a")
    print(
        f"[GRPO] HF args — per_device_train_batch_size = {hf_bs}, "
        f"gradient_accumulation_steps = {hf_gas}"
    )

    # 动态构建 GRPOTrainer 的初始化参数
    trainer_kwargs = {}
    trainer_sig = inspect.signature(GRPOTrainer.__init__)
    if "model" in trainer_sig.parameters:
        trainer_kwargs["model"] = peft_model
    if "ref_model" in trainer_sig.parameters:
        trainer_kwargs["ref_model"] = ref_model
    if "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    # 动态确定奖励函数的参数名
    expects_reward_fn = "reward_fn" in trainer_sig.parameters
    expects_reward_function = "reward_function" in trainer_sig.parameters
    expects_reward_funcs = "reward_funcs" in trainer_sig.parameters

    if expects_reward_fn:
        trainer_kwargs["reward_fn"] = reward_fn
    elif expects_reward_function:
        trainer_kwargs["reward_function"] = reward_fn
    elif expects_reward_funcs:
        trainer_kwargs["reward_funcs"] = [reward_fn]

    if "args" in trainer_sig.parameters:
        trainer_kwargs["args"] = hf_config
    if "train_dataset" in trainer_sig.parameters:
        trainer_kwargs["train_dataset"] = rl_dataset

    # 6. 初始化并执行训练
    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.add_callback(_RewardLoggingCallback(reward_buffer))

    # 再次在 trainer 上强制覆盖关键 batch 配置（尤其是在 resume_from_checkpoint 时，
    # 某些库可能会参考 checkpoint 中保存的旧参数进行展示/打印）。
    try:
        if hasattr(trainer, "args"):
            setattr(
                trainer.args,
                "per_device_train_batch_size",
                grpo_cfg.mini_batch_size,
            )
            setattr(
                trainer.args,
                "gradient_accumulation_steps",
                grpo_cfg.gradient_accumulation_steps,
            )
            t_bs = getattr(trainer.args, "per_device_train_batch_size", "n/a")
            t_gas = getattr(trainer.args, "gradient_accumulation_steps", "n/a")
            print(
                f"[GRPO] Trainer args — per_device_train_batch_size = {t_bs}, "
                f"gradient_accumulation_steps = {t_gas}"
            )
    except Exception:
        pass

    train_kwargs = {}
    train_sig = inspect.signature(trainer.train)
    if "resume_from_checkpoint" in train_sig.parameters:
        train_kwargs["resume_from_checkpoint"] = resume_from_checkpoint

    trainer.train(**train_kwargs)

    # 7. 保存模型
    final_model = trainer.model
    final_model.save_pretrained(str(training_cfg.finetuned_model_dir))
    tokenizer.save_pretrained(str(training_cfg.finetuned_model_dir))
    merge_and_save(final_model, tokenizer, training_cfg)

    return {"grpo_steps": grpo_cfg.steps}
