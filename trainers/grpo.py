"""基于 TRL 的 GRPO（Group Relative Policy Optimization）训练脚本。

GRPO 是强化学习阶段的一种策略梯度方法，支持对同一条 prompt 生成多条
样本并相互比较得分。此模块主要负责：

1. 载入 LoRA 适配器与参考模型；
2. 构造奖励函数，将项目自定义的 `batch_reward` 注入 TRL；
3. 动态适配不同版本 TRL 的配置/初始化签名。"""

from __future__ import annotations

import inspect
from dataclasses import fields
from typing import Optional

from peft import PeftModel
from trl.trainer.grpo_config import GRPOConfig as HFGRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer

from ..config import ProjectConfig
from ..data import build_grpo_dataset, build_sft_datasets, load_dataset_source
from ..modeling import load_base_model, merge_and_save
from ..reward import batch_reward
from ..utils import set_global_seed


def _reward_function(
    samples,
    *,
    references,
    metadatas,
    **_,
) -> list[float]:  # type: ignore[override]
    """包装项目级奖励函数，符合 TRL 对回调签名的要求。"""
    return batch_reward(samples, references, metadatas)


def run_grpo_training(
    project: ProjectConfig,
    *,
    resume_from_checkpoint: Optional[str] = None,
) -> dict:
    """执行 GRPO 强化学习阶段，并返回训练统计信息。"""
    training_cfg = project.training
    grpo_cfg = project.grpo

    if not grpo_cfg.enable:
        return {}

    set_global_seed(training_cfg.random_seed)

    base_model, tokenizer = load_base_model(training_cfg)
    peft_model = PeftModel.from_pretrained(
        base_model,
        training_cfg.finetuned_model_dir,
        is_trainable=True,
    )
    # 某些 PEFT 版本需要显式开启输入梯度，方便低秩权重参与优化。
    enable_input_grads = getattr(
        peft_model,
        "enable_input_require_grads",
        None,
    )
    if callable(enable_input_grads):
        enable_input_grads()
    peft_model.train()

    ref_model = None
    if not grpo_cfg.reference_free:
        ref_base, _ = load_base_model(training_cfg)
        ref_model = PeftModel.from_pretrained(
            ref_base,
            training_cfg.finetuned_model_dir,
            is_trainable=False,
        )

    if grpo_cfg.dataset is None:
        train_dataset, _ = build_sft_datasets(training_cfg, tokenizer)
        rl_source_dataset = train_dataset
    else:
        rl_source_dataset = load_dataset_source(
            grpo_cfg.dataset,
            cache_dir=training_cfg.cache_dir,
        )

    rl_dataset = build_grpo_dataset(
        rl_source_dataset,
        max_samples=grpo_cfg.steps,
    )
    prompt_lookup = {
        example["prompt"]: (example["reference"], example.get("metadata", {}))
        for example in rl_dataset.to_list()
    }

    def reward_fn(  # noqa: ANN001
        samples=None,
        prompts=None,
        completions=None,
        **_kwargs,
    ) -> list[float]:
        if samples is None:
            samples = completions or _kwargs.get("samples") or []
        if prompts is None:
            prompts = _kwargs.get("prompts") or []
        refs = []
        metas = []
        for prompt in prompts:
            reference, metadata = prompt_lookup.get(prompt, ("", {}))
            refs.append(reference)
            metas.append(metadata)
        return _reward_function(samples, references=refs, metadatas=metas)

    # `dataclasses.fields` 可动态枚举 HF 配置的字段，便于跨版本兼容。
    config_kwargs = {}
    available_fields = {field.name for field in fields(HFGRPOConfig)}
    if "learning_rate" in available_fields:
        config_kwargs["learning_rate"] = grpo_cfg.learning_rate
    if "beta" in available_fields:
        config_kwargs["beta"] = grpo_cfg.beta
    if "mini_batch_size" in available_fields:
        config_kwargs["mini_batch_size"] = grpo_cfg.mini_batch_size
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
    if "num_generations_per_prompt" in available_fields:
        config_kwargs["num_generations_per_prompt"] = (
            grpo_cfg.num_generations_per_prompt
        )

    hf_config = HFGRPOConfig(**config_kwargs)
    if hasattr(hf_config, "max_steps"):
        setattr(hf_config, "max_steps", grpo_cfg.steps)
    if not hasattr(hf_config, "unsloth_num_chunks"):
        setattr(hf_config, "unsloth_num_chunks", -1)

    trainer_kwargs = {}
    trainer_sig = inspect.signature(GRPOTrainer.__init__)
    if "model" in trainer_sig.parameters:
        trainer_kwargs["model"] = peft_model
    if "ref_model" in trainer_sig.parameters:
        trainer_kwargs["ref_model"] = ref_model
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    if "reward_funcs" in trainer_sig.parameters:
        trainer_kwargs["reward_funcs"] = [reward_fn]
    if "args" in trainer_sig.parameters:
        trainer_kwargs["args"] = hf_config
    elif "config" in trainer_sig.parameters:
        trainer_kwargs["config"] = hf_config
    if "train_dataset" in trainer_sig.parameters:
        trainer_kwargs["train_dataset"] = rl_dataset

    trainer = GRPOTrainer(**trainer_kwargs)

    train_kwargs = {}
    train_sig = inspect.signature(GRPOTrainer.train)
    if "resume_from_checkpoint" in train_sig.parameters:
        train_kwargs["resume_from_checkpoint"] = resume_from_checkpoint
    trainer.train(**train_kwargs)
    trainer.save_model(str(training_cfg.finetuned_model_dir))
    tokenizer.save_pretrained(training_cfg.finetuned_model_dir)
    merge_and_save(peft_model, tokenizer, training_cfg)

    return {"grpo_steps": grpo_cfg.steps}
