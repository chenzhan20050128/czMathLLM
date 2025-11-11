from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from trl import SFTConfig, SFTTrainer

from ..config import ProjectConfig
from ..data import build_sft_datasets
from ..modeling import (
    ensure_precision,
    load_base_model,
    merge_and_save,
    prepare_lora_model,
)
from ..utils import set_global_seed


def run_sft_training(
    project: ProjectConfig,
    *,
    resume_from_checkpoint: Optional[str] = None,
) -> dict:
    project.ensure_directories()
    training_cfg = project.training
    set_global_seed(training_cfg.random_seed)

    model, tokenizer = load_base_model(training_cfg)
    peft_model = prepare_lora_model(model, training_cfg)

    train_dataset, eval_dataset = build_sft_datasets(training_cfg, tokenizer)
    _, bf16_supported = ensure_precision()

    args_dict: dict[str, Any] = dict(
        per_device_train_batch_size=training_cfg.micro_batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        warmup_steps=training_cfg.warmup_steps,
        max_steps=training_cfg.max_steps,
        num_train_epochs=training_cfg.num_train_epochs,
        learning_rate=training_cfg.learning_rate,
        weight_decay=training_cfg.weight_decay,
        fp16=not bf16_supported,
        bf16=bf16_supported,
        logging_steps=training_cfg.logging_steps,
        optim="adamw_8bit" if training_cfg.load_in_4bit else "adamw_torch",
        lr_scheduler_type="cosine",
        seed=training_cfg.random_seed,
        output_dir=str(training_cfg.checkpoints_dir),
        eval_strategy="steps",
        eval_steps=training_cfg.eval_steps,
        save_steps=training_cfg.save_steps,
        save_total_limit=training_cfg.save_total_limit,
        dataset_num_proc=training_cfg.dataset_num_proc,
        gradient_checkpointing=training_cfg.gradient_checkpointing,
    )

    if (
        "evaluation_strategy" in SFTConfig.__init__.__code__.co_varnames
    ):  # type: ignore[attr-defined]
        args_dict["evaluation_strategy"] = args_dict.pop("eval_strategy")

    training_args = SFTConfig(**args_dict)

    trainer = SFTTrainer(
        model=peft_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        args=training_args,
        max_seq_length=training_cfg.max_seq_length,
    )

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    metrics = {**train_result.metrics}

    peft_model.save_pretrained(training_cfg.finetuned_model_dir)
    tokenizer.save_pretrained(training_cfg.finetuned_model_dir)

    merge_and_save(peft_model, tokenizer, training_cfg)

    trainer.save_state()
    trainer.save_model(Path(training_cfg.checkpoints_dir) / "last")

    return metrics
