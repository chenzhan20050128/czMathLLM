from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import unsloth  # noqa: F401  # must be imported before peft/transformers
import torch
from peft import PeftModel
from transformers import AutoTokenizer
from unsloth import FastLanguageModel, is_bfloat16_supported

from .config import TrainingConfig
from .assets import ensure_model


def _resolve_model_reference(model_ref: Optional[str]) -> str:
    if model_ref is None:
        raise ValueError("Model reference cannot be None")
    path = Path(model_ref)
    if path.exists():
        return str(path)
    return str(ensure_model(model_ref))


def load_tokenizer(config: TrainingConfig):
    tokenizer_id = config.tokenizer_id or config.base_model_local_path()
    resolved_id = _resolve_model_reference(tokenizer_id)
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_id,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(
    config: TrainingConfig,
    *,
    model_path: Optional[str] = None,
):
    import os

    os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
    resolved_path = _resolve_model_reference(
        model_path or config.base_model_local_path()
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=resolved_path,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
        load_in_8bit=config.load_in_8bit,
        full_finetuning=config.full_finetuning,
    )
    tokenizer.padding_side = "right"
    return model, tokenizer


def prepare_lora_model(model, config: TrainingConfig):
    return FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=_target_modules(model, config),
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        use_gradient_checkpointing=(
            "unsloth" if config.gradient_checkpointing else False
        ),  # type: ignore[arg-type]
        random_state=config.random_seed,
        use_rslora=config.use_rslora,
        loftq_config=None,
    )


def _target_modules(model, config: TrainingConfig) -> Sequence[str]:
    if hasattr(model, "target_modules"):
        return getattr(model, "target_modules")
    return (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


def ensure_precision() -> tuple[bool, bool]:
    bf16_supported = is_bfloat16_supported()
    return not bf16_supported, bf16_supported


def merge_and_save(
    peft_model: PeftModel,
    tokenizer,
    config: TrainingConfig,
) -> None:
    if not config.save_merged_model:
        return
    dtype = "fp16"
    if ensure_precision()[1] and config.merge_dtype == "bf16":
        dtype = "bf16"
    peft_model.save_pretrained_merged(
        config.merged_model_dir,
        tokenizer,
        save_method=f"merged_{dtype}",
    )  # type: ignore[operator]


def prepare_for_inference(model) -> None:
    FastLanguageModel.for_inference(model)


def generate_answers(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    max_new_tokens: int,
    device: Optional[str] = None,
) -> list[str]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    FastLanguageModel.for_inference(model)
    inputs = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)
