"""模型加载与推理工具函数。

注意：`unsloth` 必须优先导入，它会对 `transformers` 等库注入若干运行时补丁，
以获得更快的 LoRA 微调体验。下方的函数主要封装了 tokenizer、基座模型、
LoRA 权重的加载流程，并针对推理阶段做了轻量化设置。"""

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
    """统一解析模型引用，支持本地路径与远端仓库。"""
    if model_ref is None:
        raise ValueError("Model reference cannot be None")
    path = Path(model_ref)
    if path.exists():
        return str(path)
    return str(ensure_model(model_ref))


def load_tokenizer(config: TrainingConfig):
    """根据配置加载分词器。

    若未显式指定 tokenizer，会默认沿用基座模型，并尝试补齐 ``pad_token``。
    ``padding_side = "right"`` 可兼容大部分自回归模型的训练/推理。"""
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
    """加载基座模型并返回 `(model, tokenizer)` 二元组。

    `FastLanguageModel.from_pretrained` 是 Unsloth 对原版 `AutoModel`
    的封装，支持一键设置 4bit/8bit 量化、全参微调等高级选项。"""
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
    """基于基座模型构建 LoRA 适配器。"""
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
    """指定需要插入 LoRA 权重的模块集合。

    某些模型（如 Qwen3）会在权重中自带 `target_modules` 属性；
    若不存在则使用常见的注意力投影层作为默认值。"""
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
    """检测当前硬件是否支持 bfloat16，返回 (use_fp16, use_bf16)。"""
    bf16_supported = is_bfloat16_supported()
    return not bf16_supported, bf16_supported


def merge_and_save(
    peft_model: PeftModel,
    tokenizer,
    config: TrainingConfig,
) -> None:
    """将 LoRA 权重合并回基座并保存。

    当硬件支持 bfloat16 且用户选择 `merge_dtype="bf16"` 时，保存 bf16 精度，
    否则默认使用 fp16，避免 dtype 不兼容导致加载失败。"""
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
    """切换模型到推理模式，在 Unsloth 中会关闭多余的训练开关。"""
    FastLanguageModel.for_inference(model)


def generate_answers(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    max_new_tokens: int,
    device: Optional[str] = None,
    **generate_kwargs,
) -> list[str]:
    """批量生成答案。

    该函数演示了 PyTorch Tensor ``to(device)`` 的惯用写法，以及如何利用
    tokenizer 批量编码输入、调用 `model.generate`。额外的采样参数（例如
    ``temperature``、``top_p``）可以通过关键字参数形式追加。返回值通过
    ``batch_decode`` 解码为纯文本列表。"""
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
        **generate_kwargs,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)
