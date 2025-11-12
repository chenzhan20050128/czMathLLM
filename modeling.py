# -*- coding: utf-8 -*-
"""模型加载与推理相关的核心工具函数。

这个模块封装了使用 `unsloth` 库加载、配置和运行大语言模型的关键步骤。
`unsloth` 是一个专门为 LoRA 微调进行优化的库，能显著提升训练速度和降低显存占用。

**重要提示**: `unsloth` 必须在 `transformers` 和 `peft` 之前导入，因为它会
在运行时对这些库进行“猴子补丁”（monkey-patching），以实现其性能优化。
"""

# from __future__ import annotations: 同样是为了支持延迟解析类型注解。
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

# 导入 unsloth。`# noqa: F401` 告诉 linter（如 flake8）忽略 "unused import" 警告。
# 这里的导入是必须的，因为它有副作用（修改其他库）。
import unsloth  # noqa: F401
import torch
from peft import PeftModel
from transformers import AutoTokenizer
# `FastLanguageModel` 是 unsloth 提供的核心类，用于替代 transformers 的 `AutoModelForCausalLM`。
from unsloth import FastLanguageModel, is_bfloat16_supported

from .config import TrainingConfig
from .assets import ensure_model


def _resolve_model_reference(model_ref: Optional[str]) -> str:
    """统一解析模型引用，使其既支持本地路径，也支持 Hugging Face Hub 仓库名。

    如果 `model_ref` 是一个存在的本地路径，则直接返回该路径。
    否则，调用 `ensure_model` 函数，该函数会检查本地缓存，如果不存在则从 Hub 下载。
    """
    if model_ref is None:
        raise ValueError("模型引用不能为空")
    path = Path(model_ref)
    if path.exists():
        return str(path)
    # 如果不是本地路径，则假定为 Hugging Face Hub ID，并确保模型已下载。
    return str(ensure_model(model_ref))


def load_tokenizer(config: TrainingConfig):
    """根据配置加载分词器 (tokenizer)。

    - 如果配置中没有指定 `tokenizer_id`，则默认使用与基础模型相同的路径。
    - 确保分词器有 `pad_token`。如果缺失，通常用 `eos_token` (end-of-sequence) 来代替。
    - 设置 `padding_side = "right"` 是一个常见的做法，特别是在自回归模型的训练中，
      可以防止模型在填充部分产生不必要的注意力。
    """
    tokenizer_id = config.tokenizer_id or config.base_model_local_path()
    resolved_id = _resolve_model_reference(tokenizer_id)
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_id,
        use_fast=True,          # 尽可能使用 Rust 实现的快速分词器
        trust_remote_code=True, # 允许加载模型仓库中自定义的 Python 代码
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # 将填充添加到序列的右侧
    return tokenizer


def load_base_model(
    config: TrainingConfig,
    *, # 强制后续参数为关键字参数
    model_path: Optional[str] = None,
):
    """加载基础模型，并返回 `(model, tokenizer)` 元组。

    `FastLanguageModel.from_pretrained` 是 `unsloth` 的核心功能，它封装了
    `transformers` 的加载逻辑，并无缝集成了以下优化：
    - **4位/8位量化**: 通过 `load_in_4bit` 或 `load_in_8bit` 参数，可以一键加载
      量化后的模型，极大地减少显存占用。
    - **性能优化**: `unsloth` 会自动应用 Flash Attention 等技术来加速计算。
    - **全参数微调支持**: `full_finetuning` 参数可以方便地在全参数微调和
      参数高效微调（如 LoRA）之间切换。
    """
    import os
    # 禁用 unsloth 的匿名统计信息收集
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
    """在基础模型之上应用 LoRA 配置，使其准备好进行 LoRA 微调。

    `FastLanguageModel.get_peft_model` 是 `unsloth` 提供的另一个关键函数。
    它接收一个基础模型和 LoRA 配置，然后返回一个 `PeftModel` 对象。
    这个返回的模型已经插入了 LoRA 适配器层，并且只有这些适配器层的参数是可训练的。
    """
    return FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,                     # LoRA 秩
        target_modules=_target_modules(model, config), # 要应用 LoRA 的模块
        lora_alpha=config.lora_alpha,           # LoRA alpha
        lora_dropout=config.lora_dropout,       # LoRA dropout
        use_gradient_checkpointing="unsloth" if config.gradient_checkpointing else False, # 使用 unsloth 优化的梯度检查点
        random_state=config.random_seed,
        use_rslora=config.use_rslora,           # 是否使用 Rank-Stabilized LoRA
        loftq_config=None,                      # LoftQ 量化配置
    )


def _target_modules(model, config: TrainingConfig) -> Sequence[str]:
    """智能地确定应该在哪些模块上应用 LoRA。

    - 优先检查模型自身是否定义了 `target_modules` 属性（一些模型，如 Unsloth 优化过的
      模型，会自带这个推荐配置）。
    - 如果没有，则回退到一组常见的默认值，这些通常是 Transformer 模型中的
      注意力机制相关的线性层（query, key, value, output projections 等）。
    """
    if hasattr(model, "target_modules"):
        return getattr(model, "target_modules")
    return (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


def ensure_precision() -> tuple[bool, bool]:
    """检测当前硬件是否支持 bfloat16 精度。

    返回一个元组 `(use_fp16, use_bf16)`。
    - `bfloat16` (bf16) 是现代 NVIDIA GPU (Ampere 架构及以后) 支持的一种浮点格式，
      它具有与 `float32` 相似的动态范围，但只占用 16 位，非常适合深度学习。
    - 如果硬件不支持 bf16，则回退到使用 `float16` (fp16)。
    """
    bf16_supported = is_bfloat16_supported()
    return not bf16_supported, bf16_supported


def merge_and_save(
    peft_model: PeftModel,
    tokenizer,
    config: TrainingConfig,
) -> None:
    """将训练好的 LoRA 权重合并回基础模型，并保存为完整的模型。

    这对于部署非常有用，因为推理时可以直接加载合并后的模型，而无需处理
    基础模型和 LoRA 适配器的分离加载。
    - `unsloth` 提供了 `save_pretrained_merged` 方法来简化这个过程。
    - 它会根据硬件支持和用户配置，智能地选择保存为 `bf16` 或 `fp16` 格式。
    """
    if not config.save_merged_model:
        return

    dtype = "fp16"
    # 如果硬件支持且用户配置也希望使用 bf16，则使用 bf16
    if ensure_precision()[1] and config.merge_dtype == "bf16":
        dtype = "bf16"

    # `save_method` 参数告诉 unsloth 保存的格式
    peft_model.save_pretrained_merged(
        str(config.merged_model_dir), # unsloth 的函数需要字符串路径
        tokenizer,
        save_method=f"merged_{dtype}",
    )


def prepare_for_inference(model) -> None:
    """将模型切换到推理模式。

    `FastLanguageModel.for_inference` 是 `unsloth` 提供的一个便利函数，
    它会执行一些优化，例如合并 LoRA 权重（如果尚未合并）并禁用梯度计算，
    使模型为最高效的推理做好准备。
    """
    FastLanguageModel.for_inference(model)


def generate_answers(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    max_new_tokens: int,
    device: Optional[str] = None,
) -> list[str]:
    """使用模型为一批提示词批量生成答案。

    这是一个典型的 PyTorch 推理流程：
    1.  确定计算设备（优先使用 CUDA GPU）。
    2.  调用 `FastLanguageModel.for_inference` 确保模型处于推理状态。
    3.  使用分词器将一批文本提示（`prompts`）转换为模型输入的张量（tensors）。
        - `return_tensors="pt"`: 返回 PyTorch 张量。
        - `padding=True`: 将批次内的序列填充到相同长度。
    4.  使用 `.to(device)` 将输入张量移动到目标设备。
    5.  调用 `model.generate` 方法进行自回归文本生成。
        - `use_cache=True`: 启用 KV 缓存，加速长文本生成。
    6.  使用 `tokenizer.batch_decode` 将生成的 token ID 解码回文本字符串。
    """
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

    # `torch.no_grad()` 上下文管理器可以禁用梯度计算，减少内存消耗并加速推理
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    # 从输出中移除输入部分，只保留新生成的内容
    output_tokens = outputs[:, inputs["input_ids"].shape[1]:]

    return tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
