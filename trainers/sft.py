# -*- coding: utf-8 -*-
"""监督微调（Supervised Fine-Tuning, SFT）的训练脚本。

该模块的核心是 `run_sft_training` 函数，它 orchestrates（编排）了整个
SFT 流程。它利用了 Hugging Face `trl` 库提供的 `SFTTrainer`，这是一个
专门为在类 GPT 模型上进行指令微调而设计的强大工具。

整个流程与项目中的其他模块紧密协作：
- 从 `config` 加载配置。
- 从 `data` 构建数据集。
- 从 `modeling` 加载和准备模型。
- 从 `utils` 设置随机种子。
"""

# from __future__ import annotations: 同样是为了支持延迟解析类型注解。
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

# `SFTConfig` 和 `SFTTrainer` 是 `trl` 库中用于 SFT 的核心类。
from trl import SFTConfig, SFTTrainer

from ..config import ProjectConfig
from ..data import build_sft_datasets
from ..modeling import (
    ensure_precision,
    load_base_model,
    merge_and_save,
    prepare_lora_model,
)
from ..utils import set_global_seed, dump_dataclass


def run_sft_training(
    project: ProjectConfig,
    *, # 强制后续参数为关键字参数
    resume_from_checkpoint: Optional[str] = None,
) -> dict:
    """执行完整的监督微调流程，并返回训练过程中的指标。

    执行步骤:
    1.  **初始化**: 确保输出目录存在，并设置全局随机种子以保证可复现性。
    2.  **模型加载**: 使用 `load_base_model` 加载（可能量化的）基础模型和分词器。
    3.  **LoRA 准备**: 使用 `prepare_lora_model` 在基础模型上应用 LoRA 配置，
        得到一个 `PeftModel`，其中只有 LoRA 相关的权重是可训练的。
    4.  **数据准备**: 调用 `build_sft_datasets` 来加载、处理并分割训练和评估数据集。
    5.  **配置训练参数**: 将 `TrainingConfig` 中的配置转换为 `SFTConfig`
        （`trl` 库的训练参数类）所期望的格式。这里还处理了不同 `trl` 版本
        的参数名兼容性问题。
    6.  **初始化训练器**: 创建 `SFTTrainer` 实例，将模型、分词器、数据集和
        训练参数传入。
    7.  **执行训练**: 调用 `trainer.train()` 方法启动训练过程。此方法会处理所有
        底层的训练循环、梯度更新、日志记录和模型保存。
    8.  **保存结果**:
        - 保存最终的 LoRA 适配器权重。
        - 如果配置了 `save_merged_model`，则调用 `merge_and_save` 将 LoRA
          权 weights合并到基础模型并保存为完整模型。
        - 保存训练器的状态和最后一个检查点。
    9.  **返回指标**: 返回一个包含训练损失、步数等信息的字典。
    """
    # 1. 初始化
    project.ensure_directories()
    training_cfg = project.training
    set_global_seed(training_cfg.random_seed)

    # 将最终使用的配置保存为 JSON 文件，便于追溯
    dump_dataclass(project, training_cfg.output_dir / "project_config.json")

    # 2. 模型加载
    model, tokenizer = load_base_model(training_cfg)

    # 3. LoRA 准备
    peft_model = prepare_lora_model(model, training_cfg)

    # 4. 数据准备
    train_dataset, eval_dataset = build_sft_datasets(training_cfg, tokenizer)

    # 检查硬件精度支持
    _, bf16_supported = ensure_precision()

    # 5. 配置训练参数
    # 使用字典来构建参数，便于动态修改和传递。
    args_dict: dict[str, Any] = dict(
        per_device_train_batch_size=training_cfg.micro_batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        warmup_steps=training_cfg.warmup_steps,
        max_steps=training_cfg.max_steps,
        num_train_epochs=training_cfg.num_train_epochs,
        learning_rate=training_cfg.learning_rate,
        weight_decay=training_cfg.weight_decay,
        fp16=not bf16_supported, # 如果不支持 bf16，则使用 fp16
        bf16=bf16_supported,
        logging_steps=training_cfg.logging_steps,
        # 根据是否使用4位量化选择不同的优化器。8位 Adam 优化器能节省显存。
        optim="adamw_8bit" if training_cfg.load_in_4bit else "adamw_torch",
        lr_scheduler_type="cosine", # 使用余弦学习率调度器
        seed=training_cfg.random_seed,
        output_dir=str(training_cfg.checkpoints_dir), # 训练检查点保存目录
        eval_strategy="steps", # 按步数进行评估
        eval_steps=training_cfg.eval_steps,
        save_steps=training_cfg.save_steps,
        save_total_limit=training_cfg.save_total_limit,
        dataset_num_proc=training_cfg.dataset_num_proc,
        gradient_checkpointing=training_cfg.gradient_checkpointing,
    )

    # 兼容性处理：较新版本的 trl 将 `eval_strategy` 重命名为 `evaluation_strategy`。
    # 通过检查 `SFTConfig` 的 `__init__` 方法的参数名来动态适应。
    if "evaluation_strategy" in SFTConfig.__init__.__code__.co_varnames:
        args_dict["evaluation_strategy"] = args_dict.pop("eval_strategy")

    # 使用 `**args_dict` 将字典解包为关键字参数来初始化 SFTConfig。
    training_args = SFTConfig(**args_dict)

    # 6. 初始化训练器
    trainer = SFTTrainer(
        model=peft_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text", # 告诉训练器数据集中哪个字段包含要训练的文本
        args=training_args,
        max_seq_length=training_cfg.max_seq_length,
    )

    # 7. 执行训练
    # `resume_from_checkpoint` 参数允许从上次中断的地方继续训练。
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    metrics = {**train_result.metrics}

    # 8. 保存结果
    # 保存 LoRA 适配器权重
    peft_model.save_pretrained(str(training_cfg.finetuned_model_dir))
    tokenizer.save_pretrained(str(training_cfg.finetuned_model_dir))

    # 合并并保存完整模型
    merge_and_save(peft_model, tokenizer, training_cfg)

    # 保存训练器状态，以便未来可以恢复训练
    trainer.save_state()
    # `save_model` 会保存最后一个检查点
    trainer.save_model(str(Path(training_cfg.checkpoints_dir) / "last"))

    # 9. 返回指标
    return metrics
