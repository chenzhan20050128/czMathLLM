HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1 python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen3-4B-Thinking-2507',
    local_dir='/root/autodl-tmp/.autodl/czMathLLM/models/Qwen3-4B-Thinking-2507',
    resume_download=True,
    local_dir_use_symlinks=False,
    max_workers=8,  # 启用多线程（建议4-8个线程）
)
"


# czMathLLM

czMathLLM 是一个围绕 **Unsloth + TRL** 打造的数学教学/解题微调工具集，默认面向 Qwen3 系列模型。项目覆盖数据准备、监督式 LoRA/QLoRA 微调、GRPO 强化阶段、离线评估与推理上机的完整闭环，并通过 `cli.py` 提供一站式命令行体验。

## 项目概览

- **模型管理 (`assets.py`)**：自动下载或复用基础模型，支持通过 `MATH_LLM_MODELS` 重定向缓存目录。
- **配置体系 (`config.py`)**：`ProjectConfig` 聚合训练、GRPO、评估三大配置；`DatasetSource` 支持 HF 仓库或本地 JSON/JSONL。
- **数据流水线 (`data.py`)**：统一抽取题目/答案/推理链、推断最终答案、生成难度与标签，构建 SFT 与 RL 数据集。
- **训练执行 (`trainers/`)**：`run_sft_training` 负责监督微调，`run_grpo_training` 执行强化学习并封装奖励函数。
- **推理与评估 (`modeling.py`、`evaluation.py`)**：封装模型加载、合并 LoRA、批量生成答案与奖励计算。
- **奖励函数 (`reward.py`)**：结合 \boxed{} 解析、数值接近度与字符串重合度的组合奖励，附加难度加权。

> 完整 CLI 入口位于 `cz_math_llm/cli_core.py`，外层 `cli.py` 仅做导入与启动。

## 端到端流程快览

1. **环境准备**：安装与 GPU 匹配的 PyTorch、Unsloth、TRL 等依赖。
2. **数据配置**：编写或复用 `DatasetSource` 描述（可 JSON 定义），自动完成字段清洗与元数据构建。
3. **SFT 训练**：按权重混合推理/指令数据，创建 LoRA 适配器并保存检查点与日志。
4. **（可选）GRPO 强化**：基于自定义或默认数据集进行奖励驱动训练，进一步提升推理表现。
5. **模型合并**：选择性地合并 LoRA 权重，得到便于部署的全量模型目录。
6. **离线评估**：按相同预处理生成评估集合，批量算分导出统计表与 Parquet。
7. **推理上线**：使用 `predict` 子命令或直接调用 `generate_answers` 进行问答测试或集成部署。

## 环境与硬件要求

- **操作系统**：Linux x86_64（已在 CUDA 12+ 驱动环境验证）。
- **Python**：建议 3.10–3.12。项目默认使用 3.12。
- **GPU**：显存 ≥ 16 GB 可运行 Qwen3-4B LoRA；32 GB 以上可适度放宽批量/序列长度。
- **依赖**：`torch 2.8.0`（CUDA 12.1+）、`unsloth`, `trl`, `datasets`, `peft`, `bitsandbytes`, `accelerate` 等，详见 `requirements.txt`。
- **Hugging Face**：若需下载受限模型，请 export `HF_TOKEN`；设置 `HF_HUB_ENABLE_HF_TRANSFER=1` 可提升下载速度。

## 安装步骤

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 若官方未提供匹配的 CUDA 轮子，请先单独安装 torch
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

可选：执行 `huggingface-cli login` 保存访问令牌；设置 `export MATH_LLM_MODELS=/path/to/cache` 控制模型缓存位置。

> 🔐 **不要将 Hugging Face 令牌写入代码或仓库。** 推荐在 shell 配置文件中设置：
> ```bash
export HF_TOKEN="---"
export MATH_LLM_PRIMARY_ENDPOINT="https://aliendao.cn"
export MATH_LLM_SECONDARY_ENDPOINT="https://hf-mirror.com"
> ```
> 如未设置，程序会默认按照 AlienDAO → HF-Mirror → 官方站 的顺序尝试下载。

## 数据准备

### 数据格式

项目会在加载阶段自动归一化字段，支持下列键名（区分大小写）：

| 语义 | 默认键名 | 兼容键 | 说明 |
| --- | --- | --- | --- |
| 题目 | `question` | `prompt`、`instruction`、`problem`、`input` | 必填 |
| 最终答案 | `final_answer` | 自动推断，或 `final`, `answer_box` 等 | 可选 |
| 参考答案 | `answer` | `response`、`completion`、`target` 等 | 必填 |
| 推理链 | `reasoning` | `rationale`、`chain_of_thought`、`cot` 等 | 无则回退到 `answer` |

本地 JSON/JSONL 会读取全部对象并调用 `_normalize_record` 生成标准字段，再进一步推断：

- 若缺失 `final_answer`，尝试解析 `\boxed{...}`、`Answer:`、中文“最终答案”等模式；否则取答案末行。
- 额外生成 `metadata`，包含题目/推理长度统计、难度标签 (`easy/medium/hard`)、主题标签（几何/代数等）。

### 数据集配置

- **快速体验**：`configs/dataset.sample.json` 直接引用本地数据目录，默认 75% 推理型（`data/OpenMathReasoning/data`）+ 25% 指令型（`data/DAPO-Math-17k-Processed/all`）。
- **自定义混合**：JSON 文件应包含 `DatasetSource` 列表，每项支持字段：`name`、`subset`、`split`、`path`（本地文件）、`weight`、`reasoning`、`max_samples`。
- **命令行覆盖**：`train/grpo/evaluate` 子命令均支持 `--dataset-config` 或 `--reasoning-source` / `--instruction-source` / `--grpo-dataset`。当同时传入时，命令行参数会覆盖默认配置。

## 训练流程

```bash
python cli.py <command> [options]
```

### 监督式微调（SFT）

示例命令假设你已在仓库根目录下准备好以下资源：

- 模型：`models/Qwen3-4B-Thinking-2507`
- 推理数据：`data/OpenMathReasoning/data`
- 指令数据：`data/DAPO-Math-17k-Processed/all`

```bash
python cli.py train \
  --base-model-path models/Qwen3-4B-Thinking-2507 \
  --output-dir outputs/local_sft_big \
  --micro-batch-size 8 \
  --gradient-accumulation-steps 4 \
  --lora-rank 64 \
  --lora-dropout 0 \
  --learning-rate 2e-5 \
  --warmup-steps 50 \
  --max-steps 1600 \
  --logging-steps 10 \
  --eval-steps 25 \
  --save-steps 50 \
  --save-total-limit 3 \
  --dataset-num-proc 4 \
  --resume-from-checkpoint outputs/local_sft/checkpoints/checkpoint-200

```

执行逻辑（`run_sft_training`）：

1. `assets.ensure_model` 确保基础模型可用，按需下载。
2. `load_base_model` 以 4-bit 量化加载模型；`prepare_lora_model` 根据配置创建 LoRA 适配器，支持 RS-LoRA、梯度检查点、AdamW 8bit 等。
3. `build_sft_datasets` 读取并拼接多数据源，使用 `format_sft_example` 生成符合 Qwen3 模板的对话文本，并随机划分训练/验证集。
4. `SFTTrainer` 负责训练与评估，周期性写入日志、保存检查点到 `outputs/<experiment>_lora` 与 `outputs/checkpoints`。
5. 训练结束后保存 LoRA 权重、分词器，并可选调用 `merge_and_save` 将 LoRA 合并为全量模型（`outputs/<experiment>_merged`）。

常用参数：`--micro-batch-size`、`--gradient-accumulation-steps`、`--learning-rate`、`--num-train-epochs`、`--load-in-4bit/--no-4bit`、`--full-finetuning`。

### GRPO 强化阶段（可选）

```bash
python cli.py train \
  --reasoning-source data/OpenMathReasoning/data \
  --instruction-source data/DAPO-Math-17k-Processed/all \
  --base-model-path models/Qwen3-4B-Thinking-2507 \
  --output-dir outputs/local_sft \
  --with-grpo --grpo-steps 300 \
  --grpo-learning-rate 5e-6 --grpo-beta 0.2
```

或单独运行：

```bash
python cli.py grpo \
  --grpo-dataset data/OpenMathReasoning/data \
  --base-model-path models/Qwen3-4B-Thinking-2507 \
  --output-dir outputs/local_sft \
  --grpo-steps 400 \
  --resume-from-checkpoint outputs/local_sft/checkpoints/last
```

执行逻辑（`run_grpo_training`）：

1. 载入上一阶段 LoRA 权重，并按需加载参考模型（`reference_free=True` 时跳过）。
2. 构建 RL 数据集：若未显式指定，默认使用 SFT 训练集；否则按 `DatasetSource` 重新加载并转化为 `prompt/reference/metadata` 结构。
3. 定制奖励：`reward_fn` 调用 `batch_reward` 根据预测与参考答案的数值/文本接近度给分，并对高难度长推理给予奖励增益。
4. 将项目配置转译为 `HFGRPOConfig`，兼容不同 TRL 版本的参数签名。
5. `GRPOTrainer` 运行强化训练，定期保存检查点，最终写回 LoRA 目录并再次尝试模型合并。

关键参数：`--grpo-steps`（或 `HFGRPOConfig.total_episodes`）、`--grpo-mini-batch`、`--grpo-learning-rate`、`--grpo-beta`、`--grpo-kl`、`--grpo-reference-free`。

### 常用 CLI 参数对照

| 子命令 | 参数 | 作用 | 默认值 |
| --- | --- | --- | --- |
| train/grpo | `--dataset-config` | 指向 JSON 数据集混合配置 | 为空则使用内置默认 mix |
| train | `--max-seq-length` | 模型上下文长度 | 4096 |
| train | `--lora-rank / --lora-alpha / --lora-dropout` | LoRA 超参 | 64 / 64 / 0.05 |
| train | `--save-merged-model` | 是否保存合并模型（fp16/bf16） | 开启 |
| grpo | `--grpo-dataset` | 单独指定强化数据源 | 默认为 SFT 训练集 |
| evaluate | `--sample-size` | 评估时采样数量 | `EvaluationConfig.sample_size` |
| predict | `--adapter-path` | 指定独立 LoRA 目录 | 不传则使用合并/原始模型 |

## 评估与度量

```bash
python cli.py evaluate \
  --dataset-config configs/dataset.sample.json \
  --model-path outputs/sample_run_merged \
  --sample-size 100 \
  --save-path outputs/eval.parquet
```

评估流程：

1. 使用 `build_sft_datasets` 复现训练阶段的清洗与切分，并采样固定数量问题。
2. `format_inference_prompt` 生成系统提示 + 用户问题格式，调用 `generate_answers` 批量推理。
3. 通过 `batch_reward` 计算奖励分数，并输出 `question/reference/generation/reward` 字段的数据框，可直接 `describe()` 获取统计值。
4. 如指定 `--save-path`，将结果保存为 Parquet 供进一步分析。

## 推理与模型发布

```bash
python cli.py predict \
  --model-path outputs/sample_run_merged \
  --question "请证明勾股定理并给出示例。" \
  --max-new-tokens 512
```

使用单独 LoRA 适配器：

```bash
python cli.py predict \
  --base-model-id unsloth/Qwen3-4B-Instruct \
  --adapter-path outputs/sample_run_lora \
  --question "计算 \int_0^1 x^2 dx 并解释步骤"
```

`prepare_for_inference` 会切换到推理模式，自动选择 GPU/CPU；`generate_answers` 支持批量输入和可配置的 `max_new_tokens`。

## 输出目录约定

| 路径 | 说明 |
| --- | --- |
| `outputs/<experiment>_lora` | LoRA 权重（`adapter_config.json`, `adapter_model.safetensors` 等）与分词器 |
| `outputs/<experiment>_merged` | 合并后的全量权重（若 `save_merged_model=True`） |
| `outputs/checkpoints` | 逐步保存的训练检查点、最新状态 `last` |
| `outputs/logs`（需自行创建） | 推荐将 TensorBoard/自定义日志写入此处 |

## 配置参考

### TrainingConfig 关键字段

- `base_model_id` / `base_model_path`：HF 模型 ID 或本地路径；若同时提供，以 `base_model_path` 为准。
- `load_in_4bit` / `load_in_8bit` / `full_finetuning`：控制量化与微调模式。
- `micro_batch_size`、`gradient_accumulation_steps`、`learning_rate`、`warmup_steps` 等：SFT 训练核心超参。
- `dataset_mix`：`DatasetSource` 序列，包含权重与是否推理型标记。
- `save_merged_model`、`merge_dtype`：控制是否合并 LoRA 以及输出精度（`fp16`/`bf16`）。

### GRPOConfig 与 EvaluationConfig

- `GRPOConfig.steps`：RL 总步数，同步决定数据集截断长度。
- `reference_free`：关闭参考模型，可在资源受限时减少显存。
- `num_generations_per_prompt`：TR L 版本支持时可配置每次生成数量。
- `EvaluationConfig.system_prompt`：可自定义评估时的系统提示，保持与生产一致。

## 环境变量一览

- `MATH_LLM_MODELS`：模型缓存根目录，默认 `./models`。
- `MATH_LLM_PRIMARY_ENDPOINT`：首选镜像端点，默认 `https://aliendao.cn`。
- `MATH_LLM_SECONDARY_ENDPOINT`：第二镜像端点，默认 `https://hf-mirror.com`。
- `HF_TOKEN`：访问私有或受限模型所需。
- `HF_HUB_ENABLE_HF_TRANSFER`：设置为 `1` 时启用加速下载。
- `CUDA_VISIBLE_DEVICES`：控制可见 GPU，与 `--micro-batch-size` 协调设置显存占用。

## 调试与最佳实践

- **显存溢出**：降低 `--micro-batch-size` 或 `--max-seq-length`，必要时关闭 `--save-merged-model` 减少显存峰值。
- **数据质量**：检查 JSONL 是否包含空行或无效字段；`_normalize_record` 在字段缺失时会抛出报错。
- **随机性控制**：`TrainingConfig.random_seed` 统一设置 Python/NumPy/PyTorch 随机种子，保证复现性。
- **多机下载**：提前运行 `python -c "from cz_math_llm.assets import ensure_model; ensure_model('unsloth/Qwen3-4B-Instruct')"` 预热缓存。
- **奖励调优**：根据任务特点定制 `reward.py`，例如加入维度打分、格式校验或引用外部判题器。

## 示例数据

- `data/OpenMathReasoning/data`：完整推理数据（Parquet），与命令行示例一致。
- `data/DAPO-Math-17k-Processed/all`：指令数据（Parquet），用于补充监督信号。
- `data/sample_reasoning.jsonl`：含 `<think>...</think>` 推理链的数学题示例。
- `data/sample_instruction.jsonl`：简单指令 Q&A，适合测试混合数据管线。
- `configs/dataset.sample.json`：数据混合样板，可复制后替换 `name/subset/split/weight`。

## 下一步

- 替换示例数据为真实数学题库（如 AMC/AIME、竞赛题等），并扩充难度标签。
- 针对不同 GPU 资源调节 LoRA/GRPO 超参，记录实验配置（建议配合 `utils.dump_dataclass` 输出配置快照）。
- 结合自定义评估指标（BLEU、符号对齐等）或引入外部判题器，完善 `reward.py`。
- 将 CLI 流程封装为 CI/CD 任务或 Notebook，便于协同调试。

祝训练顺利，Enjoy math teaching with Qwen3! 🎓



```text

(base) root@autodl-container-7702429a5b-ca7e9638:~/autodl-tmp/.autodl/czMathLLM# tree
.
├── README.md
├── cli.py
├── configs
│   └── dataset.sample.json
├── cz_math_llm
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-312.pyc
│   │   ├── assets.cpython-312.pyc
│   │   ├── cli_core.cpython-312.pyc
│   │   ├── config.cpython-312.pyc
│   │   ├── data.cpython-312.pyc
│   │   ├── evaluation.cpython-312.pyc
│   │   ├── modeling.cpython-312.pyc
│   │   ├── prompts.cpython-312.pyc
│   │   ├── reward.cpython-312.pyc
│   │   └── utils.cpython-312.pyc
│   ├── assets.py
│   ├── cli_core.py
│   ├── config.py
│   ├── data.py
│   ├── evaluation.py
│   ├── modeling.py
│   ├── prompts.py
│   ├── quick_check.log
│   ├── reward.py
│   ├── trainers
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-312.pyc
│   │   │   ├── grpo.cpython-312.pyc
│   │   │   └── sft.cpython-312.pyc
│   │   ├── grpo.py
│   │   └── sft.py
│   └── utils.py
├── data
│   ├── DAPO-Math-17k-Processed
│   │   ├── README.md
│   │   ├── all
│   │   │   └── train-00000-of-00001.parquet
│   │   ├── cn
│   │   │   └── train-00000-of-00001.parquet
│   │   ├── create_dataset.py
│   │   └── en
│   │       └── train-00000-of-00001.parquet
│   ├── OpenMathReasoning
│   │   ├── README.md
│   │   └── data
│   │       └── cot-00000-of-00001.parquet
│   ├── OpenMathReasoningFull
│   │   ├── README.md
│   │   ├── data
│   │   │   ├── additional_problems-00000-of-00001.parquet
│   │   │   ├── cot-00000-of-00144.parquet
│   │   │   ├── cot-00001-of-00144.parquet
│   │   │   ├── cot-00002-of-00144.parquet
│   │   │   ├── cot-00003-of-00144.parquet
│   │   │   ├── cot-00004-of-00144.parquet
│   │   │   ├── cot-00005-of-00144.parquet
│   │   │   ├── cot-00006-of-00144.parquet
│   │   │   ├── cot-00007-of-00144.parquet
│   │   │   ├── cot-00008-of-00144.parquet
│   │   │   ├── cot-00010-of-00144.parquet
│   │   │   ├── cot-00011-of-00144.parquet
│   │   │   ├── cot-00012-of-00144.parquet
│   │   │   ├── cot-00013-of-00144.parquet
│   │   │   ├── cot-00014-of-00144.parquet
│   │   │   ├── cot-00015-of-00144.parquet
│   │   │   ├── cot-00016-of-00144.parquet
│   │   │   ├── cot-00017-of-00144.parquet
│   │   │   ├── cot-00018-of-00144.parquet
│   │   │   ├── cot-00019-of-00144.parquet
│   │   │   ├── cot-00020-of-00144.parquet
│   │   │   ├── cot-00021-of-00144.parquet
│   │   │   ├── cot-00022-of-00144.parquet
│   │   │   ├── cot-00023-of-00144.parquet
│   │   │   ├── cot-00024-of-00144.parquet
│   │   │   ├── cot-00025-of-00144.parquet
│   │   │   ├── cot-00026-of-00144.parquet
│   │   │   ├── cot-00027-of-00144.parquet
│   │   │   ├── cot-00028-of-00144.parquet
│   │   │   ├── cot-00029-of-00144.parquet
│   │   │   ├── cot-00030-of-00144.parquet
│   │   │   ├── cot-00031-of-00144.parquet
│   │   │   ├── cot-00032-of-00144.parquet
│   │   │   ├── cot-00033-of-00144.parquet
│   │   │   ├── cot-00034-of-00144.parquet
│   │   │   ├── cot-00035-of-00144.parquet
│   │   │   ├── cot-00036-of-00144.parquet
│   │   │   ├── cot-00037-of-00144.parquet
│   │   │   ├── cot-00038-of-00144.parquet
│   │   │   ├── cot-00039-of-00144.parquet
│   │   │   ├── cot-00040-of-00144.parquet
│   │   │   ├── cot-00041-of-00144.parquet
│   │   │   ├── cot-00042-of-00144.parquet
│   │   │   ├── cot-00043-of-00144.parquet
│   │   │   ├── cot-00044-of-00144.parquet
│   │   │   ├── cot-00045-of-00144.parquet
│   │   │   ├── cot-00046-of-00144.parquet
│   │   │   ├── cot-00047-of-00144.parquet
│   │   │   ├── cot-00048-of-00144.parquet
│   │   │   ├── cot-00049-of-00144.parquet
│   │   │   ├── cot-00050-of-00144.parquet
│   │   │   ├── cot-00051-of-00144.parquet
│   │   │   ├── cot-00052-of-00144.parquet
│   │   │   ├── cot-00053-of-00144.parquet
│   │   │   ├── cot-00054-of-00144.parquet
│   │   │   ├── cot-00055-of-00144.parquet
│   │   │   ├── cot-00056-of-00144.parquet
│   │   │   ├── cot-00057-of-00144.parquet
│   │   │   ├── cot-00058-of-00144.parquet
│   │   │   ├── cot-00059-of-00144.parquet
│   │   │   ├── cot-00060-of-00144.parquet
│   │   │   ├── cot-00061-of-00144.parquet
│   │   │   ├── cot-00062-of-00144.parquet
│   │   │   ├── cot-00063-of-00144.parquet
│   │   │   ├── cot-00064-of-00144.parquet
│   │   │   ├── cot-00065-of-00144.parquet
│   │   │   ├── cot-00066-of-00144.parquet
│   │   │   ├── cot-00067-of-00144.parquet
│   │   │   ├── cot-00068-of-00144.parquet
│   │   │   ├── cot-00069-of-00144.parquet
│   │   │   ├── cot-00070-of-00144.parquet
│   │   │   ├── cot-00071-of-00144.parquet
│   │   │   ├── cot-00072-of-00144.parquet
│   │   │   ├── cot-00073-of-00144.parquet
│   │   │   ├── cot-00074-of-00144.parquet
│   │   │   ├── cot-00075-of-00144.parquet
│   │   │   ├── cot-00076-of-00144.parquet
│   │   │   ├── cot-00077-of-00144.parquet
│   │   │   ├── cot-00078-of-00144.parquet
│   │   │   ├── cot-00079-of-00144.parquet
│   │   │   ├── cot-00080-of-00144.parquet
│   │   │   ├── cot-00081-of-00144.parquet
│   │   │   ├── cot-00082-of-00144.parquet
│   │   │   ├── cot-00083-of-00144.parquet
│   │   │   ├── cot-00084-of-00144.parquet
│   │   │   ├── cot-00085-of-00144.parquet
│   │   │   ├── cot-00086-of-00144.parquet
│   │   │   ├── cot-00087-of-00144.parquet
│   │   │   ├── cot-00088-of-00144.parquet
│   │   │   ├── cot-00089-of-00144.parquet
│   │   │   ├── cot-00090-of-00144.parquet
│   │   │   ├── cot-00091-of-00144.parquet
│   │   │   ├── cot-00092-of-00144.parquet
│   │   │   ├── cot-00093-of-00144.parquet
│   │   │   ├── cot-00094-of-00144.parquet
│   │   │   ├── cot-00095-of-00144.parquet
│   │   │   ├── cot-00096-of-00144.parquet
│   │   │   ├── cot-00097-of-00144.parquet
│   │   │   ├── cot-00098-of-00144.parquet
│   │   │   ├── cot-00099-of-00144.parquet
│   │   │   ├── cot-00100-of-00144.parquet
│   │   │   ├── cot-00101-of-00144.parquet
│   │   │   ├── cot-00102-of-00144.parquet
│   │   │   ├── cot-00103-of-00144.parquet
│   │   │   ├── cot-00104-of-00144.parquet
│   │   │   ├── cot-00105-of-00144.parquet
│   │   │   ├── cot-00106-of-00144.parquet
│   │   │   ├── cot-00107-of-00144.parquet
│   │   │   ├── cot-00108-of-00144.parquet
│   │   │   ├── cot-00109-of-00144.parquet
│   │   │   ├── cot-00110-of-00144.parquet
│   │   │   ├── cot-00111-of-00144.parquet
│   │   │   ├── cot-00112-of-00144.parquet
│   │   │   ├── cot-00113-of-00144.parquet
│   │   │   ├── cot-00114-of-00144.parquet
│   │   │   ├── cot-00115-of-00144.parquet
│   │   │   ├── cot-00116-of-00144.parquet
│   │   │   ├── cot-00117-of-00144.parquet
│   │   │   ├── cot-00118-of-00144.parquet
│   │   │   ├── cot-00119-of-00144.parquet
│   │   │   ├── cot-00120-of-00144.parquet
│   │   │   ├── cot-00121-of-00144.parquet
│   │   │   ├── cot-00122-of-00144.parquet
│   │   │   ├── cot-00123-of-00144.parquet
│   │   │   ├── cot-00124-of-00144.parquet
│   │   │   ├── cot-00125-of-00144.parquet
│   │   │   ├── cot-00126-of-00144.parquet
│   │   │   ├── cot-00127-of-00144.parquet
│   │   │   ├── cot-00128-of-00144.parquet
│   │   │   ├── cot-00129-of-00144.parquet
│   │   │   ├── cot-00130-of-00144.parquet
│   │   │   ├── cot-00131-of-00144.parquet
│   │   │   ├── cot-00132-of-00144.parquet
│   │   │   ├── cot-00133-of-00144.parquet
│   │   │   ├── cot-00134-of-00144.parquet
│   │   │   ├── cot-00135-of-00144.parquet
│   │   │   ├── cot-00136-of-00144.parquet
│   │   │   ├── cot-00137-of-00144.parquet
│   │   │   ├── cot-00138-of-00144.parquet
│   │   │   ├── cot-00139-of-00144.parquet
│   │   │   ├── cot-00140-of-00144.parquet
│   │   │   ├── cot-00141-of-00144.parquet
│   │   │   ├── cot-00142-of-00144.parquet
│   │   │   ├── cot-00143-of-00144.parquet
│   │   │   ├── genselect-00000-of-00014.parquet
│   │   │   ├── genselect-00001-of-00014.parquet
│   │   │   ├── genselect-00002-of-00014.parquet
│   │   │   ├── genselect-00003-of-00014.parquet
│   │   │   ├── genselect-00004-of-00014.parquet
│   │   │   ├── genselect-00005-of-00014.parquet
│   │   │   ├── genselect-00006-of-00014.parquet
│   │   │   ├── genselect-00007-of-00014.parquet
│   │   │   ├── genselect-00008-of-00014.parquet
│   │   │   ├── genselect-00009-of-00014.parquet
│   │   │   ├── genselect-00010-of-00014.parquet
│   │   │   ├── genselect-00011-of-00014.parquet
│   │   │   ├── genselect-00012-of-00014.parquet
│   │   │   ├── genselect-00013-of-00014.parquet
│   │   │   ├── tir-00000-of-00072.parquet
│   │   │   ├── tir-00001-of-00072.parquet
│   │   │   ├── tir-00002-of-00072.parquet
│   │   │   ├── tir-00003-of-00072.parquet
│   │   │   ├── tir-00004-of-00072.parquet
│   │   │   ├── tir-00005-of-00072.parquet
│   │   │   ├── tir-00006-of-00072.parquet
│   │   │   ├── tir-00007-of-00072.parquet
│   │   │   ├── tir-00008-of-00072.parquet
│   │   │   ├── tir-00009-of-00072.parquet
│   │   │   ├── tir-00010-of-00072.parquet
│   │   │   ├── tir-00011-of-00072.parquet
│   │   │   ├── tir-00012-of-00072.parquet
│   │   │   ├── tir-00013-of-00072.parquet
│   │   │   ├── tir-00014-of-00072.parquet
│   │   │   ├── tir-00015-of-00072.parquet
│   │   │   ├── tir-00016-of-00072.parquet
│   │   │   ├── tir-00017-of-00072.parquet
│   │   │   ├── tir-00018-of-00072.parquet
│   │   │   ├── tir-00019-of-00072.parquet
│   │   │   ├── tir-00020-of-00072.parquet
│   │   │   ├── tir-00021-of-00072.parquet
│   │   │   ├── tir-00022-of-00072.parquet
│   │   │   ├── tir-00023-of-00072.parquet
│   │   │   ├── tir-00024-of-00072.parquet
│   │   │   ├── tir-00025-of-00072.parquet
│   │   │   ├── tir-00026-of-00072.parquet
│   │   │   ├── tir-00027-of-00072.parquet
│   │   │   ├── tir-00028-of-00072.parquet
│   │   │   ├── tir-00029-of-00072.parquet
│   │   │   ├── tir-00030-of-00072.parquet
│   │   │   ├── tir-00031-of-00072.parquet
│   │   │   ├── tir-00032-of-00072.parquet
│   │   │   ├── tir-00033-of-00072.parquet
│   │   │   ├── tir-00034-of-00072.parquet
│   │   │   ├── tir-00035-of-00072.parquet
│   │   │   ├── tir-00036-of-00072.parquet
│   │   │   ├── tir-00037-of-00072.parquet
│   │   │   ├── tir-00038-of-00072.parquet
│   │   │   ├── tir-00039-of-00072.parquet
│   │   │   ├── tir-00040-of-00072.parquet
│   │   │   ├── tir-00041-of-00072.parquet
│   │   │   ├── tir-00042-of-00072.parquet
│   │   │   ├── tir-00043-of-00072.parquet
│   │   │   ├── tir-00044-of-00072.parquet
│   │   │   ├── tir-00045-of-00072.parquet
│   │   │   ├── tir-00046-of-00072.parquet
│   │   │   ├── tir-00047-of-00072.parquet
│   │   │   ├── tir-00048-of-00072.parquet
│   │   │   ├── tir-00049-of-00072.parquet
│   │   │   ├── tir-00050-of-00072.parquet
│   │   │   ├── tir-00051-of-00072.parquet
│   │   │   ├── tir-00052-of-00072.parquet
│   │   │   ├── tir-00053-of-00072.parquet
│   │   │   ├── tir-00054-of-00072.parquet
│   │   │   ├── tir-00055-of-00072.parquet
│   │   │   ├── tir-00056-of-00072.parquet
│   │   │   ├── tir-00057-of-00072.parquet
│   │   │   ├── tir-00058-of-00072.parquet
│   │   │   ├── tir-00059-of-00072.parquet
│   │   │   ├── tir-00060-of-00072.parquet
│   │   │   ├── tir-00061-of-00072.parquet
│   │   │   ├── tir-00062-of-00072.parquet
│   │   │   ├── tir-00063-of-00072.parquet
│   │   │   ├── tir-00064-of-00072.parquet
│   │   │   ├── tir-00065-of-00072.parquet
│   │   │   ├── tir-00066-of-00072.parquet
│   │   │   ├── tir-00067-of-00072.parquet
│   │   │   ├── tir-00068-of-00072.parquet
│   │   │   ├── tir-00069-of-00072.parquet
│   │   │   ├── tir-00070-of-00072.parquet
│   │   │   └── tir-00071-of-00072.parquet
│   │   ├── download_dataset.py
│   │   └── results.png
│   ├── sample_instruction.jsonl
│   └── sample_reasoning.jsonl
├── law-finetune-code.txt
├── models
│   └── Qwen3-4B-Thinking-2507
│       ├── LICENSE
│       ├── README.md
│       ├── config.json
│       ├── generation_config.json
│       ├── merges.txt
│       ├── model-00001-of-00003.safetensors
│       ├── model-00002-of-00003.safetensors
│       ├── model-00003-of-00003.safetensors
│       ├── model.safetensors.index.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── vocab.json
├── outputs
│   ├── local_sft
│   │   ├── checkpoints
│   │   │   ├── README.md
│   │   │   ├── checkpoint-200
│   │   │   │   ├── README.md
│   │   │   │   ├── adapter_config.json
│   │   │   │   ├── adapter_model.safetensors
│   │   │   │   ├── added_tokens.json
│   │   │   │   ├── chat_template.jinja
│   │   │   │   ├── merges.txt
│   │   │   │   ├── optimizer.pt
│   │   │   │   ├── rng_state.pth
│   │   │   │   ├── scheduler.pt
│   │   │   │   ├── special_tokens_map.json
│   │   │   │   ├── tokenizer.json
│   │   │   │   ├── tokenizer_config.json
│   │   │   │   ├── trainer_state.json
│   │   │   │   ├── training_args.bin
│   │   │   │   └── vocab.json
│   │   │   ├── last
│   │   │   │   ├── README.md
│   │   │   │   ├── adapter_config.json
│   │   │   │   ├── adapter_model.safetensors
│   │   │   │   ├── added_tokens.json
│   │   │   │   ├── chat_template.jinja
│   │   │   │   ├── merges.txt
│   │   │   │   ├── special_tokens_map.json
│   │   │   │   ├── tokenizer.json
│   │   │   │   ├── tokenizer_config.json
│   │   │   │   ├── training_args.bin
│   │   │   │   └── vocab.json
│   │   │   └── trainer_state.json
│   │   ├── qwen_math_tutor_lora
│   │   │   ├── README.md
│   │   │   ├── adapter_config.json
│   │   │   ├── adapter_model.safetensors
│   │   │   ├── added_tokens.json
│   │   │   ├── chat_template.jinja
│   │   │   ├── merges.txt
│   │   │   ├── special_tokens_map.json
│   │   │   ├── tokenizer.json
│   │   │   ├── tokenizer_config.json
│   │   │   ├── training_args.bin
│   │   │   └── vocab.json
│   │   └── qwen_math_tutor_merged
│   │       ├── added_tokens.json
│   │       ├── chat_template.jinja
│   │       ├── merges.txt
│   │       ├── model-00001-of-00003.safetensors
│   │       ├── model-00002-of-00003.safetensors
│   │       ├── model-00003-of-00003.safetensors
│   │       ├── model.safetensors.index.json
│   │       ├── special_tokens_map.json
│   │       ├── tokenizer.json
│   │       ├── tokenizer_config.json
│   │       └── vocab.json
│   ├── local_sft_big
│   │   ├── checkpoints
│   │   │   └── runs
│   │   │       ├── Nov12_11-58-55_autodl-container-7702429a5b-ca7e9638
│   │   │       │   └── events.out.tfevents.1762919936.autodl-container-7702429a5b-ca7e9638.26652.0
│   │   │       └── Nov12_12-13-56_autodl-container-7702429a5b-ca7e9638
│   │   │           └── events.out.tfevents.1762920839.autodl-container-7702429a5b-ca7e9638.4000.0
│   │   ├── qwen_math_tutor_lora
│   │   └── qwen_math_tutor_merged
│   ├── qwen3_4b_test
│   │   ├── checkpoints
│   │   ├── qwen_math_tutor_lora
│   │   └── qwen_math_tutor_merged
│   ├── test_run
│   │   ├── checkpoints
│   │   ├── qwen_math_tutor_lora
│   │   └── qwen_math_tutor_merged
│   └── token_test
│       ├── checkpoints
│       ├── qwen_math_tutor_lora
│       └── qwen_math_tutor_merged
├── quick_check.log
├── qwenFineTuning.txt
├── requirements.txt
├── tmp_qwen_test
│   └── config.json
├── trainer_output
│   ├── README.md
│   ├── checkpoint-16
│   │   ├── README.md
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   ├── added_tokens.json
│   │   ├── chat_template.jinja
│   │   ├── merges.txt
│   │   ├── optimizer.pt
│   │   ├── rng_state.pth
│   │   ├── scheduler.pt
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── trainer_state.json
│   │   ├── training_args.bin
│   │   └── vocab.json
│   ├── checkpoint-4
│   │   ├── README.md
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   ├── added_tokens.json
│   │   ├── chat_template.jinja
│   │   ├── merges.txt
│   │   ├── optimizer.pt
│   │   ├── rng_state.pth
│   │   ├── scheduler.pt
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── trainer_state.json
│   │   ├── training_args.bin
│   │   └── vocab.json
│   ├── checkpoint-8
│   │   ├── README.md
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   ├── added_tokens.json
│   │   ├── chat_template.jinja
│   │   ├── merges.txt
│   │   ├── optimizer.pt
│   │   ├── rng_state.pth
│   │   ├── scheduler.pt
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── trainer_state.json
│   │   ├── training_args.bin
│   │   └── vocab.json
│   └── runs
│       ├── Nov11_21-40-47_autodl-container-7702429a5b-ca7e9638
│       │   └── events.out.tfevents.1762868448.autodl-container-7702429a5b-ca7e9638.32383.0
│       ├── Nov11_21-42-43_autodl-container-7702429a5b-ca7e9638
│       │   └── events.out.tfevents.1762868564.autodl-container-7702429a5b-ca7e9638.32864.0
│       ├── Nov11_21-45-16_autodl-container-7702429a5b-ca7e9638
│       │   └── events.out.tfevents.1762868717.autodl-container-7702429a5b-ca7e9638.33265.0
│       ├── Nov11_21-48-03_autodl-container-7702429a5b-ca7e9638
│       │   └── events.out.tfevents.1762868884.autodl-container-7702429a5b-ca7e9638.34023.0
│       ├── Nov12_10-18-31_autodl-container-7702429a5b-ca7e9638
│       │   └── events.out.tfevents.1762913912.autodl-container-7702429a5b-ca7e9638.7912.0
│       ├── Nov12_10-23-28_autodl-container-7702429a5b-ca7e9638
│       │   └── events.out.tfevents.1762914209.autodl-container-7702429a5b-ca7e9638.10371.0
│       ├── Nov12_10-28-24_autodl-container-7702429a5b-ca7e9638
│       │   └── events.out.tfevents.1762914505.autodl-container-7702429a5b-ca7e9638.14042.0
│       ├── Nov12_10-33-23_autodl-container-7702429a5b-ca7e9638
│       │   └── events.out.tfevents.1762914804.autodl-container-7702429a5b-ca7e9638.14796.0
│       ├── Nov12_10-42-00_autodl-container-7702429a5b-ca7e9638
│       │   └── events.out.tfevents.1762915322.autodl-container-7702429a5b-ca7e9638.15719.0
│       └── Nov12_10-56-04_autodl-container-7702429a5b-ca7e9638
│           └── events.out.tfevents.1762916166.autodl-container-7702429a5b-ca7e9638.17027.0
├── unsloth_compiled_cache
│   ├── UnslothAlignPropTrainer.py
│   ├── UnslothBCOTrainer.py
│   ├── UnslothCPOTrainer.py
│   ├── UnslothDDPOTrainer.py
│   ├── UnslothDPOTrainer.py
│   ├── UnslothGKDTrainer.py
│   ├── UnslothGRPOTrainer.py
│   ├── UnslothIterativeSFTTrainer.py
│   ├── UnslothKTOTrainer.py
│   ├── UnslothNashMDTrainer.py
│   ├── UnslothORPOTrainer.py
│   ├── UnslothOnlineDPOTrainer.py
│   ├── UnslothPPOTrainer.py
│   ├── UnslothPRMTrainer.py
│   ├── UnslothRLOOTrainer.py
│   ├── UnslothRewardTrainer.py
│   ├── UnslothSFTTrainer.py
│   ├── UnslothXPOTrainer.py
│   └── __pycache__
│       ├── UnslothAlignPropTrainer.cpython-312.pyc
│       ├── UnslothBCOTrainer.cpython-312.pyc
│       ├── UnslothCPOTrainer.cpython-312.pyc
│       ├── UnslothDDPOTrainer.cpython-312.pyc
│       ├── UnslothDPOTrainer.cpython-312.pyc
│       ├── UnslothGKDTrainer.cpython-312.pyc
│       ├── UnslothGRPOTrainer.cpython-312.pyc
│       ├── UnslothIterativeSFTTrainer.cpython-312.pyc
│       ├── UnslothKTOTrainer.cpython-312.pyc
│       ├── UnslothNashMDTrainer.cpython-312.pyc
│       ├── UnslothORPOTrainer.cpython-312.pyc
│       ├── UnslothOnlineDPOTrainer.cpython-312.pyc
│       ├── UnslothPPOTrainer.cpython-312.pyc
│       ├── UnslothPRMTrainer.cpython-312.pyc
│       ├── UnslothRLOOTrainer.cpython-312.pyc
│       ├── UnslothRewardTrainer.cpython-312.pyc
│       ├── UnslothSFTTrainer.cpython-312.pyc
│       └── UnslothXPOTrainer.cpython-312.pyc
└── unsloth_training_checkpoints

61 directories, 433 files

```
