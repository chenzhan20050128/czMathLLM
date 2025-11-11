from __future__ import annotations

from typing import Optional

import pandas as pd

from .config import ProjectConfig
from .data import build_sft_datasets
from .modeling import generate_answers, load_base_model
from .prompts import format_inference_prompt
from .reward import batch_reward


def evaluate_model(
    project: ProjectConfig,
    *,
    model_path: Optional[str] = None,
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    training_cfg = project.training
    evaluation_cfg = project.evaluation

    model, tokenizer = load_base_model(training_cfg, model_path=str(model_path) if model_path else None)

    _, eval_dataset = build_sft_datasets(training_cfg, tokenizer)
    limit = sample_size or evaluation_cfg.sample_size
    dataset = eval_dataset.select(range(min(len(eval_dataset), limit)))

    batch = dataset[:]
    questions = batch["question"]
    prompts = [format_inference_prompt(q, system_prompt=evaluation_cfg.system_prompt) for q in questions]
    generations = generate_answers(model, tokenizer, prompts, max_new_tokens=evaluation_cfg.max_new_tokens)

    references = batch["final_answer"]
    metadatas = batch.get("metadata") or [{} for _ in references]
    rewards = batch_reward(generations, references, metadatas)

    df = pd.DataFrame(
        {
            "question": questions,
            "reference": references,
            "generation": generations,
            "reward": rewards,
        }
    )
    return df
