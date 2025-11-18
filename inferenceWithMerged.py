#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""使用 Unsloth 加速加载合并后的完整模型并进行推理示例。

特点:
- 直接加载已合并模型目录 (无需 LoRA adapter)。
- 使用 `FastLanguageModel.for_inference` 启用 unsloth 的推理优化。
- 支持可选的 bf16 / fp16 / fp32 dtype 选择。

运行示例:
    python inferenceWithMerged.py \
        --model-dir /path/to/qwen_math_grpo_merged \
        --prompt "已知 f(x)=x^2, 求 f'(3)." \
        --max-new-tokens 64
        
        python outputs/local_sft_big/inferenceWithMerged.py \
  --model-dir outputs/local_sft_big/qwen_math_grpo_merged \
  --prompt "黑板上画有一个凸 2011 边形。Peter 画它的对角线，要求每条新画的对角线与已画的对角线相交不超过一次。Peter 最多能画多少条对角线？" \
  --max-new-tokens 200000 \
  --stream \
  --dtype auto
"""
from __future__ import annotations
import os
import argparse
import torch
import unsloth  # noqa: F401  (必须在 transformers 前导入)
from unsloth import FastLanguageModel, is_bfloat16_supported


def parse_args():
    p = argparse.ArgumentParser(
        description="Unsloth accelerated inference for merged model"
    )
    p.add_argument("--model-dir", required=True, help="合并后模型目录")
    p.add_argument("--prompt", required=True, help="输入的提示语")
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="生成最大新 token 数",
    )
    p.add_argument(
        "--dtype",
        choices=["auto", "fp16", "bf16", "float32"],
        default="auto",
        help="推理数据类型",
    )
    p.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="使用4bit量化 (可降低显存)",
    )
    p.add_argument("--load-in-8bit", action="store_true", help="使用8bit量化")
    p.add_argument("--device", default="auto", help="设备: auto|cuda|cpu")
    p.add_argument(
        "--tokenizer-dir",
        default=None,
        help="备用 tokenizer 目录 (若合并目录内缺失 tokenizer 文件时指定)",
    )
    p.add_argument("--stream", action="store_true", help="启用逐步流式输出")
    p.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    p.add_argument("--top-p", type=float, default=0.9, help="nucleus sampling 截断概率")
    p.add_argument("--top-k", type=int, default=50, help="top-k 过滤")
    p.add_argument(
        "--do-sample", action="store_true", help="启用随机采样 (默认 greedy)"
    )
    p.add_argument(
        "--repetition-penalty", type=float, default=1.05, help="重复惩罚系数"
    )
    return p.parse_args()


def pick_dtype(choice: str):
    if choice == "auto":
        return "bfloat16" if is_bfloat16_supported() else "float16"
    mapping = {"fp16": "float16", "bf16": "bfloat16", "float32": "float32"}
    return mapping.get(choice, "float16")


def main():
    args = parse_args()
    os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

    dtype = pick_dtype(args.dtype)
    print(f"[Inference] Using dtype: {dtype}")

    print(f"[Inference] Loading merged model from: {args.model_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_dir,
        max_seq_length=8192,  # 根据训练时的最大长度调整
        dtype=dtype,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        full_finetuning=False,
        trust_remote_code=True,
    )

    # 若合并目录缺失 tokenizer 相关文件且提供了备用目录，尝试重新加载分词器
    if args.tokenizer_dir:
        import pathlib

        tk_dir = pathlib.Path(args.model_dir)
        needed = ["tokenizer.json", "tokenizer_config.json", "vocab.json"]
        if not all((tk_dir / f).exists() for f in needed):
            from transformers import AutoTokenizer

            print(
                "[Inference][Warn] 合并目录缺失 tokenizer 文件，改用备用目录加载 tokenizer。"
            )
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_dir, use_fast=True, trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

    # 准备推理模式 (合并可能的残余适配器、禁用梯度等)
    FastLanguageModel.for_inference(model)

    prompt = args.prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    model.to(device)
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.do_sample,
    )

    if args.stream:
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            tokenizer=tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        print("[Inference] Streaming generation...")
        # 异步生成: 使用线程方式让 streamer 迭代输出
        import threading

        def _generate():  # 在线程中调用 generate
            model.generate(**inputs, streamer=streamer, **gen_kwargs)

        t = threading.Thread(target=_generate)
        t.start()
        partial = []
        for piece in streamer:
            partial.append(piece)
            print(piece, end="", flush=True)
        t.join()
        print("\n[Result]", "".join(partial))
    else:
        print("[Inference] Generating (non-stream)...")
        output_ids = model.generate(**inputs, **gen_kwargs)
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("[Result]", text)


if __name__ == "__main__":
    main()
