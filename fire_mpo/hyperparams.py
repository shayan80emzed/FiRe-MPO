"""Shared training hyperparameters (paper Table 6 defaults)."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch

DEBUG = False
GLOBAL_TRAIN_BATCH_SIZE = 8

# Paper / scripts defaults (no cluster-specific absolute paths)
MODEL_PATH = "FreedomIntelligence/HuatuoGPT-Vision-7B"
PROCESSOR_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

# Paper Table 6
PAPER_BETA = 0.1
PAPER_ALPHA = 0.01
PAPER_GAMMA = 0.1
PAPER_LAMBDA = 0.5
PAPER_LR = 1e-6
PAPER_LORA_R = 128
PAPER_LORA_ALPHA = 256
PAPER_LORA_DROPOUT = 0.05

UNSLOTH_LOAD_KWARGS = dict(
    max_seq_length=4096,
    load_in_4bit=False,
    load_in_8bit=False,
    dtype=torch.bfloat16,
    gpu_memory_utilization=0.95,
)

UNSLOTH_PEFT_KWARGS = dict(
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=PAPER_LORA_R,
    lora_alpha=PAPER_LORA_ALPHA,
    lora_dropout=PAPER_LORA_DROPOUT,
    bias="none",
    random_state=3407,
    use_rslora=True,
    loftq_config=None,
)

PROCESSOR_KWARGS = dict(
    max_pixels=512 * 28 * 28,
)


def processor_for_model(base_model_path: str) -> str:
    name = base_model_path.rstrip("/").split("/")[-1]
    if "HuatuoGPT-Vision-7B" in name:
        return "Qwen/Qwen2.5-VL-7B-Instruct"
    return base_model_path


def _resolve_learning_rate(overrides: Dict[str, Any]):
    lr = overrides.pop("learning_rate", None)
    if lr is not None:
        return float(lr), overrides
    env = os.environ.get("LEARNING_RATE")
    if env is None:
        return PAPER_LR, overrides
    return float(env), overrides


def _resolve_per_device_and_grad_accum(overrides: Dict[str, Any]):
    overrides.pop("gradient_accumulation_steps", None)
    pd = overrides.pop("per_device_train_batch_size", None)
    if pd is not None:
        pd = int(pd)
    else:
        env = os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE")
        pd = int(env) if env is not None else 4
    ga = int(GLOBAL_TRAIN_BATCH_SIZE / pd)
    if pd * ga != GLOBAL_TRAIN_BATCH_SIZE:
        raise ValueError(
            f"per_device_train_batch_size={pd} must divide GLOBAL_TRAIN_BATCH_SIZE={GLOBAL_TRAIN_BATCH_SIZE}"
        )
    return pd, ga, overrides


def common_training_kwargs(run_name: Optional[str] = None, **overrides) -> Dict[str, Any]:
    overrides = dict(overrides)
    learning_rate, overrides = _resolve_learning_rate(overrides)
    per_device, grad_accum, overrides = _resolve_per_device_and_grad_accum(overrides)
    base = dict(
        bf16=True,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        num_train_epochs=1,
        logging_steps=10,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        dataloader_num_workers=8,
        dataloader_pin_memory=False,
        dataset_num_proc=0,
        report_to="none" if DEBUG else "wandb",
        run_name=run_name,
        max_prompt_length=None,
        max_completion_length=None,
        max_length=None,
        save_strategy="epoch",
        remove_unused_columns=False,
        beta=PAPER_BETA,
    )
    base.update(overrides)
    return base
