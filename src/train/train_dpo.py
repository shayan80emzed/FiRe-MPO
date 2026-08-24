"""
DPO training script.

- Unsloth: FastVisionModel + built-in LoRA (lower memory / faster).
- HuggingFace: Qwen2.5-VL + PEFT LoRA — same stack as `train_dpov.py` without DPOv terms
  (`QwenDPOTrainer` + `QwenDataCollatorForPreference`).
"""


import argparse
import os
import sys

import torch
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import DPOConfig

# Add project root to Python path for absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.dataset.dpo_dataset import DPODataset
from src.trainer.qwen_dpo_trainer import QwenDPOTrainer, QwenDataCollatorForPreference
from src.train.hyperparams import (
    MODEL_PATH,
    PROCESSOR_NAME,
    UNSLOTH_LOAD_KWARGS,
    UNSLOTH_PEFT_KWARGS,
    PROCESSOR_KWARGS,
    common_training_kwargs,
)
from utils.train import find_target_linear_names


def _run_training_hf(args):
    """Standard Transformers + PEFT — mirrors `train_dpov.py` but DPO-only (no DPOv loss)."""
    model_path = args.base_model_path or MODEL_PATH
    processor_name = args.processor_name or PROCESSOR_NAME
    print("Loading VLM with HuggingFace + PEFT (no Unsloth)...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    lora_cfg = LoraConfig(
        target_modules=find_target_linear_names(model),
        r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    processor = AutoProcessor.from_pretrained(processor_name, **PROCESSOR_KWARGS)
    processor.tokenizer.padding_side = "left"

    print("Loading Medical VQA DPO dataset...")
    dataset = DPODataset(data_path=args.data_path)

    train_kw = dict(
        run_name=os.path.basename(args.output_dir.rstrip("/")) or "huatuo-dpo-hf",
    )
    if args.learning_rate is not None:
        train_kw["learning_rate"] = args.learning_rate
    if args.per_device_train_batch_size is not None:
        train_kw["per_device_train_batch_size"] = args.per_device_train_batch_size

    training_args = DPOConfig(
        output_dir=args.output_dir,
        **common_training_kwargs(**train_kw),
    )

    trainer = QwenDPOTrainer(
        model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=lora_cfg,
        data_collator=QwenDataCollatorForPreference(
            pad_token_id=processor.tokenizer.pad_token_id
        ),
    )
    trainer.train()


def _run_training_unsloth(args):
    os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
    from unsloth import FastVisionModel

    model_path = args.base_model_path or MODEL_PATH
    processor_name = args.processor_name or PROCESSOR_NAME
    print("Loading model with Unsloth FastVisionModel...")
    model, _ = FastVisionModel.from_pretrained(
        model_name=model_path,
        **UNSLOTH_LOAD_KWARGS,
    )

    # peft_kwargs = dict(UNSLOTH_PEFT_KWARGS)
    # peft_kwargs["target_modules"] = find_target_linear_names(model)
    model = FastVisionModel.get_peft_model(
        model,
        # **peft_kwargs,
        **UNSLOTH_PEFT_KWARGS,
    )

    processor = AutoProcessor.from_pretrained(processor_name, **PROCESSOR_KWARGS)
    processor.tokenizer.padding_side = "left"

    print("Loading Medical VQA DPO dataset...")
    dataset = DPODataset(data_path=args.data_path)
    dataset = _make_disk_backed(dataset, args.data_path)

    train_kw = dict(
        run_name=os.path.basename(args.output_dir.rstrip("/")) or "huatuo-dpo-unsloth",
    )
    # Seed both parameter init and the data sampler, matching fire_mpo/train.py so the
    # DPO baseline is seed-controlled the same way as the fine-grained methods.
    if args.seed is not None:
        train_kw["seed"] = args.seed
        train_kw["data_seed"] = args.seed
    if args.max_steps is not None:
        train_kw["max_steps"] = args.max_steps
    if args.learning_rate is not None:
        train_kw["learning_rate"] = args.learning_rate
    if args.per_device_train_batch_size is not None:
        train_kw["per_device_train_batch_size"] = args.per_device_train_batch_size

    training_args = DPOConfig(
        output_dir=args.output_dir,
        **common_training_kwargs(**train_kw),
    )

    from fire_mpo.train import _make_disk_backed  # shared tokenization cache
    from src.trainer.qwen_dpo_trainer import _make_unsloth_safe_dpo_collator

    trainer = QwenDPOTrainer(
        model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=None,
        data_collator=_make_unsloth_safe_dpo_collator(
            model, processor, processor.tokenizer.pad_token_id
        ),
    )
    trainer.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed; sets both seed and data_seed.")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the DPO preference dataset JSON",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Optimizer learning rate (overrides LEARNING_RATE env if set; otherwise env is required).",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Base model path/model id. Defaults to MODEL_PATH from hyperparams.py.",
    )
    parser.add_argument(
        "--processor_name",
        type=str,
        default=None,
        help="Processor name/path. Defaults to PROCESSOR_NAME from hyperparams.py.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=None,
        help="Train batch per device; gradient_accumulation_steps is set to int(8 / this) (overrides env).",
    )
    parser.add_argument(
        "--no_unsloth",
        action="store_true",
        help="Use HuggingFace Qwen2.5-VL + PEFT (same pattern as train_dpov.py, DPO-only). Default is Unsloth.",
    )
    args = parser.parse_args()

    if args.no_unsloth:
        _run_training_hf(args)
    else:
        _run_training_unsloth(args)


if __name__ == "__main__":
    main()
