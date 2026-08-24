"""
FiRe-MPO training entrypoint (paper Eq. 3–5, Table 6).

Supports paper-named flags (--alpha, --gamma, --lambda_) and legacy
(--rrpo_alpha, --rrpo_alpha_v3, --tkl_share) for transition.
"""

from __future__ import annotations

import os

os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")

from unsloth import FastVisionModel  # noqa: E402

import argparse
import sys
from pathlib import Path

import yaml
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fire_mpo.data import PreferenceDataset  # noqa: E402
from fire_mpo.hyperparams import (  # noqa: E402
    MODEL_PATH,
    PAPER_ALPHA,
    PAPER_GAMMA,
    PAPER_LAMBDA,
    PROCESSOR_KWARGS,
    PROCESSOR_NAME,
    UNSLOTH_LOAD_KWARGS,
    UNSLOTH_PEFT_KWARGS,
    common_training_kwargs,
    processor_for_model,
)
from fire_mpo.trainer import (  # noqa: E402
    FiReMPOConfig,
    FiReMPOTrainer,
    _make_unsloth_safe_fire_mpo_collator,
)


def add_special_tokens(processor):
    special_tokens_dict = {"additional_special_tokens": ["<mask>", "</mask>"]}
    num_added = processor.tokenizer.add_special_tokens(special_tokens_dict)
    if num_added > 0:
        print(f"Added {num_added} special tokens (<mask>, </mask>)")
    return processor


def _load_yaml_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_hparams(args) -> dict:
    """Merge YAML defaults < CLI (including legacy aliases)."""
    cfg = _load_yaml_config(args.config)
    loss = cfg.get("loss", cfg)

    alpha = args.alpha
    if alpha is None and args.rrpo_alpha is not None:
        alpha = args.rrpo_alpha
    if alpha is None:
        alpha = loss.get("alpha", PAPER_ALPHA)

    gamma = args.gamma
    if gamma is None and args.rrpo_alpha_v3 is not None:
        gamma = args.rrpo_alpha_v3
    if gamma is None:
        gamma = loss.get("gamma", PAPER_GAMMA)

    lambda_ = args.lambda_
    if lambda_ is None and args.tkl_share is not None:
        lambda_ = args.tkl_share
    if lambda_ is None:
        lambda_ = loss.get("lambda", loss.get("lambda_", PAPER_LAMBDA))

    alpha_v1 = args.alpha_v1 if args.alpha_v1 is not None else args.rrpo_alpha_v1
    if alpha_v1 is None:
        alpha_v1 = loss.get("alpha_v1")
    alpha_v2 = args.alpha_v2 if args.alpha_v2 is not None else args.rrpo_alpha_v2
    if alpha_v2 is None:
        alpha_v2 = loss.get("alpha_v2")

    return {
        "alpha": float(alpha) if alpha is not None else None,
        "gamma": float(gamma) if gamma is not None else None,
        "lambda_": float(lambda_) if lambda_ is not None else PAPER_LAMBDA,
        "alpha_v1": float(alpha_v1) if alpha_v1 is not None else None,
        "alpha_v2": float(alpha_v2) if alpha_v2 is not None else None,
        "train": cfg.get("train", {}),
        "model": cfg.get("model", {}),
    }


def _make_disk_backed(dataset, data_path: str):
    """
    Round-trip the in-memory preference dataset through an on-disk Arrow store.

    ``PreferenceDataset`` builds itself with ``Dataset.from_list``, which produces an
    in-memory dataset with no ``cache_files``. HuggingFace ``datasets`` only caches
    ``.map()`` output to disk when the source dataset is disk-backed, so with an
    in-memory source every run re-runs TRL's tokenization pass from scratch -- roughly
    ten minutes per run, dominated by decoding the 1024x1024 images.

    Making the source disk-backed lets those ``.map()`` results be cached and reused by
    later runs over the same preference file, which matters when sweeping seeds and
    hyperparameters over an otherwise identical corpus. Tokenization does not depend on
    seed, alpha, gamma or lambda, so the cache is shared across all of them.

    The cache key is the preference file's path, size and mtime, so editing or
    regenerating the corpus invalidates it automatically. Failure here is non-fatal:
    the original in-memory dataset is returned and the run proceeds uncached.
    """
    import hashlib

    from datasets import load_from_disk

    try:
        stat = os.stat(data_path)
        key = hashlib.sha1(
            f"{os.path.abspath(data_path)}:{stat.st_size}:{int(stat.st_mtime)}".encode()
        ).hexdigest()[:16]
        cache_dir = ROOT / ".hf_dataset_cache" / key

        if cache_dir.exists():
            print(f"Reusing disk-backed dataset cache: {cache_dir}")
            return load_from_disk(str(cache_dir))

        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"Materializing disk-backed dataset cache: {cache_dir}")
        dataset.save_to_disk(str(cache_dir))
        return load_from_disk(str(cache_dir))
    except Exception as exc:  # pragma: no cover - caching is best-effort
        print(f"Dataset disk cache unavailable ({exc}); continuing in memory.")
        return dataset


def run_training(args):
    hp = _resolve_hparams(args)
    model_cfg = hp["model"]
    train_cfg = hp["train"]

    model_path = args.base_model_path or model_cfg.get("base_model_path") or MODEL_PATH
    processor_name = (
        (args.processor_name or "").strip()
        or model_cfg.get("processor_name")
        or processor_for_model(model_path)
        or PROCESSOR_NAME
    )
    print(f"base_model_path={model_path!r}\nprocessor_name={processor_name!r}")
    print(
        f"FiRe-MPO loss: α={hp['alpha']}, γ={hp['gamma']}, λ={hp['lambda_']} "
        f"(ablation v1={hp['alpha_v1']}, v2={hp['alpha_v2']})"
    )

    print("Loading model with Unsloth FastVisionModel...")
    model, _ = FastVisionModel.from_pretrained(model_name=model_path, **UNSLOTH_LOAD_KWARGS)
    model = FastVisionModel.get_peft_model(model, **UNSLOTH_PEFT_KWARGS)

    processor = AutoProcessor.from_pretrained(processor_name, **PROCESSOR_KWARGS)
    processor.tokenizer.padding_side = "left"
    processor = add_special_tokens(processor)

    data_path = args.data_path
    if not data_path:
        raise ValueError("--data_path is required (or set via config)")
    print(f"Loading preference dataset from {data_path}")
    dataset = PreferenceDataset(data_path=data_path)
    dataset = _make_disk_backed(dataset, data_path)

    visual_pref = bool(
        (hp["gamma"] is not None and hp["gamma"] > 0)
        or (hp["alpha_v1"] is not None and hp["alpha_v1"] > 0)
        or (hp["alpha_v2"] is not None and hp["alpha_v2"] > 0)
    )
    dataset_num_proc = 0 if visual_pref else None

    train_kw = dict(
        run_name=os.path.basename(args.output_dir.rstrip("/")) or "fire-mpo",
        num_train_epochs=args.num_train_epochs or train_cfg.get("num_train_epochs", 1),
        **({} if dataset_num_proc is None else {"dataset_num_proc": dataset_num_proc}),
    )
    lr = args.learning_rate if args.learning_rate is not None else train_cfg.get("learning_rate")
    if lr is not None:
        train_kw["learning_rate"] = lr
    pd = (
        args.per_device_train_batch_size
        if args.per_device_train_batch_size is not None
        else train_cfg.get("per_device_train_batch_size")
    )
    if pd is not None:
        train_kw["per_device_train_batch_size"] = pd

    # Seed both the parameter/dropout RNG and the data sampler. `seed` alone leaves the
    # shuffle order fixed, so runs that differ only in `seed` would still see identical
    # batch ordering -- a real source of run-to-run variance that must vary for seed
    # replication to mean anything.
    if args.seed is not None:
        train_kw["seed"] = args.seed
        train_kw["data_seed"] = args.seed
    if args.max_steps is not None:
        train_kw["max_steps"] = args.max_steps
    if args.gradient_checkpointing is not None:
        train_kw["gradient_checkpointing"] = bool(args.gradient_checkpointing)
    if args.dataloader_num_workers is not None:
        train_kw["dataloader_num_workers"] = args.dataloader_num_workers

    fire_args = FiReMPOConfig(
        alpha=hp["alpha"],
        gamma=hp["gamma"],
        lambda_=hp["lambda_"],
        alpha_v1=hp["alpha_v1"],
        alpha_v2=hp["alpha_v2"],
        output_dir=args.output_dir,
        **common_training_kwargs(**train_kw),
    )

    trainer = FiReMPOTrainer(
        model,
        args=fire_args,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=None,
        data_collator=_make_unsloth_safe_fire_mpo_collator(
            model, processor, processor.tokenizer.pad_token_id
        ),
    )
    trainer.train()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train FiRe-MPO (arXiv:2606.12590)")
    p.add_argument("--config", type=str, default=None, help="YAML config (paper/ablation).")
    p.add_argument("--data_path", type=str, default=None, help="Preference JSON path.")
    p.add_argument("--output_dir", type=str, default="models/fire-mpo")
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--base_model_path", type=str, default=None)
    p.add_argument("--processor_name", type=str, default=None)
    p.add_argument("--per_device_train_batch_size", type=int, default=None)
    p.add_argument("--num_train_epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed; sets both seed and data_seed. Default: HF default (42).")
    p.add_argument("--max_steps", type=int, default=None,
                   help="Cap training steps (used by the cost benchmark).")
    p.add_argument("--gradient_checkpointing", type=int, default=None, choices=[0, 1],
                   help="Trade compute for memory. Default on; safe to disable on a "
                        "large-VRAM GPU for a substantial speedup.")
    p.add_argument("--dataloader_num_workers", type=int, default=None)
    # Paper names
    p.add_argument("--alpha", type=float, default=None, help="α: bidirectional KL weight (default 0.01).")
    p.add_argument("--gamma", type=float, default=None, help="γ: visual dual-pair weight (default 0.1).")
    p.add_argument("--lambda_", type=float, default=None, dest="lambda_", help="λ: FKL share (default 0.5).")
    p.add_argument("--alpha_v1", type=float, default=None, help="Ablation visual term v1.")
    p.add_argument("--alpha_v2", type=float, default=None, help="Ablation visual term v2.")
    # Legacy aliases
    p.add_argument("--rrpo_alpha", type=float, default=None, help="Legacy alias for --alpha.")
    p.add_argument("--rrpo_alpha_v1", type=float, default=None, help="Legacy alias for --alpha_v1.")
    p.add_argument("--rrpo_alpha_v2", type=float, default=None, help="Legacy alias for --alpha_v2.")
    p.add_argument("--rrpo_alpha_v3", type=float, default=None, help="Legacy alias for --gamma.")
    p.add_argument("--tkl_share", type=float, default=None, help="Legacy alias for --lambda_.")
    return p


def main():
    args = build_parser().parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
