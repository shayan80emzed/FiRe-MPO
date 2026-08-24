#!/usr/bin/env python3
"""
Smoke checks for the FiRe-MPO refactor (no GPU train).

1. Config aliases (α/γ/λ ↔ rrpo_*)
2. Bidirectional KL loss shapes (Eq. 4)
3. Preference JSON schema + path resolution
4. Legacy import shims
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_config_aliases():
    from fire_mpo.trainer import FiReMPOConfig

    # Paper names → legacy
    c = FiReMPOConfig(alpha=0.01, gamma=0.1, lambda_=0.5, output_dir="/tmp/fire-mpo-smoke")
    assert c.rrpo_alpha == 0.01
    assert c.rrpo_alpha_v3 == 0.1
    assert float(c.tkl_share) == 0.5

    # Legacy → paper names
    c2 = FiReMPOConfig(
        rrpo_alpha=0.02,
        rrpo_alpha_v3=0.2,
        tkl_share=0.25,
        output_dir="/tmp/fire-mpo-smoke",
    )
    assert c2.alpha == 0.02
    assert c2.gamma == 0.2
    assert c2.lambda_ == 0.25
    print("OK config aliases")


def test_bidirectional_kl():
    from fire_mpo.loss import compute_bidirectional_kl, combine_fire_mpo_losses

    B, S, V = 2, 8, 32
    policy = torch.randn(B, S, V)
    ref = torch.randn(B, S, V)
    mask = torch.zeros(B, S)
    mask[:, 2:6] = 1

    fkl, rkl, mixed = compute_bidirectional_kl(policy, ref, mask, mask, lambda_=0.5)
    assert fkl.shape == (B,)
    assert rkl.shape == (B,)
    assert mixed.shape == (B,)
    assert torch.allclose(mixed, 0.5 * fkl + 0.5 * rkl)

    base = torch.tensor([1.0, 1.0])
    visual = torch.tensor([0.5, 0.5])
    total = combine_fire_mpo_losses(
        base, visual_rank_loss=visual, gamma=0.1, mixed_kl=mixed, alpha=0.01
    )
    expected = base + 0.1 * visual + 0.01 * mixed
    assert torch.allclose(total, expected)
    print("OK bidirectional KL + combine")


def test_preference_json():
    from fire_mpo.pipeline.paths import preference_json

    path = preference_json(
        "HuatuoGPT-Vision-7B",
        "slake",
        "rrpo_with_medsam3_rejected.json",
    )
    assert path.exists(), f"Missing preference file: {path}"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list) and len(data) > 0
    sample = data[0]
    for key in ("prompt", "chosen", "rejected", "image_path"):
        assert key in sample, f"missing {key}"
    # FiRe-MPO visual path should include rejected_image_path on most rows
    has_rej = sum(1 for x in data if x.get("rejected_image_path"))
    print(f"OK preference JSON ({len(data)} rows, {has_rej} with rejected_image_path)")
    print(f"  sample id={sample.get('id')} qid={sample.get('qid')}")
    print(f"  image_path={sample['image_path']}")


def test_legacy_imports():
    from src.trainer.rrpo_trainer import RRPOTrainer, RRPOConfig, FiReMPOTrainer
    from src.train.hyperparams import MODEL_PATH, PAPER_ALPHA, common_training_kwargs

    assert RRPOTrainer is FiReMPOTrainer
    assert RRPOConfig is not None
    assert PAPER_ALPHA == 0.01
    assert "Huatuo" in MODEL_PATH or "Qwen" in MODEL_PATH or "/" in MODEL_PATH
    kw = common_training_kwargs(run_name="smoke", learning_rate=1e-6, per_device_train_batch_size=4)
    assert kw["learning_rate"] == 1e-6
    assert kw["beta"] == 0.1
    print("OK legacy imports + hyperparams")


def main():
    test_config_aliases()
    test_bidirectional_kl()
    test_preference_json()
    test_legacy_imports()
    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
