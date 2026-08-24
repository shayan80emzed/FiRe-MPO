"""
Custom trainer implementations.

Prefer:

    from fire_mpo.trainer import FiReMPOTrainer, FiReMPOConfig
    from src.trainer.qwen_dpo_trainer import QwenDPOTrainer
"""

from importlib import import_module

__all__ = [
    "RRPOTrainer",
    "RRPOConfig",
    "DataCollatorForRRPO",
    "FiReMPOTrainer",
    "FiReMPOConfig",
]


def __getattr__(name: str):
    if name in {
        "RRPOTrainer",
        "RRPOConfig",
        "DataCollatorForRRPO",
        "FiReMPOTrainer",
        "FiReMPOConfig",
        "DataCollatorForFiReMPO",
        "_make_unsloth_safe_rrpo_collator",
        "_make_unsloth_safe_fire_mpo_collator",
    }:
        mod = import_module("src.trainer.rrpo_trainer")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
