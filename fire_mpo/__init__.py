"""FiRe-MPO: Fine-grained Regularized Medical Preference Optimization (arXiv:2606.12590).

Keep this package ``__init__`` lightweight so ``python -m fire_mpo.train`` can
import Unsloth before TRL/transformers via the train module.
"""

from importlib import import_module

__all__ = [
    "FiReMPOConfig",
    "FiReMPOTrainer",
    "DataCollatorForFiReMPO",
]


def __getattr__(name: str):
    if name in {
        "FiReMPOConfig",
        "FiReMPOTrainer",
        "DataCollatorForFiReMPO",
        "_make_unsloth_safe_fire_mpo_collator",
    }:
        mod = import_module("fire_mpo.trainer.fire_mpo_trainer")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
