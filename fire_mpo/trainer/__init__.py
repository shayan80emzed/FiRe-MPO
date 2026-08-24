"""FiRe-MPO trainer (lazy exports)."""

from importlib import import_module

__all__ = [
    "FiReMPOConfig",
    "FiReMPOTrainer",
    "DataCollatorForFiReMPO",
    "_make_unsloth_safe_fire_mpo_collator",
    "RRPOConfig",
    "RRPOTrainer",
    "DataCollatorForRRPO",
    "_make_unsloth_safe_rrpo_collator",
]

_ALIASES = {
    "RRPOConfig": "FiReMPOConfig",
    "RRPOTrainer": "FiReMPOTrainer",
    "DataCollatorForRRPO": "DataCollatorForFiReMPO",
    "_make_unsloth_safe_rrpo_collator": "_make_unsloth_safe_fire_mpo_collator",
}


def __getattr__(name: str):
    real = _ALIASES.get(name, name)
    mod = import_module("fire_mpo.trainer.fire_mpo_trainer")
    if hasattr(mod, real):
        return getattr(mod, real)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
