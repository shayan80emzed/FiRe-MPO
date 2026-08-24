"""
Compatibility shim: re-export FiRe-MPO trainer under legacy RRPO names.
"""

from fire_mpo.trainer.fire_mpo_trainer import (  # noqa: F401
    FiReMPOConfig as RRPOConfig,
    FiReMPOTrainer as RRPOTrainer,
    DataCollatorForFiReMPO as DataCollatorForRRPO,
    _make_unsloth_safe_fire_mpo_collator as _make_unsloth_safe_rrpo_collator,
    FiReMPOConfig,
    FiReMPOTrainer,
    DataCollatorForFiReMPO,
    _make_unsloth_safe_fire_mpo_collator,
)

__all__ = [
    "RRPOConfig",
    "RRPOTrainer",
    "DataCollatorForRRPO",
    "_make_unsloth_safe_rrpo_collator",
    "FiReMPOConfig",
    "FiReMPOTrainer",
    "DataCollatorForFiReMPO",
    "_make_unsloth_safe_fire_mpo_collator",
]
