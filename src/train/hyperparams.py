"""
Shared hyperparameters — re-exports from fire_mpo.hyperparams (paper Table 6).
Legacy import path: ``from src.train.hyperparams import ...``
"""

from fire_mpo.hyperparams import *  # noqa: F401,F403
from fire_mpo.hyperparams import (  # noqa: F401
    DEBUG,
    GLOBAL_TRAIN_BATCH_SIZE,
    MODEL_PATH,
    PAPER_ALPHA,
    PAPER_BETA,
    PAPER_GAMMA,
    PAPER_LAMBDA,
    PAPER_LR,
    PROCESSOR_KWARGS,
    PROCESSOR_NAME,
    UNSLOTH_LOAD_KWARGS,
    UNSLOTH_PEFT_KWARGS,
    common_training_kwargs,
    processor_for_model,
)
