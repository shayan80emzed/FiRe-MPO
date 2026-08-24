"""
Legacy entrypoint: delegates to ``fire_mpo.train`` (FiRe-MPO).

Prefer: ``python -m fire_mpo.train`` or ``scripts/train_fire_mpo.sh``.
"""

from fire_mpo.train import main

if __name__ == "__main__":
    main()
