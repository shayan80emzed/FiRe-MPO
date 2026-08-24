# FiRe-MPO (med-align)

Code companion to **Analyzing and Improving Fine-grained Preference Optimization in Medical LVLMs** ([arXiv:2606.12590](https://arxiv.org/pdf/2606.12590)).

## Paper → code map

| Paper | Code |
|-------|------|
| Eq. 3 ranking loss `(v,y+)≻(v,y−)` + `γ·(v′,y−)≻(v′,y+)` | [`fire_mpo/trainer/fire_mpo_trainer.py`](fire_mpo/trainer/fire_mpo_trainer.py) |
| Eq. 4 bidirectional token-wise KL (`λ` FKL + `(1−λ)` RKL) | [`fire_mpo/loss/fire_mpo.py`](fire_mpo/loss/fire_mpo.py) |
| Eq. 5 `L = L_rank + α L_KL` | `FiReMPOTrainer.get_batch_loss_metrics` |
| Fig. 4 on-policy preference construction | `python -m fire_mpo.pipeline.build_text_prefs` |
| MedSAM3 lesion corruption `v′` | `python -m fire_mpo.pipeline.enrich_medsam3_prompt` → `corrupt_lesions` |
| Table 6 hyperparameters | [`configs/fire_mpo_default.yaml`](configs/fire_mpo_default.yaml) |
| Tables 2–5 ablations | [`configs/ablations/`](configs/ablations/) |

Hyperparameters (paper names): **α** `--alpha`, **γ** `--gamma`, **λ** `--lambda_`, **β** DPO `beta`. Legacy aliases (`rrpo_alpha`, `rrpo_alpha_v3`, `tkl_share`) still work.

## Layout

```
fire_mpo/                 # package: train, loss, trainer, pipeline, eval
  pipeline/medsam3/       # MedSAM3 enrich + lesion corruption backends
configs/                  # paper default + ablation YAMLs
scripts/
  train_fire_mpo.sh       # main training entry
  rrpo_training.sh        # shim → train_fire_mpo.sh
  baselines/              # DPO / eval / Green / VGMED / inference
  smoke_fire_mpo.py       # refactor smoke checks
src/                      # compatibility shims + DPO/SFT helpers
utils/                    # preference builders + eval helpers
templates/                # prompt / conversation templates
preference_dataset/       # on-policy JSON (do not move)
medsam3_rejected_images/  # corrupted v′ images
models/                   # checkpoints
MedSAM3/                  # MedSAM3 install (segmentation)
GREEN/                    # Green Score dependency
figures/                  # paper figure exports
experiments/              # inference / eval outputs
```

## Quick start

```bash
python scripts/smoke_fire_mpo.py

./scripts/train_fire_mpo.sh
CONFIG=configs/ablations/wo_rkl.yaml ./scripts/train_fire_mpo.sh

python -m fire_mpo.pipeline.build_text_prefs \
  --model HuatuoGPT-Vision-7B --dataset slake --csv path/to/train.csv
python -m fire_mpo.pipeline.enrich_medsam3_prompt \
  --model HuatuoGPT-Vision-7B --dataset slake
python -m fire_mpo.pipeline.corrupt_lesions \
  --model HuatuoGPT-Vision-7B --dataset slake
```

## Evaluation

```bash
./scripts/baselines/evaluation.sh
./scripts/baselines/green_score.sh
./scripts/baselines/visual_grounding.sh
```

## Compatibility

```python
from src.trainer.rrpo_trainer import RRPOTrainer, RRPOConfig  # → FiReMPO*
```
