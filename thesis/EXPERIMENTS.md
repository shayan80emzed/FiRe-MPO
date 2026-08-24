# Experiment plan and state

**Status: paused 2026-08-06. All infrastructure is built and validated; no matrix runs
have completed yet.** Training was stopped mid-first-run to prioritize thesis writing.
Nothing was lost — checkpoints save only at epoch end, so the driver re-runs it.

To resume, jump to [Resuming](#resuming).

---

## 1. What is already done

### Blockers found and fixed

| Problem | Fix |
|---|---|
| **16 GB cgroup cap** on `user-1000.slice` OOM-killed every run during tokenization, before the GPU was touched. This is why the original experiments ran on the cluster, not here. | `/etc/systemd/system/user-1000.slice.d/99-i4health-shayan-no-memory-cap.conf` sets `MemoryMax=infinity`, mirroring the existing root exemption. The global 16 GB rule in `user-.slice.d/` is untouched, so other users are unaffected. Persists across reboot. |
| **No seed support.** `fire_mpo/train.py` and `src/train/train_dpo.py` used hand-rolled argparse with no seed, so every run used HF's default 42. Three "seeds" would have been three identical runs. | `--seed` added to both, wired to **`seed` and `data_seed`**. `seed` alone leaves shuffle order fixed; the sampler needs `data_seed`. Verified at config level. |
| **Tokenization re-ran every time** (~10 min/run). `Dataset.from_list` produces an in-memory dataset, and HF only caches `.map()` against disk-backed sources. | `_make_disk_backed()` in `fire_mpo/train.py` round-trips through an Arrow store keyed on the corpus file's path+size+mtime. Confirmed: the second run skipped tokenization entirely. Saves ~7 h across the matrix. |

### Measured (this is thesis item 3)

FiRe-MPO, Qwen3-VL-4B, SLAKE, batch 8, gradient checkpointing on:

- **4.3–4.6 s/step** steady state (the first step is ~74 s of warmup — do not mistake it for the step cost)
- **~139 GB peak** of the 143 GB card
- 610 steps/epoch → **~44 min per run**; HuatuoGPT-7B expected ~65 min

Two consequences, both load-bearing:

- **Gradient checkpointing cannot be disabled.** Tested: it OOMs at 143 GB.
- **Runs cannot be co-scheduled.** At 139 GB peak, two jobs would race for the same pool
  and OOM tens of minutes in. The driver is therefore strictly sequential.

### Results already obtained

**Style gap (`experiments/analysis/style_gap.py` → `style_gap.json`)** — this settles the
measurement half of item 6, and needed no GPU. HuatuoGPT SLAKE:

| construction | len(y⁺) | len(y⁻) | mean \|Δlen\| | within 2 words | edit distance |
|---|---|---|---|---|---|
| Style Agnostic | 1.7 | 1.9 | 0.6 | 94.4% | 0.894 |
| GT Style | 1.7 | 13.5 | 11.8 | 1.0% | 0.978 |
| LVLM Style | 13.1 | 13.5 | 1.3 | 88.1% | 0.210 |

Under GT Style, length alone separates the pair in 99% of cases and the two responses are
98% disjoint — the reward-hacking shortcut, quantified. Under LVLM Style, 88% of pairs are
within two words and edit distance falls to 0.21, leaving only the clinical span as signal.
VQA-RAD shows the same pattern.

**D1/D2/D3 mapping confirmed** from the docstring of `fire_mpo/pipeline/build_ablation_dpo.py`:
D1 = Style Agnostic, D2 = GT Style, D3 = LVLM Style. (Getting this backwards would invert
the conclusion of thesis §7.6.)

### Evaluation efficiency

`utils/correctness_evaluator.py` now has a regex fast path: when the ground truth is a bare
yes/no, the first standalone yes/no token in the response decides the verdict with no API
call. Falls back to the LLM judge when the model produced no yes/no token, rather than
guessing. 11/11 unit cases pass, including the traps (`nodule`, `normal`, `Yesterday` must
not match). Removes **33.5% of SLAKE** and **55.7% of VQA-RAD** calls.

Accuracy runs also use `NUM_RETURN_SEQUENCES=1` instead of 4 — a further ~4× on eval GPU
time. The 4-sample setting is only needed for Pass@4, which is not in this matrix.

---

## 2. Scripts

| File | Purpose |
|---|---|
| `experiments/run_matrix.sh <phase>` | Sequential, resumable driver. Phases `1a`, `1b`, `2`, `3b`. Skips any run whose checkpoint exists. |
| `experiments/chain_phases.sh` | Waits for 1a, then runs 1b → 2 → 3b. |
| `experiments/eval_runs.sh` | Inference + scoring for every completed checkpoint. |
| `experiments/aggregate_results.py` | Scored CSVs → mean±std across seeds + paired McNemar vs FiRe-MPO. |
| `experiments/analysis/style_gap.py` | Style-gap measurement (already run). |

Baselines are configurations of the same trainer, not separate code:

| Baseline | Configuration |
|---|---|
| FiRe-MPO | `ALPHA=0.01 GAMMA=0.1 LAMBDA=0.5` (default) |
| RRPO | `GAMMA=0 LAMBDA=1` (forward KL only, no visual term) |
| MASK-DPO | `GAMMA=0 ALPHA=0` (localized rewards, no regularizer) |
| DPO | `scripts/baselines/dpo_training.sh` on the D3 corpus |

mDPO is excluded — Shayan supplies those results separately.

---

## 3. The matrix

| Phase | Content | Runs | Est. |
|---|---|---|---|
| 1a | Seeds {0,1,2} × 4 methods, **Qwen3-VL-4B**, SLAKE | 12 | ~9 h |
| 1b | Seeds {0,1,2} × 4 methods, **HuatuoGPT-7B**, SLAKE | 12 | ~13 h |
| 2 | λ ∈ {0, 0.25, 0.75, 1}, α ∈ {0.001, 0.1}, γ ∈ {0.05, 0.5}, Qwen3, seed 0 | 8 | ~6 h |
| 3b | 3 constructions × 3 seeds, HuatuoGPT, SLAKE | 9 | ~10 h |
| | **training total** | **41** | **~38 h** |
| | evaluation (inference + scoring) | | ~7 h |

Seeds are `{0, 1, 2}` — deliberately not 42, so the existing checkpoints in `models/`
remain an independent check rather than being silently folded in as "seed 1".

Phase 2 omits the default point (λ=0.5, α=0.01, γ=0.1) because phase 1a's seed-0 FiRe-MPO
run already covers it.

---

## 4. Resuming

**Run under tmux, not through an editor session.** The drivers were previously launched as
children of the Claude/Cursor process tree, so closing the session would kill ~2 days of
work. `setsid` did not survive either.

```bash
cd /data/shayan/med-align
tmux new-session -d -s firempo \
  './experiments/run_matrix.sh 1a > logs/runs/phase1a_driver.log 2>&1; \
   ./experiments/chain_phases.sh'

tmux attach -t firempo          # watch
tail -f logs/runs/phase1a_driver.log
```

Then, once checkpoints exist:

```bash
./experiments/eval_runs.sh                      # inference + scoring
dpo_env/bin/python experiments/aggregate_results.py
```

Outputs land in `experiments/eval/` (per-run CSVs) and `experiments/results/`
(`runs.json`, `summary.json`, `significance.json`, `per_question/`).

Everything is resumable: `run_matrix.sh` skips runs with checkpoints, `eval_runs.sh` skips
already-scored runs. Killing and restarting costs only the in-flight run.

### Still unverified

**Seeds have not been shown to diverge behaviourally.** Config-level plumbing is confirmed
(`seed` and `data_seed` both reach `FiReMPOConfig`), but no two seeded runs have completed,
so the end-to-end check is outstanding. Do this first on resume:

```bash
# after qwen-slake-firempo-s0 and -s1 finish
dpo_env/bin/python - <<'PY'
import json, glob
for s in (0, 1):
    f = glob.glob(f"experiments/runs/qwen-slake-firempo-s{s}/checkpoint-*/trainer_state.json")
    lh = [e for e in json.load(open(f[0]))["log_history"] if "loss" in e][:5]
    print(s, [round(e["loss"], 5) for e in lh])
PY
```

The two loss sequences must differ. If they are identical, the seed is not reaching the
sampler and the remaining 22 runs of phase 1 would be wasted.

---

## 5. Known gaps not yet addressed

- **Results durability.** The previous VQA-accuracy, GREEN and inference outputs were
  produced on a cluster and no longer exist; only `experiments/visual_grounding/` survived,
  because `experiments/` is gitignored. `aggregate_results.py` writes small JSON summaries
  intended to be tracked — make sure `experiments/results/` is added to git, or the same
  loss will recur.
- **Git history is unreadable** (`git status` fails with `bad object HEAD`), so there is no
  commit trail for any of this.
- **Phase 1 covers SLAKE only.** VQA-RAD is where the smallest claimed gains live
  (+1.05 Open on Qwen3) and is a third the size per run, so it is the best candidate for an
  extension (+24 runs, ~8 h).
