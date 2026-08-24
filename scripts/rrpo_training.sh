#!/bin/bash
# Compatibility shim: old RRPO_ALPHA / RRPO_ALPHA_V3 / TKL_SHARE → FiRe-MPO train script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Map legacy env vars to paper names
export ALPHA="${ALPHA:-${RRPO_ALPHA:-0.01}}"
export GAMMA="${GAMMA:-${RRPO_ALPHA_V3:-0.1}}"
export LAMBDA="${LAMBDA:-${TKL_SHARE:-0.5}}"
if [[ -n "${RRPO_ALPHA_V1:-}" ]]; then export ALPHA_V1="${ALPHA_V1:-$RRPO_ALPHA_V1}"; fi
if [[ -n "${RRPO_ALPHA_V2:-}" ]]; then export ALPHA_V2="${ALPHA_V2:-$RRPO_ALPHA_V2}"; fi

echo "[shim] scripts/rrpo_training.sh → scripts/train_fire_mpo.sh (α=$ALPHA γ=$GAMMA λ=$LAMBDA)"
exec "$SCRIPT_DIR/train_fire_mpo.sh" "$@"
