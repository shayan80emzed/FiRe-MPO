#!/bin/bash
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/baselines/green_score.sh" "$@"
