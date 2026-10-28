#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(dirname "$project_root")"
level_zero="$workspace_root/profiling/third_party/level_zero/usr"
ocloc="$workspace_root/profiling/third_party/ocloc/usr"
cd "$project_root"
export CPATH="$level_zero/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$level_zero/lib/x86_64-linux-gnu${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$level_zero/lib/x86_64-linux-gnu:$ocloc/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$ocloc/bin:$PATH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
# Read-only control commands never initialize a GPU or run preflight.
if [[ "${1:-}" == "--check" ]]; then set -- check; fi
if [[ "${1:-}" == "--matched-ring" ]]; then shift; set -- resume "$@"; fi
if [[ "$#" -eq 0 ]]; then set -- status; fi
exec .venv/bin/python -u local/experiment_queue.py "$@"
