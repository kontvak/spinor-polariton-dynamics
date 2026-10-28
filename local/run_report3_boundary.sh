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
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 MPLBACKEND=Agg
exec .venv/bin/python -u - <<'PYRUN'
from numerics import run_report3_boundary_queue
r=run_report3_boundary_queue(hours=1.5)
print('BOUNDARY QUEUE',r['status'],flush=True)
PYRUN
