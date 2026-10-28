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
import json
from numerics import ROOT,run_report3_night,write_json
try:
    r=run_report3_night(hours=9.5)
    print('NIGHT FINISHED',r['status'],len(r['results']),flush=True)
except BaseException as e:
    p=ROOT/'results/data/report3/night_queue.json'
    r=json.loads(p.read_text()) if p.exists() else {}
    r.update(status='RUN_INCOMPLETE' if isinstance(e,KeyboardInterrupt) else 'FAILED',error=str(e))
    write_json(p,r)
    raise
PYRUN
