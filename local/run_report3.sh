#!/usr/bin/env bash
# Local Intel Arc environment already tested in ../profiling; no thermal guard.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(dirname "$project_root")"
level_zero="$workspace_root/profiling/third_party/level_zero/usr"
ocloc="$workspace_root/profiling/third_party/ocloc/usr"
cd "$project_root"
for probe in "$project_root"/results/runs/report3/*/probe_maps.npy; do
    if [[ -f "$probe" ]] && fuser "$probe" >/dev/null 2>&1; then
        echo "A report3 calculation is already active; use its saved progress." >&2
        exit 1
    fi
done
export CPATH="$level_zero/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$level_zero/lib/x86_64-linux-gnu${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$level_zero/lib/x86_64-linux-gnu:$ocloc/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$ocloc/bin:$PATH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 MPLBACKEND=Agg
mkdir -p results/runs/report3
exec flock -n results/runs/report3/pilots.lock .venv/bin/python -u - <<'PYRUN'
import json
from numerics import ROOT,run_report3_threshold_stage,write_json
try:
    result=run_report3_threshold_stage()
    print(result['status'],result.get('selected_p'),flush=True)
except BaseException as error:
    path=ROOT/'results/data/report3/threshold_selection.json'
    report=json.loads(path.read_text()) if path.exists() else {}
    report.update(status='RUN_INCOMPLETE' if isinstance(error,KeyboardInterrupt) else 'FAILED',
                  error_type=type(error).__name__,error=str(error))
    write_json(path,report)
    raise
PYRUN
