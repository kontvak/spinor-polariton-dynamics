# Polariton Condensate: Spin and Reservoir Dynamics

Numerical study of a driven polariton condensate with active/inactive reservoirs, two circular components, internal Josephson coupling and TE–TM splitting. A continuation of [polariton-condensate-dynamics](https://github.com/kontvak/polariton-condensate-dynamics).

[English report](latex/report_en.pdf) (24 pages) · [Russian report](latex/report_ru.pdf) (26 pages) · [Results](docs/journal/results_overview.md) · [Methods](docs/methods.md)

## Start

Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab polariton_spin.ipynb
```

The notebook is the main interface. Long calculations are opt-in. Units are µm, ps and meV. Homogeneous dynamics use DOP853; spatial dynamics use Fourier Strang splitting with an exact kinetic/TE–TM step and joint local RK4.

## Project layout

| Path | Contents |
|---|---|
| `model.py` | Equations, coefficients, pumping, initial states and analytic limits |
| `numerics.py` | Integrators, diagnostics, checkpoints and experiment series |
| `plotting.py` | Report figures and animations from saved data |
| `polariton_spin.ipynb` | Interactive calculations and inspection |
| `config.yaml` | Common coefficients and scenarios |
| `scripts/` | Experiment queue, plan, report builder and storage cleanup |
| `latex/` | Current English/Russian Markdown sources, LaTeX and PDFs |
| `docs/` | Calculation methods |
| `docs/journal/` | Versioned research journal, numerical checks and launch logs |
| `results/figures/report3/` | Current 16 report PNGs and six GIFs |
| `results/diagnostics/report3/` | J scan, controls and supplementary plots |
| `results/data/` | Portable numerical archives and JSON summaries |
| `results/runs/` | Local restart checkpoints; excluded from Git |
| `local/` | Specifications, figure instructions and private working notes; excluded from Git |
| `test_project.py` | Compact physical and numerical checks |

## Figures and reports

```bash
MPLBACKEND=Agg python -c 'from plotting import make_report3_figures; make_report3_figures()'
# Include GIFs when needed:
MPLBACKEND=Agg python -c 'from plotting import make_report3_figures; make_report3_figures(animations=True)'

python scripts/build_report.py
python scripts/build_report.py --ru
cd latex
latexmk -pdf report_en.tex
latexmk -pdf report_ru.tex
```

The reports use A4, 12 pt, five chapters, conclusions, two appendices and 16 figures. Figure names, order, English captions and source parameters are in [figure_manifest.json](results/data/report3/figure_manifest.json). The working figure requirements remain in the ignored `local/` folder. PNGs are 160 mm wide at 300 dpi, except the compact 105 mm angular-response plot. GIFs start at zero, last 30 seconds and use saved frames without time interpolation.

Numerical NPZ archives are retained locally and excluded from Git. A fresh clone includes PDFs, images and JSON summaries; redrawing requires the local archives or recomputation. Only the current report figures and diagnostic plots are kept; earlier numerical results remain available locally.

## Intel Arc queue

`scripts/run_report3_queue.sh` is the single environment-aware launcher; experiment parameters and priorities are in `scripts/experiments.yaml`.

On the calculation machine, the launcher uses PyTorch XPU and compiler dependencies from the neighboring `profiling` project. Optional XPU installation is described in `requirements.txt`. The queue uses `scripts/experiments.yaml`; one GPU worker executes independent experiments sequentially.

```bash
bash scripts/run_report3_queue.sh status
bash scripts/run_report3_queue.sh eta
bash scripts/run_report3_queue.sh check
bash scripts/run_report3_queue.sh start --note "Start next batch"
bash scripts/run_report3_queue.sh pause
bash scripts/run_report3_queue.sh resume --note "Continue saved states"
```

Example of adding a case without starting it:

```bash
bash scripts/run_report3_queue.sh add spot_J009 --source production_C1_spot \
  --until 3000 --set physical.J=0.009 --priority 10 --note "Refine J scan"
```

`pause --after-current` finishes the active case first. `extend ID --until 4500` changes its target; `note ID "text"` records a comment; `disable/enable ID` changes scheduling. Lower priority numbers run first. Start/resume launch a background worker. Logs are in `docs/journal/logs/`. Physical checkpoint parameters are validated before continuation. Execution completion and scientific acceptance are separate statuses.

## Storage and validation

The working project was reduced from hundreds of GiB to approximately 13 GiB. Final fields, selected maps, configurations, aggregate histories, compact phase/animation data and physical restart states are preserved. Deleted dense time windows cannot be reanalysed in full without rerunning them. Continuing a checkpoint produces a fresh diagnostic window.

```bash
python scripts/compact_results.py             # estimate only
python scripts/compact_results.py --prepare   # preserve compact traces and write a plan
python scripts/compact_results.py --apply     # explicit removal after validation
python -m pytest -q
```

The cleaner refuses an active queue. Protect a control run with `keep_dense_fields: true` in the plan, or `add --keep-dense-fields`; the sampling interval is set separately, for example `--set save.probe=1.25`. Automatic deletion in the queue is not enabled.

The sampled 2D spot scan contains running, bounded-phase and fixed-phase regimes. Selected high-J examples use 256² and lack a complete independent convergence series. The equal-pump TE–TM ring still evolves in polarization at 6000 ps. Reference grid/time controls do not imply convergence at every parameter value; measured scopes and remaining limitations are recorded in the results overview.
