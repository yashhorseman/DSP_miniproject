# FFT-Based Vibration Fault Detection (MTE 3121 DSP Mini Project)

Simulation-only study of healthy, imbalanced, misaligned and bearing-fault vibration signals using the FFT, digital filtering and related DSP tools.
Manipal Institute of Technology, Dept. of Mechatronics. Team: Dhanush Chittoory, Polimera Devashish Reddy, Bhuvan Chand Rayani, Yash Ralhan.

All results come from synthetic signals, so they apply only to the chosen signal model. They are not validation on real machinery.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python vibration_analysis.py         # ~10 s: figures/, results/
python -m unittest discover -s tests -v          # 20 unit tests
python build_notebook.py             # regenerate vibration_analysis.ipynb
python build_report.py               # regenerate Report.docx from REPORT.md
```

The random seed is fixed (`np.random.seed(42)`), so every run reproduces the same numbers (except the FFT timing table in Step 15, which depends on the machine).

## What is where

| Path | Content |
|---|---|
| `vibration_analysis.py` | The whole pipeline (core steps 0-7, extended studies 8-15), fully commented |
| `vibration_analysis.ipynb` | Same pipeline as a notebook, generated from the script |
| `REPORT.md` / `Report.docx` | Report draft (Markdown source and the Word version with inline figures) |
| `VIVA_NOTES.md` | Concepts and likely viva questions with answers |
| `figures/` | 19 PNG figures at 200 dpi |
| `results/` | `summary.txt` (all printed output), `peak_table.csv`, `mc_amplitudes.csv` |
| `tests/` | Unit tests (standard `unittest`, no extra dependencies) |

## Pipeline

Core (committed in the synopsis): signal model, time-domain statistics, Butterworth filtering, FFT spectra, peak detection and verification, toy classifier, noise sweep, 50-run statistics.

Extended studies: impulsive bearing model with real BPFO/BPFI, Welch averaging, STFT for a fault that starts mid-record, drifting speed with order tracking, filter comparison, severity sweep, combined faults, DFT/Parseval cross-checks.

## Configuration

All parameters are in the `CFG` dictionary at the top of `vibration_analysis.py` (sampling rate, shaft frequency, noise level, fault amplitudes, filter cut-offs, bearing geometry, ...).

## Notes

- The bearing geometry (9 balls, 7.94 mm ball, 39.04 mm pitch) is a typical 6205-type value assumed for the simulation, not measured data.
- The toy classifier thresholds are set from the simulated model itself, so its accuracy says nothing about real machines.
