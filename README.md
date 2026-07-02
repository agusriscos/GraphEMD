# GraphEMD

Repository for the paper *Empirical Mode Decomposition and Graph Transformation of Financial Series* (Agustín M. de los Riscos, Ana Lazcano, Julio E. Sandubete).

## Paper status
The paper is in progress and scheduled for submission on **Wednesday, 8 July 2026** to [*Information Sciences*](https://www.sciencedirect.com/journal/information-sciences) (Elsevier, Q1). The code and data pipelines in this repository are already implemented and ready to reproduce the results.

Latest version of the paper (download): [emd_graphs_agustin_riscos.pdf](paper/emd_graphs_agustin_riscos.pdf)  
Direct download (raw): https://github.com/agusriscos/GraphEMD/raw/main/paper/emd_graphs_agustin_riscos.pdf

## Contents

| Directory | Description |
|------------|-------------|
| `src/python/GraphEMD/` | Library: IMF→graph transformation, synthetic signals, and other transformations |
| `src/python/CommonUtils/` | Minimal shared utilities (`DictClass`) |
| `scripts/` | Paper pipelines: empirical panel, emdsynth, MSCI exploration |
| `analysis/` | Jupyter notebooks for experiments |

## Installation

```bash
cd GraphEMD
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e src/python
```

## Main paper pipelines

```bash
# Full empirical panel (MSCI + XLE/XLP/XLV/XAUUSD)
PYTHONPATH=src/python python scripts/run_vmd_all_assets.py

# Synthetic benchmark (EMD/EEMD/CEEMDAN/VMD)
PYTHONPATH=src/python python scripts/emdsynth/run_emdsynth_decompositions.py

# Figures for the paper (output in PAPER/figures/)
PYTHONPATH=src/python python scripts/generate_decomposition_panel_figure.py
PYTHONPATH=src/python python scripts/generate_ica_panel_figure.py
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
