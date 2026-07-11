# GraphEMD

Repository for the paper *Empirical Mode Decomposition and Graph Transformation of Financial Series* (Agustín M. de los Riscos, Ana Lazcano, Julio E. Sandubete).

---

> [!IMPORTANT]
> **Paper status**
>
> The paper is in progress and scheduled for **first submission on Wednesday, 8 July 2026** to [*Information Sciences*](https://www.sciencedirect.com/journal/information-sciences) (Elsevier, Q1).
>
> The code and data pipelines in this repository are already implemented and ready to reproduce the results.
>
> **Latest draft:** [emd_graphs_agustin_riscos.pdf](paper/emd_graphs_agustin_riscos.pdf) · [direct download](https://github.com/agusriscos/GraphEMD/raw/main/paper/emd_graphs_agustin_riscos.pdf)

---

## Contents

| Directory | Description |
|------------|-------------|
| `data/` | Input price series (parquet) for the empirical panel in the paper |
| `src/python/GraphEMD/` | Library: IMF→graph transformation, synthetic signals, and other transformations |
| `src/python/CommonUtils/` | Minimal shared utilities (`DictClass`) |
| `scripts/` | Paper pipelines: empirical panel, emdsynth, MSCI exploration |
| `analysis/` | Jupyter notebooks for experiments |

## Input data (`data/`)

Daily OHLCV price series (parquet) for the empirical panel, sourced from Yahoo Finance over **2012-01-12 → 2026-04-20**. ETFs are aligned to MSCI World trading days.

| File | Yahoo symbol | Observations |
|------|--------------|--------------|
| `msci_world.parquet` | `^MSWORLD` | 3,587 |
| `xle.parquet` | `XLE` | 3,587 |
| `xlp.parquet` | `XLP` | 3,587 |
| `xlv.parquet` | `XLV` | 3,587 |
| `xauusd.parquet` | `GC=F` | 3,584 |

Each file has columns `Open`, `High`, `Low`, `Close`, `Volume`, `Dividends`, `Stock Splits` (plus `Capital Gains` for MSCI World). Pipelines use `Close` as the input series.

To regenerate from Yahoo Finance:

```bash
PYTHONPATH=src/python python scripts/download_msci_world.py
PYTHONPATH=src/python python scripts/xle_etf_analysis/01_download_xle.py
PYTHONPATH=src/python python scripts/xlp_analysis/01_download_xlp.py
PYTHONPATH=src/python python scripts/xlv_analysis/01_download_xlv.py
PYTHONPATH=src/python python scripts/xauusd_analysis/01_download_xauusd.py
```

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
