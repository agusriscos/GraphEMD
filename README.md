# GraphEMD

Repository for the paper *Representation-dependent topology of visibility and recurrence networks under adaptive decomposition and coordinate compression* (Agustín M. de los Riscos, Julio E. Sandubete, Ana Lazcano).

---

> [!IMPORTANT]
> **Paper status**
>
> Manuscript submitted to [*Physica A: Statistical Mechanics and its Applications*](https://www.sciencedirect.com/journal/physica-a-statistical-mechanics-and-its-applications).
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

The `data/` folder contains the **input price series** used in the empirical section of the paper. Each file is a daily OHLCV history downloaded from Yahoo Finance and stored in parquet format.

All series share the common window **2012-01-12 → 2026-04-20**. ETFs (XLE, XLP, XLV) are aligned to MSCI World trading days (3,587 observations). COMEX gold futures (`GC=F`) has 3,584 observations over the same calendar range.

| File | Asset | Yahoo symbol | Observations |
|------|-------|--------------|--------------|
| `msci_world.parquet` | MSCI World index | `^MSWORLD` | 3,587 |
| `xle.parquet` | Energy Select Sector SPDR (XLE) | `XLE` | 3,587 |
| `xlp.parquet` | Consumer Staples Select Sector SPDR (XLP) | `XLP` | 3,587 |
| `xlv.parquet` | Health Care Select Sector SPDR (XLV) | `XLV` | 3,587 |
| `xauusd.parquet` | COMEX gold futures | `GC=F` | 3,584 |

Columns: `Open`, `High`, `Low`, `Close`, `Volume`, `Dividends`, `Stock Splits` (and `Capital Gains` for MSCI World). The `Close` price is the primary series used in the decomposition and graph pipelines.

In the adaptive decomposition, the slow residual component is stored separately and is excluded from both FastICA and network construction. The final FastICA dimensions are \(k=(4,5,4,5,5)\) for MSCI World, XLE, XLP, XLV and GC=F, respectively.

To regenerate the files from Yahoo Finance, use the download scripts under `scripts/` (e.g. `scripts/download_msci_world.py`, `scripts/xle_etf_analysis/01_download_xle.py`).

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
# Full empirical panel (MSCI + XLE/XLP/XLV/GC=F)
PYTHONPATH=src/python python scripts/run_vmd_all_assets.py

# Synthetic benchmark (EMD/EEMD/CEEMDAN/VMD)
PYTHONPATH=src/python python scripts/emdsynth/run_emdsynth_decompositions.py

# Figures for the paper (output in PAPER/figures/)
PYTHONPATH=src/python python scripts/generate_decomposition_panel_figure.py
PYTHONPATH=src/python python scripts/generate_ica_panel_figure.py
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
