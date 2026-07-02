#!/usr/bin/env python3
r"""
Figure for the paper: each original CEEMDAN IMF versus its linear reconstruction
from dimensionality reduction (``imfs_reconstruidas_aprox.parquet``).

Layout: 8 rows $\times$ 2 columns (shared time axis). Left column: original IMF;
right column: $\hat{\mathrm{IMF}}$ from \texttt{inverse\_transform}.

Example::

    PYTHONPATH=src/python python \\
        scripts/exploration/generate_imfs_original_vs_reconstructed_figure.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_ORIG = _REPO_ROOT / "data" / "20abr26" / "msci_world_imfs_ceemdan.parquet"
_DEFAULT_RECON = (
    _REPO_ROOT
    / "docs"
    / "20abr26"
    / "out"
    / "imfs_dim_red"
    / "k4"
    / "fastica"
    / "imfs_reconstruidas_aprox.parquet"
)
_DEFAULT_OUT = (
    _REPO_ROOT
    / "docs"
    / "20abr26"
    / "images"
    / "english"
    / "imf_original_vs_reconstructed_k4.png"
)

logger = logging.getLogger(__name__)


def _columnas_imf_ordenadas(df: pd.DataFrame) -> List[str]:
    """
    Return ``IMF_*`` names sorted by index.

    Parameters
    ----------
    df : pd.DataFrame
        Table with columns ``IMF_1``, ...

    Returns
    -------
    list of str
        Sorted names.
    """
    cols = [c for c in df.columns if c.startswith("IMF_")]
    return sorted(cols, key=lambda x: int(x.split("_")[1]))


def generar_figura(
    ruta_original: Path,
    ruta_recon: Path,
    ruta_salida: Path,
    k_reduccion: int,
    dpi: int = 200,
    figsize_ancho: float = 14.0,
    figsize_alto_por_fila: float = 1.25,
) -> None:
    r"""
    Save a PNG comparing IMF$_j$ and $\hat{\mathrm{IMF}}_j$ for $j=1,\ldots,8$.

    Parameters
    ----------
    ruta_original : Path
        CEEMDAN parquet with ``IMF_1``, ...
    ruta_recon : Path
        Parquet ``imfs_reconstruidas_aprox.parquet``.
    ruta_salida : Path
        PNG path.
    k_reduccion : int
        Dimension $k$ used in the reduction (title only).
    dpi : int, optional
        Resolution.
    figsize_ancho : float, optional
        Figure width in inches.
    figsize_alto_por_fila : float, optional
        Subplot row height in inches.
    """
    df_o = pd.read_parquet(ruta_original, engine="pyarrow")
    df_r = pd.read_parquet(ruta_recon, engine="pyarrow")
    nombres = _columnas_imf_ordenadas(df_o)
    nombres_r = _columnas_imf_ordenadas(df_r)
    if nombres != nombres_r:
        raise ValueError(
            f"Different IMF columns: original {nombres}, recon {nombres_r}."
        )
    if len(df_o) != len(df_r):
        raise ValueError(
            f"Longitudes distintas: {len(df_o)} vs {len(df_r)}."
        )

    T = len(df_o)
    t = np.arange(T, dtype=np.float64)
    n_imf = len(nombres)

    fig, axes = plt.subplots(
        n_imf,
        2,
        figsize=(figsize_ancho, figsize_alto_por_fila * n_imf + 0.6),
        sharex=True,
        constrained_layout=True,
    )
    fig.suptitle(
        f"MSCI World CEEMDAN: each IMF vs rank-{k_reduccion} linear reconstruction "
        r"($\hat{\mathrm{IMF}}$ from inverse transform)",
        fontsize=12,
    )

    color_orig = "#1f77b4"
    color_recon = "#ff7f0e"

    for i, col in enumerate(nombres):
        y_o = np.asarray(df_o[col].values, dtype=np.float64)
        y_r = np.asarray(df_r[col].values, dtype=np.float64)
        axes[i, 0].plot(t, y_o, color=color_orig, linewidth=0.55, label="Original")
        axes[i, 0].set_ylabel(col.replace("_", " "), fontsize=8)
        axes[i, 0].tick_params(axis="y", labelsize=7)
        axes[i, 0].grid(True, alpha=0.25)

        axes[i, 1].plot(t, y_r, color=color_recon, linewidth=0.55, label="Reconstructed")
        axes[i, 1].set_ylabel(f"Recon. {col}", fontsize=8)
        axes[i, 1].tick_params(axis="y", labelsize=7)
        axes[i, 1].grid(True, alpha=0.25)
        if i == 0:
            axes[i, 0].legend(loc="upper right", fontsize=7)
            axes[i, 1].legend(loc="upper right", fontsize=7)

    axes[-1, 0].set_xlabel("Time index", fontsize=9)
    axes[-1, 1].set_xlabel("Time index", fontsize=9)
    axes[0, 0].set_title("Original IMFs", fontsize=10)
    axes[0, 1].set_title(r"Reconstructed $\hat{\mathrm{IMF}}$ (same scale)", fontsize=10)

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura: %s (T=%d, k=%d)", ruta_salida, T, k_reduccion)


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser(
        description="Figure: original IMF vs reconstructed (8 rows × 2 columns).",
    )
    parser.add_argument("--parquet-original", type=Path, default=_DEFAULT_ORIG)
    parser.add_argument("--parquet-recon", type=Path, default=_DEFAULT_RECON)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Reduced dimension (legend/title only).",
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.parquet_original.is_file():
        logger.error("Missing %s", args.parquet_original)
        sys.exit(1)
    if not args.parquet_recon.is_file():
        logger.error("Missing %s (run reduce_ceemdan_imf_dimensionality.py).", args.parquet_recon)
        sys.exit(1)

    generar_figura(
        args.parquet_original,
        args.parquet_recon,
        args.out,
        k_reduccion=args.k,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
