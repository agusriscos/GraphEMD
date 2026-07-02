#!/usr/bin/env python3
"""
Generate figures from the reduced parquet (``imfs_reducidas.parquet``).

By default (**reduced series only**): only $Z_1$--$Z_k$ in a single column,
without IMFs or residue (illustrates the transformation result).

Full panel mode (``--con-panel-completo``): 9$\times$2 grid with original
IMFs, $Z_j$, and residue (extended comparison).

Example::

    PYTHONPATH=src/python python \\
        scripts/exploration/generate_imfs_original_vs_reduced_figure.py
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_ORIG = _REPO_ROOT / "data" / "20abr26" / "msci_world_imfs_ceemdan.parquet"
_DEFAULT_RED = (
    _REPO_ROOT
    / "docs"
    / "20abr26"
    / "out"
    / "imfs_dim_red"
    / "k4"
    / "fastica"
    / "imfs_reducidas.parquet"
)
_DEFAULT_OUT = (
    _REPO_ROOT
    / "docs"
    / "20abr26"
    / "images"
    / "english"
    / "imf_original_vs_reduced_fastica_k4.png"
)

logger = logging.getLogger(__name__)


def _columnas_imf_ordenadas(df: pd.DataFrame) -> List[str]:
    """
    Return ``IMF_*`` names sorted by numeric index.

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


def _columnas_z_ordenadas(df: pd.DataFrame) -> List[str]:
    """
    Return ``Z_*`` names sorted by index.

    Parameters
    ----------
    df : pd.DataFrame
        Table with columns ``Z_1``, ...

    Returns
    -------
    list of str
        Sorted names.
    """
    cols = [c for c in df.columns if re.match(r"^Z_\d+$", c)]
    return sorted(cols, key=lambda x: int(x.split("_")[1]))


def cargar_pares_series(
    ruta_original: Path,
    ruta_reducido: Path,
) -> Tuple[np.ndarray, List[str], np.ndarray, List[str], np.ndarray, np.ndarray]:
    """
    Load aligned matrices (same $T$) of original IMFs, reduced $Z$, and residues.

    Parameters
    ----------
    ruta_original : Path
        CEEMDAN parquet with ``IMF_1``, ..., ``Residuo``.
    ruta_reducido : Path
        Parquet with ``Z_1``, ..., ``Residuo``.

    Returns
    -------
    X_imf : np.ndarray
        Shape ``(T, p)`` with $p$ oscillatory IMFs.
    nombres_imf : list of str
        IMF names.
    Z : np.ndarray
        Shape ``(T, k)``.
    nombres_z : list of str
        ``Z_j`` names.
    residuo_original : np.ndarray
        CEEMDAN residue from the original parquet.
    residuo_reducido : np.ndarray
        ``Residuo`` column from the reduced parquet (must match the original).
    """
    df_o = pd.read_parquet(ruta_original, engine="pyarrow")
    df_r = pd.read_parquet(ruta_reducido, engine="pyarrow")
    nombres_imf = _columnas_imf_ordenadas(df_o)
    nombres_z = _columnas_z_ordenadas(df_r)
    if len(df_o) != len(df_r):
        raise ValueError(
            f"Longitudes distintas: original {len(df_o)}, reducido {len(df_r)}."
        )
    X_imf = np.asarray(df_o[nombres_imf].values, dtype=np.float64)
    Z = np.asarray(df_r[nombres_z].values, dtype=np.float64)
    if "Residuo" not in df_o.columns:
        raise ValueError("Original parquet must include Residuo column.")
    residuo_original = np.asarray(df_o["Residuo"].values, dtype=np.float64)
    if "Residuo" not in df_r.columns:
        raise ValueError("Reduced parquet must include Residuo column.")
    residuo_reducido = np.asarray(df_r["Residuo"].values, dtype=np.float64)
    return X_imf, nombres_imf, Z, nombres_z, residuo_original, residuo_reducido


def generar_figura_solo_reducidas(
    ruta_reducido: Path,
    ruta_salida: Path,
    titulo: str,
    dpi: int = 200,
    figsize_ancho: float = 10.0,
    figsize_alto_por_fila: float = 1.45,
) -> None:
    """
    Figure with only the $Z_j$ series (no IMFs or residue).

    Parameters
    ----------
    ruta_reducido : Path
        Parquet with ``Z_1``, ... (``Residuo`` is ignored).
    ruta_salida : Path
        PNG output path.
    titulo : str
        Global title (may contain matplotlib mathtext syntax).
    dpi : int, optional
        Resolution.
    figsize_ancho : float, optional
        Width in inches.
    figsize_alto_por_fila : float, optional
        Height per subplot.
    """
    df_r = pd.read_parquet(ruta_reducido, engine="pyarrow")
    nombres_z = _columnas_z_ordenadas(df_r)
    if not nombres_z:
        raise ValueError("No Z_* columns in the reduced parquet.")
    Z = np.asarray(df_r[nombres_z].values, dtype=np.float64)
    T, k = Z.shape[0], Z.shape[1]
    t = np.arange(T, dtype=np.float64)

    fig, axes = plt.subplots(
        k,
        1,
        figsize=(figsize_ancho, figsize_alto_por_fila * k + 0.5),
        sharex=True,
        constrained_layout=True,
    )
    if k == 1:
        axes = np.array([axes])
    fig.suptitle(titulo, fontsize=12)
    color_z = "#d62728"
    for i in range(k):
        axes[i].plot(t, Z[:, i], color=color_z, linewidth=0.65)
        axes[i].set_ylabel(nombres_z[i].replace("_", " "), fontsize=9)
        axes[i].tick_params(axis="y", labelsize=8)
        axes[i].grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time index", fontsize=10)

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure (reduced only): %s (T=%d, k=%d)", ruta_salida, T, k)


def generar_figura_panel_completo(
    ruta_original: Path,
    ruta_reducido: Path,
    ruta_salida: Path,
    titulo_derecha: str,
    dpi: int = 200,
    figsize_ancho: float = 14.0,
    figsize_alto_por_fila: float = 1.35,
) -> None:
    """
    9$\times$2 panel: IMFs, $Z_j$, residue (extended comparison).

    Parameters
    ----------
    ruta_original : Path
        CEEMDAN IMF parquet.
    ruta_reducido : Path
        Reduced parquet.
    ruta_salida : Path
        Path to the PNG file.
    titulo_derecha : str
        Right-column title.
    dpi : int, optional
        Resolution.
    figsize_ancho : float, optional
        Figure width in inches.
    figsize_alto_por_fila : float, optional
        Height per subplot row.
    """
    X_imf, nombres_imf, Z, nombres_z, residuo_original, residuo_reducido = (
        cargar_pares_series(ruta_original, ruta_reducido)
    )
    T = X_imf.shape[0]
    n_imf = X_imf.shape[1]
    k = Z.shape[1]
    t = np.arange(T, dtype=np.float64)

    n_filas = n_imf + 1
    fig, axes = plt.subplots(
        n_filas,
        2,
        figsize=(figsize_ancho, figsize_alto_por_fila * n_filas + 0.8),
        sharex=True,
        constrained_layout=True,
    )
    fig.suptitle(
        "MSCI World (CEEMDAN): original oscillatory IMFs vs dimension-reduced channels",
        fontsize=12,
    )

    color_imf = "#1f77b4"
    color_z = "#d62728"
    color_r = "#2ca02c"

    for i in range(n_imf):
        axes[i, 0].plot(t, X_imf[:, i], color=color_imf, linewidth=0.6)
        axes[i, 0].set_ylabel(nombres_imf[i].replace("_", " "), fontsize=8)
        axes[i, 0].tick_params(axis="y", labelsize=7)
        axes[i, 0].grid(True, alpha=0.25)
        if i < k:
            axes[i, 1].plot(t, Z[:, i], color=color_z, linewidth=0.6)
            axes[i, 1].set_ylabel(nombres_z[i].replace("_", " "), fontsize=8)
        else:
            axes[i, 1].set_visible(False)
        axes[i, 1].tick_params(axis="y", labelsize=7)
        axes[i, 1].grid(True, alpha=0.25)

    axes[n_filas - 1, 0].plot(t, residuo_original, color=color_r, linewidth=0.7)
    axes[n_filas - 1, 0].set_ylabel("Residue", fontsize=8)
    axes[n_filas - 1, 0].set_xlabel("Time index", fontsize=9)
    axes[n_filas - 1, 0].grid(True, alpha=0.25)
    axes[n_filas - 1, 0].tick_params(axis="y", labelsize=7)

    axes[n_filas - 1, 1].plot(t, residuo_reducido, color=color_r, linewidth=0.7)
    axes[n_filas - 1, 1].set_ylabel("Residue (unchanged)", fontsize=8)
    axes[n_filas - 1, 1].set_xlabel("Time index", fontsize=9)
    axes[n_filas - 1, 1].grid(True, alpha=0.25)
    axes[n_filas - 1, 1].tick_params(axis="y", labelsize=7)

    axes[0, 0].set_title("Original CEEMDAN IMFs", fontsize=10)
    axes[0, 1].set_title(titulo_derecha, fontsize=10)

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure (full panel): %s (T=%d, IMF=%d, k=%d)", ruta_salida, T, n_imf, k)


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser(
        description="Figure: reduced series only (default) or IMF+Z+residue panel."
    )
    parser.add_argument("--parquet-original", type=Path, default=_DEFAULT_ORIG)
    parser.add_argument("--parquet-reducido", type=Path, default=_DEFAULT_RED)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument(
        "--titulo",
        type=str,
        default="MSCI World: dimension-reduced channels (FastICA, $k=4$)",
        help="Figure title (reduced-series-only mode).",
    )
    parser.add_argument(
        "--titulo-reduccion",
        type=str,
        default=r"Reduced channels (FastICA, $k=4$)",
        help="Right-column title (full panel mode).",
    )
    parser.add_argument(
        "--con-panel-completo",
        action="store_true",
        help="Generate 9×2 comparison with original IMFs and residue (requires original parquet).",
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.parquet_reducido.is_file():
        logger.error("Missing reduced parquet: %s", args.parquet_reducido)
        sys.exit(1)

    if args.con_panel_completo:
        if not args.parquet_original.is_file():
            logger.error("Full panel requires original parquet: %s", args.parquet_original)
            sys.exit(1)
        generar_figura_panel_completo(
            args.parquet_original,
            args.parquet_reducido,
            args.out,
            titulo_derecha=args.titulo_reduccion,
            dpi=args.dpi,
        )
    else:
        generar_figura_solo_reducidas(
            args.parquet_reducido,
            args.out,
            titulo=args.titulo,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
