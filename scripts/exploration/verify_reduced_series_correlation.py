#!/usr/bin/env python3
"""
Verify that ``Z_*`` columns in reduced parquets are not linearly correlated
(Pearson and Spearman) with each other.

The residue is excluded from this computation: only the derived-channel matrix matters.

Writes an aggregated JSON report and optionally prints a summary.

Example::

    PYTHONPATH=src/python python \
        scripts/exploration/verify_reduced_series_correlation.py
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_BASE = _REPO_ROOT / "docs" / "20abr26" / "out" / "imfs_dim_red"
_DEFAULT_OUT = _DEFAULT_BASE / "verificacion_correlacion_z.json"

logger = logging.getLogger(__name__)


def _columnas_z(df: pd.DataFrame) -> List[str]:
    """
    Return ``Z_*`` names sorted numerically.

    Parameters
    ----------
    df : pd.DataFrame
        Table with columns ``Z_1``, ...

    Returns
    -------
    list of str
        Sorted columns.
    """
    cols = [c for c in df.columns if re.match(r"^Z_\d+$", c)]
    return sorted(cols, key=lambda x: int(x.split("_")[1]))


def _metricas_matriz_corr(c: np.ndarray) -> Dict[str, Any]:
    """
    Statistics for the off-diagonal part of a correlation matrix.

    Parameters
    ----------
    c : np.ndarray
        Square symmetric matrix (e.g. Pearson).

    Returns
    -------
    dict
        Maximum absolute off-diagonal value, mean, and Frobenius norm
        of the off-diagonal part.
    """
    n = c.shape[0]
    if n < 2:
        return {
            "max_abs_fuera_diagonal": 0.0,
            "media_abs_fuera_diagonal": 0.0,
            "frobenius_offdiag": 0.0,
        }
    mask = ~np.eye(n, dtype=bool)
    off = c[mask]
    return {
        "max_abs_fuera_diagonal": float(np.max(np.abs(off))),
        "media_abs_fuera_diagonal": float(np.mean(np.abs(off))),
        "frobenius_offdiag": float(np.linalg.norm(off)),
    }


def analizar_parquet_reducido(ruta: Path) -> Dict[str, Any]:
    """
    Compute Pearson and Spearman correlations among ``Z_*`` columns.

    Parameters
    ----------
    ruta : Path
        Parquet with ``Z_1``, ...

    Returns
    -------
    dict
        Keys ``pearson``, ``spearman`` with matrices and aggregate metrics.
    """
    df = pd.read_parquet(ruta, engine="pyarrow")
    cols_z = _columnas_z(df)
    if len(cols_z) < 2:
        return {
            "ruta": str(ruta.resolve()),
            "n_z": len(cols_z),
            "advertencia": "Fewer than two Z columns: cross-correlation does not apply.",
        }
    Z = np.asarray(df[cols_z].values, dtype=np.float64)
    pearson = np.corrcoef(Z.T)
    spearman = stats.spearmanr(Z, axis=0).correlation
    if spearman is None:
        spearman = np.eye(len(cols_z))
    spearman = np.asarray(spearman, dtype=np.float64)

    return {
        "ruta": str(ruta.resolve()),
        "n_muestras": int(Z.shape[0]),
        "columnas_z": cols_z,
        "pearson": {
            "matriz": np.round(pearson, 12).tolist(),
            **_metricas_matriz_corr(pearson),
        },
        "spearman": {
            "matriz": np.round(spearman, 12).tolist(),
            **_metricas_matriz_corr(spearman),
        },
    }


def recopilar_parquets(base: Path) -> List[Path]:
    """
    List all ``imfs_reducidas.parquet`` files under ``base``.

    Parameters
    ----------
    base : Path
        Root directory (p. ej. ``imfs_dim_red``).

    Returns
    -------
    list of Path
        Paths found.
    """
    if not base.is_dir():
        return []
    return sorted(base.rglob("imfs_reducidas.parquet"))


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser(
        description="Verify Pearson/Spearman correlation among reduced Z_* columns."
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=_DEFAULT_BASE,
        help="Directory containing k3/, k4/, ... with fastica/ and pca/.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=_DEFAULT_OUT,
        help="Path of the aggregated JSON report.",
    )
    parser.add_argument(
        "--umbral-pearson",
        type=float,
        default=1e-6,
        help="Maximum allowed off-diagonal |ρ| for Pearson (linear correlation).",
    )
    parser.add_argument(
        "--umbral-spearman",
        type=float,
        default=0.15,
        help=(
            "Maximum allowed off-diagonal Spearman |ρ|; may be >0 even when "
            "Pearson is ~0 (nonlinear monotonic relationships between scores)."
        ),
    )
    parser.add_argument(
        "--fallar-si-spearman",
        action="store_true",
        help="If set, exit with code 2 when the Spearman threshold fails.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parquets = recopilar_parquets(args.base)
    if not parquets:
        logger.error("No imfs_reducidas.parquet found under %s", args.base)
        sys.exit(1)

    resultados: List[Dict[str, Any]] = []
    for p in parquets:
        bloque = analizar_parquet_reducido(p)
        if "pearson" in bloque:
            mx_p = bloque["pearson"]["max_abs_fuera_diagonal"]
            mx_s = bloque["spearman"]["max_abs_fuera_diagonal"]
            bloque["ok_pearson"] = bool(mx_p <= args.umbral_pearson)
            bloque["ok_spearman"] = bool(mx_s <= args.umbral_spearman)
            rel = p.relative_to(args.base) if p.is_relative_to(args.base) else p
            logger.info(
                "%s | Pearson max|ρ|=%.3e (ok=%s) Spearman max|ρ|=%.4f (ok=%s)",
                rel,
                mx_p,
                bloque["ok_pearson"],
                mx_s,
                bloque["ok_spearman"],
            )
        resultados.append(bloque)

    informe = {
        "umbral_pearson_max_abs": args.umbral_pearson,
        "umbral_spearman_max_abs": args.umbral_spearman,
        "nota": (
            "Residuo is excluded. Pearson measures linear correlation (FastICA/PCA with "
            "whitening usually leave Pearson ~0 between Z_*). Spearman measures "
            "monotonic rank association and may be non-zero without "
            "contradicting linear orthogonality."
        ),
        "resultados": resultados,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(informe, indent=2), encoding="utf-8")
    logger.info("Report: %s", args.out_json.resolve())

    fallos_p = [r for r in resultados if "ok_pearson" in r and not r["ok_pearson"]]
    fallos_s = [r for r in resultados if "ok_spearman" in r and not r["ok_spearman"]]
    if fallos_p:
        logger.error(
            "%d files exceed the Pearson threshold (linear correlation).",
            len(fallos_p),
        )
        sys.exit(2)
    if args.fallar_si_spearman and fallos_s:
        logger.error(
            "%d files exceed the Spearman threshold.", len(fallos_s)
        )
        sys.exit(2)
    if fallos_s:
        logger.warning(
            "%d files exceed the Spearman threshold (informational; Pearson OK).",
            len(fallos_s),
        )
    logger.info(
        "Pearson: all paths satisfy max|ρ| <= %.1e.", args.umbral_pearson
    )


if __name__ == "__main__":
    main()
