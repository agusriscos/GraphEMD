#!/usr/bin/env python3
"""
Comprueba que las columnas ``Z_*`` de los parquets reducidos no estén
correlacionadas linealmente (Pearson y Spearman) entre sí.

El residuo se excluye de este cómputo: solo interesa la matriz de canales derivados.

Escribe un informe JSON agregado y opcionalmente imprime un resumen.

Ejemplo::

    PYTHONPATH=src/python python \\
        scripts/GraphEMD/exploracion/verificar_correlacion_series_reducidas.py
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

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_BASE = _REPO_ROOT / "docs" / "20abr26" / "out" / "imfs_dim_red"
_DEFAULT_OUT = _DEFAULT_BASE / "verificacion_correlacion_z.json"

logger = logging.getLogger(__name__)


def _columnas_z(df: pd.DataFrame) -> List[str]:
    """
    Devuelve nombres ``Z_*`` ordenados numéricamente.

    Parameters
    ----------
    df : pd.DataFrame
        Tabla con columnas ``Z_1``, ...

    Returns
    -------
    list of str
        Columnas ordenadas.
    """
    cols = [c for c in df.columns if re.match(r"^Z_\d+$", c)]
    return sorted(cols, key=lambda x: int(x.split("_")[1]))


def _metricas_matriz_corr(c: np.ndarray) -> Dict[str, Any]:
    """
    Estadísticos de la parte fuera de la diagonal de una matriz de correlación.

    Parameters
    ----------
    c : np.ndarray
        Matriz cuadrada simétrica (p. ej. Pearson).

    Returns
    -------
    dict
        Máximo valor absoluto fuera de diagonal, media, norma de Frobenius
        de la parte fuera de diagonal.
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
    Calcula correlaciones Pearson y Spearman entre columnas ``Z_*``.

    Parameters
    ----------
    ruta : Path
        Parquet con ``Z_1``, ...

    Returns
    -------
    dict
        Claves ``pearson``, ``spearman`` con matrices y métricas agregadas.
    """
    df = pd.read_parquet(ruta, engine="pyarrow")
    cols_z = _columnas_z(df)
    if len(cols_z) < 2:
        return {
            "ruta": str(ruta.resolve()),
            "n_z": len(cols_z),
            "advertencia": "Menos de dos columnas Z: correlación cruzada no aplica.",
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
    Lista todos los ``imfs_reducidas.parquet`` bajo ``base``.

    Parameters
    ----------
    base : Path
        Directorio raíz (p. ej. ``imfs_dim_red``).

    Returns
    -------
    list of Path
        Rutas encontradas.
    """
    if not base.is_dir():
        return []
    return sorted(base.rglob("imfs_reducidas.parquet"))


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser(
        description="Verifica correlación (Pearson/Spearman) entre columnas Z_* reducidas."
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=_DEFAULT_BASE,
        help="Directorio que contiene k3/, k4/, ... con fastica/ y pca/.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=_DEFAULT_OUT,
        help="Ruta del informe JSON agregado.",
    )
    parser.add_argument(
        "--umbral-pearson",
        type=float,
        default=1e-6,
        help="Máximo |ρ| de Pearson fuera de diagonal admitido (correlación lineal).",
    )
    parser.add_argument(
        "--umbral-spearman",
        type=float,
        default=0.15,
        help=(
            "Máximo |ρ| de Spearman fuera de diagonal admitido; puede ser >0 aunque "
            "Pearson sea ~0 (relaciones monótonas no lineales entre scores)."
        ),
    )
    parser.add_argument(
        "--fallar-si-spearman",
        action="store_true",
        help="Si se indica, el proceso termina con código 2 si falla el umbral Spearman.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parquets = recopilar_parquets(args.base)
    if not parquets:
        logger.error("No se encontró ningún imfs_reducidas.parquet bajo %s", args.base)
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
            "Se excluye Residuo. Pearson mide correlación lineal (FastICA/PCA con "
            "blanqueamiento suelen dejar Pearson ~0 entre Z_*). Spearman mide "
            "asociación monótona de rangos y puede ser distinta de cero sin "
            "contradecir ortogonalidad lineal."
        ),
        "resultados": resultados,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(informe, indent=2), encoding="utf-8")
    logger.info("Informe: %s", args.out_json.resolve())

    fallos_p = [r for r in resultados if "ok_pearson" in r and not r["ok_pearson"]]
    fallos_s = [r for r in resultados if "ok_spearman" in r and not r["ok_spearman"]]
    if fallos_p:
        logger.error(
            "%d archivos superan el umbral Pearson (correlación lineal).",
            len(fallos_p),
        )
        sys.exit(2)
    if args.fallar_si_spearman and fallos_s:
        logger.error(
            "%d archivos superan el umbral Spearman.", len(fallos_s)
        )
        sys.exit(2)
    if fallos_s:
        logger.warning(
            "%d archivos superan el umbral Spearman (informativo; Pearson OK).",
            len(fallos_s),
        )
    logger.info(
        "Pearson: todas las rutas cumplen max|ρ| <= %.1e.", args.umbral_pearson
    )


if __name__ == "__main__":
    main()
