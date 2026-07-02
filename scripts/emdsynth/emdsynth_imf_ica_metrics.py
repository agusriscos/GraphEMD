#!/usr/bin/env python3
"""
FastICA source correlation metrics applied to the synthetic IMF block.

Reads ``{escenario}_imfs_{metodo}.parquet`` files produced by
``run_emdsynth_decompositions.py``, takes the first ``min(8, n_imf)``
``IMF_*`` columns (same cap as the native pair summary), standardizes, and fits
FastICA with ``n_components = min(k_objetivo, p)`` and default ``k_objetivo=4``
(aligned with the rank-four MSCI block). The EMD residue is not included in the
fit or correlated pairs. Reuses ``_estadisticos_corr_pares_imfs`` from the
decomposition script for ``\\bar{\\rho}`` and ``\\pi_{0.05}`` among ``Z`` columns
(equivalent to correlating rows of ``Z.T``).

Example::

    PYTHONPATH=src/python python3 \\
        scripts/emdsynth/emdsynth_imf_ica_metrics.py \\
        --entrada scripts/emdsynth/out
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_ENTRADA = _REPO_ROOT / "scripts" / "emdsynth" / "out"

logger = logging.getLogger(__name__)


def _cargar_modulo_descomposiciones() -> Any:
    """
    Load ``run_emdsynth_decompositions`` as a module to reuse metrics.

    Returns
    -------
    module
        Module with ``_estadisticos_corr_pares_imfs``.
    """
    ruta = Path(__file__).resolve().parent / "run_emdsynth_decompositions.py"
    spec = importlib.util.spec_from_file_location("emdsynth_decomposiciones", ruta)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {ruta}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _columnas_imf_ordenadas(df: pd.DataFrame) -> List[str]:
    """
    Return ``IMF_*`` names sorted numerically.

    Parameters
    ----------
    df : pd.DataFrame
        Table with columns ``IMF_1``, ...

    Returns
    -------
    list of str
        Sorted list.
    """
    cols = [c for c in df.columns if c.startswith("IMF_")]
    if not cols:
        raise ValueError("No IMF_* columns in the parquet.")
    return sorted(cols, key=lambda x: int(x.split("_")[1]))


def extraer_bloque_imf_capado(
    df: pd.DataFrame, max_imf: int = 8
) -> Tuple[np.ndarray, bool, int]:
    """
    Extract matrix ``(T, p_use)`` with the first ``min(max_imf, n_imf)`` IMFs.

    Parameters
    ----------
    df : pd.DataFrame
        IMF parquet plus optional ``Residuo``.
    max_imf : int
        Intrinsic-mode cap aligned with the native summary.

    Returns
    -------
    X : np.ndarray
        Capped oscillatory block, shape ``(T, p_use)``.
    hay_residuo : bool
        True if a ``Residuo`` column exists.
    n_imf_total : int
        Number of ``IMF_*`` columns in the file (before capping).
    """
    nombres = _columnas_imf_ordenadas(df)
    n_total = len(nombres)
    use = min(max_imf, n_total)
    nombres_use = nombres[:use]
    X = np.asarray(df[nombres_use].values, dtype=np.float64)
    hay_residuo = "Residuo" in df.columns
    return X, hay_residuo, n_total


def ajustar_ica_fuentes(
    X: np.ndarray,
    random_state: int,
    n_components: int,
) -> np.ndarray:
    """
    Scale columns and fit FastICA with fixed ``n_components`` (rank reduction).

    Parameters
    ----------
    X : np.ndarray
        Capped IMFs, shape ``(T, p)``.
    random_state : int
        Seed for FastICA.
    n_components : int
        Output dimension ``k`` (must satisfy ``1 <= k <= p``).

    Returns
    -------
    np.ndarray
        Sources ``Z``, shape ``(T, k)``.
    """
    p = X.shape[1]
    if p < 1:
        raise ValueError("At least one IMF column is required.")
    k = int(n_components)
    if k < 1 or k > p:
        raise ValueError(f"n_components must be in [1, p]; received k={k}, p={p}.")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        ica = FastICA(
            n_components=k,
            algorithm="parallel",
            whiten="unit-variance",
            max_iter=8000,
            random_state=random_state,
            tol=1e-5,
        )
        return np.asarray(ica.fit_transform(Xs), dtype=np.float64)


def metricas_corr_tras_ica(
    mod: Any,
    Z: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, int, float, int]:
    """
    Mean |r| correlation, pair count, and fraction with p < alpha between sources.

    Parameters
    ----------
    mod : module
        Module with ``_estadisticos_corr_pares_imfs``.
    Z : np.ndarray
        ICA sources, shape ``(T, k)``.
    alpha : float
        p-value threshold (without multiple-comparison correction).

    Returns
    -------
    media_abs_r : float
        Mean |r_ij| on the upper triangle.
    n_pares : int
        Number of valid pairs.
    frac_p : float
        Fraction with p < alpha.
    k : int
        Number of sources (columns of ``Z``).
    """
    k = Z.shape[1]
    if k < 2:
        return 0.0, 0, float("nan"), k
    media_abs_r, n_pares, frac_p = mod._estadisticos_corr_pares_imfs(
        Z.T,
        max_filas=min(8, k),
        alpha=alpha,
    )
    return float(media_abs_r), int(n_pares), float(frac_p), k


def procesar_parquet(
    mod: Any,
    ruta: Path,
    random_state: int,
    max_imf: int,
    k_reduccion_objetivo: int,
) -> Dict[str, Any]:
    """
    Compute ICA metrics for a rank-reduced IMF parquet.

    Parameters
    ----------
    mod : module
        Decomposition module (correlation function).
    ruta : Path
        Path to the parquet ``*_imfs_*.parquet``.
    random_state : int
        FastICA seed.
    max_imf : int
        Maximum number of IMFs fed to ICA (first columns in order).
    k_reduccion_objetivo : int
        Desired output dimension (e.g. 4); effective ``k = min(k, p)``.

    Returns
    -------
    dict
        Scenario metadata, ``p_imf_entrada``, ``k_reduccion``, reported ``N``, metrics.
    """
    df = pd.read_parquet(ruta, engine="pyarrow")
    X, hay_residuo, n_imf_total = extraer_bloque_imf_capado(df, max_imf=max_imf)
    p_entrada = X.shape[1]
    k = min(int(k_reduccion_objetivo), p_entrada)
    Z = ajustar_ica_fuentes(X, random_state=random_state, n_components=k)
    media_abs_r, n_pares, frac_p, k_out = metricas_corr_tras_ica(mod, Z)
    if k_out != k:
        raise RuntimeError("Inconsistent ICA output dimension.")
    nombre = ruta.name
    # "{escenario}_imfs_{metodo}.parquet"
    base = nombre.replace("_imfs_", "|").replace(".parquet", "")
    partes = base.split("|", 1)
    escenario = partes[0]
    metodo = partes[1] if len(partes) > 1 else "unknown"
    n_reporte = k + (1 if hay_residuo else 0)
    return {
        "escenario": escenario,
        "metodo": metodo,
        "ruta_parquet": str(ruta.resolve()),
        "n_imf_archivo": int(n_imf_total),
        "p_imf_entrada_ica": int(p_entrada),
        "k_reduccion_ica": int(k),
        "k_reduccion_objetivo": int(k_reduccion_objetivo),
        "n_componentes_reporte": int(n_reporte),
        "hay_residuo": bool(hay_residuo),
        "corr_promedio_pares_ica": media_abs_r,
        "n_pares_corr": n_pares,
        "frac_pares_p_lt_005": frac_p,
    }


def listar_parquets_imfs(dir_entrada: Path) -> List[Path]:
    """
    List ``*_imfs_*.parquet`` files under ``dir_entrada``.

    Parameters
    ----------
    dir_entrada : Path
        Output directory from ``run_emdsynth_decompositions``.

    Returns
    -------
    list of Path
        Sorted paths.
    """
    if not dir_entrada.is_dir():
        return []
    out = sorted(dir_entrada.glob("*_imfs_*.parquet"))
    return out


def main(argv: Optional[List[str]] = None) -> None:
    """
    CLI: process all IMF parquets and write JSON plus a console summary.

    Parameters
    ----------
    argv : list of str, optional
        Arguments (default ``sys.argv[1:]``).
    """
    p = argparse.ArgumentParser(
        description="FastICA source correlation on synthetic IMFs (emdsynth)."
    )
    p.add_argument(
        "--entrada",
        type=Path,
        default=_DEFAULT_ENTRADA,
        help="Directory with parquets *_imfs_*.parquet.",
    )
    p.add_argument(
        "--salida-json",
        type=Path,
        default=None,
        help="JSON path (default: entrada/metricas_ica_emdsynth.json).",
    )
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--max-imf",
        type=int,
        default=8,
        help="Maximum number of IMFs fed to ICA; aligned with the native cap.",
    )
    p.add_argument(
        "--k-reduccion",
        type=int,
        default=4,
        help="Target FastICA output dimension (k); effective min(k, p) with p IMF columns.",
    )
    p.add_argument(
        "--latex",
        action="store_true",
        help="Print tabular rows ready to paste into LaTeX.",
    )
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    dir_in = args.entrada.resolve()
    rutas = listar_parquets_imfs(dir_in)
    if not rutas:
        logger.error(
            "No *_imfs_*.parquet parquets found in %s. "
            "Run run_emdsynth_decompositions.py first.",
            dir_in,
        )
        sys.exit(1)

    mod = _cargar_modulo_descomposiciones()
    filas: List[Dict[str, Any]] = []
    for ruta in rutas:
        try:
            fila = procesar_parquet(
                mod,
                ruta,
                random_state=args.random_state,
                max_imf=args.max_imf,
                k_reduccion_objetivo=args.k_reduccion,
            )
            filas.append(fila)
            logger.info(
                "%s %s: p=%s k=%s N=%s rho_bar=%.4e pi=%.2f",
                fila["escenario"],
                fila["metodo"],
                fila["p_imf_entrada_ica"],
                fila["k_reduccion_ica"],
                fila["n_componentes_reporte"],
                fila["corr_promedio_pares_ica"],
                fila["frac_pares_p_lt_005"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failure in %s: %s", ruta, exc)
            sys.exit(1)

    out_json = (
        args.salida_json
        if args.salida_json is not None
        else dir_in / "metricas_ica_emdsynth.json"
    )
    payload = {
        "random_state": args.random_state,
        "max_imf_entrada": args.max_imf,
        "k_reduccion_objetivo": args.k_reduccion,
        "filas": filas,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("JSON saved to %s", out_json)

    if args.latex:
        _imprimir_latex_tabla(filas)


def _imprimir_latex_tabla(filas: List[Dict[str, Any]]) -> None:
    """
    Print LaTeX rows ordered like ``tab:synthetic_emd_summary``.

    Parameters
    ----------
    filas : list of dict
        Output of ``procesar_parquet`` for all scenarios and methods.
    """
    orden_escenarios = [
        "frecuencias_cercanas",
        "chirp_lineal",
        "burst_sobre_portadora",
        "combinado_ruido",
        "superposicion_multicomponente",
    ]
    orden_metodos = ["emd", "eemd", "ceemdan", "vmd"]
    idx = {(f["escenario"], f["metodo"]): f for f in filas}
    etiquetas = [
        "Close tones",
        "Linear chirp",
        "Burst on carrier",
        "Close tones + chirp + noise",
        "MULTI",
    ]
    for esc, lab in zip(orden_escenarios, etiquetas):
        partes = []
        for met in orden_metodos:
            f = idx.get((esc, met))
            if f is None:
                partes.extend(["?", "?", "?"])
                continue
            rho = f["corr_promedio_pares_ica"]
            pi = f["frac_pares_p_lt_005"]
            nrep = f["n_componentes_reporte"]
            if np.isnan(pi):
                pi_s = "---"
            else:
                pi_s = f"{pi:.2f}"
            partes.append(str(nrep))
            partes.append(f"{rho:.3f}")
            partes.append(pi_s)
        print(f"{lab} & " + " & ".join(partes) + r" \\")


if __name__ == "__main__":
    main()
