"""
Script to apply independent component analysis (ICA) to CEEMDAN IMF components of XLE (Energy Select Sector SPDR Fund).

1. Load data required to apply independent component analysis (ICA) to IMF components of XLE.
2. Compute optimal ICA parameters (number of independent components, etc.) for XLE.
2. Apply independent component analysis (ICA) to XLE CEEMDAN IMF components with the computed optimal parameters.
3. Document implemented logic and results in this script.
4. Do not change any existing code for now

Implemented logic
---------------------
- **Load**: ``xle_imfs_ceemdan.parquet`` (8 oscillatory IMFs + ``Residuo``; the residual is not included in the ICA fit).
- **ICA block**: ``(T, p)`` matrix with 8 columns ``IMF_1``…``IMF_8``; column-wise standardization and **FastICA**
  (``sklearn``, ``algorithm=parallel``, ``whiten=unit-variance``, ``random_state=42``), aligned with
  ``reduce_ceemdan_imf_dimensionality.py`` and the MSCI block with ``k=4``.
- **Calibration of** ``k``: for each ``k ∈ {2,…,p−1}`` ICA is fit and
  ``max|corr(Z_i,Z_j)|``, Frobenius reconstruction error, and mean ``R²`` are recorded. Choose the
  **smallest** independent ``k`` with ``R²`` ≥ 72% of the maximum and bounded Frobenius error
  relative to the ``R²`` elbow (reference ``err(k_rodilla)``).
- **Final output**: sources ``Z_1…Z_k`` + unmixed ``Residuo``, ``modelo_ica.npz`` model,
  JSON metrics, and approximate IMF reconstruction.

Results obtained (run 2026-05-17, T=3587, p=8 IMF)
-------------------------------------------------------------

**Native IMFs:** ``max|corr|`` between IMFs ≈ 0.28 (IMF_2–IMF_3); mean |r| ≈ 0.059.

**Calibration:** grid ``k=2…7``; automatic selection by ``R²`` elbow + 90% threshold +
composite score (see ``detalle_seleccion`` in JSON).

**Applied output:** ``Z_1…Z_4`` + ``Residuo`` in ``ica/fastica/imfs_reducidas.parquet``;
``xle_ica_parametros.json`` with full grid.

**Figure:** ``figures/xle_componentes_ica_ceemdan.png`` (time panel of ``Z_j`` and residual).
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_EXPLORATION = _REPO_ROOT / "scripts" / "exploration"
_DIR_DATOS = _REPO_ROOT / "data" / "GraphEMD" / "xle_etf_analysis"
_RUTA_IMFS_XLE = _DIR_DATOS / "xle_imfs_ceemdan.parquet"
_DIR_SALIDA_ICA = _DIR_DATOS / "ica" / "fastica"
_RUTA_PARQUET_ICA = _DIR_SALIDA_ICA / "imfs_reducidas.parquet"
_RUTA_PARAMETROS_JSON = _DIR_DATOS / "xle_ica_parametros.json"
_RUTA_XLE = _DIR_DATOS / "xle.parquet"
_DIR_FIGURAS = _DIR_DATOS / "figures"
_RUTA_FIGURA_ICA = _DIR_FIGURAS / "xle_componentes_ica_ceemdan.png"

RANDOM_STATE: int = 42
K_MIN_CALIBRACION: int = 2
UMBRAL_CORR_Z_INDEPENDENCIA: float = 0.01
FRACCION_R2_OBJETIVO: float = 0.72
FACTOR_ERROR_FROBENIUS_RODILLA: float = 1.15

if str(_EXPLORATION) not in sys.path:
    sys.path.insert(0, str(_EXPLORATION))

from reduce_ceemdan_imf_dimensionality import (  # noqa: E402
    _correlation_metrics,
    fit_fastica,
    write_references,
    extract_imf_block_and_residual,
    save_ica_outputs,
    imf_reconstruction_metrics,
    reconstruct_imfs_from_ica_z,
    select_optimal_k_from_ica_grid,
)

logger = logging.getLogger(__name__)


def cargar_imfs_xle(ruta_imfs: Path = _RUTA_IMFS_XLE) -> pd.DataFrame:
    """
    Load the CEEMDAN IMF parquet for XLE.

    Parameters
    ----------
    ruta_imfs : Path
        Path to ``xle_imfs_ceemdan.parquet``.

    Returns
    -------
    pd.DataFrame
        Table with IMFs and residual.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not ruta_imfs.is_file():
        raise FileNotFoundError(
            f"Not found: {ruta_imfs}. Run 03_ceemdan_xle.py first."
        )
    df = pd.read_parquet(ruta_imfs, engine="pyarrow")
    logger.info("XLE IMFs: %d rows, columns %s", len(df), list(df.columns))
    return df


def evaluar_k_ica(
    X: np.ndarray,
    nombres_imf: list[str],
    k: int,
    random_state: int,
) -> dict[str, Any]:
    """
    Fit FastICA with ``k`` components and return independence and reconstruction metrics.

    Parameters
    ----------
    X : np.ndarray
        IMF block ``(T, p)``.
    nombres_imf : list[str]
        IMF column names.
    k : int
        Number of independent components.
    random_state : int
        FastICA seed.

    Returns
    -------
    dict
        Metrics for the calibration grid.
    """
    ica, scaler, z = fit_fastica(X, k, random_state)
    x_hat = reconstruct_imfs_from_ica_z(z, ica, scaler)
    rec = imf_reconstruction_metrics(X, x_hat, nombres_imf)
    corr = _correlation_metrics(z)
    return {
        "n_components": int(k),
        "max_abs_corr_Z": float(corr["max_abs_fuera_diagonal"]),
        "media_abs_corr_Z": float(corr["media_abs_fuera_diagonal"]),
        "error_relativo_frobenius": float(rec["error_relativo_frobenius"]),
        "rmse_global_reconstruccion": float(rec["rmse_global"]),
        "r2_medio_columnas": float(rec["r2_medio_columnas"]),
    }


def calibrar_n_components_ica(
    X: np.ndarray,
    nombres_imf: list[str],
    random_state: int = RANDOM_STATE,
    k_min: int = K_MIN_CALIBRACION,
) -> dict[str, Any]:
    """
    Sweep ``k`` and select the optimal number of independent components.

    Parameters
    ----------
    X : np.ndarray
        IMF block ``(T, p)``.
    nombres_imf : list[str]
        IMF labels.
    random_state : int
        FastICA random seed.
    k_min : int
        Minimum ``k`` to evaluate (``k < p``).

    Returns
    -------
    dict
        ``mejor``, ``detalle_seleccion``, ``rejilla`` blocks and native IMF correlation.
    """
    p = int(X.shape[1])
    if p < 2:
        raise ValueError("At least 2 IMFs are required to calibrate ICA.")
    if k_min < 1 or k_min >= p:
        raise ValueError(f"k_min={k_min} invalid for p={p} IMF.")

    rejilla: list[dict[str, Any]] = []
    for k in range(k_min, p):
        logger.info("ICA calibration: evaluating k=%d ...", k)
        rejilla.append(evaluar_k_ica(X, nombres_imf, k, random_state))

    df_rejilla = pd.DataFrame(rejilla)
    seleccion = select_optimal_k_from_ica_grid(
        df_rejilla,
        p=p,
        umbral_corr_z=UMBRAL_CORR_Z_INDEPENDENCIA,
        fraccion_r2_objetivo=FRACCION_R2_OBJETIVO,
        factor_error_frobenius_rodilla=FACTOR_ERROR_FROBENIUS_RODILLA,
    )
    mejor_k = int(seleccion["n_components"])
    corr_nativa = _correlation_metrics(X)

    return {
        "criterio_seleccion": seleccion["criterio_seleccion"],
        "p_imf_entrada": p,
        "k_min_evaluado": k_min,
        "k_max_evaluado": p - 1,
        "random_state": random_state,
        "mejor": {
            "n_components": mejor_k,
            **{
                k: float(v) if isinstance(v, (float, np.floating)) else v
                for k, v in seleccion.items()
                if k
                not in (
                    "n_components",
                    "criterio_seleccion",
                    "detalle_seleccion",
                )
            },
        },
        "detalle_seleccion": seleccion["detalle_seleccion"],
        "correlacion_imfs_nativas": corr_nativa,
        "rejilla": rejilla,
    }


def aplicar_ica_xle(
    X: np.ndarray,
    residuo: Optional[np.ndarray],
    nombres_imf: list[str],
    n_components: int,
    dir_salida: Path,
    meta: dict[str, Any],
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """
    Fit FastICA with ``n_components`` and persist parquet, model, and metrics.

    Parameters
    ----------
    X : np.ndarray
        Bloque IMF.
    residuo : np.ndarray or None
        Untransformed CEEMDAN residual.
    nombres_imf : list[str]
        Input IMF names.
    n_components : int
        Optimal ``k`` from calibration.
    dir_salida : Path
        ``ica/fastica/`` folder.
    meta : dict
        Metadata for the metrics JSON.
    random_state : int
        FastICA random seed.

    Returns
    -------
    dict
        Model, sources, and written paths.
    """
    if n_components >= X.shape[1]:
        raise ValueError(
            f"n_components={n_components} must be < number of IMFs ({X.shape[1]})."
        )
    ica, scaler, z = fit_fastica(X, n_components, random_state)
    save_ica_outputs(
        dir_salida,
        z,
        residuo,
        ica,
        scaler,
        nombres_imf,
        meta,
        X,
        guardar_parquet_recon=True,
    )
    return {
        "n_components": int(n_components),
        "forma_Z": list(z.shape),
        "dir_salida": str(dir_salida.resolve()),
        "parquet_reducido": str((dir_salida / "imfs_reducidas.parquet").resolve()),
        "modelo": str((dir_salida / "modelo_ica.npz").resolve()),
        "metricas": str((dir_salida / "metricas_reduccion.json").resolve()),
    }


def _columnas_z_ordenadas(df: pd.DataFrame) -> list[str]:
    """
    Return ``Z_1``, ``Z_2``, … in numeric order and ``Residuo`` at the end if present.

    Parameters
    ----------
    df : pd.DataFrame
        Table with ICA sources.

    Returns
    -------
    list[str]
        Sorted column names.
    """
    cols_z = sorted(
        [c for c in df.columns if c.startswith("Z_")],
        key=lambda x: int(x.split("_")[1]),
    )
    if "Residuo" in df.columns:
        cols_z.append("Residuo")
    return cols_z


def cargar_indice_temporal_xle(ruta_xle: Path = _RUTA_XLE) -> pd.Index:
    """
    Load the date index from ``xle.parquet`` for the time axis.

    Parameters
    ----------
    ruta_xle : Path
        price parquet XLE.

    Returns
    -------
    pd.Index
        Time index (dates).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not ruta_xle.is_file():
        raise FileNotFoundError(f"Not found: {ruta_xle}.")
    df = pd.read_parquet(ruta_xle, engine="pyarrow")
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index
    for columna in ("Date", "date", "Datetime"):
        if columna in df.columns:
            return pd.to_datetime(df[columna])
    return pd.RangeIndex(len(df))


def generar_panel_componentes_ica(
    df_componentes: pd.DataFrame,
    indice: pd.Index,
    ruta_salida: Path = _RUTA_FIGURA_ICA,
    k: Optional[int] = None,
    dpi: int = 150,
) -> Path:
    """
    Generate a stacked panel with ICA sources and the CEEMDAN residual.

    Parameters
    ----------
    df_componentes : pd.DataFrame
        Columns ``Z_1``…``Z_k`` and optionally ``Residuo``.
    indice : pd.Index
        Time axis (dates or integer index).
    ruta_salida : Path
        PNG path.
    k : int, optional
        Number of ICA components (title only).
    dpi : int
        Figure resolution.

    Returns
    -------
    Path
        Path of the saved file.
    """
    columnas = _columnas_z_ordenadas(df_componentes)
    if not columnas:
        raise ValueError("No Z_* or Residuo columns in the DataFrame.")
    if len(indice) != len(df_componentes):
        raise ValueError(
            f"Index length ({len(indice)}) differs from rows ({len(df_componentes)})."
        )

    n = len(columnas)
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, nombre in zip(axes, columnas):
        ax.plot(indice, df_componentes[nombre].values, linewidth=0.6, color="C0")
        ax.set_ylabel(nombre, fontsize=9)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Fecha")
    titulo_k = f"k={k}" if k is not None else f"k={len([c for c in columnas if c.startswith('Z_')])}"
    fig.suptitle(
        f"XLE: independent FastICA components ({titulo_k}) + CEEMDAN residual",
        fontsize=11,
        y=1.002,
    )
    fig.tight_layout()
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("ICA components figure: %s", ruta_salida)
    return ruta_salida


def generar_figura_desde_parquet(
    ruta_parquet: Path = _RUTA_PARQUET_ICA,
    ruta_xle: Path = _RUTA_XLE,
    ruta_salida: Path = _RUTA_FIGURA_ICA,
    k: Optional[int] = None,
) -> Path:
    """
    Load the reduced ICA parquet and write the PNG panel.

    Parameters
    ----------
    ruta_parquet : Path
        ``imfs_reducidas.parquet``.
    ruta_xle : Path
        prices XLE for the time index.
    ruta_salida : Path
        Output PNG.
    k : int, optional
        ICA components (title); if None, inferred from parquet.

    Returns
    -------
    Path
        Figure path.
    """
    if not ruta_parquet.is_file():
        raise FileNotFoundError(
            f"Not found: {ruta_parquet}. Run script 07 or the full ICA pipeline first."
        )
    df_z = pd.read_parquet(ruta_parquet, engine="pyarrow")
    indice = cargar_indice_temporal_xle(ruta_xle)
    if k is None:
        k = len([c for c in df_z.columns if c.startswith("Z_")])
    return generar_panel_componentes_ica(df_z, indice, ruta_salida, k=k)


def guardar_parametros_ica(
    calibracion: dict[str, Any],
    aplicacion: dict[str, Any],
    ruta_json: Path = _RUTA_PARAMETROS_JSON,
) -> None:
    """
    Write JSON with calibration and final application result.

    Parameters
    ----------
    calibracion : dict
        Output of :func:`calibrar_n_components_ica`.
    aplicacion : dict
        Output of :func:`aplicar_ica_xle`.
    ruta_json : Path
        Output JSON file.
    """
    payload = {
        "parquet_entrada": str(_RUTA_IMFS_XLE.resolve()),
        "calibracion": calibracion,
        "aplicacion": aplicacion,
    }
    ruta_json.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_json, "w", encoding="utf-8") as archivo:
        json.dump(payload, archivo, indent=2, ensure_ascii=False)
    logger.info("ICA parameters: %s", ruta_json)


def main(
    ruta_imfs: Optional[Path] = None,
    dir_salida: Optional[Path] = None,
    forzar_k: Optional[int] = None,
) -> dict[str, Any]:
    """
    Run calibration and FastICA application on XLE.

    Parameters
    ----------
    ruta_imfs : Path, optional
        CEEMDAN IMF parquet.
    dir_salida : Path, optional
        ``ica/fastica/`` directory.
    forzar_k : int, optional
        If set, skip automatic selection and use this ``k``.

    Returns
    -------
    dict
        Calibration, application, and paths.
    """
    ruta = ruta_imfs or _RUTA_IMFS_XLE
    salida = dir_salida or _DIR_SALIDA_ICA

    df = cargar_imfs_xle(ruta)
    x, nombres_imf, residuo = extract_imf_block_and_residual(df)

    calibracion = calibrar_n_components_ica(x, nombres_imf)
    k_optimo = int(forzar_k) if forzar_k is not None else int(calibracion["mejor"]["n_components"])
    logger.info("ICA k selected for XLE: %d", k_optimo)

    meta = {
        "activo": "XLE",
        "parquet_entrada": str(ruta.resolve()),
        "n_muestras_temporales": int(x.shape[0]),
        "columnas_imf_entrada": nombres_imf,
        "calibracion_mejor_k": calibracion["mejor"],
    }

    salida.parent.mkdir(parents=True, exist_ok=True)
    write_references(salida.parent)

    aplicacion = aplicar_ica_xle(
        x,
        residuo,
        nombres_imf,
        k_optimo,
        salida,
        meta,
    )
    guardar_parametros_ica(calibracion, aplicacion)

    ruta_figura = generar_figura_desde_parquet(
        ruta_parquet=salida / "imfs_reducidas.parquet",
        k=k_optimo,
    )

    return {
        "k_optimo": k_optimo,
        "calibracion": calibracion,
        "aplicacion": aplicacion,
        "ruta_parametros": str(_RUTA_PARAMETROS_JSON),
        "ruta_figura": str(ruta_figura),
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="FastICA on CEEMDAN IMFs of XLE with k calibration."
    )
    parser.add_argument(
        "--parquet-imfs",
        type=Path,
        default=None,
        help="IMF parquet (default xle_imfs_ceemdan.parquet).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="FastICA output directory.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Fix k and skip automatic calibration selection.",
    )
    parser.add_argument(
        "--solo-figura",
        action="store_true",
        help="Only generate the PNG panel from existing imfs_reducidas.parquet.",
    )
    parser.add_argument(
        "--figura",
        type=Path,
        default=None,
        help="PNG path (default figures/xle_componentes_ica_ceemdan.png).",
    )
    args = parser.parse_args()
    try:
        if args.solo_figura:
            ruta = generar_figura_desde_parquet(
                ruta_parquet=(args.out_dir or _DIR_SALIDA_ICA) / "imfs_reducidas.parquet",
                ruta_salida=args.figura or _RUTA_FIGURA_ICA,
                k=args.k,
            )
            logger.info("Figure generated: %s", ruta)
            sys.exit(0)
        resultado = main(
            ruta_imfs=args.parquet_imfs,
            dir_salida=args.out_dir,
            forzar_k=args.k,
        )
        logger.info(
            "Done. k=%d. Parameters: %s. Figura: %s",
            resultado["k_optimo"],
            resultado["ruta_parametros"],
            resultado["ruta_figura"],
        )
    except Exception:
        logger.exception("Error in ICA for XLE")
        sys.exit(1)
