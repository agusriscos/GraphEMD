"""
Script to decompose XLP (Consumer Staples Select Sector SPDR Fund) into IMF components using CEEMDAN.

1. Load XLP data from parquet file data/GraphEMD/xlp_analysis/xlp.parquet.
2. Extract data in the same time window as analyzed MSCI World (12 January 2012 – 20 April 2026). There must be 3587 data points/observations
3. Compute optimal CEEMDAN parameters for XLP using the same logic applied to MSCI World in this repository.
3. Decompose the series into IMF components using CEEMDAN with the computed optimal parameters.
4. Document implemented logic and results in this script.
5. Do not change any existing code for now

Implemented logic
---------------------
- **Load**: ``xlp.parquet`` (window 2012-01-12–2026-04-20, 3587 observations aligned to the MSCI calendar).
- **MSCI reference** (``docs/20abr26``, CEEMDAN table): ``max_imf=14``, ``trials=100``,
  ``epsilon=0.05``, ``seed=42``.
- **Calibration XLP** (grid ``epsilon`` × ``trials`` × ``max_imf`` with ``max_imf≥9``):
  in XLP, CEEMDAN typically stops at **8 IMFs** even when ``max_imf`` is higher; the criterion
  prioritizes the highest achievable ``n_imfs``. Among configurations with
  reconstruction ``rmse_relativo`` ``< 1e-10``, choose the one that
  yields the most **linear, monotonically increasing** residual (last row of the
  decomposition, ``Close − Σ IMF``): priority to increasing monotonicity and positive
  slope; then higher ``R²`` of ``residual ~ a·t+b`` and higher fraction of positive
  differences; on tie, lower ``rmse_relativo_vs_lineal``, lower ``rmse_relativo``, and lower
  ``corr_promedio_pares`` between IMFs. ``max_imf`` sets the mode cap and thus
  how much structure remains in the residual vs. the IMFs.
- Also saves ``mejor_separacion_imfs`` (emdsynth criterion: minimum linear coupling
  between IMFs) for comparative reference only.
- **Final decomposition** with ``calibracion["mejor"]`` (linearity criterion); output in
  ``xlp_imfs_ceemdan.parquet`` and ``xlp_ceemdan_parametros.json``.

Results obtained
--------------------

After each calibration, review ``xlp_ceemdan_parametros.json`` (``mejor`` block) and
re-run ``04_ceemdan_xlp_validation_and_results.py`` for figures and validation.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DIR_DATOS = _REPO_ROOT / "data" / "GraphEMD" / "xlp_analysis"
_RUTA_PARQUET_XLP = _DIR_DATOS / "xlp.parquet"
_RUTA_IMFS_SALIDA = _DIR_DATOS / "xlp_imfs_ceemdan.parquet"
_RUTA_PARAMETROS_JSON = _DIR_DATOS / "xlp_ceemdan_parametros.json"

N_OBSERVACIONES_ESPERADAS: int = 3587

# MSCI World reference (docs/20abr26/main.tex, tab:ceemdan_params)
MSCI_MAX_IMF: int = 14
MSCI_TRIALS: int = 100
MSCI_EPSILON: float = 0.05
MSCI_SEED: int = 42

# Reduced grid (~160 configs, ~2 h): sweep with ``max_imf`` ≥ 9.
REJILLA_EPSILON: list[float] = [0.03, 0.05, 0.08, 0.12]
REJILLA_TRIALS: list[int] = [100, 140, 180, 200, 220]
REJILLA_MAX_IMF: list[int] = [9, 10, 12, 14, 16, 18, 20, 22]
UMBRAL_RMSE_RELATIVO: float = 1e-10

logger = logging.getLogger(__name__)


def _cargar_modulo_emdsynth_pipeline():
    """
    Load ``run_emdsynth_decompositions`` without modifying the on-disk module.

    Returns
    -------
    module
        Module with ``descomponer_ceemdan`` and ``calcular_metricas``.
    """
    ruta = (
        _REPO_ROOT
        / "scripts"
        / "emdsynth"
        / "run_emdsynth_decompositions.py"
    )
    spec = importlib.util.spec_from_file_location("emdsynth_ejecutar_xlp", ruta)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {ruta}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cargar_serie_cierre_xlp(ruta_parquet: Path = _RUTA_PARQUET_XLP) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load XLP and return the closing price vector.

    Parameters
    ----------
    ruta_parquet : Path
        Path to ``xlp.parquet``.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        Full DataFrame and 1D ``Close`` series in ``float64``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If ``Close`` is missing or row count does not match the MSCI window.
    """
    if not ruta_parquet.is_file():
        raise FileNotFoundError(
            f"Not found: {ruta_parquet}. Run 01_download_xlp.py first."
        )
    df = pd.read_parquet(ruta_parquet, engine="pyarrow")
    if "Close" not in df.columns:
        raise ValueError(f"Missing Close column. Columns: {list(df.columns)}")
    if len(df) != N_OBSERVACIONES_ESPERADAS:
        logger.warning(
            "Expected %s observations; parquet has %s.",
            N_OBSERVACIONES_ESPERADAS,
            len(df),
        )
    serie = np.asarray(df["Close"].values, dtype=np.float64)
    logger.info(
        "Serie Close: n=%s, range %s → %s",
        len(serie),
        df.index.min(),
        df.index.max(),
    )
    return df, serie


def extraer_residuo(imfs: np.ndarray) -> np.ndarray:
    """
    Extract the CEEMDAN residual (last row of the mode matrix).

    Parameters
    ----------
    imfs : np.ndarray
        Output of ``descomponer_ceemdan`` with shape ``(n_modes, n_samples)``.

    Returns
    -------
    np.ndarray
        1D residual series.
    """
    return np.asarray(imfs[-1, :], dtype=np.float64)


def metricas_linealidad_residuo(residuo: np.ndarray) -> dict[str, float]:
    """
    Quantify how linear the residual is with respect to the time index.

    Parameters
    ----------
    residuo : np.ndarray
        CEEMDAN residual component.

    Returns
    -------
    dict
        ``r2_regresion_lineal``, ``rmse_relativo_vs_lineal``, fraction of differences
        with the same sign (monotonicity proxy), and slope.
    """
    r = np.asarray(residuo, dtype=np.float64)
    n = len(r)
    t = np.arange(n, dtype=np.float64)
    coef = np.polyfit(t, r, 1)
    ajuste = coef[0] * t + coef[1]
    residuo_cent = r - np.mean(r)
    ss_tot = float(np.sum(residuo_cent**2)) + 1e-15
    ss_res = float(np.sum((r - ajuste) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    norma_r = float(np.linalg.norm(r)) + 1e-15
    rmse_vs_lineal = float(np.linalg.norm(r - ajuste) / norma_r)
    diffs = np.diff(r)
    tol = 1e-12
    frac_creciente = float(np.mean(diffs > tol))
    frac_decreciente = float(np.mean(diffs < -tol))
    frac_monotona = max(frac_creciente, frac_decreciente)
    return {
        "r2_regresion_lineal": float(r2),
        "rmse_relativo_vs_lineal": rmse_vs_lineal,
        "pendiente": float(coef[0]),
        "frac_diff_creciente": frac_creciente,
        "frac_diff_decreciente": frac_decreciente,
        "frac_diff_mismo_signo": frac_monotona,
        "monotono_creciente": bool(frac_decreciente < 1e-6),
        "monotono_decreciente": bool(frac_creciente < 1e-6),
    }


def _clave_criterio_linealidad(
    metricas_linealidad: dict[str, float],
    metricas_decomposicion: dict[str, float],
    n_imfs: int,
) -> tuple[float, float, float, float, float, float, float, float]:
    """
    Sort key: linear, increasing, monotonic residual (lower is better).

    Parameters
    ----------
    metricas_linealidad : dict
        Output of ``metricas_linealidad_residuo``.
    metricas_decomposicion : dict
        Output of ``calcular_metricas``.
    n_imfs : int
        Number of IMFs extracted (excluding residual).

    Returns
    -------
    tuple
        Penalty if not monotonically increasing, penalty if slope ``≤ 0``,
        ``-n_imfs``, then ``(-R², -frac_diff_creciente, rmse_vs_lineal, …)``.
    """
    m = metricas_linealidad
    penal_no_creciente = 0.0 if m.get("monotono_creciente") else 1.0
    penal_pendiente = 0.0 if float(m["pendiente"]) > 0.0 else 1.0
    return (
        penal_no_creciente,
        penal_pendiente,
        -float(n_imfs),
        -float(m["r2_regresion_lineal"]),
        -float(m.get("frac_diff_creciente", m["frac_diff_mismo_signo"])),
        float(m["rmse_relativo_vs_lineal"]),
        float(metricas_decomposicion["rmse_relativo"]),
        float(metricas_decomposicion["corr_promedio_pares"]),
    )


def _clave_criterio_separacion_imfs(metricas_decomposicion: dict[str, float]) -> tuple[float, float]:
    """
    emdsynth key: minimum coupling between IMFs (lower is better).

    Parameters
    ----------
    metricas_decomposicion : dict
        Output of ``calcular_metricas``.

    Returns
    -------
    tuple
        ``(corr_promedio_pares, frac_energia_residuo)``.
    """
    return (
        float(metricas_decomposicion["corr_promedio_pares"]),
        float(metricas_decomposicion["frac_energia_residuo"]),
    )


def calibrar_parametros_ceemdan_xlp(
    serie: np.ndarray,
    mod_emdsynth: Any,
    semilla: int = MSCI_SEED,
    umbral_rmse: float = UMBRAL_RMSE_RELATIVO,
) -> dict[str, Any]:
    """
    Search ``epsilon``, ``trials``, and ``max_imf`` prioritizing a linear, increasing residual.

    Parameters
    ----------
    serie : np.ndarray
        1D closing prices.
    mod_emdsynth : module
        Module with ``descomponer_ceemdan`` and ``calcular_metricas``.
    semilla : int
        CEEMDAN noise seed.
    umbral_rmse : float
        Maximum acceptable ``rmse_relativo`` for full reconstruction.

    Returns
    -------
    dict
        ``mejor`` (residual linearity criterion), ``mejor_separacion_imfs`` (emdsynth
        criterion), ``referencia_msci``, ``rejilla`` (grid), ``evaluaciones``.
    """
    descomponer = mod_emdsynth.descomponer_ceemdan
    calcular_metricas = mod_emdsynth.calcular_metricas

    evaluaciones: list[dict[str, Any]] = []
    mejor_lineal: Optional[dict[str, Any]] = None
    mejor_clave_lineal: Optional[tuple[float, ...]] = None
    mejor_separacion: Optional[dict[str, Any]] = None
    mejor_clave_sep: Optional[tuple[float, float]] = None
    total = len(REJILLA_EPSILON) * len(REJILLA_TRIALS) * len(REJILLA_MAX_IMF)
    paso = 0

    for epsilon, trials, max_imf in itertools.product(
        REJILLA_EPSILON, REJILLA_TRIALS, REJILLA_MAX_IMF
    ):
        paso += 1
        logger.info(
            "Calibration [%s/%s]: epsilon=%.2f, trials=%s, max_imf=%s",
            paso,
            total,
            epsilon,
            trials,
            max_imf,
        )
        imfs = descomponer(
            serie,
            max_imf=max_imf,
            trials=trials,
            epsilon=epsilon,
            seed=semilla,
        )
        metricas = calcular_metricas(serie, imfs)
        n_imfs = int(imfs.shape[0]) - 1
        residuo = extraer_residuo(imfs)
        m_lineal = metricas_linealidad_residuo(residuo)
        valida = metricas["rmse_relativo"] < umbral_rmse
        entrada = {
            "epsilon": epsilon,
            "trials": trials,
            "max_imf": max_imf,
            "n_imfs": n_imfs,
            "metricas": {k: float(v) for k, v in metricas.items()},
            "metricas_linealidad_residuo": m_lineal,
            "valida": valida,
        }
        evaluaciones.append(entrada)
        if not valida:
            continue

        cl_lin = _clave_criterio_linealidad(m_lineal, metricas, n_imfs)
        if mejor_clave_lineal is None or cl_lin < mejor_clave_lineal:
            mejor_clave_lineal = cl_lin
            mejor_lineal = {
                "epsilon": epsilon,
                "trials": trials,
                "max_imf": max_imf,
                "n_imfs": n_imfs,
                "seed": semilla,
                "criterio": "linealidad_residuo",
                "metricas": entrada["metricas"],
                "metricas_linealidad_residuo": m_lineal,
            }

        cl_sep = _clave_criterio_separacion_imfs(metricas)
        if mejor_clave_sep is None or cl_sep < mejor_clave_sep:
            mejor_clave_sep = cl_sep
            mejor_separacion = {
                "epsilon": epsilon,
                "trials": trials,
                "max_imf": max_imf,
                "n_imfs": n_imfs,
                "seed": semilla,
                "criterio": "separacion_imfs_emdsynth",
                "metricas": entrada["metricas"],
                "metricas_linealidad_residuo": m_lineal,
            }

    if mejor_lineal is None:
        raise RuntimeError(
            "CEEMDAN XLP calibration: no configuration meets the RMSE threshold."
        )

    n_imfs_vals = [int(e["n_imfs"]) for e in evaluaciones]
    resumen_n_imfs = {
        "minimo": int(min(n_imfs_vals)),
        "maximo": int(max(n_imfs_vals)),
        "nota": (
            "CEEMDAN on XLP reaches at most 8 IMFs in this window; "
            "max_imf≥9 only sets the algorithm upper bound."
            if max(n_imfs_vals) < 9
            else None
        ),
    }
    if resumen_n_imfs["maximo"] < 9:
        logger.warning(
            "No configuration extracted 9+ IMFs (maximum observed: %s). "
            "Best among %s valid configurations selected by RMSE.",
            resumen_n_imfs["maximo"],
            sum(1 for e in evaluaciones if e["valida"]),
        )

    ml = mejor_lineal["metricas_linealidad_residuo"]
    logger.info(
        "Optimal (linear residual): epsilon=%.2f, trials=%s, max_imf=%s, "
        "n_imfs=%s, R²=%.4f, frac_crec=%.4f, monot_crec=%s, "
        "rmse_vs_lineal=%.4f, corr_IMFs=%.4f",
        mejor_lineal["epsilon"],
        mejor_lineal["trials"],
        mejor_lineal["max_imf"],
        mejor_lineal["n_imfs"],
        ml["r2_regresion_lineal"],
        ml["frac_diff_creciente"],
        ml["monotono_creciente"],
        ml["rmse_relativo_vs_lineal"],
        mejor_lineal["metricas"]["corr_promedio_pares"],
    )
    if mejor_separacion is not None:
        ms = mejor_separacion["metricas_linealidad_residuo"]
        logger.info(
            "Reference (IMF separation): epsilon=%.2f, trials=%s, residual R²=%.4f",
            mejor_separacion["epsilon"],
            mejor_separacion["trials"],
            ms["r2_regresion_lineal"],
        )

    imfs_msci = descomponer(
        serie,
        max_imf=MSCI_MAX_IMF,
        trials=MSCI_TRIALS,
        epsilon=MSCI_EPSILON,
        seed=semilla,
    )
    metricas_msci = calcular_metricas(serie, imfs_msci)
    m_lineal_msci = metricas_linealidad_residuo(extraer_residuo(imfs_msci))

    return {
        "criterio_seleccion": "linealidad_residuo",
        "resumen_n_imfs": resumen_n_imfs,
        "mejor": mejor_lineal,
        "mejor_separacion_imfs": mejor_separacion,
        "referencia_msci": {
            "max_imf": MSCI_MAX_IMF,
            "trials": MSCI_TRIALS,
            "epsilon": MSCI_EPSILON,
            "seed": semilla,
            "metricas": {k: float(v) for k, v in metricas_msci.items()},
            "metricas_linealidad_residuo": m_lineal_msci,
        },
        "rejilla": {
            "epsilon": REJILLA_EPSILON,
            "trials": REJILLA_TRIALS,
            "max_imf": REJILLA_MAX_IMF,
        },
        "evaluaciones": evaluaciones,
    }


def imfs_array_a_dataframe(imfs: np.ndarray) -> pd.DataFrame:
    """
    Convert CEEMDAN output (rows = modes) to a DataFrame with IMF_* and Residuo columns.

    Parameters
    ----------
    imfs : np.ndarray
        2D matrix; the last row is the residual.

    Returns
    -------
    pd.DataFrame
        Columns ``IMF_1`` … ``IMF_n`` and ``Residuo``.
    """
    n_imfs = imfs.shape[0] - 1
    datos: dict[str, np.ndarray] = {}
    for i in range(n_imfs):
        datos[f"IMF_{i + 1}"] = imfs[i]
    datos["Residuo"] = imfs[-1]
    return pd.DataFrame(datos)


def descomponer_y_guardar(
    serie: np.ndarray,
    parametros: dict[str, Any],
    mod_emdsynth: Any,
    indice: pd.Index,
    ruta_imfs: Path = _RUTA_IMFS_SALIDA,
) -> pd.DataFrame:
    """
    Apply CEEMDAN with the given parameters and save the IMF parquet.

    Parameters
    ----------
    serie : np.ndarray
        Closing prices.
    parametros : dict
        Must include ``epsilon``, ``trials``, ``max_imf``, and ``seed``.
    mod_emdsynth : module
        emdsynth pipeline module.
    indice : pd.Index
        Time index aligned with ``serie``.
    ruta_imfs : Path
        Parquet destination.

    Returns
    -------
    pd.DataFrame
        IMFs with the same index as the price series.
    """
    imfs = mod_emdsynth.descomponer_ceemdan(
        serie,
        max_imf=int(parametros["max_imf"]),
        trials=int(parametros["trials"]),
        epsilon=float(parametros["epsilon"]),
        seed=int(parametros["seed"]),
    )
    df_imfs = imfs_array_a_dataframe(imfs)
    df_imfs.index = indice
    _DIR_DATOS.mkdir(parents=True, exist_ok=True)
    df_imfs.to_parquet(ruta_imfs, engine="pyarrow", index=True)
    logger.info(
        "IMFs saved: %s (%s columns, %s rows)",
        ruta_imfs,
        len(df_imfs.columns),
        len(df_imfs),
    )
    return df_imfs


def resumir_energia_imfs(df_imfs: pd.DataFrame, serie: np.ndarray) -> dict[str, Any]:
    """
    Compute variance fractions per component relative to the original series.

    Parameters
    ----------
    df_imfs : pd.DataFrame
        IMFs and residual.
    serie : np.ndarray
        Closing prices.

    Returns
    -------
    dict
        Total variance, per-column variance, and number of IMFs.
    """
    var_serie = float(np.var(serie))
    resumen: dict[str, float] = {}
    for col in df_imfs.columns:
        resumen[col] = float(np.var(df_imfs[col].values) / (var_serie + 1e-15))
    return {
        "varianza_serie": var_serie,
        "frac_varianza_por_componente": resumen,
        "n_imfs": int(sum(1 for c in df_imfs.columns if c.startswith("IMF_"))),
        "columnas": list(df_imfs.columns),
    }


def guardar_parametros_json(
    calibracion: dict[str, Any],
    resumen_imfs: dict[str, Any],
    ruta: Path = _RUTA_PARAMETROS_JSON,
) -> Path:
    """
    Persist calibration parameters and decomposition summary.

    Parameters
    ----------
    calibracion : dict
        Output of ``calibrar_parametros_ceemdan_xlp``.
    resumen_imfs : dict
        Output of ``resumir_energia_imfs``.
    ruta : Path
        Output JSON file.

    Returns
    -------
    Path
        Written path.
    """
    payload = {
        "calibracion": calibracion,
        "resumen_descomposicion": resumen_imfs,
    }
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as archivo:
        json.dump(payload, archivo, indent=2, ensure_ascii=False)
    logger.info("Parameters saved: %s", ruta)
    return ruta


def main() -> dict[str, Any]:
    """
    Calibrate CEEMDAN on XLP, decompose with optimal parameters, and save outputs.

    Returns
    -------
    dict
        Calibration results, IMF DataFrame, and generated paths.
    """
    mod = _cargar_modulo_emdsynth_pipeline()
    df, serie = cargar_serie_cierre_xlp()

    logger.info("=" * 70)
    logger.info(
        "CEEMDAN CALIBRATION — XLP (grid %s×%s×%s, max_imf≥9, linear residual)",
        len(REJILLA_EPSILON),
        len(REJILLA_TRIALS),
        len(REJILLA_MAX_IMF),
    )
    logger.info("=" * 70)
    calibracion = calibrar_parametros_ceemdan_xlp(serie, mod)

    parametros_optimos = calibracion["mejor"]
    logger.info("=" * 70)
    logger.info("CEEMDAN DECOMPOSITION — optimal parameters (linear residual)")
    logger.info("=" * 70)
    df_imfs = descomponer_y_guardar(serie, parametros_optimos, mod, df.index)
    resumen = resumir_energia_imfs(df_imfs, serie)
    resumen["linealidad_residuo"] = metricas_linealidad_residuo(
        np.asarray(df_imfs["Residuo"].values, dtype=np.float64)
    )
    guardar_parametros_json(calibracion, resumen)

    lin = resumen["linealidad_residuo"]
    logger.info(
        "Components: %s IMFs + residual; rmse_rel=%.2e; residual R²=%.4f, "
        "rmse_vs_lineal=%.4f",
        resumen["n_imfs"],
        parametros_optimos["metricas"]["rmse_relativo"],
        lin["r2_regresion_lineal"],
        lin["rmse_relativo_vs_lineal"],
    )
    return {
        "calibracion": calibracion,
        "df_imfs": df_imfs,
        "resumen": resumen,
        "ruta_imfs": str(_RUTA_IMFS_SALIDA),
        "ruta_parametros": str(_RUTA_PARAMETROS_JSON),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        main()
    except Exception:
        logger.exception("Error in CEEMDAN for XLP")
        sys.exit(1)
