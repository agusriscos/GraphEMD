"""
Script to validate and analyze CEEMDAN decomposition results for XLV (Health Care Select Sector SPDR Fund).

1. Load data required to validate and analyze decomposition results for XLV. On one hand, closing price data for XLV and on the other hand IMF component data from the CEEMDAN decomposition of XLV.
2. Obtain statistical information on energy, variance, frequency, and mean amplitude of the IMF components.
3. Validate CEEMDAN decomposition results for XLV by comparing them with MSCI World decomposition results in this repository. Compute reconstruction error of the original series from the IMF components. Assess the decomposition residual for XLV (check whether it represents the long-term trend of the original series (monotonically increasing or decreasing)).
4. Plot IMF components and the original XLV series. Also plot CEEMDAN vs. EEMD decomposition residual for XLV
5. Document implemented logic and results in this script.
6. Do not change any existing code for now

Implemented logic
-------------------
- **Load**: ``xlv.parquet``, ``xlv_imfs_ceemdan.parquet``; MSCI reference in
  ``data/20abr26/msci_world.parquet`` and ``msci_world_imfs_ceemdan.parquet``.
- **IMF metrics** (``docs/20abr26``, IMF table): energy ``Σx²``, sample variance,
  frequency = number of local maxima (``scipy.signal.find_peaks``, cycle proxy),
  mean amplitude = ``mean(|x|)``.
- **Validation**: reconstruction ``Σ(IMF_k)+Residuo`` vs. ``Close`` (absolute and relative RMSE);
  CEEMDAN residual = ``Close − Σ IMF``; monotonicity (first-difference sign),
  Spearman correlation with time and linear regression ``R²`` of the residual.
- **EEMD XLV**: default MSCI parameters (``info_msci_world_data``); figure
  ``imf_decomposition`` via ``exportar_figuras_documento_20abr26``; bottom panel =
  CEEMDAN residual magnitude vs. EEMD gap (``|Close−Σ IMF|`` without residual).
- **Composite trend**: sum of the two lowest-frequency IMFs plus the residual
  (e.g. ``IMF_7 + IMF_8 + Residuo``); comparative figure with the isolated residual.
- **Figures**: ``figures/xlv_imfs_panel.png``, ``figures/xlv_imf_decomposition_ceemdan_vs_eemd.png``,
  ``figures/xlv_tendencia_compuesta_ceemdan.png``.

Results obtained (run 2026-05-16, n=3587)
-------------------------------------------------

**Reconstruction XLV:** ``rmse_relativo≈8.7×10⁻¹⁷`` (numerical identity IMFs+residual).

**IMFs XLV (MSCI-like pattern):** IMF_1 — energy 156, 1282 cycles, amp. 0.15;
IMF_8 — energy 27367, 2 cycles, amp. 2.24; 8 IMFs total.

**Residual XLV:** mean 28.36, range [19.76, 48.28]; not strictly monotonic
(``R²`` temporal linear 0.41; Spearman with time 0.39); the long-term trend is
in the residual but with more roughness than MSCI.

**MSCI (reference, IMFs trimmed to 3587 rows):** ``rmse_relativo≈4.1×10⁻⁵``;
residual with linear ``R²`` 0.95 and Spearman≈1.0 (smoother trend).

**Figures:** ``figures/xlv_imfs_panel.png``,
``figures/xlv_imf_decomposition_ceemdan_vs_eemd.png`` (CEEMDAN vs EEMD gap).

**Outputs:** ``xlv_imf_metricas_ceemdan.csv``, ``xlv_validacion_ceemdan.json``,
``xlv_imfs_eemd.parquet`` (EEMD cache).
"""

from __future__ import annotations

import importlib.util
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
from scipy.signal import find_peaks
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=UserWarning)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC_PYTHON = _REPO_ROOT / "src" / "python"
_EXPLORATION = _REPO_ROOT / "scripts" / "exploration"
_DIR_DATOS = _REPO_ROOT / "data" / "GraphEMD" / "xlv_analysis"
_DIR_FIGURAS = _DIR_DATOS / "figures"
_RUTA_XLV = _DIR_DATOS / "xlv.parquet"
_RUTA_IMFS_XLV = _DIR_DATOS / "xlv_imfs_ceemdan.parquet"
_RUTA_IMFS_EEMD_XLV = _DIR_DATOS / "xlv_imfs_eemd.parquet"
_RUTA_MSCI = _REPO_ROOT / "data" / "20abr26" / "msci_world.parquet"
_RUTA_IMFS_MSCI = _REPO_ROOT / "data" / "20abr26" / "msci_world_imfs_ceemdan.parquet"
_RUTA_METRICAS_CSV = _DIR_DATOS / "xlv_imf_metricas_ceemdan.csv"
_RUTA_VALIDACION_JSON = _DIR_DATOS / "xlv_validacion_ceemdan.json"
_RUTA_FIGURA_TENDENCIA_COMPUESTA = (
    _DIR_FIGURAS / "xlv_tendencia_compuesta_ceemdan.png"
)
N_IMFS_BAJA_FRECUENCIA_TENDENCIA: int = 2
N_OBSERVACIONES_ESPERADAS: int = 3587

logger = logging.getLogger(__name__)


def _asegurar_paths_import() -> None:
    """
    Add required paths to ``sys.path`` to import ``GraphEMD`` and exploration modules.
    """
    for ruta in (_SRC_PYTHON, _EXPLORATION):
        s = str(ruta)
        if s not in sys.path:
            sys.path.insert(0, s)


def _cargar_modulo_info_msci() -> Any:
    """
    Load ``info_msci_world_data`` without modifying existing files.

    Returns
    -------
    module
        Module with decomposition functions and figures from document 20abr26.
    """
    _asegurar_paths_import()
    ruta = _EXPLORATION / "info_msci_world_data.py"
    spec = importlib.util.spec_from_file_location("info_msci_world_data_xlv", ruta)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {ruta}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cargar_xlv() -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load precios XLV prices and the closing series.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        OHLCV DataFrame and ``Close`` vector.

    Raises
    ------
    FileNotFoundError
        If the price parquet is missing.
    """
    if not _RUTA_XLV.is_file():
        raise FileNotFoundError(f"Run 01_download_xlv.py: missing {_RUTA_XLV}")
    df = pd.read_parquet(_RUTA_XLV, engine="pyarrow")
    serie = np.asarray(df["Close"].values, dtype=np.float64)
    return df, serie


def cargar_imfs_xlv() -> pd.DataFrame:
    """
    Load the CEEMDAN IMF parquet for XLV.

    Returns
    -------
    pd.DataFrame
        IMFs and residual.

    Raises
    ------
    FileNotFoundError
        If the IMF parquet is missing.
    """
    if not _RUTA_IMFS_XLV.is_file():
        raise FileNotFoundError(f"Run 03_ceemdan_xlv.py: missing {_RUTA_IMFS_XLV}")
    return pd.read_parquet(_RUTA_IMFS_XLV, engine="pyarrow")


def _alinear_serie_e_imfs(
    serie: np.ndarray,
    df_imfs: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Trim series and IMFs to the minimum common length.

    Parameters
    ----------
    serie : np.ndarray
        Closing prices.
    df_imfs : pd.DataFrame
        IMF components.

    Returns
    -------
    tuple
        Series and IMFs with the same number of rows.
    """
    n = min(len(serie), len(df_imfs))
    if len(serie) != n or len(df_imfs) != n:
        logger.warning(
            "Alignment by minimum length: series=%s, IMFs=%s → %s",
            len(serie),
            len(df_imfs),
            n,
        )
    return np.asarray(serie[:n], dtype=np.float64), df_imfs.iloc[:n].copy()


def cargar_referencia_msci() -> tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
    """
    Load MSCI World series and IMFs if available in ``data/20abr26``.

    Returns
    -------
    tuple
        ``(serie_close, df_imfs)`` or ``(None, None)`` if files are missing.
    """
    if not _RUTA_MSCI.is_file() or not _RUTA_IMFS_MSCI.is_file():
        logger.warning("MSCI reference not available in data/20abr26.")
        return None, None
    df_m = pd.read_parquet(_RUTA_MSCI, engine="pyarrow")
    df_i = pd.read_parquet(_RUTA_IMFS_MSCI, engine="pyarrow")
    serie = np.asarray(df_m["Close"].values, dtype=np.float64)
    return _alinear_serie_e_imfs(serie, df_i)


def _columnas_imf(df: pd.DataFrame) -> list[str]:
    """
    Return sorted IMF column names.

    Parameters
    ----------
    df : pd.DataFrame
        Decomposition table.

    Returns
    -------
    list[str]
        ``IMF_1``, ``IMF_2``, ...
    """
    return sorted(
        [c for c in df.columns if c.startswith("IMF_")],
        key=lambda x: int(x.split("_")[1]),
    )


def calcular_metricas_componente(serie: np.ndarray) -> dict[str, float]:
    """
    Compute energy, variance, cycles (local maxima), and mean amplitude.

    Parameters
    ----------
    serie : np.ndarray
        1D component.

    Returns
    -------
    dict
        Metrics aligned with ``docs/20abr26`` (IMF table).
    """
    x = np.asarray(serie, dtype=np.float64)
    picos, _ = find_peaks(x)
    return {
        "energia": float(np.sum(x**2)),
        "varianza": float(np.var(x, ddof=1)),
        "frecuencia_ciclos": float(len(picos)),
        "amplitud_media": float(np.mean(np.abs(x))),
    }


def tabla_metricas_imfs(df_imfs: pd.DataFrame) -> pd.DataFrame:
    """
    Build the per-IMF metrics table (excludes residual).

    Parameters
    ----------
    df_imfs : pd.DataFrame
        CEEMDAN decomposition.

    Returns
    -------
    pd.DataFrame
        One row per IMF with energy, variance, frequency, and amplitude.
    """
    filas = []
    for col in _columnas_imf(df_imfs):
        m = calcular_metricas_componente(df_imfs[col].values)
        m["componente"] = col
        filas.append(m)
    return pd.DataFrame(filas).set_index("componente")


def validar_reconstruccion(
    serie: np.ndarray,
    df_imfs: pd.DataFrame,
) -> dict[str, Any]:
    """
    Evaluate error between the original series and the sum of all modes.

    Parameters
    ----------
    serie : np.ndarray
        Closing prices.
    df_imfs : pd.DataFrame
        IMFs + residual.

    Returns
    -------
    dict
        Absolute and relative reconstruction errors.
    """
    recon = np.sum(
        [np.asarray(df_imfs[c].values, dtype=np.float64) for c in df_imfs.columns],
        axis=0,
    )
    diff = serie - recon
    norma = float(np.linalg.norm(serie)) + 1e-15
    return {
        "rmse_absoluto": float(np.sqrt(np.mean(diff**2))),
        "rmse_relativo": float(np.linalg.norm(diff) / norma),
        "error_max_abs": float(np.max(np.abs(diff))),
        "error_media_abs": float(np.mean(np.abs(diff))),
    }


def evaluar_residuo(
    serie: np.ndarray,
    df_imfs: pd.DataFrame,
) -> dict[str, Any]:
    """
    Analyze the CEEMDAN residual and its relationship to the long-term trend.

    Parameters
    ----------
    serie : np.ndarray
        Closing prices.
    df_imfs : pd.DataFrame
        Must include the ``Residuo`` column.

    Returns
    -------
    dict
        Residual statistics, monotonicity, and temporal linear fit.
    """
    if "Residuo" not in df_imfs.columns:
        raise ValueError("Missing Residuo column in df_imfs.")
    cols_imf = _columnas_imf(df_imfs)
    suma_imfs = np.sum(
        [np.asarray(df_imfs[c].values, dtype=np.float64) for c in cols_imf], axis=0
    )
    residuo = np.asarray(df_imfs["Residuo"].values, dtype=np.float64)
    residuo_implicito = serie - suma_imfs
    diffs = np.diff(residuo)
    tol = 1e-10
    monotono_creciente = bool(np.all(diffs >= -tol))
    monotono_decreciente = bool(np.all(diffs <= tol))
    monotono_estricto = monotono_creciente or monotono_decreciente
    t = np.arange(len(residuo), dtype=np.float64)
    coef = np.polyfit(t, residuo, 1)
    tendencia_lineal = coef[0] * t + coef[1]
    ss_res = float(np.sum((residuo - tendencia_lineal) ** 2))
    ss_tot = float(np.sum((residuo - np.mean(residuo)) ** 2)) + 1e-15
    r2_lineal = 1.0 - ss_res / ss_tot
    rho, p_spearman = spearmanr(t, residuo)
    max_gap_implicito = float(np.max(np.abs(residuo - residuo_implicito)))
    return {
        "media": float(np.mean(residuo)),
        "std": float(np.std(residuo, ddof=1)),
        "minimo": float(np.min(residuo)),
        "maximo": float(np.max(residuo)),
        "monotono_creciente": monotono_creciente,
        "monotono_decreciente": monotono_decreciente,
        "monotono_global": monotono_estricto,
        "pendiente_regresion_temporal": float(coef[0]),
        "r2_regresion_lineal": float(r2_lineal),
        "spearman_tiempo": float(rho),
        "p_valor_spearman": float(p_spearman),
        "max_abs_residuo_menos_implicito": max_gap_implicito,
        "interpretacion": (
            "Long-term trend (monotonically increasing)"
            if monotono_creciente
            else (
                "Long-term trend (monotonically decreasing)"
                if monotono_decreciente
                else "Not strictly monotonic; check visual smoothness"
            )
        ),
    }


def construir_tendencia_compuesta(
    df_imfs: pd.DataFrame,
    n_ultimas_imf: int = N_IMFS_BAJA_FRECUENCIA_TENDENCIA,
) -> tuple[np.ndarray, list[str]]:
    """
    Sum the lowest-frequency IMFs and the residual as a long-term trend proxy.

    Parameters
    ----------
    df_imfs : pd.DataFrame
        CEEMDAN decomposition with ``IMF_*`` and ``Residuo`` columns.
    n_ultimas_imf : int
        Number of lowest-frequency IMFs to include (e.g. 2 → IMF_7+IMF_8).

    Returns
    -------
    tuple[np.ndarray, list[str]]
        Composite trend series and names of summed components.

    Raises
    ------
    ValueError
        If the ``Residuo`` column is missing.
    """
    if "Residuo" not in df_imfs.columns:
        raise ValueError("Missing Residuo column in df_imfs.")
    cols_imf = _columnas_imf(df_imfs)
    if not cols_imf:
        raise ValueError("No IMF columns in df_imfs.")
    n_usar = min(n_ultimas_imf, len(cols_imf))
    seleccion = cols_imf[-n_usar:]
    componentes = seleccion + ["Residuo"]
    tendencia = np.sum(
        [np.asarray(df_imfs[c].values, dtype=np.float64) for c in componentes],
        axis=0,
    )
    return tendencia, componentes


def _metricas_serie_tendencia(serie_tendencia: np.ndarray) -> dict[str, Any]:
    """
    Compute monotonicity and temporal linear fit of a 1D series.

    Parameters
    ----------
    serie_tendencia : np.ndarray
        Trend or residual series.

    Returns
    -------
    dict
        Statistics, monotonicity, linear ``R²``, and Spearman with time.
    """
    y = np.asarray(serie_tendencia, dtype=np.float64)
    diffs = np.diff(y)
    tol = 1e-10
    monotono_creciente = bool(np.all(diffs >= -tol))
    monotono_decreciente = bool(np.all(diffs <= tol))
    t = np.arange(len(y), dtype=np.float64)
    coef = np.polyfit(t, y, 1)
    ajuste = coef[0] * t + coef[1]
    ss_res = float(np.sum((y - ajuste) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-15
    r2_lineal = 1.0 - ss_res / ss_tot
    rho, p_spearman = spearmanr(t, y)
    return {
        "media": float(np.mean(y)),
        "std": float(np.std(y, ddof=1)),
        "minimo": float(np.min(y)),
        "maximo": float(np.max(y)),
        "monotono_creciente": monotono_creciente,
        "monotono_decreciente": monotono_decreciente,
        "monotono_global": monotono_creciente or monotono_decreciente,
        "pendiente_regresion_temporal": float(coef[0]),
        "r2_regresion_lineal": float(r2_lineal),
        "spearman_tiempo": float(rho),
        "p_valor_spearman": float(p_spearman),
    }


def evaluar_tendencia_compuesta(
    serie_close: np.ndarray,
    df_imfs: pd.DataFrame,
    residuo_eval: dict[str, Any],
    n_ultimas_imf: int = N_IMFS_BAJA_FRECUENCIA_TENDENCIA,
) -> dict[str, Any]:
    """
    Compare the isolated residual with the composite trend (last IMFs + residual).

    Parameters
    ----------
    serie_close : np.ndarray
        Closing prices.
    df_imfs : pd.DataFrame
        CEEMDAN IMFs.
    residuo_eval : dict
        Output of ``evaluar_residuo`` for the residual.
    n_ultimas_imf : int
        Low-frequency IMFs to sum.

    Returns
    -------
    dict
        Residual metrics, composite trend metrics, and correlation with ``Close``.
    """
    tendencia, componentes = construir_tendencia_compuesta(df_imfs, n_ultimas_imf)
    metricas_compuesta = _metricas_serie_tendencia(tendencia)
    residuo = np.asarray(df_imfs["Residuo"].values, dtype=np.float64)
    close = np.asarray(serie_close, dtype=np.float64)
    corr_residuo_compuesta = float(np.corrcoef(residuo, tendencia)[0, 1])
    corr_close_residuo = float(np.corrcoef(close, residuo)[0, 1])
    corr_close_compuesta = float(np.corrcoef(close, tendencia)[0, 1])
    return {
        "componentes_sumados": componentes,
        "expresion": " + ".join(componentes),
        "metricas_residuo": {
            "r2_regresion_lineal": residuo_eval["r2_regresion_lineal"],
            "spearman_tiempo": residuo_eval["spearman_tiempo"],
            "monotono_creciente": residuo_eval["monotono_creciente"],
            "correlacion_con_close": corr_close_residuo,
        },
        "metricas_tendencia_compuesta": metricas_compuesta,
        "correlacion_residuo_tendencia_compuesta": corr_residuo_compuesta,
        "correlacion_tendencia_compuesta_con_close": corr_close_compuesta,
        "mejora_r2_lineal_vs_residuo": float(
            metricas_compuesta["r2_regresion_lineal"]
            - residuo_eval["r2_regresion_lineal"]
        ),
        "mejora_correlacion_close_vs_residuo": float(
            corr_close_compuesta - corr_close_residuo
        ),
    }


def interpretacion_bibliografica_residuo_xlv(
    validacion_reconstruccion: dict[str, Any],
    residuo_eval: dict[str, Any],
    tendencia_compuesta_eval: dict[str, Any],
) -> dict[str, Any]:
    """
    Write the XLV residual interpretation supported by EMD/CEEMDAN literature.

    Parameters
    ----------
    validacion_reconstruccion : dict
        Reconstruction metrics ``Σ(IMF)+Residuo`` vs. ``Close``.
    residuo_eval : dict
        Isolated residual evaluation.
    tendencia_compuesta_eval : dict
        Composite trend evaluation.

    Returns
    -------
    dict
        Interpretive paragraph and bibliographic references.
    """
    r2_res = float(residuo_eval["r2_regresion_lineal"])
    r2_comp = float(
        tendencia_compuesta_eval["metricas_tendencia_compuesta"]["r2_regresion_lineal"]
    )
    corr_close_res = float(
        tendencia_compuesta_eval["metricas_residuo"]["correlacion_con_close"]
    )
    corr_close_comp = float(
        tendencia_compuesta_eval["correlacion_tendencia_compuesta_con_close"]
    )
    rmse_rel = float(validacion_reconstruccion["rmse_relativo"])
    expresion = tendencia_compuesta_eval["expresion"]
    parrafo = (
        f"The CEEMDAN decomposition of XLV is mathematically consistent: the sum of "
        f"all modes reconstructs the price with relative RMSE ≈ {rmse_rel:.2e} "
        f"(Huang et al., 1998; Torres et al., 2011). The isolated residual is not "
        f"strictly monotonic or linear (temporal R² = {r2_res:.3f}; correlation "
        f"with Close = {corr_close_res:.3f}), which does not invalidate the decomposition: in "
        f"EMD the final residual is a sifting stopping criterion (monotonic trend "
        f"in the algorithmic sense), and the operational trend is context-dependent "
        f"(Moghtaderi, Flandrin and Borgnat, 2011). In non-stationary financial assets "
        f"part of the very low-frequency dynamics often remains in the last IMFs; "
        f"therefore the long-term trajectory is better represented by {expresion}, which "
        f"correlates {corr_close_comp:.3f} with Close versus {corr_close_res:.3f} for the "
        f"residual only (linear temporal R² = {r2_comp:.3f}). Validation must "
        f"prioritize exact reconstruction and mode stability over requiring a "
        f"perfectly rectilinear residual (CEEMDAN market applications: 2LE-CEEMDAN, 2024)."
    )
    referencias = [
        {
            "clave": "Huang1998",
            "cita": (
                "Huang, N. E., Shen, Z., Long, S. R., et al. (1998). The empirical mode "
                "decomposition and the Hilbert spectrum for nonlinear and non-stationary "
                "time series analysis. Proc. R. Soc. Lond. A, 454, 903–995."
            ),
        },
        {
            "clave": "Torres2011",
            "cita": (
                "Torres, M. E., Colominas, M. A., Schlotthauer, G., Flandrin, P. (2011). "
                "A complete ensemble empirical mode decomposition with adaptive noise. "
                "ICASSP, 4144–4147."
            ),
        },
        {
            "clave": "Moghtaderi2011",
            "cita": (
                "Moghtaderi, A., Flandrin, P., Borgnat, P. (2011). Trend filtering via "
                "empirical mode decompositions. HAL ensl-00565293."
            ),
        },
        {
            "clave": "Colominas2014",
            "cita": (
                "Colominas, M. A., Schlotthauer, G., Torres, M. E. (2014). Improved "
                "complete ensemble EMD. Biomedical Signal Processing and Control, 14, 19–29."
            ),
        },
        {
            "clave": "CEEMDAN_finanzas_2024",
            "cita": (
                "2LE-CEEMDAN on stock series (2024). PLOS ONE / PMC10909190."
            ),
        },
    ]
    return {
        "parrafo": parrafo,
        "referencias": referencias,
        "conclusion": (
            "valid_decomposition_exact_reconstruction"
            if rmse_rel < 1e-6
            else "review_reconstruction"
        ),
        "residuo_como_tendencia_pura": False,
        "recomendacion_tendencia": expresion,
    }


def generar_figura_tendencia_compuesta(
    df_precios: pd.DataFrame,
    serie: np.ndarray,
    df_imfs: pd.DataFrame,
    eval_tendencia: dict[str, Any],
    ruta_salida: Path = _RUTA_FIGURA_TENDENCIA_COMPUESTA,
) -> None:
    """
    Plot Close, CEEMDAN residual, and composite trend (last IMFs + residual).

    Parameters
    ----------
    df_precios : pd.DataFrame
        Prices with time index.
    serie : np.ndarray
        ``Close`` vector.
    df_imfs : pd.DataFrame
        IMFs and residual.
    eval_tendencia : dict
        Output of ``evaluar_tendencia_compuesta``.
    ruta_salida : Path
        Output PNG.
    """
    tendencia, _ = construir_tendencia_compuesta(df_imfs)
    residuo = np.asarray(df_imfs["Residuo"].values, dtype=np.float64)
    fechas = df_precios.index
    expresion = eval_tendencia["expresion"]
    r2_res = eval_tendencia["metricas_residuo"]["r2_regresion_lineal"]
    r2_comp = eval_tendencia["metricas_tendencia_compuesta"]["r2_regresion_lineal"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(fechas, serie, color="0.35", linewidth=0.7, label="Close XLV")
    axes[0].plot(
        fechas,
        tendencia,
        color="C1",
        linewidth=1.0,
        label=f"Composite trend ({expresion})",
    )
    axes[0].set_ylabel("Price / trend (USD)")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(True, alpha=0.25)
    axes[0].set_title(
        f"XLV CEEMDAN: price and composite trend (linear R²={r2_comp:.3f})",
        fontsize=10,
    )

    axes[1].plot(
        fechas,
        residuo,
        color="C0",
        linewidth=0.9,
        label=f"Residuo (R²={r2_res:.3f})",
    )
    axes[1].plot(
        fechas,
        tendencia,
        color="C1",
        linewidth=1.0,
        label=f"Composite trend (R²={r2_comp:.3f})",
    )
    axes[1].set_ylabel("Componente (USD)")
    axes[1].set_xlabel("Fecha")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(True, alpha=0.25)
    axes[1].set_title("Isolated residual vs. composite trend", fontsize=10)

    fig.tight_layout()
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Composite trend figure: %s", ruta_salida)


def comparar_con_msci(
    tabla_xlv: pd.DataFrame,
    tabla_msci: Optional[pd.DataFrame],
    validacion_xlv: dict[str, Any],
    validacion_msci: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """
    Summarize structural similarities XLV vs. MSCI World.

    Parameters
    ----------
    tabla_xlv : pd.DataFrame
        IMF metrics XLV.
    tabla_msci : pd.DataFrame, optional
        IMF metrics MSCI.
    validacion_xlv : dict
        Reconstruction XLV.
    validacion_msci : dict, optional
        MSCI reconstruction.

    Returns
    -------
    dict
        Comparison of counts, errors, and frequency patterns.
    """
    out: dict[str, Any] = {
        "n_imfs_xlv": len(tabla_xlv),
        "validacion_xlv": validacion_xlv,
    }
    if tabla_msci is not None:
        out["n_imfs_msci"] = len(tabla_msci)
        out["validacion_msci"] = validacion_msci
        out["frecuencia_imf1_xlv"] = float(tabla_xlv.loc["IMF_1", "frecuencia_ciclos"])
        out["frecuencia_imf1_msci"] = float(tabla_msci.loc["IMF_1", "frecuencia_ciclos"])
        out["amplitud_imf8_xlv"] = float(
            tabla_xlv.loc[tabla_xlv.index[-1], "amplitud_media"]
        )
        out["amplitud_imf8_msci"] = float(
            tabla_msci.loc[tabla_msci.index[-1], "amplitud_media"]
        )
    return out


def extract_eemd_imfs_xlv(serie: np.ndarray, mod_info: Any) -> pd.DataFrame:
    """
    Obtain EEMD IMFs for XLV (parquet cache if available).

    Parameters
    ----------
    serie : np.ndarray
        Closing prices.
    mod_info : module
        ``info_msci_world_data`` module.

    Returns
    -------
    pd.DataFrame
        EEMD IMFs aligned to the length of ``serie``.
    """
    if _RUTA_IMFS_EEMD_XLV.is_file():
        logger.info("EEMD XLV loaded from cache: %s", _RUTA_IMFS_EEMD_XLV)
        return pd.read_parquet(_RUTA_IMFS_EEMD_XLV, engine="pyarrow")
    logger.info("Computing EEMD for XLV (may take several minutes)...")
    df_eemd = mod_info.extract_eemd_imfs(serie)
    if len(df_eemd) != len(serie):
        _, df_eemd = _alinear_serie_e_imfs(serie, df_eemd)
    _DIR_DATOS.mkdir(parents=True, exist_ok=True)
    df_eemd.to_parquet(_RUTA_IMFS_EEMD_XLV, engine="pyarrow", index=False)
    logger.info("EEMD XLV saved to %s", _RUTA_IMFS_EEMD_XLV)
    return df_eemd


def generar_panel_imfs(
    df_precios: pd.DataFrame,
    df_imfs: pd.DataFrame,
    serie: np.ndarray,
    ruta_salida: Path,
) -> None:
    """
    Stacked figure: Close and each IMF + residual.

    Parameters
    ----------
    df_precios : pd.DataFrame
        Prices (time index).
    df_imfs : pd.DataFrame
        Components.
    serie : np.ndarray
        Vector Close.
    ruta_salida : Path
        Output PNG.
    """
    columnas = ["Close"] + _columnas_imf(df_imfs) + ["Residuo"]
    n = len(columnas)
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.6 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, nombre in zip(axes, columnas):
        if nombre == "Close":
            y = serie
        else:
            y = df_imfs[nombre].values
        ax.plot(df_precios.index, y, linewidth=0.6)
        ax.set_ylabel(nombre, fontsize=8)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Fecha")
    fig.suptitle("XLV: closing price and CEEMDAN IMFs", fontsize=11, y=1.002)
    fig.tight_layout()
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("IMF panel figure: %s", ruta_salida)


def generar_figura_ceemdan_vs_eemd(
    df_imfs_ceemdan: pd.DataFrame,
    serie: np.ndarray,
    mod_info: Any,
    directorio: Path,
    df_imfs_eemd: Optional[pd.DataFrame] = None,
) -> None:
    """
    ``imf_decomposition.png``-style figure (CEEMDAN vs EEMD) for XLV.

    Parameters
    ----------
    df_imfs_ceemdan : pd.DataFrame
        CEEMDAN IMFs.
    serie : np.ndarray
        Close.
    mod_info : module
        Module with ``exportar_figuras_documento_20abr26``.
    directorio : Path
        Output folder.
    df_imfs_eemd : pd.DataFrame, optional
        Precomputed EEMD IMFs.
    """
    directorio.mkdir(parents=True, exist_ok=True)
    mod_info.exportar_figuras_documento_20abr26(
        df_imfs_ceemdan,
        serie,
        directorio,
        df_imfs_eemd=df_imfs_eemd,
    )
    origen = directorio / "imf_decomposition.png"
    destino = directorio / "xlv_imf_decomposition_ceemdan_vs_eemd.png"
    if origen.is_file():
        origen.replace(destino)
    logger.info("CEEMDAN vs EEMD figure: %s", destino)


def main() -> dict[str, Any]:
    """
    Validation pipeline, metrics, MSCI comparison, and figures.

    Returns
    -------
    dict
        Tables, validation, and generated paths.
    """
    mod_info = _cargar_modulo_info_msci()
    df_xlv, serie_xlv = cargar_xlv()
    df_imfs_xlv = cargar_imfs_xlv()
    if len(df_xlv) != len(df_imfs_xlv):
        raise ValueError(
            f"Different lengths: prices {len(df_xlv)} vs IMFs {len(df_imfs_xlv)}"
        )

    logger.info("=" * 70)
    logger.info("IMF METRICS — XLV (CEEMDAN)")
    logger.info("=" * 70)
    tabla_xlv = tabla_metricas_imfs(df_imfs_xlv)
    logger.info("\n%s", tabla_xlv.to_string(float_format=lambda x: f"{x:.4f}"))

    validacion_xlv = validar_reconstruccion(serie_xlv, df_imfs_xlv)
    residuo_xlv = evaluar_residuo(serie_xlv, df_imfs_xlv)
    tendencia_compuesta_xlv = evaluar_tendencia_compuesta(
        serie_xlv, df_imfs_xlv, residuo_xlv
    )
    interpretacion_xlv = interpretacion_bibliografica_residuo_xlv(
        validacion_xlv,
        residuo_xlv,
        tendencia_compuesta_xlv,
    )
    logger.info("Reconstruction XLV: %s", validacion_xlv)
    logger.info("Residuo XLV: %s", residuo_xlv)
    logger.info("XLV composite trend: %s", tendencia_compuesta_xlv["expresion"])
    logger.info(
        "Residual R²=%.3f, composite R²=%.3f, corr(Close) residual=%.3f, composite=%.3f",
        tendencia_compuesta_xlv["metricas_residuo"]["r2_regresion_lineal"],
        tendencia_compuesta_xlv["metricas_tendencia_compuesta"]["r2_regresion_lineal"],
        tendencia_compuesta_xlv["metricas_residuo"]["correlacion_con_close"],
        tendencia_compuesta_xlv["correlacion_tendencia_compuesta_con_close"],
    )

    serie_msci, df_imfs_msci = cargar_referencia_msci()
    tabla_msci: Optional[pd.DataFrame] = None
    validacion_msci: Optional[dict[str, Any]] = None
    residuo_msci: Optional[dict[str, Any]] = None
    if serie_msci is not None and df_imfs_msci is not None:
        logger.info("=" * 70)
        logger.info("REFERENCIA MSCI WORLD")
        logger.info("=" * 70)
        tabla_msci = tabla_metricas_imfs(df_imfs_msci)
        validacion_msci = validar_reconstruccion(serie_msci, df_imfs_msci)
        residuo_msci = evaluar_residuo(serie_msci, df_imfs_msci)
        logger.info("Reconstruction MSCI: %s", validacion_msci)
        logger.info("Residuo MSCI: %s", residuo_msci)

    comparacion = comparar_con_msci(
        tabla_xlv, tabla_msci, validacion_xlv, validacion_msci
    )

    _DIR_DATOS.mkdir(parents=True, exist_ok=True)
    tabla_xlv.to_csv(_RUTA_METRICAS_CSV)
    payload = {
        "n_observaciones": len(serie_xlv),
        "metricas_imf_xlv": tabla_xlv.reset_index().to_dict(orient="records"),
        "validacion_reconstruccion_xlv": validacion_xlv,
        "evaluacion_residuo_xlv": residuo_xlv,
        "evaluacion_tendencia_compuesta_xlv": tendencia_compuesta_xlv,
        "interpretacion_residuo_ceemdan_xlv": interpretacion_xlv,
        "figura_tendencia_compuesta": str(_RUTA_FIGURA_TENDENCIA_COMPUESTA),
        "comparacion_msci": comparacion,
    }
    if tabla_msci is not None:
        payload["metricas_imf_msci"] = tabla_msci.reset_index().to_dict(orient="records")
        payload["evaluacion_residuo_msci"] = residuo_msci
    with open(_RUTA_VALIDACION_JSON, "w", encoding="utf-8") as archivo:
        json.dump(payload, archivo, indent=2, ensure_ascii=False)

    df_eemd = extract_eemd_imfs_xlv(serie_xlv, mod_info)
    generar_panel_imfs(
        df_xlv,
        df_imfs_xlv,
        serie_xlv,
        _DIR_FIGURAS / "xlv_imfs_panel.png",
    )
    generar_figura_ceemdan_vs_eemd(
        df_imfs_xlv,
        serie_xlv,
        mod_info,
        _DIR_FIGURAS,
        df_imfs_eemd=df_eemd,
    )
    generar_figura_tendencia_compuesta(
        df_xlv,
        serie_xlv,
        df_imfs_xlv,
        tendencia_compuesta_xlv,
    )

    return {
        "tabla_xlv": tabla_xlv,
        "validacion_xlv": validacion_xlv,
        "residuo_xlv": residuo_xlv,
        "tendencia_compuesta_xlv": tendencia_compuesta_xlv,
        "interpretacion_xlv": interpretacion_xlv,
        "comparacion": comparacion,
        "ruta_metricas": str(_RUTA_METRICAS_CSV),
        "ruta_validacion": str(_RUTA_VALIDACION_JSON),
        "ruta_figura_tendencia_compuesta": str(_RUTA_FIGURA_TENDENCIA_COMPUESTA),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        resultado = main()
        logger.info("Proceso completado. CSV: %s", resultado["ruta_metricas"])
    except Exception:
        logger.exception("Error in CEEMDAN validation for XLV")
        sys.exit(1)
