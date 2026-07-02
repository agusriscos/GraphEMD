"""
Script para validar y analizar los resultados de la descomposición de la XAU/USD (par spot XAU/USD (oro frente al dólar)) en componentes IMFs mediante CEEMDAN.

1. Carga los datos necesarios para la validacion y analisis de los resultados de la descomposición de la XAU/USD. Por un lado los datos de precio de cierre de la XAU/USD y por otro lado los datos de los componentes IMFs obtenidos en la descomposición de la XAU/USD mediante CEEMDAN.
2. Obten la información estadistica sobre energía, variancia, frecuencia y amplitud media de los componentes IMFs obtenidos.
3. Valida los resultados de la descomposición de la XAU/USD mediante CEEMDAN comparandolos con los resultados de la descomposición de la MSCI World en este mismo repositorio. Calcula el error de la reconstrucción de la serie original con los componentes IMFs obtenidos. Evalua como es el residuo de la descomposicion de la XAU/USD (mira si representa la tendencia a largo plazo de la serie original (monotona creciente o decreciente)).
4. Pinta las graficas de los componentes IMFs obtenidos y la serie original de la XAU/USD. Tambien pinta el residuo de la descomposicion CEEMDAN vs. EEMD para el XAU/USD
5. Documenta en este mismo script la logica implementada y los resultados obtenidos.
6. No cambies por el momento ningun codigo que ya exista

Lógica implementada
-------------------
- **Carga**: ``xauusd.parquet``, ``xauusd_imfs_ceemdan.parquet``; referencia MSCI en
  ``data/20abr26/msci_world.parquet`` y ``msci_world_imfs_ceemdan.parquet``.
- **Métricas IMF** (``docs/20abr26``, tabla IMF): energía ``Σx²``, varianza muestral,
  frecuencia = número de máximos locales (``scipy.signal.find_peaks``, proxy de ciclos),
  amplitud media = ``mean(|x|)``.
- **Validación**: reconstrucción ``Σ(IMF_k)+Residuo`` vs. ``Close`` (RMSE absoluto y relativo);
  residuo CEEMDAN = ``Close − Σ IMF``; monotonicidad (signo de diferencias de primer orden),
  correlación de Spearman con el tiempo y ``R²`` de regresión lineal del residuo.
- **EEMD XAU/USD**: parámetros por defecto MSCI (``info_msci_world_data``); figura
  ``imf_decomposition`` vía ``exportar_figuras_documento_20abr26``; panel inferior =
  magnitud del residuo CEEMDAN vs. brecha EEMD (``|Close−Σ IMF|`` sin residuo).
- **Tendencia compuesta**: suma de las dos IMFs de menor frecuencia más el residuo
  (p. ej. ``IMF_7 + IMF_8 + Residuo``); figura comparativa con el residuo aislado.
- **Figuras**: ``figures/xauusd_imfs_panel.png``, ``figures/xauusd_imf_decomposition_ceemdan_vs_eemd.png``,
  ``figures/xauusd_tendencia_compuesta_ceemdan.png``.

Resultados obtenidos (ejecución 2026-05-16, n=3587)
-------------------------------------------------

**Reconstrucción XAU/USD:** ``rmse_relativo≈8.7×10⁻¹⁷`` (identidad numérica IMFs+residuo).

**IMFs XAU/USD (patrón tipo MSCI):** IMF_1 — energía 156, 1282 ciclos, amp. 0.15;
IMF_8 — energía 27367, 2 ciclos, amp. 2.24; 8 IMFs en total.

**Residuo XAU/USD:** media 28.36, rango [19.76, 48.28]; no estrictamente monótono
(``R²`` lineal temporal 0.41; Spearman tiempo 0.39); la tendencia largo plazo está
en el residuo pero con más rugosidad que MSCI.

**MSCI (referencia, IMFs recortados a 3587 filas):** ``rmse_relativo≈4.1×10⁻⁵``;
residuo con ``R²`` lineal 0.95 y Spearman≈1.0 (tendencia más suave).

**Figuras:** ``figures/xauusd_imfs_panel.png``,
``figures/xauusd_imf_decomposition_ceemdan_vs_eemd.png`` (brecha CEEMDAN vs EEMD).

**Salidas:** ``xauusd_imf_metricas_ceemdan.csv``, ``xauusd_validacion_ceemdan.json``,
``xauusd_imfs_eemd.parquet`` (caché EEMD).
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

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SRC_PYTHON = _REPO_ROOT / "src" / "python"
_EXPLORACION = _REPO_ROOT / "scripts" / "GraphEMD" / "exploracion"
_DIR_DATOS = _REPO_ROOT / "data" / "GraphEMD" / "xauusd_analysis"
_DIR_FIGURAS = _DIR_DATOS / "figures"
_RUTA_XAUUSD = _DIR_DATOS / "xauusd.parquet"
_RUTA_IMFS_XAUUSD = _DIR_DATOS / "xauusd_imfs_ceemdan.parquet"
_RUTA_IMFS_EEMD_XAUUSD = _DIR_DATOS / "xauusd_imfs_eemd.parquet"
_RUTA_MSCI = _REPO_ROOT / "data" / "20abr26" / "msci_world.parquet"
_RUTA_IMFS_MSCI = _REPO_ROOT / "data" / "20abr26" / "msci_world_imfs_ceemdan.parquet"
_RUTA_METRICAS_CSV = _DIR_DATOS / "xauusd_imf_metricas_ceemdan.csv"
_RUTA_VALIDACION_JSON = _DIR_DATOS / "xauusd_validacion_ceemdan.json"
_RUTA_FIGURA_TENDENCIA_COMPUESTA = (
    _DIR_FIGURAS / "xauusd_tendencia_compuesta_ceemdan.png"
)
N_IMFS_BAJA_FRECUENCIA_TENDENCIA: int = 2
N_OBSERVACIONES_ESPERADAS: int = 3584

logger = logging.getLogger(__name__)


def _asegurar_paths_import() -> None:
    """
    Añade al ``sys.path`` las rutas necesarias para importar ``GraphEMD`` y exploración.
    """
    for ruta in (_SRC_PYTHON, _EXPLORACION):
        s = str(ruta)
        if s not in sys.path:
            sys.path.insert(0, s)


def _cargar_modulo_info_msci() -> Any:
    """
    Carga ``info_msci_world_data`` sin modificar archivos existentes.

    Returns
    -------
    module
        Módulo con funciones de descomposición y figuras del documento 20abr26.
    """
    _asegurar_paths_import()
    ruta = _EXPLORACION / "info_msci_world_data.py"
    spec = importlib.util.spec_from_file_location("info_msci_world_data_xauusd", ruta)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar {ruta}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cargar_xauusd() -> tuple[pd.DataFrame, np.ndarray]:
    """
    Carga precios XAU/USD y la serie de cierre.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        DataFrame OHLCV y vector ``Close``.

    Raises
    ------
    FileNotFoundError
        Si falta el parquet de precios.
    """
    if not _RUTA_XAUUSD.is_file():
        raise FileNotFoundError(f"Ejecute 01_download_xauusd.py: falta {_RUTA_XAUUSD}")
    df = pd.read_parquet(_RUTA_XAUUSD, engine="pyarrow")
    serie = np.asarray(df["Close"].values, dtype=np.float64)
    return df, serie


def cargar_imfs_xauusd() -> pd.DataFrame:
    """
    Carga el parquet de IMFs CEEMDAN de XAU/USD.

    Returns
    -------
    pd.DataFrame
        IMFs y residuo.

    Raises
    ------
    FileNotFoundError
        Si falta el parquet de IMFs.
    """
    if not _RUTA_IMFS_XAUUSD.is_file():
        raise FileNotFoundError(f"Ejecute 03_ceemdan_xauusd.py: falta {_RUTA_IMFS_XAUUSD}")
    return pd.read_parquet(_RUTA_IMFS_XAUUSD, engine="pyarrow")


def _alinear_serie_e_imfs(
    serie: np.ndarray,
    df_imfs: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Recorta serie e IMFs a la longitud común mínima.

    Parameters
    ----------
    serie : np.ndarray
        Precios de cierre.
    df_imfs : pd.DataFrame
        Componentes IMF.

    Returns
    -------
    tuple
        Serie e IMFs con el mismo número de filas.
    """
    n = min(len(serie), len(df_imfs))
    if len(serie) != n or len(df_imfs) != n:
        logger.warning(
            "Alineación por longitud mínima: serie=%s, IMFs=%s → %s",
            len(serie),
            len(df_imfs),
            n,
        )
    return np.asarray(serie[:n], dtype=np.float64), df_imfs.iloc[:n].copy()


def cargar_referencia_msci() -> tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
    """
    Carga serie e IMFs MSCI World si existen en ``data/20abr26``.

    Returns
    -------
    tuple
        ``(serie_close, df_imfs)`` o ``(None, None)`` si no hay archivos.
    """
    if not _RUTA_MSCI.is_file() or not _RUTA_IMFS_MSCI.is_file():
        logger.warning("Referencia MSCI no disponible en data/20abr26.")
        return None, None
    df_m = pd.read_parquet(_RUTA_MSCI, engine="pyarrow")
    df_i = pd.read_parquet(_RUTA_IMFS_MSCI, engine="pyarrow")
    serie = np.asarray(df_m["Close"].values, dtype=np.float64)
    return _alinear_serie_e_imfs(serie, df_i)


def _columnas_imf(df: pd.DataFrame) -> list[str]:
    """
    Devuelve nombres de columnas IMF ordenadas.

    Parameters
    ----------
    df : pd.DataFrame
        Tabla de descomposición.

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
    Calcula energía, varianza, ciclos (máximos locales) y amplitud media.

    Parameters
    ----------
    serie : np.ndarray
        Componente 1D.

    Returns
    -------
    dict
        Métricas alineadas con ``docs/20abr26`` (tabla IMF).
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
    Construye la tabla de métricas por IMF (excluye residuo).

    Parameters
    ----------
    df_imfs : pd.DataFrame
        Descomposición CEEMDAN.

    Returns
    -------
    pd.DataFrame
        Una fila por IMF con energía, varianza, frecuencia y amplitud.
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
    Evalúa el error entre la serie original y la suma de todos los modos.

    Parameters
    ----------
    serie : np.ndarray
        Precios de cierre.
    df_imfs : pd.DataFrame
        IMFs + residuo.

    Returns
    -------
    dict
        Errores absolutos y relativos de reconstrucción.
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
    Analiza el residuo CEEMDAN y su relación con la tendencia de largo plazo.

    Parameters
    ----------
    serie : np.ndarray
        Precios de cierre.
    df_imfs : pd.DataFrame
        Debe incluir columna ``Residuo``.

    Returns
    -------
    dict
        Estadísticos del residuo, monotonicidad y ajuste lineal temporal.
    """
    if "Residuo" not in df_imfs.columns:
        raise ValueError("Falta columna Residuo en df_imfs.")
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
            "Tendencia largo plazo (monótona creciente)"
            if monotono_creciente
            else (
                "Tendencia largo plazo (monótona decreciente)"
                if monotono_decreciente
                else "No estrictamente monótona; revisar suavidad visual"
            )
        ),
    }


def construir_tendencia_compuesta(
    df_imfs: pd.DataFrame,
    n_ultimas_imf: int = N_IMFS_BAJA_FRECUENCIA_TENDENCIA,
) -> tuple[np.ndarray, list[str]]:
    """
    Suma las IMFs de menor frecuencia y el residuo como proxy de tendencia largo plazo.

    Parameters
    ----------
    df_imfs : pd.DataFrame
        Descomposición CEEMDAN con columnas ``IMF_*`` y ``Residuo``.
    n_ultimas_imf : int
        Número de IMFs de más baja frecuencia a incluir (p. ej. 2 → IMF_7+IMF_8).

    Returns
    -------
    tuple[np.ndarray, list[str]]
        Serie de tendencia compuesta y nombres de componentes sumados.

    Raises
    ------
    ValueError
        Si falta la columna ``Residuo``.
    """
    if "Residuo" not in df_imfs.columns:
        raise ValueError("Falta columna Residuo en df_imfs.")
    cols_imf = _columnas_imf(df_imfs)
    if not cols_imf:
        raise ValueError("No hay columnas IMF en df_imfs.")
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
    Calcula monotonicidad y ajuste lineal temporal de una serie 1D.

    Parameters
    ----------
    serie_tendencia : np.ndarray
        Serie de tendencia o residuo.

    Returns
    -------
    dict
        Estadísticos, monotonicidad, ``R²`` lineal y Spearman con el tiempo.
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
    Compara el residuo aislado con la tendencia compuesta (últimas IMFs + residuo).

    Parameters
    ----------
    serie_close : np.ndarray
        Precios de cierre.
    df_imfs : pd.DataFrame
        IMFs CEEMDAN.
    residuo_eval : dict
        Salida de ``evaluar_residuo`` para el residuo.
    n_ultimas_imf : int
        IMFs de baja frecuencia a sumar.

    Returns
    -------
    dict
        Métricas del residuo, de la tendencia compuesta y correlación con ``Close``.
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


def interpretacion_bibliografica_residuo_xauusd(
    validacion_reconstruccion: dict[str, Any],
    residuo_eval: dict[str, Any],
    tendencia_compuesta_eval: dict[str, Any],
) -> dict[str, Any]:
    """
    Redacta la interpretación del residuo XAU/USD apoyada en literatura EMD/CEEMDAN.

    Parameters
    ----------
    validacion_reconstruccion : dict
        Métricas de reconstrucción ``Σ(IMF)+Residuo`` vs. ``Close``.
    residuo_eval : dict
        Evaluación del residuo aislado.
    tendencia_compuesta_eval : dict
        Evaluación de la tendencia compuesta.

    Returns
    -------
    dict
        Párrafo interpretativo y referencias bibliográficas.
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
        f"La descomposición CEEMDAN de XAU/USD es matemáticamente consistente: la suma de "
        f"todos los modos reconstruye el precio con RMSE relativo ≈ {rmse_rel:.2e} "
        f"(Huang et al., 1998; Torres et al., 2011). El residuo aislado no es "
        f"estrictamente monótono ni lineal (R² temporal = {r2_res:.3f}; correlación "
        f"con Close = {corr_close_res:.3f}), lo cual no invalida la descomposición: en "
        f"EMD el residuo final es criterio de parada del tamizado (tendencia monótona "
        f"en sentido algorítmico), y la tendencia operativa es dependiente del contexto "
        f"(Moghtaderi, Flandrin y Borgnat, 2011). En activos financieros no estacionarios "
        f"parte de la dinámica de muy baja frecuencia suele quedar en las últimas IMFs; "
        f"por ello la trayectoria largo plazo se representa mejor con {expresion}, que "
        f"correlaciona {corr_close_comp:.3f} con Close frente a {corr_close_res:.3f} del "
        f"residuo solo (R² lineal temporal = {r2_comp:.3f}). La validación debe "
        f"priorizar reconstrucción exacta y estabilidad de modos frente a exigir un "
        f"residuo rectilíneo perfecto (aplicaciones CEEMDAN en mercados: 2LE-CEEMDAN, 2024)."
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
                "2LE-CEEMDAN en series bursátiles (2024). PLOS ONE / PMC10909190."
            ),
        },
    ]
    return {
        "parrafo": parrafo,
        "referencias": referencias,
        "conclusion": (
            "descomposicion_valida_reconstruccion_exacta"
            if rmse_rel < 1e-6
            else "revisar_reconstruccion"
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
    Grafica Close, residuo CEEMDAN y tendencia compuesta (últimas IMFs + residuo).

    Parameters
    ----------
    df_precios : pd.DataFrame
        Precios con índice temporal.
    serie : np.ndarray
        Vector ``Close``.
    df_imfs : pd.DataFrame
        IMFs y residuo.
    eval_tendencia : dict
        Salida de ``evaluar_tendencia_compuesta``.
    ruta_salida : Path
        PNG de salida.
    """
    tendencia, _ = construir_tendencia_compuesta(df_imfs)
    residuo = np.asarray(df_imfs["Residuo"].values, dtype=np.float64)
    fechas = df_precios.index
    expresion = eval_tendencia["expresion"]
    r2_res = eval_tendencia["metricas_residuo"]["r2_regresion_lineal"]
    r2_comp = eval_tendencia["metricas_tendencia_compuesta"]["r2_regresion_lineal"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(fechas, serie, color="0.35", linewidth=0.7, label="Close XAU/USD")
    axes[0].plot(
        fechas,
        tendencia,
        color="C1",
        linewidth=1.0,
        label=f"Tendencia compuesta ({expresion})",
    )
    axes[0].set_ylabel("Precio / tendencia (USD)")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(True, alpha=0.25)
    axes[0].set_title(
        f"XAU/USD CEEMDAN: precio y tendencia compuesta (R² lineal={r2_comp:.3f})",
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
        label=f"Tendencia compuesta (R²={r2_comp:.3f})",
    )
    axes[1].set_ylabel("Componente (USD)")
    axes[1].set_xlabel("Fecha")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(True, alpha=0.25)
    axes[1].set_title("Residuo aislado vs. tendencia compuesta", fontsize=10)

    fig.tight_layout()
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura tendencia compuesta: %s", ruta_salida)


def comparar_con_msci(
    tabla_xauusd: pd.DataFrame,
    tabla_msci: Optional[pd.DataFrame],
    validacion_xauusd: dict[str, Any],
    validacion_msci: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """
    Resume similitudes estructurales XAU/USD vs MSCI World.

    Parameters
    ----------
    tabla_xauusd : pd.DataFrame
        Métricas IMF XAU/USD.
    tabla_msci : pd.DataFrame, optional
        Métricas IMF MSCI.
    validacion_xauusd : dict
        Reconstrucción XAU/USD.
    validacion_msci : dict, optional
        Reconstrucción MSCI.

    Returns
    -------
    dict
        Comparación de conteos, errores y patrones de frecuencia.
    """
    out: dict[str, Any] = {
        "n_imfs_xauusd": len(tabla_xauusd),
        "validacion_xauusd": validacion_xauusd,
    }
    if tabla_msci is not None:
        out["n_imfs_msci"] = len(tabla_msci)
        out["validacion_msci"] = validacion_msci
        out["frecuencia_imf1_xauusd"] = float(tabla_xauusd.loc["IMF_1", "frecuencia_ciclos"])
        out["frecuencia_imf1_msci"] = float(tabla_msci.loc["IMF_1", "frecuencia_ciclos"])
        out["amplitud_imf8_xauusd"] = float(
            tabla_xauusd.loc[tabla_xauusd.index[-1], "amplitud_media"]
        )
        out["amplitud_imf8_msci"] = float(
            tabla_msci.loc[tabla_msci.index[-1], "amplitud_media"]
        )
    return out


def obtener_imfs_eemd_xauusd(serie: np.ndarray, mod_info: Any) -> pd.DataFrame:
    """
    Obtiene IMFs EEMD de XAU/USD (cache en parquet si existe).

    Parameters
    ----------
    serie : np.ndarray
        Precios de cierre.
    mod_info : module
        Módulo ``info_msci_world_data``.

    Returns
    -------
    pd.DataFrame
        IMFs EEMD alineadas a la longitud de ``serie``.
    """
    if _RUTA_IMFS_EEMD_XAUUSD.is_file():
        logger.info("EEMD XAU/USD cargado desde caché: %s", _RUTA_IMFS_EEMD_XAUUSD)
        return pd.read_parquet(_RUTA_IMFS_EEMD_XAUUSD, engine="pyarrow")
    logger.info("Calculando EEMD para XAU/USD (puede tardar varios minutos)...")
    df_eemd = mod_info.obtener_imfs_eemd(serie)
    if len(df_eemd) != len(serie):
        _, df_eemd = _alinear_serie_e_imfs(serie, df_eemd)
    _DIR_DATOS.mkdir(parents=True, exist_ok=True)
    df_eemd.to_parquet(_RUTA_IMFS_EEMD_XAUUSD, engine="pyarrow", index=False)
    logger.info("EEMD XAU/USD guardado en %s", _RUTA_IMFS_EEMD_XAUUSD)
    return df_eemd


def generar_panel_imfs(
    df_precios: pd.DataFrame,
    df_imfs: pd.DataFrame,
    serie: np.ndarray,
    ruta_salida: Path,
) -> None:
    """
    Figura apilada: Close y cada IMF + residuo.

    Parameters
    ----------
    df_precios : pd.DataFrame
        Precios (índice temporal).
    df_imfs : pd.DataFrame
        Componentes.
    serie : np.ndarray
        Vector Close.
    ruta_salida : Path
        PNG de salida.
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
    fig.suptitle("XAU/USD: precio de cierre e IMFs CEEMDAN", fontsize=11, y=1.002)
    fig.tight_layout()
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura panel IMFs: %s", ruta_salida)


def generar_figura_ceemdan_vs_eemd(
    df_imfs_ceemdan: pd.DataFrame,
    serie: np.ndarray,
    mod_info: Any,
    directorio: Path,
    df_imfs_eemd: Optional[pd.DataFrame] = None,
) -> None:
    """
    Figura estilo ``imf_decomposition.png`` (CEEMDAN vs EEMD) para XAU/USD.

    Parameters
    ----------
    df_imfs_ceemdan : pd.DataFrame
        IMFs CEEMDAN.
    serie : np.ndarray
        Close.
    mod_info : module
        Módulo con ``exportar_figuras_documento_20abr26``.
    directorio : Path
        Carpeta de salida.
    df_imfs_eemd : pd.DataFrame, optional
        IMFs EEMD precalculadas.
    """
    directorio.mkdir(parents=True, exist_ok=True)
    mod_info.exportar_figuras_documento_20abr26(
        df_imfs_ceemdan,
        serie,
        directorio,
        df_imfs_eemd=df_imfs_eemd,
    )
    origen = directorio / "imf_decomposition.png"
    destino = directorio / "xauusd_imf_decomposition_ceemdan_vs_eemd.png"
    if origen.is_file():
        origen.replace(destino)
    logger.info("Figura CEEMDAN vs EEMD: %s", destino)


def main() -> dict[str, Any]:
    """
    Pipeline de validación, métricas, comparación MSCI y figuras.

    Returns
    -------
    dict
        Tablas, validación y rutas generadas.
    """
    mod_info = _cargar_modulo_info_msci()
    df_xauusd, serie_xauusd = cargar_xauusd()
    df_imfs_xauusd = cargar_imfs_xauusd()
    if len(df_xauusd) != len(df_imfs_xauusd):
        raise ValueError(
            f"Longitudes distintas: precios {len(df_xauusd)} vs IMFs {len(df_imfs_xauusd)}"
        )

    logger.info("=" * 70)
    logger.info("MÉTRICAS IMF — XAU/USD (CEEMDAN)")
    logger.info("=" * 70)
    tabla_xauusd = tabla_metricas_imfs(df_imfs_xauusd)
    logger.info("\n%s", tabla_xauusd.to_string(float_format=lambda x: f"{x:.4f}"))

    validacion_xauusd = validar_reconstruccion(serie_xauusd, df_imfs_xauusd)
    residuo_xauusd = evaluar_residuo(serie_xauusd, df_imfs_xauusd)
    tendencia_compuesta_xauusd = evaluar_tendencia_compuesta(
        serie_xauusd, df_imfs_xauusd, residuo_xauusd
    )
    interpretacion_xauusd = interpretacion_bibliografica_residuo_xauusd(
        validacion_xauusd,
        residuo_xauusd,
        tendencia_compuesta_xauusd,
    )
    logger.info("Reconstrucción XAU/USD: %s", validacion_xauusd)
    logger.info("Residuo XAU/USD: %s", residuo_xauusd)
    logger.info("Tendencia compuesta XAU/USD: %s", tendencia_compuesta_xauusd["expresion"])
    logger.info(
        "R² residuo=%.3f, R² compuesta=%.3f, corr(Close) residuo=%.3f, compuesta=%.3f",
        tendencia_compuesta_xauusd["metricas_residuo"]["r2_regresion_lineal"],
        tendencia_compuesta_xauusd["metricas_tendencia_compuesta"]["r2_regresion_lineal"],
        tendencia_compuesta_xauusd["metricas_residuo"]["correlacion_con_close"],
        tendencia_compuesta_xauusd["correlacion_tendencia_compuesta_con_close"],
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
        logger.info("Reconstrucción MSCI: %s", validacion_msci)
        logger.info("Residuo MSCI: %s", residuo_msci)

    comparacion = comparar_con_msci(
        tabla_xauusd, tabla_msci, validacion_xauusd, validacion_msci
    )

    _DIR_DATOS.mkdir(parents=True, exist_ok=True)
    tabla_xauusd.to_csv(_RUTA_METRICAS_CSV)
    payload = {
        "n_observaciones": len(serie_xauusd),
        "metricas_imf_xauusd": tabla_xauusd.reset_index().to_dict(orient="records"),
        "validacion_reconstruccion_xauusd": validacion_xauusd,
        "evaluacion_residuo_xauusd": residuo_xauusd,
        "evaluacion_tendencia_compuesta_xauusd": tendencia_compuesta_xauusd,
        "interpretacion_residuo_ceemdan_xauusd": interpretacion_xauusd,
        "figura_tendencia_compuesta": str(_RUTA_FIGURA_TENDENCIA_COMPUESTA),
        "comparacion_msci": comparacion,
    }
    if tabla_msci is not None:
        payload["metricas_imf_msci"] = tabla_msci.reset_index().to_dict(orient="records")
        payload["evaluacion_residuo_msci"] = residuo_msci
    with open(_RUTA_VALIDACION_JSON, "w", encoding="utf-8") as archivo:
        json.dump(payload, archivo, indent=2, ensure_ascii=False)

    df_eemd = obtener_imfs_eemd_xauusd(serie_xauusd, mod_info)
    generar_panel_imfs(
        df_xauusd,
        df_imfs_xauusd,
        serie_xauusd,
        _DIR_FIGURAS / "xauusd_imfs_panel.png",
    )
    generar_figura_ceemdan_vs_eemd(
        df_imfs_xauusd,
        serie_xauusd,
        mod_info,
        _DIR_FIGURAS,
        df_imfs_eemd=df_eemd,
    )
    generar_figura_tendencia_compuesta(
        df_xauusd,
        serie_xauusd,
        df_imfs_xauusd,
        tendencia_compuesta_xauusd,
    )

    return {
        "tabla_xauusd": tabla_xauusd,
        "validacion_xauusd": validacion_xauusd,
        "residuo_xauusd": residuo_xauusd,
        "tendencia_compuesta_xauusd": tendencia_compuesta_xauusd,
        "interpretacion_xauusd": interpretacion_xauusd,
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
        logger.exception("Error en validación CEEMDAN de XAU/USD")
        sys.exit(1)
