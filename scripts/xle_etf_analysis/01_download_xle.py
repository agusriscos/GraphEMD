"""
Script to download XLE (Energy Select Sector SPDR Fund) data from Yahoo Finance.

Implemented logic
---------------------
- Download history of ``XLE`` with ``yfinance`` (``period="max"``).
- Trim to inclusive dates ``2012-01-12`` … ``2026-04-20`` (same window as ``docs/20abr26``).
- If ``data/20abr26/msci_world.parquet``, align the index of XLE to MSCI World
  (same trading days) to obtain exactly 3587 observations; otherwise only the date trim is used.
- Metrics on ``Close``: min/max, cumulative return ``(C_T/C_0 - 1)×100``,
  daily returns in % (``pct_change×100``), mean and standard deviation of those returns,
  30-day rolling volatility as rolling ``std`` of daily returns (same definition as
  ``analysis/16dic25/02_emd_conditions_analysis.ipynb``) and its maximum.

Results obtained (run 2026-05-16)
-------------------------------------------
- Observations: 3587
- Range: 2012-01-12 → 2026-04-20
- Close minimum (USD): 9.2453
- Close maximum (USD): 62.5600
- Cumulative return (%): 160.21
- Daily return — mean (%): 0.0414
- Daily return — std dev (%): 1.7100
- 30d rolling volatility — mean (%): 1.4842
- 30d rolling volatility — max (%): 8.1207
- Parquet: ``data/GraphEMD/xle_etf_analysis/xle.parquet``
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf

FECHA_INICIO: str = "2012-01-12"
FECHA_FIN: str = "2026-04-20"
N_OBSERVACIONES_ESPERADAS: int = 3587
VENTANA_VOLATILIDAD: int = 30
SIMBOLO_XLE: str = "XLE"
NOMBRE_ARCHIVO_PARQUET: str = "xle.parquet"

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DIR_DATOS = _REPO_ROOT / "data" / "GraphEMD" / "xle_etf_analysis"
_RUTA_MSCI_REFERENCIA = _REPO_ROOT / "data" / "20abr26" / "msci_world.parquet"

logger = logging.getLogger(__name__)


def _ruta_script() -> Path:
    """
    Return the absolute path of this script.

    Returns
    -------
    Path
        Path to the ``download_xle.py``.
    """
    return Path(__file__).resolve()


def descargar_xle_yahoo() -> pd.DataFrame:
    """
    Download the full historical series for XLE data from Yahoo Finance.

    Returns
    -------
    pd.DataFrame
        OHLCV with a datetime index (timezone-aware if Yahoo provides it).

    Raises
    ------
    ValueError
        If the download is empty or fails.
    """
    logger.info("Descargando %s data from Yahoo Finance...", SIMBOLO_XLE)
    ticker = yf.Ticker(SIMBOLO_XLE)
    df = ticker.history(period="max")
    if df is None or df.empty:
        raise ValueError(f"Could not download data for {SIMBOLO_XLE}.")
    logger.info(
        "Raw download: %s records, %s → %s",
        len(df),
        df.index.min(),
        df.index.max(),
    )
    return df


def _normalizar_indice_fechas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the index to calendar dates (no time) for series comparison.

    Parameters
    ----------
    df : pd.DataFrame
        Series with a ``DatetimeIndex`` index.

    Returns
    -------
    pd.DataFrame
        Copy with index at UTC midnight normalized to ``datetime64[ns]`` per day.
    """
    out = df.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(out.index))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    out.index = idx.normalize()
    return out.loc[~out.index.duplicated(keep="last")]


def recortar_ventana_msci(
    df: pd.DataFrame,
    fecha_inicio: str = FECHA_INICIO,
    fecha_fin: str = FECHA_FIN,
) -> pd.DataFrame:
    """
    Filter the DataFrame to the MSCI World study time window.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data for XLE.
    fecha_inicio : str
        Inclusive start date (``YYYY-MM-DD``).
    fecha_fin : str
        Inclusive end date (``YYYY-MM-DD``).

    Returns
    -------
    pd.DataFrame
        Subset between ``fecha_inicio`` and ``fecha_fin``.
    """
    df_norm = _normalizar_indice_fechas(df)
    inicio = pd.Timestamp(fecha_inicio)
    fin = pd.Timestamp(fecha_fin)
    mascara = (df_norm.index >= inicio) & (df_norm.index <= fin)
    recorte = df_norm.loc[mascara]
    logger.info(
        "Trim %s–%s: %s observations (%s → %s)",
        fecha_inicio,
        fecha_fin,
        len(recorte),
        recorte.index.min() if len(recorte) else "—",
        recorte.index.max() if len(recorte) else "—",
    )
    return recorte


def alinear_a_calendario_msci(
    df_xle: pd.DataFrame,
    ruta_msci: Path = _RUTA_MSCI_REFERENCIA,
) -> pd.DataFrame:
    """
    Reindexa XLE to the date calendar of ``msci_world.parquet``.

    Parameters
    ----------
    df_xle : pd.DataFrame
        XLE already trimmed to the time window.
    ruta_msci : Path
        Path to the MSCI World price parquet.

    Returns
    -------
    pd.DataFrame
        XLE with the same index as MSCI (only dates present in both).

    Raises
    ------
    FileNotFoundError
        If the reference file does not exist.
    ValueError
        If rows are missing after alignment relative to the MSCI calendar.
    """
    if not ruta_msci.is_file():
        raise FileNotFoundError(f"Not found: calendario MSCI: {ruta_msci}")
    df_msci = pd.read_parquet(ruta_msci, engine="pyarrow")
    fechas_msci = _normalizar_indice_fechas(df_msci).index
    xle_norm = _normalizar_indice_fechas(df_xle)
    alineado = xle_norm.reindex(fechas_msci)
    faltantes = alineado["Close"].isna().sum()
    if faltantes > 0:
        logger.warning(
            "%s MSCI dates without XLE quote; those rows are dropped.",
            faltantes,
        )
        alineado = alineado.dropna(subset=["Close"])
    logger.info(
        "MSCI alignment: %s observations (MSCI reference: %s)",
        len(alineado),
        len(fechas_msci),
    )
    return alineado


def calcular_metricas_precio(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute descriptive statistics on prices and daily returns.

    Parameters
    ----------
    df : pd.DataFrame
        Must include the ``Close`` column.

    Returns
    -------
    dict
        Metrics: close min/max, cumulative return, mean and std of daily
        returns (%), 30d rolling volatility (mean and max in %).
    """
    if "Close" not in df.columns:
        raise ValueError(f"Missing Close column. Columns: {list(df.columns)}")
    close = df["Close"].astype(float)
    retornos_pct = close.pct_change() * 100.0
    retornos_validos = retornos_pct.dropna()
    vol_30 = retornos_pct.rolling(window=VENTANA_VOLATILIDAD).std()
    vol_30_valida = vol_30.dropna()

    precio_inicial = float(np.asarray(close.iloc[0], dtype=np.float64))
    precio_final = float(np.asarray(close.iloc[-1], dtype=np.float64))
    retorno_cumulativo = (precio_final / precio_inicial - 1.0) * 100.0

    return {
        "n_observaciones": len(close),
        "fecha_inicio": str(close.index.min()),
        "fecha_fin": str(close.index.max()),
        "close_min": float(close.min()),
        "close_max": float(close.max()),
        "retorno_cumulativo_pct": retorno_cumulativo,
        "retorno_diario_media_pct": float(retornos_validos.mean()),
        "retorno_diario_std_pct": float(retornos_validos.std()),
        "volatilidad_30d_media_pct": float(vol_30_valida.mean())
        if len(vol_30_valida)
        else float("nan"),
        "volatilidad_30d_max_pct": float(vol_30_valida.max())
        if len(vol_30_valida)
        else float("nan"),
    }


def guardar_parquet(df: pd.DataFrame, directorio: Path = _DIR_DATOS) -> Path:
    """
    Persist the DataFrame to parquet.

    Parameters
    ----------
    df : pd.DataFrame
        Data to save.
    directorio : Path
        Destination folder.

    Returns
    -------
    Path
        Path of the written file.
    """
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / NOMBRE_ARCHIVO_PARQUET
    df.to_parquet(ruta, engine="pyarrow", index=True)
    logger.info("Saved: %s (%s rows)", ruta, len(df))
    return ruta


def _formatear_bloque_resultados(metricas: dict[str, Any]) -> str:
    """
    Generate text to document results in the module docstring.

    Parameters
    ----------
    metricas : dict
        Output of ``calcular_metricas_precio``.

    Returns
    -------
    str
        Formatted text block with metrics.
    """
    return (
        f"- Observations: {metricas['n_observaciones']}\n"
        f"- Range: {metricas['fecha_inicio']} → {metricas['fecha_fin']}\n"
        f"- Close minimum (USD): {metricas['close_min']:.4f}\n"
        f"- Close maximum (USD): {metricas['close_max']:.4f}\n"
        f"- Cumulative return (%): {metricas['retorno_cumulativo_pct']:.2f}\n"
        f"- Daily return — mean (%): {metricas['retorno_diario_media_pct']:.4f}\n"
        f"- Daily return — std dev (%): {metricas['retorno_diario_std_pct']:.4f}\n"
        f"- 30d rolling volatility — mean (%): {metricas['volatilidad_30d_media_pct']:.4f}\n"
        f"- 30d rolling volatility — max (%): {metricas['volatilidad_30d_max_pct']:.4f}"
    )


def imprimir_resumen(metricas: dict[str, Any], ruta_parquet: Path) -> None:
    """
    Log a readable summary of the computed metrics.

    Parameters
    ----------
    metricas : dict
        Series metrics.
    ruta_parquet : Path
        Generated parquet file.
    """
    logger.info("Parquet file: %s", ruta_parquet)
    logger.info("\n%s", _formatear_bloque_resultados(metricas))


def main() -> dict[str, Any]:
    """
    Download XLE, align to the MSCI window, save parquet, and compute metrics.

    Returns
    -------
    dict
        Dictionary with metrics and run metadata.
    """
    df_bruto = descargar_xle_yahoo()
    df_recorte = recortar_ventana_msci(df_bruto)

    if _RUTA_MSCI_REFERENCIA.is_file():
        df_final = alinear_a_calendario_msci(df_recorte)
    else:
        logger.warning(
            "Not found: %s; using date trim only.",
            _RUTA_MSCI_REFERENCIA,
        )
        df_final = df_recorte

    if len(df_final) != N_OBSERVACIONES_ESPERADAS:
        logger.warning(
            "Expected %s observations; got %s.",
            N_OBSERVACIONES_ESPERADAS,
            len(df_final),
        )

    metricas = calcular_metricas_precio(df_final)
    ruta = guardar_parquet(df_final)
    imprimir_resumen(metricas, ruta)
    return {"metricas": metricas, "ruta_parquet": str(ruta)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        main()
    except Exception:
        logger.exception("Error downloading or analyzing XLE")
        sys.exit(1)
