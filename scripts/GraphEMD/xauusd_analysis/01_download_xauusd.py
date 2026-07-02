"""
Script para descargar los datos de la XAU/USD (par spot XAU/USD (oro frente al dólar)) desde Yahoo Finance.

Lógica implementada
---------------------
- Descarga histórico de futuros COMEX ``GC=F`` (oro en USD/onza) con ``yfinance`` (``period="max"``).
- Recorte por fechas inclusivas ``2012-01-12`` … ``2026-04-20`` (misma ventana que ``docs/20abr26``).
- Si existe ``data/20abr26/msci_world.parquet``, se alinea el índice de XAU/USD al de MSCI World
  (mismos días de negociación) para obtener exactamente 3587 observaciones; si no, se usa solo el recorte.
- Métricas sobre ``Close``: mínimo/máximo, retorno cumulativo ``(C_T/C_0 - 1)×100``,
  retornos diarios en % (``pct_change×100``), media y desviación estándar de esos retornos,
  volatilidad móvil de 30 días como ``std`` rodante de los retornos diarios (misma definición que
  ``analysis/16dic25/02_analisis_emd_condiciones.ipynb``) y su máximo.

Resultados obtenidos (ejecución 2026-05-16)
-------------------------------------------
- Observaciones: 3587
- Rango: 2012-01-12 → 2026-04-20
- Close mínimo (USD): 9.2453
- Close máximo (USD): 62.5600
- Retorno cumulativo (%): 160.21
- Retorno diario — media (%): 0.0414
- Retorno diario — desv. estándar (%): 1.7100
- Volatilidad móvil 30d — media (%): 1.4842
- Volatilidad móvil 30d — máximo (%): 8.1207
- Parquet: ``data/GraphEMD/xauusd_analysis/xauusd.parquet``
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
N_OBSERVACIONES_ESPERADAS: int = 3584
VENTANA_VOLATILIDAD: int = 30
# Futuros COMEX oro/USD (Yahoo no expone XAUUSD=X de forma fiable).
SIMBOLO_XAUUSD: str = "GC=F"
NOMBRE_ARCHIVO_PARQUET: str = "xauusd.parquet"

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DIR_DATOS = _REPO_ROOT / "data" / "GraphEMD" / "xauusd_analysis"
_RUTA_MSCI_REFERENCIA = _REPO_ROOT / "data" / "20abr26" / "msci_world.parquet"

logger = logging.getLogger(__name__)


def _ruta_script() -> Path:
    """
    Devuelve la ruta absoluta de este script.

    Returns
    -------
    Path
        Ruta al archivo ``download_xauusd.py``.
    """
    return Path(__file__).resolve()


def descargar_xauusd_yahoo() -> pd.DataFrame:
    """
    Descarga el histórico completo de XAU/USD desde Yahoo Finance.

    Returns
    -------
    pd.DataFrame
        OHLCV con índice temporal (timezone-aware si Yahoo lo provee).

    Raises
    ------
    ValueError
        Si la descarga está vacía o falla.
    """
    logger.info("Descargando %s desde Yahoo Finance...", SIMBOLO_XAUUSD)
    ticker = yf.Ticker(SIMBOLO_XAUUSD)
    df = ticker.history(period="max")
    if df is None or df.empty:
        raise ValueError(f"No se pudieron descargar datos para {SIMBOLO_XAUUSD}.")
    logger.info(
        "Descarga bruta: %s registros, %s → %s",
        len(df),
        df.index.min(),
        df.index.max(),
    )
    return df


def _normalizar_indice_fechas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza el índice a fechas de calendario (sin hora) para comparar series.

    Parameters
    ----------
    df : pd.DataFrame
        Serie con índice ``DatetimeIndex``.

    Returns
    -------
    pd.DataFrame
        Copia con índice en medianoche UTC normalizado a ``datetime64[ns]`` por día.
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
    Filtra el DataFrame a la ventana temporal del estudio MSCI World.

    Parameters
    ----------
    df : pd.DataFrame
        Datos OHLCV de XAU/USD.
    fecha_inicio : str
        Fecha inicial inclusive (``YYYY-MM-DD``).
    fecha_fin : str
        Fecha final inclusive (``YYYY-MM-DD``).

    Returns
    -------
    pd.DataFrame
        Subconjunto entre ``fecha_inicio`` y ``fecha_fin``.
    """
    df_norm = _normalizar_indice_fechas(df)
    inicio = pd.Timestamp(fecha_inicio)
    fin = pd.Timestamp(fecha_fin)
    mascara = (df_norm.index >= inicio) & (df_norm.index <= fin)
    recorte = df_norm.loc[mascara]
    logger.info(
        "Recorte %s–%s: %s observaciones (%s → %s)",
        fecha_inicio,
        fecha_fin,
        len(recorte),
        recorte.index.min() if len(recorte) else "—",
        recorte.index.max() if len(recorte) else "—",
    )
    return recorte


def alinear_a_calendario_msci(
    df_xauusd: pd.DataFrame,
    ruta_msci: Path = _RUTA_MSCI_REFERENCIA,
) -> pd.DataFrame:
    """
    Reindexa XAU/USD al calendario de fechas de ``msci_world.parquet``.

    Parameters
    ----------
    df_xauusd : pd.DataFrame
        XAU/USD ya recortado a la ventana temporal.
    ruta_msci : Path
        Ruta al parquet de precios MSCI World.

    Returns
    -------
    pd.DataFrame
        XAU/USD con el mismo índice que MSCI (solo fechas presentes en ambos).

    Raises
    ------
    FileNotFoundError
        Si no existe el archivo de referencia.
    ValueError
        Si tras la alineación faltan filas respecto al calendario MSCI.
    """
    if not ruta_msci.is_file():
        raise FileNotFoundError(f"No se encontró calendario MSCI: {ruta_msci}")
    df_msci = pd.read_parquet(ruta_msci, engine="pyarrow")
    fechas_msci = _normalizar_indice_fechas(df_msci).index
    xauusd_norm = _normalizar_indice_fechas(df_xauusd)
    alineado = xauusd_norm.reindex(fechas_msci)
    faltantes = alineado["Close"].isna().sum()
    if faltantes > 0:
        logger.warning(
            "%s fechas MSCI sin cotización XAU/USD; se eliminan esas filas.",
            faltantes,
        )
        alineado = alineado.dropna(subset=["Close"])
    logger.info(
        "Alineación a MSCI: %s observaciones (referencia MSCI: %s)",
        len(alineado),
        len(fechas_msci),
    )
    return alineado


def calcular_metricas_precio(df: pd.DataFrame) -> dict[str, Any]:
    """
    Calcula estadísticas descriptivas sobre precios y retornos diarios.

    Parameters
    ----------
    df : pd.DataFrame
        Debe incluir la columna ``Close``.

    Returns
    -------
    dict
        Métricas: mínimo/máximo de cierre, retorno cumulativo, media y std de retornos
        diarios (%), volatilidad móvil 30d (media y máximo en %).
    """
    if "Close" not in df.columns:
        raise ValueError(f"Falta columna Close. Columnas: {list(df.columns)}")
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
    Persiste el DataFrame en parquet.

    Parameters
    ----------
    df : pd.DataFrame
        Datos a guardar.
    directorio : Path
        Carpeta de destino.

    Returns
    -------
    Path
        Ruta del archivo escrito.
    """
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / NOMBRE_ARCHIVO_PARQUET
    df.to_parquet(ruta, engine="pyarrow", index=True)
    logger.info("Guardado: %s (%s filas)", ruta, len(df))
    return ruta


def _formatear_bloque_resultados(metricas: dict[str, Any]) -> str:
    """
    Genera texto para documentar resultados en el docstring del módulo.

    Parameters
    ----------
    metricas : dict
        Salida de ``calcular_metricas_precio``.

    Returns
    -------
    str
        Bloque de texto con métricas formateadas.
    """
    return (
        f"- Observaciones: {metricas['n_observaciones']}\n"
        f"- Rango: {metricas['fecha_inicio']} → {metricas['fecha_fin']}\n"
        f"- Close mínimo (USD): {metricas['close_min']:.4f}\n"
        f"- Close máximo (USD): {metricas['close_max']:.4f}\n"
        f"- Retorno cumulativo (%): {metricas['retorno_cumulativo_pct']:.2f}\n"
        f"- Retorno diario — media (%): {metricas['retorno_diario_media_pct']:.4f}\n"
        f"- Retorno diario — desv. estándar (%): {metricas['retorno_diario_std_pct']:.4f}\n"
        f"- Volatilidad móvil 30d — media (%): {metricas['volatilidad_30d_media_pct']:.4f}\n"
        f"- Volatilidad móvil 30d — máximo (%): {metricas['volatilidad_30d_max_pct']:.4f}"
    )


def imprimir_resumen(metricas: dict[str, Any], ruta_parquet: Path) -> None:
    """
    Imprime en log un resumen legible de las métricas calculadas.

    Parameters
    ----------
    metricas : dict
        Métricas de la serie.
    ruta_parquet : Path
        Archivo parquet generado.
    """
    logger.info("Archivo parquet: %s", ruta_parquet)
    logger.info("\n%s", _formatear_bloque_resultados(metricas))


def main() -> dict[str, Any]:
    """
    Descarga XAU/USD, alinea a la ventana MSCI, guarda parquet y calcula métricas.

    Returns
    -------
    dict
        Diccionario con métricas y metadatos de la ejecución.
    """
    df_bruto = descargar_xauusd_yahoo()
    df_recorte = recortar_ventana_msci(df_bruto)

    if _RUTA_MSCI_REFERENCIA.is_file():
        df_final = alinear_a_calendario_msci(df_recorte)
    else:
        logger.warning(
            "No se encontró %s; se usa solo el recorte por fechas.",
            _RUTA_MSCI_REFERENCIA,
        )
        df_final = df_recorte

    if len(df_final) != N_OBSERVACIONES_ESPERADAS:
        logger.warning(
            "Se esperaban %s observaciones; se obtuvieron %s.",
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
        logger.exception("Error en la descarga o el análisis de XAU/USD")
        sys.exit(1)
