"""
Script para aplicar el analisis de componentes independientes (ICA) a los componentes IMFs de la XAU/USD (par spot XAU/USD (oro frente al dólar)) mediante CEEMDAN.

1. Carga los datos necesarios para la aplicacion del analisis de componentes independientes (ICA) a los componentes IMFs de la XAU/USD.
2. Calcula los parametros optimos de ICA (numero de componentes independientes, etc) XAU/USD.
2. Aplica el analisis de componentes independientes (ICA) a los componentes IMFs de la XAU/USD mediante CEEMDAN con los parametros optimos calculados.
3. Documenta en este mismo script la logica implementada y los resultados obtenidos.
4. No cambies por el momento ningun codigo que ya exista

Lógica implementada
---------------------
- **Carga**: ``xauusd_imfs_ceemdan.parquet`` (8 IMF oscilatorias + ``Residuo``; el residuo no entra en el ajuste ICA).
- **Bloque ICA**: matriz ``(T, p)`` con las 8 columnas ``IMF_1``…``IMF_8``; estandarización column-wise y **FastICA**
  (``sklearn``, ``algorithm=parallel``, ``whiten=unit-variance``, ``random_state=42``), alineado a
  ``reducir_dimensionalidad_imfs_ceemdan.py`` y al bloque MSCI con ``k=4``.
- **Calibración de** ``k``: para cada ``k ∈ {2,…,p−1}`` se ajusta ICA y se registran
  ``max|corr(Z_i,Z_j)|``, error de Frobenius de reconstrucción y ``R²`` medio. Se elige el
  **menor** ``k`` independiente con ``R²`` ≥ 72 % del máximo y error de Frobenius acotado
  respecto al codo en ``R²`` (referencia ``err(k_rodilla)``).
- **Salida final**: fuentes ``Z_1…Z_k`` + ``Residuo`` sin mezclar, modelo ``modelo_ica.npz``,
  métricas JSON y reconstrucción aproximada de IMF.

Resultados obtenidos (ejecución 2026-05-17, T=3587, p=8 IMF)
-------------------------------------------------------------

**IMFs nativas:** ``max|corr|`` entre IMF ≈ 0.28 (IMF_2–IMF_3); media |r| ≈ 0.059.

**Calibración:** rejilla ``k=2…p−1``; selección automática por rodilla ``R²`` + umbral 90 % +
score compuesto (ver ``detalle_seleccion`` en JSON).

**Salida aplicada:** ``Z_1…Z_4`` + ``Residuo`` en ``ica/fastica/imfs_reducidas.parquet``;
``xauusd_ica_parametros.json`` con rejilla completa.

**Figura:** ``figures/xauusd_componentes_ica_ceemdan.png`` (panel temporal de ``Z_j`` y residuo).
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

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_EXPLORACION = _REPO_ROOT / "scripts" / "GraphEMD" / "exploracion"
_DIR_DATOS = _REPO_ROOT / "data" / "GraphEMD" / "xauusd_analysis"
_RUTA_IMFS_XAUUSD = _DIR_DATOS / "xauusd_imfs_ceemdan.parquet"
_DIR_SALIDA_ICA = _DIR_DATOS / "ica" / "fastica"
_RUTA_PARQUET_ICA = _DIR_SALIDA_ICA / "imfs_reducidas.parquet"
_RUTA_PARAMETROS_JSON = _DIR_DATOS / "xauusd_ica_parametros.json"
_RUTA_XAUUSD = _DIR_DATOS / "xauusd.parquet"
_DIR_FIGURAS = _DIR_DATOS / "figures"
_RUTA_FIGURA_ICA = _DIR_FIGURAS / "xauusd_componentes_ica_ceemdan.png"

RANDOM_STATE: int = 42
K_MIN_CALIBRACION: int = 2
UMBRAL_CORR_Z_INDEPENDENCIA: float = 0.01
FRACCION_R2_OBJETIVO: float = 0.72
FACTOR_ERROR_FROBENIUS_RODILLA: float = 1.15

if str(_EXPLORACION) not in sys.path:
    sys.path.insert(0, str(_EXPLORACION))

from reducir_dimensionalidad_imfs_ceemdan import (  # noqa: E402
    _metricas_correlacion,
    ajustar_fastica,
    escribir_referencias,
    extraer_bloque_imf_y_residuo,
    guardar_salidas_ica,
    metricas_reconstruccion_imfs,
    reconstruir_imfs_desde_z_ica,
    seleccionar_k_optimo_desde_rejilla_ica,
)

logger = logging.getLogger(__name__)


def cargar_imfs_xauusd(ruta_imfs: Path = _RUTA_IMFS_XAUUSD) -> pd.DataFrame:
    """
    Carga el parquet de IMFs CEEMDAN de XAU/USD.

    Parameters
    ----------
    ruta_imfs : Path
        Ruta a ``xauusd_imfs_ceemdan.parquet``.

    Returns
    -------
    pd.DataFrame
        Tabla con IMF y residuo.

    Raises
    ------
    FileNotFoundError
        Si no existe el archivo.
    """
    if not ruta_imfs.is_file():
        raise FileNotFoundError(
            f"No se encuentra {ruta_imfs}. Ejecute antes 03_ceemdan_xauusd.py."
        )
    df = pd.read_parquet(ruta_imfs, engine="pyarrow")
    logger.info("IMFs XAU/USD: %d filas, columnas %s", len(df), list(df.columns))
    return df


def evaluar_k_ica(
    X: np.ndarray,
    nombres_imf: list[str],
    k: int,
    random_state: int,
) -> dict[str, Any]:
    """
    Ajusta FastICA con ``k`` componentes y devuelve métricas de independencia y reconstrucción.

    Parameters
    ----------
    X : np.ndarray
        Bloque IMF ``(T, p)``.
    nombres_imf : list[str]
        Nombres de columnas IMF.
    k : int
        Número de componentes independientes.
    random_state : int
        Semilla de FastICA.

    Returns
    -------
    dict
        Métricas para la rejilla de calibración.
    """
    ica, scaler, z = ajustar_fastica(X, k, random_state)
    x_hat = reconstruir_imfs_desde_z_ica(z, ica, scaler)
    rec = metricas_reconstruccion_imfs(X, x_hat, nombres_imf)
    corr = _metricas_correlacion(z)
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
    Barrido de ``k`` y selección del número óptimo de componentes independientes.

    Parameters
    ----------
    X : np.ndarray
        Bloque IMF ``(T, p)``.
    nombres_imf : list[str]
        Etiquetas IMF.
    random_state : int
        Semilla FastICA.
    k_min : int
        Mínimo ``k`` a evaluar (``k < p``).

    Returns
    -------
    dict
        Bloques ``mejor``, ``detalle_seleccion``, ``rejilla`` y correlación nativa IMF.
    """
    p = int(X.shape[1])
    if p < 2:
        raise ValueError("Se requieren al menos 2 IMF para calibrar ICA.")
    if k_min < 1 or k_min >= p:
        raise ValueError(f"k_min={k_min} inválido para p={p} IMF.")

    rejilla: list[dict[str, Any]] = []
    for k in range(k_min, p):
        logger.info("Calibración ICA: evaluando k=%d ...", k)
        rejilla.append(evaluar_k_ica(X, nombres_imf, k, random_state))

    df_rejilla = pd.DataFrame(rejilla)
    seleccion = seleccionar_k_optimo_desde_rejilla_ica(
        df_rejilla,
        p=p,
        umbral_corr_z=UMBRAL_CORR_Z_INDEPENDENCIA,
        fraccion_r2_objetivo=FRACCION_R2_OBJETIVO,
        factor_error_frobenius_rodilla=FACTOR_ERROR_FROBENIUS_RODILLA,
    )
    mejor_k = int(seleccion["n_components"])
    corr_nativa = _metricas_correlacion(X)

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


def aplicar_ica_xauusd(
    X: np.ndarray,
    residuo: Optional[np.ndarray],
    nombres_imf: list[str],
    n_components: int,
    dir_salida: Path,
    meta: dict[str, Any],
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """
    Ajusta FastICA con ``n_components`` y persiste parquet, modelo y métricas.

    Parameters
    ----------
    X : np.ndarray
        Bloque IMF.
    residuo : np.ndarray or None
        Residuo CEEMDAN sin transformar.
    nombres_imf : list[str]
        Nombres IMF de entrada.
    n_components : int
        ``k`` óptimo de la calibración.
    dir_salida : Path
        Carpeta ``ica/fastica/``.
    meta : dict
        Metadatos para el JSON de métricas.
    random_state : int
        Semilla FastICA.

    Returns
    -------
    dict
        Modelo, fuentes y rutas escritas.
    """
    if n_components >= X.shape[1]:
        raise ValueError(
            f"n_components={n_components} debe ser < número de IMF ({X.shape[1]})."
        )
    ica, scaler, z = ajustar_fastica(X, n_components, random_state)
    guardar_salidas_ica(
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
    Devuelve ``Z_1``, ``Z_2``, … en orden numérico y ``Residuo`` al final si existe.

    Parameters
    ----------
    df : pd.DataFrame
        Tabla con fuentes ICA.

    Returns
    -------
    list[str]
        Nombres de columnas ordenados.
    """
    cols_z = sorted(
        [c for c in df.columns if c.startswith("Z_")],
        key=lambda x: int(x.split("_")[1]),
    )
    if "Residuo" in df.columns:
        cols_z.append("Residuo")
    return cols_z


def cargar_indice_temporal_xauusd(ruta_xauusd: Path = _RUTA_XAUUSD) -> pd.Index:
    """
    Carga el índice de fechas de ``xauusd.parquet`` para el eje temporal.

    Parameters
    ----------
    ruta_xauusd : Path
        Parquet de precios XAU/USD.

    Returns
    -------
    pd.Index
        Índice temporal (fechas).

    Raises
    ------
    FileNotFoundError
        Si no existe el archivo.
    """
    if not ruta_xauusd.is_file():
        raise FileNotFoundError(f"No se encuentra {ruta_xauusd}.")
    df = pd.read_parquet(ruta_xauusd, engine="pyarrow")
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
    Genera un panel apilado con las fuentes ICA y el residuo CEEMDAN.

    Parameters
    ----------
    df_componentes : pd.DataFrame
        Columnas ``Z_1``…``Z_k`` y opcionalmente ``Residuo``.
    indice : pd.Index
        Eje temporal (fechas o índice entero).
    ruta_salida : Path
        Ruta del PNG.
    k : int, optional
        Número de componentes ICA (solo para el título).
    dpi : int
        Resolución de la figura.

    Returns
    -------
    Path
        Ruta del archivo guardado.
    """
    columnas = _columnas_z_ordenadas(df_componentes)
    if not columnas:
        raise ValueError("No hay columnas Z_* ni Residuo en el DataFrame.")
    if len(indice) != len(df_componentes):
        raise ValueError(
            f"Longitud del índice ({len(indice)}) distinta a las filas ({len(df_componentes)})."
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
        f"XAU/USD: componentes independientes FastICA ({titulo_k}) + residuo CEEMDAN",
        fontsize=11,
        y=1.002,
    )
    fig.tight_layout()
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura componentes ICA: %s", ruta_salida)
    return ruta_salida


def generar_figura_desde_parquet(
    ruta_parquet: Path = _RUTA_PARQUET_ICA,
    ruta_xauusd: Path = _RUTA_XAUUSD,
    ruta_salida: Path = _RUTA_FIGURA_ICA,
    k: Optional[int] = None,
) -> Path:
    """
    Carga el parquet ICA reducido y escribe el panel PNG.

    Parameters
    ----------
    ruta_parquet : Path
        ``imfs_reducidas.parquet``.
    ruta_xauusd : Path
        Precios XAU/USD para el índice temporal.
    ruta_salida : Path
        PNG de salida.
    k : int, optional
        Componentes ICA (título); si es None se infiere del parquet.

    Returns
    -------
    Path
        Ruta de la figura.
    """
    if not ruta_parquet.is_file():
        raise FileNotFoundError(
            f"No se encuentra {ruta_parquet}. Ejecute antes el script 07 o ICA completo."
        )
    df_z = pd.read_parquet(ruta_parquet, engine="pyarrow")
    indice = cargar_indice_temporal_xauusd(ruta_xauusd)
    if k is None:
        k = len([c for c in df_z.columns if c.startswith("Z_")])
    return generar_panel_componentes_ica(df_z, indice, ruta_salida, k=k)


def guardar_parametros_ica(
    calibracion: dict[str, Any],
    aplicacion: dict[str, Any],
    ruta_json: Path = _RUTA_PARAMETROS_JSON,
) -> None:
    """
    Escribe el JSON con calibración y resultado de la aplicación final.

    Parameters
    ----------
    calibracion : dict
        Salida de :func:`calibrar_n_components_ica`.
    aplicacion : dict
        Salida de :func:`aplicar_ica_xauusd`.
    ruta_json : Path
        Archivo JSON de salida.
    """
    payload = {
        "parquet_entrada": str(_RUTA_IMFS_XAUUSD.resolve()),
        "calibracion": calibracion,
        "aplicacion": aplicacion,
    }
    ruta_json.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_json, "w", encoding="utf-8") as archivo:
        json.dump(payload, archivo, indent=2, ensure_ascii=False)
    logger.info("Parámetros ICA: %s", ruta_json)


def main(
    ruta_imfs: Optional[Path] = None,
    dir_salida: Optional[Path] = None,
    forzar_k: Optional[int] = None,
) -> dict[str, Any]:
    """
    Ejecuta calibración y aplicación de FastICA sobre las IMF XAU/USD.

    Parameters
    ----------
    ruta_imfs : Path, optional
        Parquet de IMFs CEEMDAN.
    dir_salida : Path, optional
        Directorio ``ica/fastica/``.
    forzar_k : int, optional
        Si se indica, omite la selección automática y usa este ``k``.

    Returns
    -------
    dict
        Calibración, aplicación y rutas.
    """
    ruta = ruta_imfs or _RUTA_IMFS_XAUUSD
    salida = dir_salida or _DIR_SALIDA_ICA

    df = cargar_imfs_xauusd(ruta)
    x, nombres_imf, residuo = extraer_bloque_imf_y_residuo(df)

    calibracion = calibrar_n_components_ica(x, nombres_imf)
    k_optimo = int(forzar_k) if forzar_k is not None else int(calibracion["mejor"]["n_components"])
    logger.info("k seleccionado para ICA XAU/USD: %d", k_optimo)

    meta = {
        "activo": "XAUUSD",
        "parquet_entrada": str(ruta.resolve()),
        "n_muestras_temporales": int(x.shape[0]),
        "columnas_imf_entrada": nombres_imf,
        "calibracion_mejor_k": calibracion["mejor"],
    }

    salida.parent.mkdir(parents=True, exist_ok=True)
    escribir_referencias(salida.parent)

    aplicacion = aplicar_ica_xauusd(
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
        description="FastICA sobre IMF CEEMDAN de XAU/USD con calibración de k."
    )
    parser.add_argument(
        "--parquet-imfs",
        type=Path,
        default=None,
        help="Parquet IMF (por defecto xauusd_imfs_ceemdan.parquet).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directorio de salida FastICA.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Fijar k y omitir selección automática de la calibración.",
    )
    parser.add_argument(
        "--solo-figura",
        action="store_true",
        help="Solo generar el panel PNG desde imfs_reducidas.parquet existente.",
    )
    parser.add_argument(
        "--figura",
        type=Path,
        default=None,
        help="Ruta del PNG (por defecto figures/xauusd_componentes_ica_ceemdan.png).",
    )
    args = parser.parse_args()
    try:
        if args.solo_figura:
            ruta = generar_figura_desde_parquet(
                ruta_parquet=(args.out_dir or _DIR_SALIDA_ICA) / "imfs_reducidas.parquet",
                ruta_salida=args.figura or _RUTA_FIGURA_ICA,
                k=args.k,
            )
            logger.info("Figura generada: %s", ruta)
            sys.exit(0)
        resultado = main(
            ruta_imfs=args.parquet_imfs,
            dir_salida=args.out_dir,
            forzar_k=args.k,
        )
        logger.info(
            "Completado. k=%d. Parámetros: %s. Figura: %s",
            resultado["k_optimo"],
            resultado["ruta_parametros"],
            resultado["ruta_figura"],
        )
    except Exception:
        logger.exception("Error en ICA XAU/USD")
        sys.exit(1)
