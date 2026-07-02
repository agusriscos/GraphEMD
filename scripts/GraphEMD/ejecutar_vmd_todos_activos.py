#!/usr/bin/env python3
"""
Pipeline VMD para todas las señales financieras del proyecto GraphEMD.

Ejecuta, por activo (MSCI World, XLE, XLP, XLV, XAU/USD):

1. Calibración de hiperparámetros VMD (rejilla K × alpha × DC).
2. Descomposición y persistencia de modos en parquet.
3. Validación (reconstrucción, residuo, métricas por IMF, figura panel).
4. Transformaciones a grafos (HVG, NVG, recurrencia).
5. FastICA con selección automática del k óptimo.

Ejemplo::

    PYTHONPATH=src/python python scripts/GraphEMD/ejecutar_vmd_todos_activos.py

    PYTHONPATH=src/python python scripts/GraphEMD/ejecutar_vmd_todos_activos.py \\
        --activos xle,xauusd --solo-paso ica
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC_PYTHON = _REPO_ROOT / "src" / "python"
_EXPLORACION = _REPO_ROOT / "scripts" / "GraphEMD" / "exploracion"
_EMDSYNTH = _REPO_ROOT / "scripts" / "GraphEMD" / "emdsynth"

N_OBSERVACIONES_ESPERADAS: int = 3587
UMBRAL_RMSE_RELATIVO: float = 1e-10

REJILLA_VMD_K: List[int] = [6, 8, 10, 12]
REJILLA_VMD_ALPHA: List[float] = [1000.0, 2000.0, 5000.0]
REJILLA_VMD_DC: List[int] = [0, 1]

TAU_MAX: int = 50
DIM_MAX: int = 10
UMBRAL_PERCENTIL_RECURRENCIA: float = 10.0
RANDOM_STATE_RECURRENCIA: int = 42

RANDOM_STATE_ICA: int = 42
K_MIN_CALIBRACION_ICA: int = 2
UMBRAL_CORR_Z: float = 0.01
FRACCION_R2_OBJETIVO: float = 0.72
FACTOR_ERROR_FROB_RODILLA: float = 1.15

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfigActivo:
    """
    Configuración de rutas y metadatos para un activo financiero.

    Attributes
    ----------
    id_activo : str
        Identificador corto (p. ej. ``xle``).
    nombre : str
        Etiqueta legible.
    dir_datos : Path
        Directorio de datos del activo.
    ruta_precios : Path
        Parquet con precios de cierre.
    prefijo : str
        Prefijo de archivos de salida (p. ej. ``xle`` o ``msci_world``).
    columna_precio : str
        Nombre de la columna de precios.
    """

    id_activo: str
    nombre: str
    dir_datos: Path
    ruta_precios: Path
    prefijo: str
    columna_precio: str = "Close"


ACTIVOS: Tuple[ConfigActivo, ...] = (
    ConfigActivo(
        id_activo="msci_world",
        nombre="MSCI World",
        dir_datos=_REPO_ROOT / "data" / "20abr26",
        ruta_precios=_REPO_ROOT / "data" / "20abr26" / "msci_world.parquet",
        prefijo="msci_world",
    ),
    ConfigActivo(
        id_activo="xle",
        nombre="XLE",
        dir_datos=_REPO_ROOT / "data" / "GraphEMD" / "xle_etf_analysis",
        ruta_precios=_REPO_ROOT
        / "data"
        / "GraphEMD"
        / "xle_etf_analysis"
        / "xle.parquet",
        prefijo="xle",
    ),
    ConfigActivo(
        id_activo="xlp",
        nombre="XLP",
        dir_datos=_REPO_ROOT / "data" / "GraphEMD" / "xlp_analysis",
        ruta_precios=_REPO_ROOT / "data" / "GraphEMD" / "xlp_analysis" / "xlp.parquet",
        prefijo="xlp",
    ),
    ConfigActivo(
        id_activo="xlv",
        nombre="XLV",
        dir_datos=_REPO_ROOT / "data" / "GraphEMD" / "xlv_analysis",
        ruta_precios=_REPO_ROOT / "data" / "GraphEMD" / "xlv_analysis" / "xlv.parquet",
        prefijo="xlv",
    ),
    ConfigActivo(
        id_activo="xauusd",
        nombre="XAU/USD",
        dir_datos=_REPO_ROOT / "data" / "GraphEMD" / "xauusd_analysis",
        ruta_precios=_REPO_ROOT
        / "data"
        / "GraphEMD"
        / "xauusd_analysis"
        / "xauusd.parquet",
        prefijo="xauusd",
    ),
)


def _asegurar_paths() -> None:
    """
    Añade al ``sys.path`` las rutas de importación del proyecto.
    """
    for ruta in (_SRC_PYTHON, _EXPLORACION):
        s = str(ruta)
        if s not in sys.path:
            sys.path.insert(0, s)


def _cargar_modulo_emdsynth() -> Any:
    """
    Carga el pipeline emdsynth con ``descomponer_vmd`` y ``calcular_metricas``.

    Returns
    -------
    module
        Módulo ``ejecutar_descomposiciones_emdsynth``.
    """
    ruta = _EMDSYNTH / "ejecutar_descomposiciones_emdsynth.py"
    spec = importlib.util.spec_from_file_location("emdsynth_vmd_activos", ruta)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar {ruta}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cargar_modulo_validacion() -> Any:
    """
    Carga funciones de validación desde el script 04 de XLE.

    Returns
    -------
    module
        Módulo con ``validar_reconstruccion``, ``evaluar_residuo``, etc.
    """
    ruta = (
        _REPO_ROOT
        / "scripts"
        / "GraphEMD"
        / "xle_etf_analysis"
        / "04_ceemdan_xle_validation_and_results.py"
    )
    spec = importlib.util.spec_from_file_location("validacion_vmd_activos", ruta)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar {ruta}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cargar_modulo_ica() -> Any:
    """
    Carga utilidades ICA desde ``reducir_dimensionalidad_imfs_ceemdan``.

    Returns
    -------
    module
        Módulo con FastICA, selección de k y persistencia.
    """
    _asegurar_paths()
    ruta = _EXPLORACION / "reducir_dimensionalidad_imfs_ceemdan.py"
    spec = importlib.util.spec_from_file_location("ica_vmd_activos", ruta)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar {ruta}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rutas_activo(cfg: ConfigActivo) -> Dict[str, Path]:
    """
    Devuelve rutas estándar de salida VMD para un activo.

    Parameters
    ----------
    cfg : ConfigActivo
        Configuración del activo.

    Returns
    -------
    dict
        Mapa de nombres lógicos a rutas.
    """
    d = cfg.dir_datos
    p = cfg.prefijo
    return {
        "imfs": d / f"{p}_imfs_vmd.parquet",
        "parametros": d / f"{p}_vmd_parametros.json",
        "validacion": d / f"{p}_validacion_vmd.json",
        "metricas_imf_csv": d / f"{p}_imf_metricas_vmd.csv",
        "dir_grafos": d / "grafos_vmd",
        "resumen_grafos_csv": d / f"{p}_resumen_grafos_vmd.csv",
        "param_recurrencia_csv": d / f"{p}_parametros_recurrencia_vmd.csv",
        "manifest_grafos": d / f"{p}_grafos_vmd_manifest.json",
        "dir_ica": d / "ica_vmd" / "fastica",
        "parametros_ica": d / f"{p}_ica_vmd_parametros.json",
        "dir_figuras": d / "figures",
        "figura_panel": d / "figures" / f"{p}_imfs_panel_vmd.png",
        "figura_ica": d / "figures" / f"{p}_componentes_ica_vmd.png",
        "figura_comparativa_residuos": d
        / "figures"
        / f"{p}_imf_decomposition_ceemdan_eemd_vmd.png",
        "imfs_ceemdan": d / f"{p}_imfs_ceemdan.parquet",
        "imfs_eemd": d / f"{p}_imfs_eemd.parquet",
    }


def cargar_serie_precios(cfg: ConfigActivo) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Carga precios de cierre del activo.

    Parameters
    ----------
    cfg : ConfigActivo
        Configuración del activo.

    Returns
    -------
    tuple
        DataFrame de precios e índice, vector 1D de cierre.

    Raises
    ------
    FileNotFoundError
        Si no existe el parquet de precios.
    """
    if not cfg.ruta_precios.is_file():
        raise FileNotFoundError(
            f"No se encontró {cfg.ruta_precios}. Ejecute el script de descarga."
        )
    df = pd.read_parquet(cfg.ruta_precios, engine="pyarrow")
    if cfg.columna_precio not in df.columns:
        raise ValueError(
            f"Falta columna {cfg.columna_precio} en {cfg.ruta_precios}. "
            f"Columnas: {list(df.columns)}"
        )
    if len(df) != N_OBSERVACIONES_ESPERADAS:
        logger.warning(
            "%s: se esperaban %s observaciones; hay %s.",
            cfg.nombre,
            N_OBSERVACIONES_ESPERADAS,
            len(df),
        )
    serie = np.asarray(df[cfg.columna_precio].values, dtype=np.float64)
    logger.info(
        "%s: serie cargada n=%s, rango %s → %s",
        cfg.nombre,
        len(serie),
        df.index.min(),
        df.index.max(),
    )
    return df, serie


def imfs_array_a_dataframe(imfs: np.ndarray) -> pd.DataFrame:
    """
    Convierte modos VMD (última fila = residuo) en DataFrame IMF_* + Residuo.

    Parameters
    ----------
    imfs : np.ndarray
        Matriz ``(n_modos, n_muestras)``.

    Returns
    -------
    pd.DataFrame
        Columnas ``IMF_1`` … ``IMF_K`` y ``Residuo``.
    """
    n_imfs = imfs.shape[0] - 1
    datos: Dict[str, np.ndarray] = {}
    for i in range(n_imfs):
        datos[f"IMF_{i + 1}"] = imfs[i]
    datos["Residuo"] = imfs[-1]
    return pd.DataFrame(datos)


def _columnas_imf_ordenadas(df: pd.DataFrame) -> List[str]:
    """
    Devuelve nombres ``IMF_*`` ordenados numéricamente.

    Parameters
    ----------
    df : pd.DataFrame
        Tabla con columnas IMF.

    Returns
    -------
    list of str
        Columnas IMF en orden.
    """
    cols = [c for c in df.columns if c.startswith("IMF_")]
    return sorted(cols, key=lambda x: int(x.split("_")[1]))


def _alinear_df_imfs_longitud(df_imfs: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Recorta o valida un DataFrame de IMFs para que coincida con ``n`` muestras.

    Parameters
    ----------
    df_imfs : pd.DataFrame
        IMFs CEEMDAN/EEMD/VMD.
    n : int
        Longitud objetivo (p. ej. precios de cierre).

    Returns
    -------
    pd.DataFrame
        Tabla alineada a ``n`` filas.

    Raises
    ------
    ValueError
        Si hay menos filas que ``n``.
    """
    if len(df_imfs) == n:
        return df_imfs
    if len(df_imfs) > n:
        logger.warning(
            "IMFs con %s filas; se recortan las primeras %s para alinear a la señal "
            "(convención ``_alinear_serie_e_imfs`` en validación CEEMDAN).",
            len(df_imfs),
            n,
        )
        return df_imfs.iloc[:n].copy()
    raise ValueError(
        f"IMFs con {len(df_imfs)} filas; se requieren al menos {n} para alinear."
    )


def _brecha_residuo_implicito(serie: np.ndarray, df_imfs: pd.DataFrame) -> np.ndarray:
    """
    Calcula ``|x - Σ IMF_*|`` (misma definición que CEEMDAN/EEMD en el panel de residuos).

    Parameters
    ----------
    serie : np.ndarray
        Señal original.
    df_imfs : pd.DataFrame
        Descomposición con columnas ``IMF_*``.

    Returns
    -------
    np.ndarray
        Magnitud de la brecha de reconstrucción sin columna ``Residuo``.
    """
    cols = _columnas_imf_ordenadas(df_imfs)
    suma = np.sum(
        [np.asarray(df_imfs[c].values, dtype=np.float64) for c in cols],
        axis=0,
    )
    return np.abs(serie - suma)


def _suma_imfs_sin_residuo(df_imfs: pd.DataFrame) -> np.ndarray:
    """
    Suma todas las columnas ``IMF_*`` (excluye ``Residuo``).

    Parameters
    ----------
    df_imfs : pd.DataFrame
        Descomposición con columnas IMF.

    Returns
    -------
    np.ndarray
        Suma de modos oscilatorios sin el residuo explícito.
    """
    cols = _columnas_imf_ordenadas(df_imfs)
    return np.sum(
        [np.asarray(df_imfs[c].values, dtype=np.float64) for c in cols],
        axis=0,
    )


def _suma_imfs_oscilatorias_vmd(df_imfs: pd.DataFrame, vmd_dc: int) -> np.ndarray:
    """
    Suma de IMFs excluyendo el modo de tendencia en VMD.

    Con ``DC=1`` omite ``IMF_1`` (modo DC); en caso contrario suma todos los ``IMF_*``.

    Parameters
    ----------
    df_imfs : pd.DataFrame
        Descomposición VMD.
    vmd_dc : int
        Valor de ``DC`` usado en la calibración.

    Returns
    -------
    np.ndarray
        Suma de modos oscilatorios según la convención VMD.
    """
    cols = _columnas_imf_ordenadas(df_imfs)
    if vmd_dc == 1 and len(cols) >= 2:
        cols_osc = cols[1:]
    else:
        cols_osc = cols
    return np.sum(
        [np.asarray(df_imfs[c].values, dtype=np.float64) for c in cols_osc],
        axis=0,
    )


def _brecha_tendencia_vmd_dc1(serie: np.ndarray, df_imfs: pd.DataFrame) -> np.ndarray:
    """
    Brecha de tendencia VMD con ``DC=1``: ``|x - Σ_{i≥2} IMF_i|`` ≈ modo DC (``IMF_1``).

    Parameters
    ----------
    serie : np.ndarray
        Señal original.
    df_imfs : pd.DataFrame
        Descomposición VMD con ``DC=1``.

    Returns
    -------
    np.ndarray
        Magnitud asociada al modo de baja frecuencia / tendencia.
    """
    cols = _columnas_imf_ordenadas(df_imfs)
    if len(cols) < 2:
        return _brecha_residuo_implicito(serie, df_imfs)
    cols_osc = cols[1:]
    suma_osc = np.sum(
        [np.asarray(df_imfs[c].values, dtype=np.float64) for c in cols_osc],
        axis=0,
    )
    return np.abs(serie - suma_osc)


def calibrar_vmd_activo(
    serie: np.ndarray,
    mod_emdsynth: Any,
    dc_fijo: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Busca hiperparámetros VMD que minimicen el acoplamiento lineal entre modos.

    Parameters
    ----------
    serie : np.ndarray
        Precios de cierre 1D.
    mod_emdsynth : module
        Módulo emdsynth con ``descomponer_vmd`` y ``calcular_metricas``.
    dc_fijo : int, optional
        Si se indica (0 o 1), fija ``DC`` y solo barre ``K`` y ``alpha``.

    Returns
    -------
    dict
        Bloques ``mejor``, ``rejilla`` y ``evaluaciones``.
    """
    descomponer_vmd = mod_emdsynth.descomponer_vmd
    calcular_metricas = mod_emdsynth.calcular_metricas
    rejilla_dc = [int(dc_fijo)] if dc_fijo is not None else REJILLA_VMD_DC

    def _clave(mets: Dict[str, float]) -> Tuple[float, float]:
        return (
            float(mets["corr_promedio_pares"]),
            float(mets["frac_energia_residuo"]),
        )

    evaluaciones: List[Dict[str, Any]] = []
    mejor: Optional[Dict[str, Any]] = None
    mejor_clave: Optional[Tuple[float, float]] = None

    for k_vmd, alpha_vmd, dc_vmd in itertools.product(
        REJILLA_VMD_K, REJILLA_VMD_ALPHA, rejilla_dc
    ):
        imfs = descomponer_vmd(
            serie,
            k=k_vmd,
            alpha=alpha_vmd,
            dc=dc_vmd,
        )
        mets = calcular_metricas(serie, imfs)
        fila = {
            "K": k_vmd,
            "alpha": alpha_vmd,
            "DC": dc_vmd,
            "metricas": {k: float(v) for k, v in mets.items()},
        }
        evaluaciones.append(fila)
        if mets["rmse_relativo"] >= UMBRAL_RMSE_RELATIVO:
            continue
        cl = _clave(mets)
        if mejor_clave is None or cl < mejor_clave:
            mejor_clave = cl
            mejor = {
                "K": k_vmd,
                "alpha": alpha_vmd,
                "DC": dc_vmd,
                "metricas": {k: float(v) for k, v in mets.items()},
            }

    if mejor is None:
        raise RuntimeError(f"Calibración VMD sin configuración válida para la señal.")

    criterio = "min_corr_promedio_pares_luego_frac_energia_residuo"
    if dc_fijo is not None:
        criterio = f"dc_fijo={dc_fijo}; " + criterio

    return {
        "criterio_seleccion": criterio,
        "dc_fijo": dc_fijo,
        "mejor": mejor,
        "rejilla": {
            "K": REJILLA_VMD_K,
            "alpha": REJILLA_VMD_ALPHA,
            "DC": rejilla_dc,
        },
        "evaluaciones": evaluaciones,
    }


def ejecutar_vmd_activo(
    cfg: ConfigActivo,
    mod_emdsynth: Any,
    calibrar: bool = True,
    parametros_fijos: Optional[Dict[str, Any]] = None,
    dc_fijo: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Calibra (opcional) y descompone la señal con VMD.

    Parameters
    ----------
    cfg : ConfigActivo
        Activo a procesar.
    mod_emdsynth : module
        Módulo emdsynth.
    calibrar : bool
        Si True, ejecuta rejilla de calibración.
    parametros_fijos : dict, optional
        Hiperparámetros VMD si ``calibrar=False``.
    dc_fijo : int, optional
        Fija ``DC`` durante la calibración (p. ej. 1 para modo de tendencia).

    Returns
    -------
    dict
        Calibración, DataFrame IMFs y rutas escritas.
    """
    rutas = _rutas_activo(cfg)
    df_precios, serie = cargar_serie_precios(cfg)

    if calibrar:
        logger.info("=" * 60)
        logger.info("CALIBRACIÓN VMD — %s (DC fijo=%s)", cfg.nombre, dc_fijo)
        logger.info("=" * 60)
        calibracion = calibrar_vmd_activo(serie, mod_emdsynth, dc_fijo=dc_fijo)
    else:
        dc_def = int(dc_fijo) if dc_fijo is not None else 1
        params = parametros_fijos or {"K": 6, "alpha": 5000.0, "DC": dc_def}
        calibracion = {"mejor": params, "criterio_seleccion": "parametros_fijos"}

    mejor = calibracion["mejor"]
    imfs = mod_emdsynth.descomponer_vmd(
        serie,
        k=int(mejor["K"]),
        alpha=float(mejor["alpha"]),
        dc=int(mejor["DC"]),
    )
    df_imfs = imfs_array_a_dataframe(imfs)
    df_imfs.index = df_precios.index

    cfg.dir_datos.mkdir(parents=True, exist_ok=True)
    df_imfs.to_parquet(rutas["imfs"], engine="pyarrow", index=True)

    var_serie = float(np.var(serie))
    resumen = {
        "n_imfs": int(sum(1 for c in df_imfs.columns if c.startswith("IMF_"))),
        "varianza_serie": var_serie,
        "frac_varianza_por_componente": {
            col: float(np.var(df_imfs[col].values) / (var_serie + 1e-15))
            for col in df_imfs.columns
        },
        "columnas": list(df_imfs.columns),
    }

    payload = {
        "activo": cfg.id_activo,
        "nombre": cfg.nombre,
        "metodo": "VMD",
        "calibracion": calibracion,
        "resumen_descomposicion": resumen,
    }
    with open(rutas["parametros"], "w", encoding="utf-8") as archivo:
        json.dump(payload, archivo, indent=2, ensure_ascii=False)

    m = mejor.get("metricas", mod_emdsynth.calcular_metricas(serie, imfs))
    logger.info(
        "%s VMD: K=%s alpha=%s DC=%s n_modos=%s rmse_rel=%.2e corr=%.4f",
        cfg.nombre,
        mejor["K"],
        mejor["alpha"],
        mejor["DC"],
        int(m["n_modos"]),
        m["rmse_relativo"],
        m["corr_promedio_pares"],
    )
    logger.info("IMFs guardadas: %s", rutas["imfs"])

    return {
        "cfg": cfg,
        "df_precios": df_precios,
        "serie": serie,
        "df_imfs": df_imfs,
        "calibracion": calibracion,
        "rutas": rutas,
    }


def generar_figura_comparativa_residuos(
    cfg: ConfigActivo,
    serie: np.ndarray,
    df_imfs_vmd: pd.DataFrame,
    vmd_dc: int,
    ruta_salida: Path,
) -> Optional[Path]:
    """
    Figura estilo CEEMDAN vs EEMD añadiendo la brecha de tendencia VMD (``DC=1``).

    Parameters
    ----------
    cfg : ConfigActivo
        Activo.
    serie : np.ndarray
        Precios de cierre.
    df_imfs_vmd : pd.DataFrame
        Modos VMD recién calculados.
    vmd_dc : int
        Valor de ``DC`` usado en VMD (1 activa comparación por ``IMF_1``).
    ruta_salida : Path
        PNG de salida.

    Returns
    -------
    Path or None
        Ruta escrita, o None si faltan parquets CEEMDAN/EEMD.
    """
    rutas = _rutas_activo(cfg)
    if not rutas["imfs_ceemdan"].is_file():
        logger.warning(
            "%s: sin %s; se omite figura comparativa.",
            cfg.nombre,
            rutas["imfs_ceemdan"],
        )
        return None

    df_ceemdan = _alinear_df_imfs_longitud(
        pd.read_parquet(rutas["imfs_ceemdan"], engine="pyarrow"),
        len(serie),
    )
    brecha_ceemdan = _brecha_residuo_implicito(serie, df_ceemdan)
    suma_ceemdan = _suma_imfs_sin_residuo(df_ceemdan)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(serie, label="Original", linewidth=0.8, color="C0")
    axes[0].plot(
        suma_ceemdan,
        label="Σ IMFs (CEEMDAN, sin residuo)",
        linewidth=0.8,
        alpha=0.85,
        color="C1",
    )
    axes[1].plot(
        brecha_ceemdan,
        label="|Original − Σ IMFs| (CEEMDAN)",
        linewidth=0.7,
        color="C1",
    )

    if rutas["imfs_eemd"].is_file():
        df_eemd = _alinear_df_imfs_longitud(
            pd.read_parquet(rutas["imfs_eemd"], engine="pyarrow"),
            len(serie),
        )
        suma_eemd = _suma_imfs_sin_residuo(df_eemd)
        brecha_eemd = _brecha_residuo_implicito(serie, df_eemd)
        axes[0].plot(
            suma_eemd,
            label="Σ IMFs (EEMD, sin residuo)",
            linewidth=0.8,
            alpha=0.85,
            color="C2",
        )
        axes[1].plot(
            brecha_eemd,
            label="|Original − Σ IMFs| (EEMD)",
            linewidth=0.7,
            color="C2",
            alpha=0.9,
        )

    suma_vmd = _suma_imfs_oscilatorias_vmd(df_imfs_vmd, vmd_dc=vmd_dc)
    etiqueta_suma_vmd = (
        "Σ IMF_{2..K} (VMD, DC=1)" if vmd_dc == 1 else "Σ IMFs (VMD, sin residuo)"
    )
    axes[0].plot(
        suma_vmd,
        label=etiqueta_suma_vmd,
        linewidth=0.8,
        alpha=0.85,
        color="C3",
    )

    if vmd_dc == 1:
        brecha_vmd = _brecha_tendencia_vmd_dc1(serie, df_imfs_vmd)
        axes[1].plot(
            brecha_vmd,
            label="|Original − Σ IMF_{2..K}| (VMD, DC=1)",
            linewidth=0.7,
            color="C3",
            alpha=0.9,
        )
    else:
        brecha_vmd = _brecha_residuo_implicito(serie, df_imfs_vmd)
        axes[1].plot(
            brecha_vmd,
            label="|Original − Σ IMFs| (VMD)",
            linewidth=0.7,
            color="C3",
            alpha=0.9,
        )

    axes[0].set_ylabel("Level")
    axes[0].legend(loc="upper left", fontsize=7)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("|Original − Σ IMFs|")
    axes[1].set_xlabel("Time index")
    axes[1].legend(loc="upper left", fontsize=7)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura comparativa residuos: %s", ruta_salida)
    return ruta_salida


def validar_vmd_activo(
    cfg: ConfigActivo,
    df_precios: pd.DataFrame,
    serie: np.ndarray,
    df_imfs: pd.DataFrame,
    mod_validacion: Any,
    vmd_dc: int = 1,
) -> Dict[str, Any]:
    """
    Valida la descomposición VMD y genera métricas y figura panel.

    Parameters
    ----------
    cfg : ConfigActivo
        Activo.
    df_precios : pd.DataFrame
        Precios con índice temporal.
    serie : np.ndarray
        Vector de cierre.
    df_imfs : pd.DataFrame
        Modos VMD.
    mod_validacion : module
        Funciones de validación CEEMDAN reutilizadas.
    vmd_dc : int
        Valor de ``DC`` usado en la descomposición VMD.

    Returns
    -------
    dict
        Informe de validación y rutas.
    """
    rutas = _rutas_activo(cfg)
    recon = mod_validacion.validar_reconstruccion(serie, df_imfs)
    residuo = mod_validacion.evaluar_residuo(serie, df_imfs)
    if vmd_dc == 1:
        brecha_tendencia = _brecha_tendencia_vmd_dc1(serie, df_imfs)
        imf1 = np.asarray(df_imfs["IMF_1"].values, dtype=np.float64)
        residuo["brecha_tendencia_dc1_std"] = float(np.std(brecha_tendencia))
        residuo["brecha_tendencia_dc1_media"] = float(np.mean(brecha_tendencia))
        residuo["corr_brecha_tendencia_imf1"] = float(
            np.corrcoef(brecha_tendencia, imf1)[0, 1]
        )
        residuo["interpretacion_dc1"] = (
            "Con DC=1, IMF_1 captura la tendencia; la brecha |x-ΣIMF_{2..K}| es comparable "
            "al residuo implícito CEEMDAN/EEMD."
        )
    tabla = mod_validacion.tabla_metricas_imfs(df_imfs)

    rutas["metricas_imf_csv"].parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(rutas["metricas_imf_csv"])

    generar_panel_imfs_vmd(
        df_precios,
        df_imfs,
        serie,
        cfg.nombre,
        rutas["figura_panel"],
    )
    ruta_comparativa = generar_figura_comparativa_residuos(
        cfg,
        serie,
        df_imfs,
        vmd_dc=vmd_dc,
        ruta_salida=rutas["figura_comparativa_residuos"],
    )

    informe = {
        "activo": cfg.id_activo,
        "nombre": cfg.nombre,
        "metodo": "VMD",
        "n_observaciones": int(len(serie)),
        "n_imfs": int(sum(1 for c in df_imfs.columns if c.startswith("IMF_"))),
        "reconstruccion": recon,
        "residuo": residuo,
        "vmd_dc": int(vmd_dc),
        "figura_comparativa_residuos": (
            str(ruta_comparativa) if ruta_comparativa is not None else None
        ),
        "metricas_imf": tabla.reset_index().to_dict(orient="records"),
    }
    with open(rutas["validacion"], "w", encoding="utf-8") as archivo:
        json.dump(informe, archivo, indent=2, ensure_ascii=False)

    logger.info(
        "%s validación: rmse_rel=%.2e, residuo R²=%.4f, Spearman=%.4f",
        cfg.nombre,
        recon["rmse_relativo"],
        residuo["r2_regresion_lineal"],
        residuo["spearman_tiempo"],
    )
    return informe


def generar_panel_imfs_vmd(
    df_precios: pd.DataFrame,
    df_imfs: pd.DataFrame,
    serie: np.ndarray,
    nombre_activo: str,
    ruta_salida: Path,
) -> None:
    """
    Genera figura apilada con precio de cierre y modos VMD.

    Parameters
    ----------
    df_precios : pd.DataFrame
        Precios (índice temporal).
    df_imfs : pd.DataFrame
        Modos VMD.
    serie : np.ndarray
        Vector de cierre.
    nombre_activo : str
        Etiqueta para el título.
    ruta_salida : Path
        Ruta del PNG.
    """
    cols_imf = sorted(
        [c for c in df_imfs.columns if c.startswith("IMF_")],
        key=lambda x: int(x.split("_")[1]),
    )
    columnas = (
        ["Close"] + cols_imf + (["Residuo"] if "Residuo" in df_imfs.columns else [])
    )
    n = len(columnas)
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.6 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, nombre in zip(axes, columnas):
        y = serie if nombre == "Close" else df_imfs[nombre].values
        ax.plot(df_precios.index, y, linewidth=0.6)
        ax.set_ylabel(nombre, fontsize=8)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Fecha")
    fig.tight_layout()
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura panel VMD: %s", ruta_salida)


def _cargar_obtener_grafos_all_imf() -> Any:
    """
    Carga ``obtener_grafos_all_imf`` sin ejecutar ``GraphEMD.data`` (depende de torch).

    Returns
    -------
    callable
        Función ``obtener_grafos_all_imf``.
    """
    ruta = _SRC_PYTHON / "GraphEMD" / "data" / "graph_imf_transform_utils.py"
    spec = importlib.util.spec_from_file_location("graph_imf_transform_utils_vmd", ruta)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar {ruta}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.obtener_grafos_all_imf


def ejecutar_grafos_vmd_activo(
    cfg: ConfigActivo,
    df_imfs: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Aplica HVG, NVG y recurrencia a cada modo VMD.

    Parameters
    ----------
    cfg : ConfigActivo
        Activo.
    df_imfs : pd.DataFrame
        Modos VMD.

    Returns
    -------
    dict
        Resumen y rutas de grafos.
    """
    _asegurar_paths()
    obtener_grafos_all_imf = _cargar_obtener_grafos_all_imf()
    from ejecutar_salidas_subseccion_grafos_ceemdan_20abr26 import (  # noqa: E402
        calcular_tabla_parametros_recurrencia,
    )

    rutas = _rutas_activo(cfg)
    rutas["dir_grafos"].mkdir(parents=True, exist_ok=True)

    df_params = calcular_tabla_parametros_recurrencia(
        df_imfs,
        umbral_percentil=UMBRAL_PERCENTIL_RECURRENCIA,
        random_state=RANDOM_STATE_RECURRENCIA,
    )
    df_params.to_csv(rutas["param_recurrencia_csv"], index=False)

    resultados = obtener_grafos_all_imf(
        df_imfs=str(rutas["imfs"]),
        carpeta_salida_base=str(rutas["dir_grafos"]),
        tau_max=TAU_MAX,
        dim_max=DIM_MAX,
        umbral_percentil=UMBRAL_PERCENTIL_RECURRENCIA,
        random_state=RANDOM_STATE_RECURRENCIA,
    )

    filas: List[Dict[str, Any]] = []
    for id_imf, tipos in resultados.items():
        fila: Dict[str, Any] = {"componente": id_imf}
        for tipo_grafo in ("hvg", "nvg", "recurrencia"):
            info = tipos.get(tipo_grafo, {})
            if isinstance(info, dict) and info.get("exito"):
                fila[f"nodos_{tipo_grafo}"] = info.get("num_nodes")
                fila[f"enlaces_{tipo_grafo}"] = info.get("num_edges")
                if tipo_grafo == "recurrencia":
                    fila["tau"] = info.get("tau")
                    fila["dim_embedding"] = info.get("dim_embedding")
                    fila["epsilon"] = info.get("umbral_recurrencia")
            else:
                fila[f"error_{tipo_grafo}"] = (
                    info.get("error", "fallo") if isinstance(info, dict) else "fallo"
                )
        filas.append(fila)

    df_resumen = pd.DataFrame(filas)
    df_resumen.to_csv(rutas["resumen_grafos_csv"], index=False)

    manifest = {
        "activo": cfg.id_activo,
        "metodo": "VMD",
        "ruta_imfs": str(rutas["imfs"]),
        "dir_grafos": str(rutas["dir_grafos"]),
        "parametros_recurrencia": df_params.to_dict(orient="records"),
        "resumen_por_componente": df_resumen.to_dict(orient="records"),
        "detalle_grafos": resultados,
    }
    with open(rutas["manifest_grafos"], "w", encoding="utf-8") as archivo:
        json.dump(manifest, archivo, indent=2, ensure_ascii=False)

    logger.info(
        "%s grafos VMD: %d componentes en %s",
        cfg.nombre,
        len(filas),
        rutas["dir_grafos"],
    )
    return {
        "resumen": df_resumen,
        "parametros_recurrencia": df_params,
        "manifest": manifest,
    }


def evaluar_k_ica(
    mod_ica: Any,
    x: np.ndarray,
    nombres_imf: List[str],
    k: int,
) -> Dict[str, Any]:
    """
    Evalúa un valor de k en la rejilla ICA.

    Parameters
    ----------
    mod_ica : module
        Utilidades ICA.
    x : np.ndarray
        Bloque IMF ``(T, p)``.
    nombres_imf : list of str
        Nombres de columnas IMF.
    k : int
        Número de componentes.

    Returns
    -------
    dict
        Métricas de independencia y reconstrucción.
    """
    ica, scaler, z = mod_ica.ajustar_fastica(x, k, RANDOM_STATE_ICA)
    x_hat = mod_ica.reconstruir_imfs_desde_z_ica(z, ica, scaler)
    rec = mod_ica.metricas_reconstruccion_imfs(x, x_hat, nombres_imf)
    corr = mod_ica._metricas_correlacion(z)
    return {
        "n_components": int(k),
        "max_abs_corr_Z": float(corr["max_abs_fuera_diagonal"]),
        "media_abs_corr_Z": float(corr["media_abs_fuera_diagonal"]),
        "error_relativo_frobenius": float(rec["error_relativo_frobenius"]),
        "rmse_global_reconstruccion": float(rec["rmse_global"]),
        "r2_medio_columnas": float(rec["r2_medio_columnas"]),
    }


def ejecutar_ica_vmd_activo(
    cfg: ConfigActivo,
    df_imfs: pd.DataFrame,
    mod_ica: Any,
) -> Dict[str, Any]:
    """
    Calibra k y aplica FastICA sobre modos VMD oscilatorios.

    Parameters
    ----------
    cfg : ConfigActivo
        Activo.
    df_imfs : pd.DataFrame
        Modos VMD.
    mod_ica : module
        Utilidades ICA.

    Returns
    -------
    dict
        k óptimo, calibración y rutas de salida.
    """
    rutas = _rutas_activo(cfg)
    x, nombres_imf, residuo = mod_ica.extraer_bloque_imf_y_residuo(df_imfs)
    p = int(x.shape[1])
    if p < 2:
        raise ValueError(f"{cfg.nombre}: se requieren al menos 2 IMF para ICA.")

    rejilla: List[Dict[str, Any]] = []
    for k in range(K_MIN_CALIBRACION_ICA, p):
        logger.info("%s ICA: evaluando k=%d ...", cfg.nombre, k)
        rejilla.append(evaluar_k_ica(mod_ica, x, nombres_imf, k))

    df_rejilla = pd.DataFrame(rejilla)
    seleccion = mod_ica.seleccionar_k_optimo_desde_rejilla_ica(
        df_rejilla,
        p=p,
        umbral_corr_z=UMBRAL_CORR_Z,
        fraccion_r2_objetivo=FRACCION_R2_OBJETIVO,
        factor_error_frobenius_rodilla=FACTOR_ERROR_FROB_RODILLA,
    )
    k_optimo = int(seleccion["n_components"])

    meta = {
        "activo": cfg.id_activo,
        "nombre": cfg.nombre,
        "metodo_descomposicion": "VMD",
        "parquet_entrada": str(rutas["imfs"].resolve()),
        "n_muestras_temporales": int(x.shape[0]),
        "columnas_imf_entrada": nombres_imf,
        "calibracion_mejor_k": seleccion,
    }

    rutas["dir_ica"].mkdir(parents=True, exist_ok=True)
    mod_ica.escribir_referencias(rutas["dir_ica"].parent)

    ica, scaler, z = mod_ica.ajustar_fastica(x, k_optimo, RANDOM_STATE_ICA)
    mod_ica.guardar_salidas_ica(
        rutas["dir_ica"],
        z,
        residuo,
        ica,
        scaler,
        nombres_imf,
        meta,
        x,
        guardar_parquet_recon=True,
    )

    calibracion = {
        "criterio_seleccion": seleccion["criterio_seleccion"],
        "p_imf_entrada": p,
        "k_min_evaluado": K_MIN_CALIBRACION_ICA,
        "k_max_evaluado": p - 1,
        "random_state": RANDOM_STATE_ICA,
        "mejor": seleccion,
        "detalle_seleccion": seleccion["detalle_seleccion"],
        "correlacion_imfs_nativas": mod_ica._metricas_correlacion(x),
        "rejilla": rejilla,
    }
    aplicacion = {
        "n_components": k_optimo,
        "forma_Z": list(z.shape),
        "dir_salida": str(rutas["dir_ica"].resolve()),
        "parquet_reducido": str(
            (rutas["dir_ica"] / "imfs_reducidas.parquet").resolve()
        ),
        "modelo": str((rutas["dir_ica"] / "modelo_ica.npz").resolve()),
        "metricas": str((rutas["dir_ica"] / "metricas_reduccion.json").resolve()),
    }
    payload = {
        "parquet_entrada": str(rutas["imfs"].resolve()),
        "calibracion": calibracion,
        "aplicacion": aplicacion,
    }
    with open(rutas["parametros_ica"], "w", encoding="utf-8") as archivo:
        json.dump(payload, archivo, indent=2, ensure_ascii=False)

    df_z = pd.read_parquet(
        rutas["dir_ica"] / "imfs_reducidas.parquet", engine="pyarrow"
    )
    generar_panel_ica_vmd(
        df_z,
        df_imfs.index,
        cfg.nombre,
        k_optimo,
        rutas["figura_ica"],
    )

    logger.info(
        "%s ICA VMD: k_optimo=%d guardado en %s", cfg.nombre, k_optimo, rutas["dir_ica"]
    )
    return {
        "k_optimo": k_optimo,
        "calibracion": calibracion,
        "aplicacion": aplicacion,
    }


def generar_panel_ica_vmd(
    df_componentes: pd.DataFrame,
    indice: pd.Index,
    nombre_activo: str,
    k: int,
    ruta_salida: Path,
) -> None:
    """
    Genera panel temporal de fuentes ICA y residuo VMD.

    Parameters
    ----------
    df_componentes : pd.DataFrame
        Columnas ``Z_1``…``Z_k`` y opcional ``Residuo``.
    indice : pd.Index
        Eje temporal.
    nombre_activo : str
        Etiqueta del activo.
    k : int
        Número de componentes ICA.
    ruta_salida : Path
        Ruta del PNG.
    """
    cols_z = sorted(
        [c for c in df_componentes.columns if c.startswith("Z_")],
        key=lambda x: int(x.split("_")[1]),
    )
    if "Residuo" in df_componentes.columns:
        cols_z.append("Residuo")
    n = len(cols_z)
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, nombre in zip(axes, cols_z):
        ax.plot(indice, df_componentes[nombre].values, linewidth=0.6)
        ax.set_ylabel(nombre, fontsize=9)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Fecha")
    fig.tight_layout()
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura ICA VMD: %s", ruta_salida)


def procesar_activo(
    cfg: ConfigActivo,
    mod_emdsynth: Any,
    mod_validacion: Any,
    mod_ica: Any,
    pasos: Tuple[str, ...],
    calibrar: bool,
    dc_fijo: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Ejecuta los pasos solicitados del pipeline VMD para un activo.

    Parameters
    ----------
    cfg : ConfigActivo
        Activo.
    mod_emdsynth : module
        Módulo VMD.
    mod_validacion : module
        Validación.
    mod_ica : module
        ICA.
    pasos : tuple of str
        Subconjunto de ``vmd``, ``validacion``, ``grafos``, ``ica``.
    calibrar : bool
        Calibrar hiperparámetros VMD.
    dc_fijo : int, optional
        Fija ``DC`` en la calibración y descomposición.

    Returns
    -------
    dict
        Resultados parciales por paso.
    """
    resultado: Dict[str, Any] = {"activo": cfg.id_activo}
    rutas = _rutas_activo(cfg)

    if "vmd" in pasos:
        salida_vmd = ejecutar_vmd_activo(
            cfg, mod_emdsynth, calibrar=calibrar, dc_fijo=dc_fijo
        )
        resultado["vmd"] = salida_vmd
        df_precios = salida_vmd["df_precios"]
        serie = salida_vmd["serie"]
        df_imfs = salida_vmd["df_imfs"]
        vmd_dc_efectivo = int(salida_vmd["calibracion"]["mejor"]["DC"])
    else:
        df_precios, serie = cargar_serie_precios(cfg)
        if not rutas["imfs"].is_file():
            raise FileNotFoundError(
                f"Falta {rutas['imfs']}. Ejecute primero el paso vmd."
            )
        df_imfs = pd.read_parquet(rutas["imfs"], engine="pyarrow")
        vmd_dc_efectivo = int(dc_fijo) if dc_fijo is not None else 1

    if "validacion" in pasos:
        resultado["validacion"] = validar_vmd_activo(
            cfg,
            df_precios,
            serie,
            df_imfs,
            mod_validacion,
            vmd_dc=vmd_dc_efectivo,
        )

    if "grafos" in pasos:
        resultado["grafos"] = ejecutar_grafos_vmd_activo(cfg, df_imfs)

    if "ica" in pasos:
        resultado["ica"] = ejecutar_ica_vmd_activo(cfg, df_imfs, mod_ica)

    return resultado


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parsea argumentos de línea de comandos.

    Parameters
    ----------
    argv : list of str, optional
        Argumentos (por defecto ``sys.argv[1:]``).

    Returns
    -------
    argparse.Namespace
        Opciones CLI.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline VMD completo para MSCI, ETFs y oro."
    )
    parser.add_argument(
        "--activos",
        type=str,
        default=",".join(a.id_activo for a in ACTIVOS),
        help="IDs separados por coma (msci_world,xle,xlp,xlv,xauusd).",
    )
    parser.add_argument(
        "--solo-paso",
        type=str,
        default="vmd,validacion,grafos,ica",
        help="Pasos a ejecutar: vmd,validacion,grafos,ica.",
    )
    parser.add_argument(
        "--sin-calibracion",
        action="store_true",
        help="Usa K=6, alpha=5000, DC=1 sin rejilla por activo.",
    )
    parser.add_argument(
        "--vmd-dc",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Fija DC en VMD (1 = modo de tendencia en IMF_1). "
            "Por defecto 1 para comparar con CEEMDAN/EEMD."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Punto de entrada: procesa todos los activos solicitados.

    Parameters
    ----------
    argv : list of str, optional
        Argumentos CLI.

    Returns
    -------
    dict
        Resultados agregados por activo.
    """
    args = _parse_args(argv)
    ids = [s.strip() for s in args.activos.split(",") if s.strip()]
    pasos = tuple(s.strip() for s in args.solo_paso.split(",") if s.strip())
    mapa = {a.id_activo: a for a in ACTIVOS}

    mod_emdsynth = _cargar_modulo_emdsynth()
    mod_validacion = _cargar_modulo_validacion()
    mod_ica = _cargar_modulo_ica()

    resultados: Dict[str, Any] = {}
    for id_activo in ids:
        if id_activo not in mapa:
            raise ValueError(f"Activo desconocido: {id_activo}. Válidos: {list(mapa)}")
        cfg = mapa[id_activo]
        logger.info("=" * 70)
        logger.info("PROCESANDO %s (%s)", cfg.nombre, cfg.id_activo)
        logger.info("=" * 70)
        resultados[id_activo] = procesar_activo(
            cfg,
            mod_emdsynth,
            mod_validacion,
            mod_ica,
            pasos=pasos,
            calibrar=not args.sin_calibracion,
            dc_fijo=args.vmd_dc,
        )

    resumen_k = {
        id_a: resultados[id_a].get("ica", {}).get("k_optimo")
        for id_a in resultados
        if "ica" in resultados[id_a]
    }
    if resumen_k:
        logger.info("Resumen k ICA óptimo: %s", resumen_k)

    return resultados


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        main()
    except Exception:
        logger.exception("Error en pipeline VMD")
        sys.exit(1)
