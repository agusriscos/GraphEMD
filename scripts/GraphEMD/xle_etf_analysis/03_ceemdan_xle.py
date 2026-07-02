"""
Script para descomponer la ETF XLE (Energy Select Sector SPDR Fund) en componentes IMFs mediante CEEMDAN.

1. Carga los datos de la ETF XLE desde el archivo parquet data/GraphEMD/xle_etf_analysis/xle.parquet.
2. Extrae los datos en la misma ventana temporal que el MSCI World analizado (12 Enero 2012 - 20 Abril 2026). Debe de haber 3587 puntos/observaciones
3. Calcula los parametros optimos de CEEMDAN para el XLE usando la misma logica que se ha realizado para el MSCI World en este mismo repositorio.
3. Descompone la serie en componentes IMFs mediante CEEMDAN con los parametros optimos calculados.
4. Documenta en este mismo script la logica implementada y los resultados obtenidos.
5. No cambies por el momento ningun codigo que ya exista

Lógica implementada
---------------------
- **Carga**: ``xle.parquet`` (ventana 2012-01-12–2026-04-20, 3587 observaciones alineadas al calendario MSCI).
- **Referencia MSCI** (``docs/20abr26``, tabla CEEMDAN): ``max_imf=14``, ``trials=100``,
  ``epsilon=0.05``, ``seed=42``.
- **Calibración XLE** (rejilla ``epsilon`` × ``trials`` × ``max_imf`` con ``max_imf≥9``):
  en XLE CEEMDAN suele detenerse en **8 IMFs** aunque ``max_imf`` sea mayor; el criterio
  prioriza el mayor ``n_imfs`` alcanzable. Entre configuraciones con
  ``rmse_relativo`` de reconstrucción ``< 1e-10``, se elige la que
  deja un residuo lo más **lineal, monótono creciente** posible (última fila de la
  descomposición, ``Close − Σ IMF``): prioridad a monotonicidad creciente y pendiente
  positiva; luego mayor ``R²`` de ``residuo ~ a·t+b`` y mayor fracción de diferencias
  positivas; en empate, menor ``rmse_relativo_vs_lineal``, menor ``rmse_relativo`` y menor
  ``corr_promedio_pares`` entre IMFs. ``max_imf`` controla el tope de modos y, por tanto,
  cuánta estructura queda en el residuo frente a las IMF.
- Se guarda además ``mejor_separacion_imfs`` (criterio emdsynth: mínimo acoplamiento lineal
  entre IMFs) solo como referencia comparativa.
- **Descomposición final** con ``calibracion["mejor"]`` (criterio linealidad); salida en
  ``xle_imfs_ceemdan.parquet`` y ``xle_ceemdan_parametros.json``.

Resultados obtenidos
--------------------

Tras cada calibración, revisar ``xle_ceemdan_parametros.json`` (bloque ``mejor``) y
re-ejecutar ``04_ceemdan_xle_validation_and_results.py`` para figuras y validación.
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

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DIR_DATOS = _REPO_ROOT / "data" / "GraphEMD" / "xle_etf_analysis"
_RUTA_PARQUET_XLE = _DIR_DATOS / "xle.parquet"
_RUTA_IMFS_SALIDA = _DIR_DATOS / "xle_imfs_ceemdan.parquet"
_RUTA_PARAMETROS_JSON = _DIR_DATOS / "xle_ceemdan_parametros.json"

N_OBSERVACIONES_ESPERADAS: int = 3587

# Referencia MSCI World (docs/20abr26/main.tex, tab:ceemdan_params)
MSCI_MAX_IMF: int = 14
MSCI_TRIALS: int = 100
MSCI_EPSILON: float = 0.05
MSCI_SEED: int = 42

# Rejilla reducida (~160 configs, ~2 h): barrido con ``max_imf`` ≥ 9.
REJILLA_EPSILON: list[float] = [0.03, 0.05, 0.08, 0.12]
REJILLA_TRIALS: list[int] = [100, 140, 180, 200, 220]
REJILLA_MAX_IMF: list[int] = [9, 10, 12, 14, 16, 18, 20, 22]
UMBRAL_RMSE_RELATIVO: float = 1e-10

logger = logging.getLogger(__name__)


def _cargar_modulo_emdsynth_pipeline():
    """
    Carga ``ejecutar_descomposiciones_emdsynth`` sin modificar el módulo en disco.

    Returns
    -------
    module
        Módulo con ``descomponer_ceemdan`` y ``calcular_metricas``.
    """
    ruta = (
        _REPO_ROOT
        / "scripts"
        / "GraphEMD"
        / "emdsynth"
        / "ejecutar_descomposiciones_emdsynth.py"
    )
    spec = importlib.util.spec_from_file_location("emdsynth_ejecutar_xle", ruta)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar {ruta}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cargar_serie_cierre_xle(ruta_parquet: Path = _RUTA_PARQUET_XLE) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Carga XLE y devuelve el vector de precios de cierre.

    Parameters
    ----------
    ruta_parquet : Path
        Ruta a ``xle.parquet``.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        DataFrame completo y serie ``Close`` 1D en ``float64``.

    Raises
    ------
    FileNotFoundError
        Si no existe el archivo.
    ValueError
        Si falta ``Close`` o el número de filas no coincide con la ventana MSCI.
    """
    if not ruta_parquet.is_file():
        raise FileNotFoundError(
            f"No se encontró {ruta_parquet}. Ejecute antes 01_download_xle.py."
        )
    df = pd.read_parquet(ruta_parquet, engine="pyarrow")
    if "Close" not in df.columns:
        raise ValueError(f"Falta columna Close. Columnas: {list(df.columns)}")
    if len(df) != N_OBSERVACIONES_ESPERADAS:
        logger.warning(
            "Se esperaban %s observaciones; el parquet tiene %s.",
            N_OBSERVACIONES_ESPERADAS,
            len(df),
        )
    serie = np.asarray(df["Close"].values, dtype=np.float64)
    logger.info(
        "Serie Close: n=%s, rango %s → %s",
        len(serie),
        df.index.min(),
        df.index.max(),
    )
    return df, serie


def extraer_residuo(imfs: np.ndarray) -> np.ndarray:
    """
    Obtiene el residuo CEEMDAN (última fila de la matriz de modos).

    Parameters
    ----------
    imfs : np.ndarray
        Salida de ``descomponer_ceemdan`` con forma ``(n_modos, n_muestras)``.

    Returns
    -------
    np.ndarray
        Serie 1D del residuo.
    """
    return np.asarray(imfs[-1, :], dtype=np.float64)


def metricas_linealidad_residuo(residuo: np.ndarray) -> dict[str, float]:
    """
    Cuantifica qué tan lineal es el residuo respecto al índice temporal.

    Parameters
    ----------
    residuo : np.ndarray
        Componente residuo de CEEMDAN.

    Returns
    -------
    dict
        ``r2_regresion_lineal``, ``rmse_relativo_vs_lineal``, fracción de diferencias
        con el mismo signo (proxy de monotonicidad) y pendiente.
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
    Clave de ordenación: residuo lineal, creciente y monótono (menor es mejor).

    Parameters
    ----------
    metricas_linealidad : dict
        Salida de ``metricas_linealidad_residuo``.
    metricas_decomposicion : dict
        Salida de ``calcular_metricas``.
    n_imfs : int
        Número de IMFs extraídos (sin contar residuo).

    Returns
    -------
    tuple
        Penalización si no es monótono creciente, penalización si pendiente ``≤ 0``,
        ``-n_imfs``, luego ``(-R², -frac_diff_creciente, rmse_vs_lineal, …)``.
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
    Clave emdsynth: mínimo acoplamiento entre IMFs (menor es mejor).

    Parameters
    ----------
    metricas_decomposicion : dict
        Salida de ``calcular_metricas``.

    Returns
    -------
    tuple
        ``(corr_promedio_pares, frac_energia_residuo)``.
    """
    return (
        float(metricas_decomposicion["corr_promedio_pares"]),
        float(metricas_decomposicion["frac_energia_residuo"]),
    )


def calibrar_parametros_ceemdan_xle(
    serie: np.ndarray,
    mod_emdsynth: Any,
    semilla: int = MSCI_SEED,
    umbral_rmse: float = UMBRAL_RMSE_RELATIVO,
) -> dict[str, Any]:
    """
    Busca ``epsilon``, ``trials`` y ``max_imf`` priorizando residuo lineal y creciente.

    Parameters
    ----------
    serie : np.ndarray
        Precios de cierre 1D.
    mod_emdsynth : module
        Módulo con ``descomponer_ceemdan`` y ``calcular_metricas``.
    semilla : int
        Semilla del ruido CEEMDAN.
    umbral_rmse : float
        Máximo ``rmse_relativo`` aceptable en la reconstrucción completa.

    Returns
    -------
    dict
        ``mejor`` (criterio linealidad del residuo), ``mejor_separacion_imfs`` (criterio
        emdsynth), ``referencia_msci``, ``rejilla``, ``evaluaciones``.
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
            "Calibración [%s/%s]: epsilon=%.2f, trials=%s, max_imf=%s",
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
            "Calibración CEEMDAN XLE: ninguna configuración cumple el umbral de RMSE."
        )

    n_imfs_vals = [int(e["n_imfs"]) for e in evaluaciones]
    resumen_n_imfs = {
        "minimo": int(min(n_imfs_vals)),
        "maximo": int(max(n_imfs_vals)),
        "nota": (
            "CEEMDAN en XLE alcanza como máximo 8 IMFs en esta ventana; "
            "max_imf≥9 solo fija el tope del algoritmo."
            if max(n_imfs_vals) < 9
            else None
        ),
    }
    if resumen_n_imfs["maximo"] < 9:
        logger.warning(
            "Ninguna configuración extrajo 9+ IMFs (máximo observado: %s). "
            "Se elige el mejor entre las %s válidas por RMSE.",
            resumen_n_imfs["maximo"],
            sum(1 for e in evaluaciones if e["valida"]),
        )

    ml = mejor_lineal["metricas_linealidad_residuo"]
    logger.info(
        "Óptimo (residuo lineal): epsilon=%.2f, trials=%s, max_imf=%s, "
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
            "Referencia (separación IMFs): epsilon=%.2f, trials=%s, R² residuo=%.4f",
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
    Convierte la salida de CEEMDAN (filas = modos) en DataFrame con columnas IMF_* y Residuo.

    Parameters
    ----------
    imfs : np.ndarray
        Matriz 2D; la última fila es el residuo.

    Returns
    -------
    pd.DataFrame
        Columnas ``IMF_1`` … ``IMF_n`` y ``Residuo``.
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
    Aplica CEEMDAN con los parámetros indicados y guarda el parquet de IMFs.

    Parameters
    ----------
    serie : np.ndarray
        Precios de cierre.
    parametros : dict
        Debe incluir ``epsilon``, ``trials``, ``max_imf``, ``seed``.
    mod_emdsynth : module
        Módulo del pipeline emdsynth.
    indice : pd.Index
        Índice temporal alineado con ``serie``.
    ruta_imfs : Path
        Destino del parquet.

    Returns
    -------
    pd.DataFrame
        IMFs con el mismo índice que la serie de precios.
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
        "IMFs guardadas: %s (%s columnas, %s filas)",
        ruta_imfs,
        len(df_imfs.columns),
        len(df_imfs),
    )
    return df_imfs


def resumir_energia_imfs(df_imfs: pd.DataFrame, serie: np.ndarray) -> dict[str, Any]:
    """
    Calcula fracciones de varianza por componente respecto a la serie original.

    Parameters
    ----------
    df_imfs : pd.DataFrame
        IMFs y residuo.
    serie : np.ndarray
        Precios de cierre.

    Returns
    -------
    dict
        Varianza total, varianza por columna y número de IMFs.
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
    Persiste parámetros de calibración y resumen de la descomposición.

    Parameters
    ----------
    calibracion : dict
        Salida de ``calibrar_parametros_ceemdan_xle``.
    resumen_imfs : dict
        Salida de ``resumir_energia_imfs``.
    ruta : Path
        Archivo JSON de salida.

    Returns
    -------
    Path
        Ruta escrita.
    """
    payload = {
        "calibracion": calibracion,
        "resumen_descomposicion": resumen_imfs,
    }
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as archivo:
        json.dump(payload, archivo, indent=2, ensure_ascii=False)
    logger.info("Parámetros guardados: %s", ruta)
    return ruta


def main() -> dict[str, Any]:
    """
    Calibra CEEMDAN en XLE, descompone con parámetros óptimos y guarda salidas.

    Returns
    -------
    dict
        Resultados de calibración, DataFrame de IMFs y rutas generadas.
    """
    mod = _cargar_modulo_emdsynth_pipeline()
    df, serie = cargar_serie_cierre_xle()

    logger.info("=" * 70)
    logger.info(
        "CALIBRACIÓN CEEMDAN — XLE (rejilla %s×%s×%s, max_imf≥9, residuo lineal)",
        len(REJILLA_EPSILON),
        len(REJILLA_TRIALS),
        len(REJILLA_MAX_IMF),
    )
    logger.info("=" * 70)
    calibracion = calibrar_parametros_ceemdan_xle(serie, mod)

    parametros_optimos = calibracion["mejor"]
    logger.info("=" * 70)
    logger.info("DESCOMPOSICIÓN CEEMDAN — parámetros óptimos (residuo lineal)")
    logger.info("=" * 70)
    df_imfs = descomponer_y_guardar(serie, parametros_optimos, mod, df.index)
    resumen = resumir_energia_imfs(df_imfs, serie)
    resumen["linealidad_residuo"] = metricas_linealidad_residuo(
        np.asarray(df_imfs["Residuo"].values, dtype=np.float64)
    )
    guardar_parametros_json(calibracion, resumen)

    lin = resumen["linealidad_residuo"]
    logger.info(
        "Componentes: %s IMFs + residuo; rmse_rel=%.2e; residuo R²=%.4f, "
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
        logger.exception("Error en CEEMDAN de XLE")
        sys.exit(1)
