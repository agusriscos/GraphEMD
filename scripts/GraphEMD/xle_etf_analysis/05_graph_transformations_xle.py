"""
Script para aplicar las transformaciones de grafos a los componentes IMFs de la ETF XLE (Energy Select Sector SPDR Fund) mediante CEEMDAN.

1. Carga los datos necesarios para la aplicacion de las transformaciones de grafos a los componentes IMFs de la ETF XLE. Por un lado los datos de los componentes IMFs obtenidos en la descomposición de la ETF XLE mediante CEEMDAN y por otro lado los datos de los grafos obtenidos en la aplicacion de las transformaciones de grafos a los componentes IMFs de la ETF XLE.
2. Aplica las transformaciones de grafos a los componentes IMFs de la ETF XLE mediante CEEMDAN.
3. Documenta en este mismo script la logica implementada y los resultados obtenidos.
4. No cambies por el momento ningun codigo que ya exista

Lógica implementada
---------------------
- **Carga**: ``xle_imfs_ceemdan.parquet`` (8 IMFs + ``Residuo``, 3587 observaciones; salida de
  ``03_ceemdan_xle.py``).
- **Visibilidad** (``ts2vg``, misma convención que MSCI en ``graph_imf_transform_utils``):
  - **HVG** (Horizontal Visibility Graph): aristas entre puntos con visibilidad horizontal
    (``HorizontalVG``, dirigido ``left_to_right``).
  - **NVG** (Natural Visibility Graph): visibilidad natural (``NaturalVG``).
- **Recurrencia** (delay embedding + umbral por percentil de distancias, ``random_state=42``):
  selección de ``tau`` (información mutua), dimensión ``d`` (false nearest neighbours),
  umbral ``epsilon`` al percentil 10 de distancias en el espacio embebido; grafo no dirigido
  con nodos = instantes y aristas donde la distancia euclídea ``< epsilon``.
- **Implementación**: ``obtener_grafos_all_imf`` guarda por componente parquet + ``.pt``
  (PyTorch Geometric) en ``data/GraphEMD/xle_etf_analysis/grafos/{hvg,nvg,recurrencia}/``.
- **Tabla auxiliar**: ``xle_parametros_recurrencia_ceemdan.csv`` (``tau``, ``d``, ``epsilon`` por IMF).

Resultados obtenidos (ejecución 2026-05-16, n=3587)
-------------------------------------------------

**9 componentes** (IMF_1–IMF_8 + Residuo): HVG, NVG y recurrencia generados sin error.

**Visibilidad:** cada serie tiene **3587 nodos** (un nodo por instante). HVG ≈ 5.9–7.2×10³
enlaces; NVG crece con la escala de la IMF (IMF_1 ≈ 10⁴, Residuo ≈ 4.15×10⁶ enlaces).

**Recurrencia** (``tau``, ``d``, ``epsilon`` en ``xle_parametros_recurrencia_ceemdan.csv``):
IMF_1–IMF_5 con ``d=4``; IMF_6 ``d=3``; IMF_7–IMF_8 y Residuo ``d=2``; ``tau`` crece
en modos lentos (IMF_7: 41, Residuo: 21). Nodos en embedding ≈ 3529–3581; enlaces
≈ 3.3×10⁴–3.5×10⁴ por componente.

**Salidas:** ``grafos/{hvg,nvg,recurrencia}/``, ``xle_resumen_grafos_imf.csv``,
``xle_parametros_recurrencia_ceemdan.csv``, ``xle_grafos_imf_manifest.json``.
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SRC_PYTHON = _REPO_ROOT / "src" / "python"
_EXPLORACION = _REPO_ROOT / "scripts" / "GraphEMD" / "exploracion"
_DIR_DATOS = _REPO_ROOT / "data" / "GraphEMD" / "xle_etf_analysis"
_RUTA_IMFS_XLE = _DIR_DATOS / "xle_imfs_ceemdan.parquet"
_DIR_GRAFOS = _DIR_DATOS / "grafos"
_RUTA_RESUMEN_CSV = _DIR_DATOS / "xle_resumen_grafos_imf.csv"
_RUTA_PARAM_RECURRENCIA_CSV = _DIR_DATOS / "xle_parametros_recurrencia_ceemdan.csv"
_RUTA_RESUMEN_JSON = _DIR_DATOS / "xle_grafos_imf_manifest.json"

# Mismos valores que MSCI (``ejecutar_salidas_subseccion_grafos_ceemdan_20abr26.py``)
TAU_MAX: int = 50
DIM_MAX: int = 10
UMBRAL_PERCENTIL_RECURRENCIA: float = 10.0
RANDOM_STATE_RECURRENCIA: int = 42

if str(_SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(_SRC_PYTHON))
if str(_EXPLORACION) not in sys.path:
    sys.path.insert(0, str(_EXPLORACION))

from GraphEMD.data.graph_imf_transform_utils import obtener_grafos_all_imf  # noqa: E402
from ejecutar_salidas_subseccion_grafos_ceemdan_20abr26 import (  # noqa: E402
    calcular_tabla_parametros_recurrencia,
)

logger = logging.getLogger(__name__)


def cargar_imfs_ceemdan_xle(
    ruta_imfs: Path = _RUTA_IMFS_XLE,
) -> pd.DataFrame:
    """
    Carga el parquet de IMFs CEEMDAN de XLE.

    Parameters
    ----------
    ruta_imfs : Path
        Ruta a ``xle_imfs_ceemdan.parquet``.

    Returns
    -------
    pd.DataFrame
        Columnas ``IMF_1`` … ``IMF_n`` y ``Residuo``.

    Raises
    ------
    FileNotFoundError
        Si no existe el archivo de IMFs.
    ValueError
        Si no hay columnas IMF ni Residuo.
    """
    if not ruta_imfs.is_file():
        raise FileNotFoundError(
            f"No se encuentra {ruta_imfs}. Ejecute antes 03_ceemdan_xle.py."
        )
    df_imfs = pd.read_parquet(ruta_imfs, engine="pyarrow")
    columnas = [c for c in df_imfs.columns if c.startswith("IMF_") or c == "Residuo"]
    if not columnas:
        raise ValueError(
            f"Sin columnas IMF/Residuo en {ruta_imfs}. Columnas: {list(df_imfs.columns)}"
        )
    logger.info(
        "IMFs XLE cargadas: %d filas, componentes %s",
        len(df_imfs),
        columnas,
    )
    return df_imfs


def resumir_resultados_grafos(resultados: dict[str, Any]) -> pd.DataFrame:
    """
    Convierte el diccionario de ``obtener_grafos_all_imf`` en tabla plana.

    Parameters
    ----------
    resultados : dict
        Salida de :func:`obtener_grafos_all_imf`.

    Returns
    -------
    pd.DataFrame
        Una fila por componente con nodos/enlaces por tipo de grafo.
    """
    filas: list[dict[str, Any]] = []
    for id_imf, tipos in resultados.items():
        fila: dict[str, Any] = {"componente": id_imf}
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
    return pd.DataFrame(filas)


def aplicar_transformaciones_grafos(
    ruta_imfs: Path = _RUTA_IMFS_XLE,
    dir_grafos: Path = _DIR_GRAFOS,
    tau_max: int = TAU_MAX,
    dim_max: int = DIM_MAX,
    umbral_percentil: float = UMBRAL_PERCENTIL_RECURRENCIA,
    random_state: int = RANDOM_STATE_RECURRENCIA,
) -> dict[str, Any]:
    """
    Genera HVG, NVG y grafos de recurrencia para cada IMF/residuo de XLE.

    Parameters
    ----------
    ruta_imfs : Path
        Parquet con las IMFs CEEMDAN.
    dir_grafos : Path
        Directorio base de salida (subcarpetas ``hvg``, ``nvg``, ``recurrencia``).
    tau_max : int
        Máximo delay para recurrencia.
    dim_max : int
        Dimensión máxima de embedding (FNN).
    umbral_percentil : float
        Percentil para el umbral de distancia en recurrencia.
    random_state : int
        Semilla para el umbral de recurrencia.

    Returns
    -------
    dict
        Resultados por componente (misma estructura que ``obtener_grafos_all_imf``).
    """
    dir_grafos.mkdir(parents=True, exist_ok=True)
    logger.info("Generando grafos en %s ...", dir_grafos)
    return obtener_grafos_all_imf(
        df_imfs=str(ruta_imfs),
        carpeta_salida_base=str(dir_grafos),
        tau_max=tau_max,
        dim_max=dim_max,
        umbral_percentil=umbral_percentil,
        random_state=random_state,
    )


def guardar_tabla_recurrencia(
    df_imfs: pd.DataFrame,
    ruta_csv: Path = _RUTA_PARAM_RECURRENCIA_CSV,
    umbral_percentil: float = UMBRAL_PERCENTIL_RECURRENCIA,
    random_state: int = RANDOM_STATE_RECURRENCIA,
) -> pd.DataFrame:
    """
    Calcula y guarda tau, d y epsilon por componente (sin reconstruir grafos).

    Parameters
    ----------
    df_imfs : pd.DataFrame
        IMFs CEEMDAN de XLE.
    ruta_csv : Path
        Ruta del CSV de salida.
    umbral_percentil : float
        Percentil del umbral de distancia.
    random_state : int
        Semilla para el cálculo del umbral.

    Returns
    -------
    pd.DataFrame
        Tabla de parámetros de recurrencia.
    """
    df_params = calcular_tabla_parametros_recurrencia(
        df_imfs,
        umbral_percentil=umbral_percentil,
        random_state=random_state,
    )
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    df_params.to_csv(ruta_csv, index=False)
    logger.info("Parámetros de recurrencia: %s", ruta_csv)
    return df_params


def guardar_manifest(
    resultados: dict[str, Any],
    df_resumen: pd.DataFrame,
    df_params: pd.DataFrame,
    ruta_json: Path = _RUTA_RESUMEN_JSON,
) -> None:
    """
    Escribe un JSON con rutas y métricas de los grafos generados.

    Parameters
    ----------
    resultados : dict
        Salida de :func:`aplicar_transformaciones_grafos`.
    df_resumen : pd.DataFrame
        Tabla resumen por componente.
    df_params : pd.DataFrame
        Parámetros de embedding de recurrencia.
    ruta_json : Path
        Archivo JSON de salida.
    """
    payload = {
        "ruta_imfs": str(_RUTA_IMFS_XLE),
        "dir_grafos": str(_DIR_GRAFOS),
        "parametros_recurrencia": df_params.to_dict(orient="records"),
        "resumen_por_componente": df_resumen.to_dict(orient="records"),
        "detalle_grafos": resultados,
    }
    ruta_json.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_json, "w", encoding="utf-8") as archivo:
        json.dump(payload, archivo, indent=2, ensure_ascii=False)
    logger.info("Manifiesto: %s", ruta_json)


def main(
    ruta_imfs: Optional[Path] = None,
    dir_grafos: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Ejecuta la pipeline de transformación IMF → grafos para XLE.

    Parameters
    ----------
    ruta_imfs : Path, optional
        Parquet de IMFs. Por defecto ``xle_imfs_ceemdan.parquet``.
    dir_grafos : Path, optional
        Carpeta base de grafos. Por defecto ``data/.../grafos``.

    Returns
    -------
    dict
        Rutas de salida y tablas resumen.
    """
    ruta = ruta_imfs or _RUTA_IMFS_XLE
    carpeta = dir_grafos or _DIR_GRAFOS

    df_imfs = cargar_imfs_ceemdan_xle(ruta)
    df_params = guardar_tabla_recurrencia(df_imfs)

    resultados = aplicar_transformaciones_grafos(
        ruta_imfs=ruta,
        dir_grafos=carpeta,
    )
    df_resumen = resumir_resultados_grafos(resultados)
    _RUTA_RESUMEN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_resumen.to_csv(_RUTA_RESUMEN_CSV, index=False)
    logger.info("Resumen grafos: %s", _RUTA_RESUMEN_CSV)

    guardar_manifest(resultados, df_resumen, df_params)

    return {
        "ruta_imfs": str(ruta),
        "dir_grafos": str(carpeta),
        "ruta_resumen_csv": str(_RUTA_RESUMEN_CSV),
        "ruta_param_recurrencia_csv": str(_RUTA_PARAM_RECURRENCIA_CSV),
        "ruta_manifest_json": str(_RUTA_RESUMEN_JSON),
        "resumen": df_resumen,
        "parametros_recurrencia": df_params,
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Transformaciones IMF→grafo (HVG, NVG, recurrencia) para XLE CEEMDAN."
    )
    parser.add_argument(
        "--parquet-imfs",
        type=Path,
        default=None,
        help="Parquet con IMF_1,...,Residuo (por defecto xle_imfs_ceemdan.parquet).",
    )
    parser.add_argument(
        "--dir-grafos",
        type=Path,
        default=None,
        help="Directorio base de salida de grafos.",
    )
    args = parser.parse_args()
    try:
        salida = main(ruta_imfs=args.parquet_imfs, dir_grafos=args.dir_grafos)
        logger.info("Completado. Resumen: %s", salida["ruta_resumen_csv"])
    except Exception:
        logger.exception("Error en transformaciones de grafos XLE")
        sys.exit(1)
