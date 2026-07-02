#!/usr/bin/env python3
"""
Aplica transformaciones serie→grafo (HVG, NVG, recurrencia) a las componentes ICA
del panel empírico y calcula las mismas métricas estructurales, de clustering y
de centralidad usadas en los grafos IMF (script ``06_obtener_metricas_grafos_*``).

Por activo se lee ``ica/fastica/imfs_reducidas.parquet`` (MSCI World:
``docs/20abr26/out/imfs_dim_red/k4/fastica/imfs_reducidas.parquet``). Las salidas
se escriben en ``grafos_ica/`` dentro del directorio de datos de cada serie.

Ejemplo::

    PYTHONPATH=src/python python scripts/GraphEMD/ejecutar_grafos_ica_panel_activos.py

    PYTHONPATH=src/python python scripts/GraphEMD/ejecutar_grafos_ica_panel_activos.py \\
        --activos xle,xlv --solo-metricas
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC_PYTHON = _REPO_ROOT / "src" / "python"
_EXPLORACION = _REPO_ROOT / "scripts" / "GraphEMD" / "exploracion"
_SCRIPT_VMD = _REPO_ROOT / "scripts" / "GraphEMD" / "ejecutar_vmd_todos_activos.py"
_SCRIPT_METRICAS = (
    _REPO_ROOT
    / "scripts"
    / "GraphEMD"
    / "xle_etf_analysis"
    / "06_obtener_metricas_grafos_xle.py"
)
_MSCI_ICA = (
    _REPO_ROOT / "docs" / "20abr26" / "out" / "imfs_dim_red" / "k4" / "fastica"
)
_DIR_SALIDA_PANEL = _REPO_ROOT / "scripts" / "GraphEMD" / "out"

TAU_MAX: int = 50
DIM_MAX: int = 10
UMBRAL_PERCENTIL_RECURRENCIA: float = 10.0
RANDOM_STATE_RECURRENCIA: int = 42
TIPOS_GRAFO: Tuple[str, ...] = ("hvg", "nvg", "recurrencia")

logger = logging.getLogger(__name__)


def _asegurar_paths() -> None:
    """
    Añade al ``sys.path`` las rutas de importación del proyecto.
    """
    for ruta in (_SRC_PYTHON, _EXPLORACION):
        texto = str(ruta)
        if texto not in sys.path:
            sys.path.insert(0, texto)


def _cargar_modulo(ruta: Path, nombre: str) -> Any:
    """
    Carga un módulo Python desde una ruta de archivo.

    Parameters
    ----------
    ruta : Path
        Archivo ``.py`` a importar.
    nombre : str
        Nombre interno del módulo.

    Returns
    -------
    module
        Módulo cargado.

    Raises
    ------
    ImportError
        Si no se puede resolver el spec o el loader.
    """
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar {ruta}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


def _columnas_ica(df: pd.DataFrame, incluir_residuo: bool = True) -> List[str]:
    """
    Devuelve las columnas ICA ordenadas (``Z_1``, …, ``Z_k`` y opcional ``Residuo``).

    Parameters
    ----------
    df : pd.DataFrame
        Parquet reducido FastICA.
    incluir_residuo : bool
        Si es True, añade ``Residuo`` al final cuando exista.

    Returns
    -------
    list of str
        Nombres de columnas a codificar.
    """
    cols_z = sorted(
        [c for c in df.columns if c.startswith("Z_")],
        key=lambda nombre: int(nombre.split("_")[1]),
    )
    if incluir_residuo and "Residuo" in df.columns:
        cols_z.append("Residuo")
    if not cols_z:
        raise ValueError(
            f"Sin columnas Z_* en el parquet. Columnas: {list(df.columns)}"
        )
    return cols_z


def _ruta_parquet_ica(cfg: Any) -> Path:
    """
    Resuelve la ruta al parquet ICA CEEMDAN de un activo.

    Parameters
    ----------
    cfg : ConfigActivo
        Configuración del activo (desde ``ejecutar_vmd_todos_activos``).

    Returns
    -------
    Path
        Ruta a ``imfs_reducidas.parquet``.
    """
    if cfg.id_activo == "msci_world":
        return _MSCI_ICA / "imfs_reducidas.parquet"
    return cfg.dir_datos / "ica" / "fastica" / "imfs_reducidas.parquet"


def _rutas_salida_activo(cfg: Any) -> Dict[str, Path]:
    """
    Devuelve rutas de salida para grafos y métricas ICA de un activo.

    Parameters
    ----------
    cfg : ConfigActivo
        Configuración del activo.

    Returns
    -------
    dict
        Claves: ``parquet_ica``, ``dir_grafos``, ``resumen_csv``, ``param_recurrencia_csv``,
        ``manifest_json``, ``metricas_csv``, ``resumen_metricas_json``.
    """
    prefijo = cfg.prefijo
    dir_datos = cfg.dir_datos
    return {
        "parquet_ica": _ruta_parquet_ica(cfg),
        "dir_grafos": dir_datos / "grafos_ica",
        "resumen_csv": dir_datos / f"{prefijo}_resumen_grafos_ica.csv",
        "param_recurrencia_csv": dir_datos / f"{prefijo}_parametros_recurrencia_ica.csv",
        "manifest_json": dir_datos / f"{prefijo}_grafos_ica_manifest.json",
        "metricas_csv": dir_datos / f"{prefijo}_metricas_grafos_ica.csv",
        "resumen_metricas_json": dir_datos / f"{prefijo}_resumen_metricas_grafos_ica.json",
    }


def calcular_tabla_parametros_recurrencia_ica(
    df_componentes: pd.DataFrame,
    columnas: List[str],
    umbral_percentil: float = UMBRAL_PERCENTIL_RECURRENCIA,
    random_state: int = RANDOM_STATE_RECURRENCIA,
    tau_max: int = TAU_MAX,
    dim_max: int = DIM_MAX,
) -> pd.DataFrame:
    """
    Calcula tau, dimensión de embedding y epsilon por componente ICA.

    Parameters
    ----------
    df_componentes : pd.DataFrame
        Series ``Z_j`` (y opcional ``Residuo``).
    columnas : list of str
        Columnas a evaluar.
    umbral_percentil : float
        Percentil para el umbral de distancias.
    random_state : int
        Semilla del umbral.
    tau_max : int
        Máximo delay en información mutua.
    dim_max : int
        Dimensión máxima en FNN.

    Returns
    -------
    pd.DataFrame
        Columnas: componente, tau, d, epsilon.
    """
    _asegurar_paths()
    from GraphEMD.data.graph_imf_transform_utils import (  # noqa: WPS433
        calcular_false_nearest_neighbors,
        calcular_matriz_recurrencia,
        construir_espacio_embedding,
        seleccionar_tau,
    )

    buf = io.StringIO()
    filas: List[Dict[str, Any]] = []
    for nombre in columnas:
        x = np.asarray(df_componentes[nombre].values, dtype=np.float64).copy()
        tau = int(seleccionar_tau(x, tau_max=tau_max))
        d = int(calcular_false_nearest_neighbors(x, tau=tau, dim_max=dim_max))
        emb = construir_espacio_embedding(x, d, tau)
        with contextlib.redirect_stdout(buf):
            _, eps = calcular_matriz_recurrencia(
                emb,
                umbral_percentil=umbral_percentil,
                random_state=random_state,
            )
        filas.append(
            {"componente": nombre, "tau": tau, "d": d, "epsilon": float(eps)}
        )
    return pd.DataFrame(filas)


def resumir_resultados_grafos(resultados: Dict[str, Any]) -> pd.DataFrame:
    """
    Convierte la salida de ``obtener_grafos_all_imf`` en tabla plana.

    Parameters
    ----------
    resultados : dict
        Diccionario por componente y tipo de grafo.

    Returns
    -------
    pd.DataFrame
        Una fila por componente con nodos/enlaces por tipo.
    """
    filas: List[Dict[str, Any]] = []
    for id_componente, tipos in resultados.items():
        fila: Dict[str, Any] = {"componente": id_componente}
        for tipo_grafo in TIPOS_GRAFO:
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


def aplicar_transformaciones_ica_activo(
    cfg: Any,
    incluir_residuo: bool = True,
    forzar: bool = False,
) -> Dict[str, Any]:
    """
    Genera HVG, NVG y recurrencia para cada componente ICA de un activo.

    Parameters
    ----------
    cfg : ConfigActivo
        Activo del panel.
    incluir_residuo : bool
        Codifica también el residuo CEEMDAN si está en el parquet.
    forzar : bool
        Regenera grafos aunque exista el manifiesto.

    Returns
    -------
    dict
        Rutas, tablas resumen y resultados por componente.
    """
    _asegurar_paths()
    from GraphEMD.data.graph_imf_transform_utils import obtener_grafos_all_imf  # noqa: WPS433

    rutas = _rutas_salida_activo(cfg)
    ruta_parquet = rutas["parquet_ica"]
    dir_grafos = rutas["dir_grafos"]

    if not ruta_parquet.is_file():
        raise FileNotFoundError(
            f"No se encuentra {ruta_parquet}. Ejecute antes el script ICA del activo."
        )

    if rutas["manifest_json"].is_file() and not forzar:
        logger.info(
            "%s: manifiesto ICA existente (%s); omitiendo transformaciones.",
            cfg.nombre,
            rutas["manifest_json"],
        )
        with open(rutas["manifest_json"], encoding="utf-8") as archivo:
            manifest = json.load(archivo)
        return {
            "activo": cfg.id_activo,
            "omitido": True,
            "manifest": manifest,
            **rutas,
        }

    df_ica = pd.read_parquet(ruta_parquet, engine="pyarrow")
    columnas = _columnas_ica(df_ica, incluir_residuo=incluir_residuo)
    logger.info(
        "%s: generando grafos ICA para %s en %s",
        cfg.nombre,
        columnas,
        dir_grafos,
    )

    dir_grafos.mkdir(parents=True, exist_ok=True)
    df_params = calcular_tabla_parametros_recurrencia_ica(df_ica, columnas)
    df_params.to_csv(rutas["param_recurrencia_csv"], index=False)

    resultados = obtener_grafos_all_imf(
        df_imfs=str(ruta_parquet),
        carpeta_salida_base=str(dir_grafos),
        tau_max=TAU_MAX,
        dim_max=DIM_MAX,
        umbral_percentil=UMBRAL_PERCENTIL_RECURRENCIA,
        random_state=RANDOM_STATE_RECURRENCIA,
        columnas_imf=columnas,
    )

    df_resumen = resumir_resultados_grafos(resultados)
    df_resumen.to_csv(rutas["resumen_csv"], index=False)

    manifest = {
        "activo": cfg.id_activo,
        "nombre": cfg.nombre,
        "metodo": "FastICA_CEEMDAN",
        "ruta_parquet_ica": str(ruta_parquet.resolve()),
        "dir_grafos": str(dir_grafos.resolve()),
        "columnas_procesadas": columnas,
        "parametros_recurrencia": df_params.to_dict(orient="records"),
        "resumen_por_componente": df_resumen.to_dict(orient="records"),
        "detalle_grafos": resultados,
    }
    with open(rutas["manifest_json"], "w", encoding="utf-8") as archivo:
        json.dump(manifest, archivo, indent=2, ensure_ascii=False)

    logger.info(
        "%s: %d componentes ICA codificados; resumen en %s",
        cfg.nombre,
        len(columnas),
        rutas["resumen_csv"],
    )
    return {
        "activo": cfg.id_activo,
        "omitido": False,
        "columnas": columnas,
        "resumen": df_resumen,
        "parametros_recurrencia": df_params,
        "manifest": manifest,
        **{k: v for k, v in rutas.items()},
    }


def id_carpeta_a_componente_ica(id_carpeta: str) -> str:
    """
    Convierte el nombre de carpeta (``z_1``, ``residuo``) al id de componente ICA.

    Parameters
    ----------
    id_carpeta : str
        Nombre del subdirectorio bajo ``grafos_ica/<tipo>/``.

    Returns
    -------
    str
        ``Z_1``, …, ``Z_k`` o ``Residuo``.
    """
    if id_carpeta == "residuo":
        return "Residuo"
    if id_carpeta.startswith("z_"):
        sufijo = id_carpeta.split("_", 1)[1]
        return f"Z_{sufijo}"
    return id_carpeta


def listar_componentes_grafos_ica(dir_grafos: Path) -> List[str]:
    """
    Lista componentes ICA disponibles a partir de las carpetas HVG.

    Parameters
    ----------
    dir_grafos : Path
        Directorio base ``grafos_ica/``.

    Returns
    -------
    list of str
        Nombres de componente ordenados.
    """
    carpeta_hvg = dir_grafos / "hvg"
    if not carpeta_hvg.is_dir():
        raise FileNotFoundError(
            f"No existe {carpeta_hvg}. Ejecute primero las transformaciones ICA."
        )
    ids = sorted(p.name for p in carpeta_hvg.iterdir() if p.is_dir())
    return [id_carpeta_a_componente_ica(nombre) for nombre in ids]


def ruta_archivo_grafo_ica(
    dir_grafos: Path,
    tipo_grafo: str,
    componente: str,
) -> Path:
    """
    Devuelve la ruta al archivo ``.pt`` del grafo ICA.

    Parameters
    ----------
    dir_grafos : Path
        Directorio base de grafos ICA.
    tipo_grafo : str
        ``hvg``, ``nvg`` o ``recurrencia``.
    componente : str
        ``Z_k`` o ``Residuo``.

    Returns
    -------
    Path
        Ruta al objeto Data serializado.
    """
    if componente == "Residuo":
        id_carpeta = "residuo"
    else:
        id_carpeta = componente.lower()
    return (
        dir_grafos
        / tipo_grafo
        / id_carpeta
        / f"grafo_{tipo_grafo}_{id_carpeta}.pt"
    )


def _calcular_metricas_grafo_seguro(
    mod_metricas: Any,
    grafo: Any,
    componente: str,
    tipo: str,
) -> Dict[str, Any]:
    """
    Calcula métricas de grafo tolerando fallos de convergencia en centralidad.

    Parameters
    ----------
    mod_metricas : module
        Módulo con funciones de métricas IMF (script 06).
    grafo : nx.Graph
        Grafo cargado desde disco.
    componente : str
        Nombre del componente ICA.
    tipo : str
        Tipo de grafo (``hvg``, ``nvg``, ``recurrencia``).

    Returns
    -------
    dict
        Fila de métricas; centralidad parcial o NaN si no converge.
    """
    fila: Dict[str, Any] = {
        "componente": componente,
        "tipo_grafo": tipo,
    }
    fila.update(mod_metricas.metricas_estructurales(grafo))
    fila.update(mod_metricas.metricas_clustering(grafo, tipo))
    try:
        fila.update(mod_metricas.metricas_centralidad(grafo, tipo))
    except Exception as exc:
        logger.warning(
            "Centralidad omitida para %s %s (%s): %s",
            componente,
            tipo,
            type(exc).__name__,
            exc,
        )
        fila.update(
            {
                "degree_centrality_media": float("nan"),
                "betweenness_centrality_media": float("nan"),
                "closeness_centrality_media": float("nan"),
                "eigenvector_centrality_media": float("nan"),
                "centralidad_omitida": True,
                "betweenness_aproximada": False,
                "closeness_muestreada": False,
            }
        )
    return fila


def calcular_metricas_ica_activo(cfg: Any) -> pd.DataFrame:
    """
    Calcula métricas de grafos ICA con la misma lógica que el script 06 IMF.

    Parameters
    ----------
    cfg : ConfigActivo
        Activo del panel.

    Returns
    -------
    pd.DataFrame
        Una fila por (componente, tipo_grafo) con columna adicional ``activo``.
    """
    mod_metricas = _cargar_modulo(_SCRIPT_METRICAS, "metricas_grafos_imf_xle")
    rutas = _rutas_salida_activo(cfg)
    dir_grafos = rutas["dir_grafos"]
    componentes = listar_componentes_grafos_ica(dir_grafos)
    filas: List[Dict[str, Any]] = []

    for componente in componentes:
        for tipo in TIPOS_GRAFO:
            ruta_pt = ruta_archivo_grafo_ica(dir_grafos, tipo, componente)
            logger.info("%s — %s %s ...", cfg.nombre, componente, tipo.upper())
            grafo = mod_metricas.cargar_grafo_desde_pt(ruta_pt)
            fila = _calcular_metricas_grafo_seguro(
                mod_metricas, grafo, componente, tipo
            )
            fila["activo"] = cfg.id_activo
            fila["nombre_activo"] = cfg.nombre
            filas.append(fila)

    return pd.DataFrame(filas)


def guardar_metricas_activo(
    cfg: Any,
    df_metricas: pd.DataFrame,
) -> Dict[str, str]:
    """
    Persiste CSV y JSON de métricas ICA por activo.

    Parameters
    ----------
    cfg : ConfigActivo
        Activo del panel.
    df_metricas : pd.DataFrame
        Métricas calculadas.

    Returns
    -------
    dict
        Rutas de salida escritas.
    """
    rutas = _rutas_salida_activo(cfg)
    rutas["metricas_csv"].parent.mkdir(parents=True, exist_ok=True)
    df_metricas.to_csv(rutas["metricas_csv"], index=False)

    omitidos_clustering = df_metricas.loc[
        df_metricas["clustering_omitido"] == True,  # noqa: E712
        ["componente", "tipo_grafo"],
    ].to_dict(orient="records")
    omitidos_centralidad = df_metricas.loc[
        df_metricas["centralidad_omitida"] == True,  # noqa: E712
        ["componente", "tipo_grafo"],
    ].to_dict(orient="records")

    payload = {
        "activo": cfg.id_activo,
        "nombre": cfg.nombre,
        "num_filas": int(len(df_metricas)),
        "componentes": sorted(df_metricas["componente"].unique().tolist()),
        "tipos_grafo": sorted(df_metricas["tipo_grafo"].unique().tolist()),
        "clustering_omitido": omitidos_clustering,
        "centralidad_omitida": omitidos_centralidad,
        "metricas": df_metricas.to_dict(orient="records"),
    }
    with open(rutas["resumen_metricas_json"], "w", encoding="utf-8") as archivo:
        json.dump(payload, archivo, indent=2, ensure_ascii=False)

    logger.info("%s: métricas ICA en %s", cfg.nombre, rutas["metricas_csv"])
    return {
        "metricas_csv": str(rutas["metricas_csv"]),
        "resumen_metricas_json": str(rutas["resumen_metricas_json"]),
    }


def procesar_activo(
    cfg: Any,
    pasos: Tuple[str, ...],
    incluir_residuo: bool = True,
    forzar_transformaciones: bool = False,
) -> Dict[str, Any]:
    """
    Ejecuta transformaciones y/o métricas ICA para un activo.

    Parameters
    ----------
    cfg : ConfigActivo
        Activo del panel.
    pasos : tuple of str
        Subconjunto de ``transformaciones``, ``metricas``.
    incluir_residuo : bool
        Incluye el residuo CEEMDAN en la codificación.
    forzar_transformaciones : bool
        Regenera grafos aunque exista manifiesto.

    Returns
    -------
    dict
        Resultados parciales por paso.
    """
    resultado: Dict[str, Any] = {"activo": cfg.id_activo, "nombre": cfg.nombre}

    if "transformaciones" in pasos:
        resultado["transformaciones"] = aplicar_transformaciones_ica_activo(
            cfg,
            incluir_residuo=incluir_residuo,
            forzar=forzar_transformaciones,
        )

    if "metricas" in pasos:
        df_metricas = calcular_metricas_ica_activo(cfg)
        resultado["metricas"] = guardar_metricas_activo(cfg, df_metricas)
        resultado["df_metricas"] = df_metricas

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
        description=(
            "Transformaciones serie→grafo y métricas para componentes ICA del panel."
        )
    )
    parser.add_argument(
        "--activos",
        type=str,
        default="msci_world,xle,xlp,xlv,xauusd",
        help="IDs separados por coma.",
    )
    parser.add_argument(
        "--solo-transformaciones",
        action="store_true",
        help="Solo genera grafos (HVG, NVG, recurrencia).",
    )
    parser.add_argument(
        "--solo-metricas",
        action="store_true",
        help="Solo calcula métricas (requiere grafos_ica/ existentes).",
    )
    parser.add_argument(
        "--sin-residuo",
        action="store_true",
        help="Codifica solo Z_j, excluyendo el residuo CEEMDAN.",
    )
    parser.add_argument(
        "--forzar",
        action="store_true",
        help="Regenera grafos aunque exista manifiesto previo.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Punto de entrada: pipeline ICA→grafos→métricas para el panel empírico.

    Parameters
    ----------
    argv : list of str, optional
        Argumentos CLI.

    Returns
    -------
    dict
        Resultados por activo y tabla panel consolidada.
    """
    args = _parse_args(argv)
    mod_vmd = _cargar_modulo(_SCRIPT_VMD, "vmd_activos_grafos_ica")
    mapa_activos = {a.id_activo: a for a in mod_vmd.ACTIVOS}

    ids = [x.strip() for x in args.activos.split(",") if x.strip()]
    desconocidos = [x for x in ids if x not in mapa_activos]
    if desconocidos:
        raise ValueError(f"Activos desconocidos: {desconocidos}")

    if args.solo_transformaciones and args.solo_metricas:
        raise ValueError("Use solo uno de --solo-transformaciones o --solo-metricas.")
    if args.solo_transformaciones:
        pasos: Tuple[str, ...] = ("transformaciones",)
    elif args.solo_metricas:
        pasos = ("metricas",)
    else:
        pasos = ("transformaciones", "metricas")

    incluir_residuo = not args.sin_residuo
    resultados: Dict[str, Any] = {}

    for id_activo in ids:
        cfg = mapa_activos[id_activo]
        logger.info("=== %s (%s) ===", cfg.nombre, id_activo)
        salida = procesar_activo(
            cfg,
            pasos=pasos,
            incluir_residuo=incluir_residuo,
            forzar_transformaciones=args.forzar,
        )
        resultados[id_activo] = salida

    salida_panel: Dict[str, Any] = {"activos": resultados}
    if "metricas" in pasos:
        tablas_panel: List[pd.DataFrame] = []
        for cfg in mod_vmd.ACTIVOS:
            ruta_csv = _rutas_salida_activo(cfg)["metricas_csv"]
            if ruta_csv.is_file():
                tablas_panel.append(pd.read_csv(ruta_csv))
        if tablas_panel:
            df_panel = pd.concat(tablas_panel, ignore_index=True)
            _DIR_SALIDA_PANEL.mkdir(parents=True, exist_ok=True)
            ruta_panel_csv = _DIR_SALIDA_PANEL / "metricas_grafos_ica_panel.csv"
            ruta_panel_json = (
                _DIR_SALIDA_PANEL / "resumen_metricas_grafos_ica_panel.json"
            )
            df_panel.to_csv(ruta_panel_csv, index=False)
            with open(ruta_panel_json, "w", encoding="utf-8") as archivo:
                json.dump(
                    {
                        "num_filas": int(len(df_panel)),
                        "activos": sorted(df_panel["activo"].unique().tolist()),
                        "metricas": df_panel.to_dict(orient="records"),
                    },
                    archivo,
                    indent=2,
                    ensure_ascii=False,
                )
            salida_panel["panel_metricas_csv"] = str(ruta_panel_csv)
            salida_panel["panel_resumen_json"] = str(ruta_panel_json)
            logger.info(
                "Panel consolidado: %s (%d filas)", ruta_panel_csv, len(df_panel)
            )

    return salida_panel


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        main()
    except Exception:
        logger.exception("Error en pipeline grafos ICA del panel")
        sys.exit(1)
