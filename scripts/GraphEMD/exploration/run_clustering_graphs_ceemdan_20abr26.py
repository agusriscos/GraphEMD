"""
Calcula coeficientes de clustering (media, mediana y desviación típica de la
distribución por nodo) para la subsubsección *Clustering coefficients and local
structure* y la tabla ``tab:clustering_coefficients`` en ``docs/20abr26/main.tex``.

Para cada componente (IMF$_1$--IMF$_8$ y ``Residuo``) se construyen HVG, NVG y
grafo de recurrencia con los mismos criterios que en
``ejecutar_metricas_estructurales_grafos_ceemdan_20abr26.py``.

Si el NVG tiene más de ``MAX_ARISTAS_NVG_CLUSTERING`` aristas, no se calculan
las métricas de clustering (coste prohibitivo; típicamente el residuo).

Salida
------
- ``docs/20abr26/out/clustering_por_componente_ceemdan.csv``
- ``docs/20abr26/out/resumen_clustering_grafos_ceemdan.csv``
- ``docs/20abr26/out/resumen_clustering_grafos_ceemdan.md``

Dependencias: mismas que ``ejecutar_metricas_estructurales_grafos_ceemdan_20abr26.py``.

Ejecución::

    PYTHONPATH=src/python python scripts/GraphEMD/exploracion/ejecutar_clustering_grafos_ceemdan_20abr26.py
"""

from __future__ import annotations

import contextlib
import io
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from ts2vg import HorizontalVG, NaturalVG

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SRC_PYTHON = _REPO_ROOT / "src" / "python"
if str(_SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(_SRC_PYTHON))

from GraphEMD.data.graph_imf_transform_utils import (
    calcular_false_nearest_neighbors,
    calcular_matriz_recurrencia,
    construir_espacio_embedding,
    seleccionar_tau,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
_BUF = io.StringIO()

MAX_ARISTAS_NVG_CLUSTERING = 2_000_000


def _grafo_visibilidad(
    x: np.ndarray, constructor: type[HorizontalVG] | type[NaturalVG]
) -> nx.Graph:
    """
    Construye un grafo no dirigido simple a partir de aristas devueltas por ts2vg.

    Parameters
    ----------
    x : np.ndarray
        Serie 1D.
    constructor : type
        ``HorizontalVG`` o ``NaturalVG``.

    Returns
    -------
    nx.Graph
        Grafo con aristas deduplicadas.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    g = constructor(directed="left_to_right").build(x)
    G = nx.Graph()
    G.add_edges_from(map(tuple, np.array(g.edges)))
    return G


def _grafo_recurrencia(
    x: np.ndarray,
    umbral_percentil: float = 10.0,
    random_state: int = 42,
) -> nx.Graph:
    """
    Construye el grafo de recurrencia como ``Graph`` de NetworkX.

    Parameters
    ----------
    x : np.ndarray
        Serie de una IMF o del residuo.
    umbral_percentil : float, optional
        Percentil para el umbral de distancia.
    random_state : int, optional
        Semilla para el umbral basado en muestreo.

    Returns
    -------
    nx.Graph
        Grafo no ponderado sin bucles.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    tau = int(seleccionar_tau(x, tau_max=50))
    d = int(calcular_false_nearest_neighbors(x, tau=tau, dim_max=10))
    emb = construir_espacio_embedding(x, d, tau)
    with contextlib.redirect_stdout(_BUF):
        mat, _eps = calcular_matriz_recurrencia(
            emb, umbral_percentil=umbral_percentil, random_state=random_state
        )
    sym = np.maximum(mat, mat.T).astype(np.uint8)
    np.fill_diagonal(sym, 0)
    return nx.from_numpy_array(sym)


def estadisticas_clustering(G: nx.Graph) -> dict[str, float]:
    """
    Calcula media, mediana y desviación típica de los coeficientes locales.

    La media coincide con ``networkx.average_clustering`` en grafos no ponderados.

    Parameters
    ----------
    G : nx.Graph
        Grafo no dirigido simple.

    Returns
    -------
    dict[str, float]
        Claves ``media``, ``mediana`` y ``desviacion_tipica`` (poblacional, ddof=0).
    """
    if G.number_of_nodes() == 0:
        return {
            "media": float("nan"),
            "mediana": float("nan"),
            "desviacion_tipica": float("nan"),
        }
    coef = nx.clustering(G)
    vals = np.fromiter(coef.values(), dtype=np.float64, count=len(coef))
    return {
        "media": float(nx.average_clustering(G)),
        "mediana": float(np.median(vals)),
        "desviacion_tipica": float(np.std(vals, ddof=0)),
    }


def _formatear_rango_resumen(r: dict[str, float], *, cuatro_decimales: bool) -> str:
    """
    Formatea un par min--max para la tabla de resumen exportada.

    Parameters
    ----------
    r : dict[str, float]
        Diccionario con claves ``min`` y ``max``.
    cuatro_decimales : bool
        Si es True, usa cuatro decimales fijos.

    Returns
    -------
    str
        Cadena ``min - max``.
    """
    if cuatro_decimales:
        return f"{r['min']:.4f} - {r['max']:.4f}"
    return f"{r['min']:.6g} - {r['max']:.6g}"


def _rango_columna(df: pd.DataFrame, nombre_columna: str) -> dict[str, float]:
    """
    Devuelve mínimo y máximo de una columna numérica ignorando NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Tabla detallada por componente.
    nombre_columna : str
        Nombre de la columna.

    Returns
    -------
    dict[str, float]
        Claves ``min`` y ``max``.
    """
    arr = np.asarray(
        pd.to_numeric(df[nombre_columna], errors="coerce"), dtype=np.float64
    )
    return {
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
    }


def main() -> None:
    """
    Calcula CSV de clustering por componente y resumen de rangos para el LaTeX.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ruta_imfs = _REPO_ROOT / "data" / "20abr26" / "msci_world_imfs_ceemdan.parquet"
    if not ruta_imfs.is_file():
        raise FileNotFoundError(f"No se encuentra {ruta_imfs}")

    df_imfs = pd.read_parquet(ruta_imfs, engine="pyarrow")
    columnas = [c for c in df_imfs.columns if c.startswith("IMF_") or c == "Residuo"]

    filas: list[dict[str, Any]] = []
    for nombre in columnas:
        x = np.asarray(df_imfs[nombre].to_numpy(), dtype=np.float64)
        logger.info("Procesando %s...", nombre)

        Gh = _grafo_visibilidad(x, HorizontalVG)
        sh = estadisticas_clustering(Gh)

        Gn = _grafo_visibilidad(x, NaturalVG)
        mn = Gn.number_of_edges()
        if mn > MAX_ARISTAS_NVG_CLUSTERING:
            logger.info(
                "  NVG con %s aristas: omitido clustering (>%s).",
                mn,
                MAX_ARISTAS_NVG_CLUSTERING,
            )
            sn = {
                "media": float("nan"),
                "mediana": float("nan"),
                "desviacion_tipica": float("nan"),
            }
        else:
            sn = estadisticas_clustering(Gn)

        Gr = _grafo_recurrencia(x)
        sr = estadisticas_clustering(Gr)

        filas.append(
            {
                "componente": nombre,
                "hvg_media": sh["media"],
                "hvg_mediana": sh["mediana"],
                "hvg_desv_tip": sh["desviacion_tipica"],
                "nvg_media": sn["media"],
                "nvg_mediana": sn["mediana"],
                "nvg_desv_tip": sn["desviacion_tipica"],
                "nvg_m": mn,
                "nvg_clustering_omitido": mn > MAX_ARISTAS_NVG_CLUSTERING,
                "rec_media": sr["media"],
                "rec_mediana": sr["mediana"],
                "rec_desv_tip": sr["desviacion_tipica"],
            }
        )

    det = pd.DataFrame(filas)
    out_dir = _REPO_ROOT / "docs" / "20abr26" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    det_path = out_dir / "clustering_por_componente_ceemdan.csv"
    det.to_csv(det_path, index=False)
    logger.info("Detalle: %s", det_path)

    sufijos = [
        ("media", "Average coefficient (range)"),
        ("mediana", "Median coefficient (range)"),
        ("desv_tip", "Standard deviation (range)"),
    ]
    filas_res: list[dict[str, str]] = []
    for suf, etiqueta in sufijos:
        rh = _rango_columna(det, f"hvg_{suf}")
        rr = _rango_columna(det, f"rec_{suf}")
        rn = _rango_columna(det, f"nvg_{suf}")
        filas_res.append(
            {
                "property": etiqueta,
                "HVG": _formatear_rango_resumen(rh, cuatro_decimales=True),
                "Recurrence": _formatear_rango_resumen(rr, cuatro_decimales=True),
                "NVG": _formatear_rango_resumen(rn, cuatro_decimales=True),
            }
        )
    res_df = pd.DataFrame(filas_res)
    res_path = out_dir / "resumen_clustering_grafos_ceemdan.csv"
    res_df.to_csv(res_path, index=False)
    logger.info("Resumen: %s", res_path)

    md_path = out_dir / "resumen_clustering_grafos_ceemdan.md"
    nota = (
        f"\n\nNVG: si el grafo supera {MAX_ARISTAS_NVG_CLUSTERING} aristas, "
        "el clustering no se calcula (NaN en el detalle); el rango de NVG "
        "usa solo componentes con valor finito."
    )
    md_path.write_text(
        "# Resumen clustering (CEEMDAN)\n\n"
        + res_df.to_string(index=False)
        + nota
        + "\n\nDetalle: `clustering_por_componente_ceemdan.csv`.\n",
        encoding="utf-8",
    )
    logger.info("Markdown: %s", md_path)


if __name__ == "__main__":
    main()
