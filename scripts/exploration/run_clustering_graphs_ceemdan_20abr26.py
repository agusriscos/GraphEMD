"""
Compute clustering coefficients (mean, median, and standard deviation of the per-node
distribution) for the *Clustering coefficients and local structure* subsubsection and
table ``tab:clustering_coefficients`` in ``docs/20abr26/main.tex``.

For each component (IMF$_1$--IMF$_8$ and ``Residuo``), HVG, NVG, and recurrence graphs
are built with the same criteria as in
``run_structural_graph_metrics_ceemdan_20abr26.py``.

If NVG has more than ``MAX_ARISTAS_NVG_CLUSTERING`` edges, clustering metrics are not
computed (prohibitive cost; typically the residue).

Output
------
- ``docs/20abr26/out/clustering_por_componente_ceemdan.csv``
- ``docs/20abr26/out/resumen_clustering_grafos_ceemdan.csv``
- ``docs/20abr26/out/resumen_clustering_grafos_ceemdan.md``

Dependencies: same as ``run_structural_graph_metrics_ceemdan_20abr26.py``.

Execution::

    PYTHONPATH=src/python python scripts/exploration/run_clustering_graphs_ceemdan_20abr26.py
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

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
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
    Build an undirected simple graph from edges returned by ts2vg.

    Parameters
    ----------
    x : np.ndarray
        1D series.
    constructor : type
        ``HorizontalVG`` o ``NaturalVG``.

    Returns
    -------
    nx.Graph
        Graph with deduplicated edges deduplicadas.
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
    Build the recurrence graph as a NetworkX ``Graph``.

    Parameters
    ----------
    x : np.ndarray
        Series for one IMF or the residue.
    umbral_percentil : float, optional
        Percentile for the distance threshold de distancia.
    random_state : int, optional
        Seed for el umbral basado en muestreo.

    Returns
    -------
    nx.Graph
        Unweighted graph sin bucles.
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
    Compute mean, median, and standard deviation of local clustering coefficients.

    The mean matches ``networkx.average_clustering`` on unweighted graphs.

    Parameters
    ----------
    G : nx.Graph
        Undirected graph simple.

    Returns
    -------
    dict[str, float]
        Keys ``media``, ``mediana``, and ``desviacion_tipica`` (population, ddof=0).
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
    Format a min--max pair for the exported summary table.

    Parameters
    ----------
    r : dict[str, float]
        Diccionario con claves ``min`` y ``max``.
    cuatro_decimales : bool
        If True, usa four fixed decimal places.

    Returns
    -------
    str
        ``min - max`` string.
    """
    if cuatro_decimales:
        return f"{r['min']:.4f} - {r['max']:.4f}"
    return f"{r['min']:.6g} - {r['max']:.6g}"


def _rango_columna(df: pd.DataFrame, nombre_columna: str) -> dict[str, float]:
    """
    Return minimum and maximum of a numeric column, ignoring NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Per-component detail table.
    nombre_columna : str
        Column name.

    Returns
    -------
    dict[str, float]
        Keys ``min`` and ``max``.
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
    Compute per-component clustering CSV and range summary for LaTeX.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ruta_imfs = _REPO_ROOT / "data" / "20abr26" / "msci_world_imfs_ceemdan.parquet"
    if not ruta_imfs.is_file():
        raise FileNotFoundError(f"Not found: {ruta_imfs}")

    df_imfs = pd.read_parquet(ruta_imfs, engine="pyarrow")
    columnas = [c for c in df_imfs.columns if c.startswith("IMF_") or c == "Residuo"]

    filas: list[dict[str, Any]] = []
    for nombre in columnas:
        x = np.asarray(df_imfs[nombre].to_numpy(), dtype=np.float64)
        logger.info("Processing %s...", nombre)

        Gh = _grafo_visibilidad(x, HorizontalVG)
        sh = estadisticas_clustering(Gh)

        Gn = _grafo_visibilidad(x, NaturalVG)
        mn = Gn.number_of_edges()
        if mn > MAX_ARISTAS_NVG_CLUSTERING:
            logger.info(
                "  NVG with %s edges: clustering skipped (>%s).",
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
    logger.info("Detail: %s", det_path)

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
    logger.info("Summary: %s", res_path)

    md_path = out_dir / "resumen_clustering_grafos_ceemdan.md"
    nota = (
        f"\n\nNVG: if the graph exceeds {MAX_ARISTAS_NVG_CLUSTERING} edges, "
        "clustering is not computed (NaN in detail); NVG range "
        "uses only components with finite values."
    )
    md_path.write_text(
        "# Clustering summary (CEEMDAN)\n\n"
        + res_df.to_string(index=False)
        + nota
        + "\n\nDetalle: `clustering_por_componente_ceemdan.csv`.\n",
        encoding="utf-8",
    )
    logger.info("Markdown: %s", md_path)


if __name__ == "__main__":
    main()
