"""
Compute summary quantities in ``docs/20abr26/main.tex`` (*General structural properties.
Connectivity and distance* subsubsection) and table ``tab:propiedades_estructurales``.

For each CEEMDAN parquet component (IMF$_1$--IMF$_8$ and ``Residuo``), HVG, NVG, and
recurrence graphs are obtained (same criteria as in the document: 10th percentile for
$\varepsilon$, FNN, MI for $\tau$).

Output
------
- ``docs/20abr26/out/metricas_estructurales_por_componente_ceemdan.csv``
- ``docs/20abr26/out/resumen_estructural_grafos_ceemdan.csv`` (min--max ranges)
- ``docs/20abr26/out/resumen_estructural_grafos_ceemdan.md`` (suggested LaTeX table)

Dependencies: ``networkx``, ``ts2vg``, ``pandas``, ``numpy``, ``scipy``, ``scikit-learn``,
and the ``GraphEMD`` package (via ``PYTHONPATH=src/python``), which imports ``torch`` and
``torch_geometric`` when loading ``graph_imf_transform_utils``.

Execution::

    PYTHONPATH=src/python python scripts/exploration/run_structural_graph_metrics_ceemdan_20abr26.py
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


def _diametro_aproximado(G: nx.Graph) -> float:
    """
    Estimate the diameter with two BFS (worst-case bound on unweighted graphs).

    Parameters
    ----------
    G : nx.Graph
        Non-empty connected graph.

    Returns
    -------
    float
        Estimated diameter (exact if the graph is a tree or satisfies
        double-sweep algorithm properties).
    """
    if G.number_of_nodes() == 0:
        return float("nan")
    if G.number_of_edges() == 0:
        return 0.0
    u = next(iter(G.nodes))
    d1 = nx.single_source_shortest_path_length(G, u)
    v = max(d1, key=d1.get)
    d2 = nx.single_source_shortest_path_length(G, v)
    return float(max(d2.values()))


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


def metricas_hvg(x: np.ndarray) -> dict[str, Any]:
    """
    Structural metrics of the HVG on series ``x``.

    Parameters
    ----------
    x : np.ndarray
        Series for one IMF or the residue.

    Returns
    -------
    dict[str, Any]
        n, m, density, components, diameter (approx.), mean degree.
    """
    G = _grafo_visibilidad(x, HorizontalVG)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    dens = nx.density(G)
    comp = nx.number_connected_components(G)
    if comp != 1:
        diam = float("nan")
    else:
        diam = _diametro_aproximado(G)
    avg_deg = (2.0 * m / n) if n else 0.0
    return {
        "n": n,
        "m": m,
        "densidad": dens,
        "componentes": comp,
        "diametro": diam,
        "grado_medio": avg_deg,
    }


def metricas_nvg(x: np.ndarray) -> dict[str, Any]:
    """
    Structural metrics of the NVG on series ``x``.

    Parameters
    ----------
    x : np.ndarray
        Series for one IMF or the residue.

    Returns
    -------
    dict[str, Any]
        n, m, density, components, diameter (approx.), mean degree.
    """
    G = _grafo_visibilidad(x, NaturalVG)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    dens = nx.density(G)
    comp = nx.number_connected_components(G)
    if comp != 1:
        diam = float("nan")
    else:
        diam = _diametro_aproximado(G)
    avg_deg = (2.0 * m / n) if n else 0.0
    return {
        "n": n,
        "m": m,
        "densidad": dens,
        "componentes": comp,
        "diametro": diam,
        "grado_medio": avg_deg,
    }


def metricas_recurrencia(
    x: np.ndarray,
    umbral_percentil: float = 10.0,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Recurrence-graph metrics (symmetric matrix, no self-loops).

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
    dict[str, Any]
        n (embedding nodes), m (undirected edges), densidad, componentes,
        diameter if connected, mean degree.
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
    G = nx.from_numpy_array(sym)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    dens = nx.density(G)
    comp = nx.number_connected_components(G)
    if comp != 1:
        diam = float("nan")
    else:
        diam = _diametro_aproximado(G)
    avg_deg = (2.0 * m / n) if n else 0.0
    return {
        "n": n,
        "m": m,
        "densidad": dens,
        "componentes": comp,
        "diametro": diam,
        "grado_medio": avg_deg,
    }


def _formatear_rango_resumen(r: dict[str, float], *, entero: bool) -> str:
    """
    Format a min--max pair for the exported summary table.

    Parameters
    ----------
    r : dict[str, float]
        Diccionario con claves ``min`` y ``max``.
    entero : bool
        If True, rounds to integers (p. ej. number of edges).

    Returns
    -------
    str
        ``min - max`` string ready for CSV or LaTeX.
    """
    if entero:
        return f"{int(round(r['min']))} - {int(round(r['max']))}"
    return f"{r['min']:.6g} - {r['max']:.6g}"


def _rango_columna(df: pd.DataFrame, nombre_columna: str) -> dict[str, float]:
    """
    Return minimum and maximum of a column, ignoring NaN in diameter metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Per-component detail table.
    nombre_columna : str
        Column name (e.g. ``hvg_densidad``).

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
    Entry point: compute metrics CSV and summary for the LaTeX document.
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
        mh = metricas_hvg(x)
        mn = metricas_nvg(x)
        mr = metricas_recurrencia(x)
        filas.append(
            {
                "componente": nombre,
                "hvg_n": mh["n"],
                "hvg_m": mh["m"],
                "hvg_densidad": mh["densidad"],
                "hvg_componentes": mh["componentes"],
                "hvg_diametro": mh["diametro"],
                "hvg_grado_medio": mh["grado_medio"],
                "nvg_n": mn["n"],
                "nvg_m": mn["m"],
                "nvg_densidad": mn["densidad"],
                "nvg_componentes": mn["componentes"],
                "nvg_diametro": mn["diametro"],
                "nvg_grado_medio": mn["grado_medio"],
                "rec_n": mr["n"],
                "rec_m": mr["m"],
                "rec_densidad": mr["densidad"],
                "rec_componentes": mr["componentes"],
                "rec_diametro": mr["diametro"],
                "rec_grado_medio": mr["grado_medio"],
            }
        )

    det = pd.DataFrame(filas)
    out_dir = _REPO_ROOT / "docs" / "20abr26" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    det_path = out_dir / "metricas_estructurales_por_componente_ceemdan.csv"
    det.to_csv(det_path, index=False)
    logger.info("Detail: %s", det_path)

    sufijos = [
        ("densidad", "Density (range)"),
        ("m", "Number of links (range)"),
        ("componentes", "Connected components"),
        ("diametro", "Diameter (range)"),
        ("grado_medio", "Average degree (range)"),
    ]
    filas_res: list[dict[str, str]] = []
    for suf, etiqueta in sufijos:
        rh = _rango_columna(det, f"hvg_{suf}")
        rr = _rango_columna(det, f"rec_{suf}")
        rn = _rango_columna(det, f"nvg_{suf}")
        es_entero = suf in ("m", "componentes")
        filas_res.append(
            {
                "property": etiqueta,
                "HVG": _formatear_rango_resumen(rh, entero=es_entero),
                "Recurrence": _formatear_rango_resumen(rr, entero=es_entero),
                "NVG": _formatear_rango_resumen(rn, entero=es_entero),
            }
        )
    res_df = pd.DataFrame(filas_res)
    res_path = out_dir / "resumen_estructural_grafos_ceemdan.csv"
    res_df.to_csv(res_path, index=False)
    logger.info("Summary: %s", res_path)

    # Text for pasting into LaTeX (4 decimals for densities and degrees when applicable)
    md_lines = [
        "# Structural summary (CEEMDAN)",
        "",
        res_df.to_string(index=False),
        "",
        "Full numeric values in `metricas_estructurales_por_componente_ceemdan.csv`.",
    ]
    md_path = out_dir / "resumen_estructural_grafos_ceemdan.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Markdown: %s", md_path)


if __name__ == "__main__":
    main()
