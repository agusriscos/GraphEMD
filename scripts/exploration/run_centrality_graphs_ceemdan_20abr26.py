"""
Compute mean (and per-node maximum) betweenness, closeness, and eigenvector centrality
for the *Centrality analysis* subsubsection and table ``tab:centralidad`` in
``docs/20abr26/main.tex``.

For each component (IMF$_1$--IMF$_8$ and ``Residuo``), HVG, NVG, and recurrence graphs
are built with the same criteria as in
``run_structural_graph_metrics_ceemdan_20abr26.py``.

If NVG exceeds ``MAX_ARISTAS_NVG_CENTRALIDAD`` edges, centralities are not computed
(typically the residue). If ``m`` exceeds ``UMBRAL_ARISTAS_BETWEENNESS_APROX``, betweenness
uses ``k`` sampling (NetworkX). If ``m`` exceeds ``UMBRAL_ARISTAS_CERCANIA_EXACTA``,
NVG closeness uses node sampling (``closeness_centrality(G, u=...)``). For large
recurrence graphs, betweenness and closeness use the same approximation criteria
(see ``UMBRAL_ARISTAS_RECURRENCIA_APROX``). For ``n <= N_MAX_EIGENVECTOR_NUMPY``,
``eigenvector_centrality_numpy`` is used (dense ``n x n`` matrix; here ``n`` is the
daily sample size).

If ``*_nvg_edges.parquet`` exist in ``out_msci_world_grafos/imfs_ceemdan_20abr26``,
NVG is loaded from disk instead of ``ts2vg``.

Output
------
- ``docs/20abr26/out/centralidad_por_componente_ceemdan.csv``
- ``docs/20abr26/out/resumen_centralidad_grafos_ceemdan.csv``
- ``docs/20abr26/out/resumen_centralidad_grafos_ceemdan.md``

Dependencies: same as ``run_structural_graph_metrics_ceemdan_20abr26.py``.

Execution::

    PYTHONPATH=src/python python scripts/exploration/run_centrality_graphs_ceemdan_20abr26.py
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

MAX_ARISTAS_NVG_CENTRALIDAD = 2_000_000
UMBRAL_ARISTAS_BETWEENNESS_APROX = 200_000
K_MUESTRA_BETWEENNESS = 600
SEMILLA_BETWEENNESS = 42
UMBRAL_ARISTAS_CERCANIA_EXACTA = 50_000
K_MUESTRA_CERCANIA = 500
SEMILLA_CERCANIA = 42
UMBRAL_ARISTAS_RECURRENCIA_APROX = 12_000
N_MAX_EIGENVECTOR_NUMPY = 4096

_DIR_NVG_PRECALC = (
    _REPO_ROOT
    / "scripts"
    / "GraphEMD"
    / "exploration"
    / "out_msci_world_grafos"
    / "imfs_ceemdan_20abr26"
)


def _grafo_nvg_desde_parquet_si_existe(
    nombre_componente: str, n_nodos: int
) -> nx.Graph | None:
    """
    Load NVG edges from precalculated parquet if available.

    Avoids ``NaturalVG.build`` on large graphs when the team has already exported
    ``{componente}_nvg_edges.parquet`` en ``_DIR_NVG_PRECALC``.

    Parameters
    ----------
    nombre_componente : str
        ``IMF_1``, ..., ``IMF_8``, or ``Residuo``.
    n_nodos : int
        Expected number of nodes (series length).

    Returns
    -------
    nx.Graph | None
        Undirected graph or None if the file is missing.
    """
    ruta = _DIR_NVG_PRECALC / f"{nombre_componente}_nvg_edges.parquet"
    if not ruta.is_file():
        return None
    ed = pd.read_parquet(ruta, columns=["source", "target"])
    g = nx.Graph()
    g.add_edges_from(
        zip(
            ed["source"].to_numpy(dtype=np.int64), ed["target"].to_numpy(dtype=np.int64)
        )
    )
    g.add_nodes_from(range(n_nodos))
    return g


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


def _media_y_max(d: dict[Any, float]) -> tuple[float, float]:
    """
    Return mean and maximum of a node $\\to$ escalar.

    Parameters
    ----------
    d : dict
        Node-to-scalar map.

    Returns
    -------
    tuple[float, float]
        Mean and maximum; NaN if ``d`` is empty.
    """
    if not d:
        return float("nan"), float("nan")
    vals = np.fromiter(d.values(), dtype=np.float64, count=len(d))
    return float(np.mean(vals)), float(np.max(vals))


def _betweenness(G: nx.Graph, *, aproximar: bool) -> tuple[float, float, bool]:
    """
    Normalized betweenness: mean and maximum over nodes.

    Parameters
    ----------
    G : nx.Graph
        Undirected graph.
    aproximar : bool
        If True, uses random sampling of ``K_MUESTRA_BETWEENNESS`` nodes.

    Returns
    -------
    tuple[float, float, bool]
        Mean, maximum, and whether approximation was used.
    """
    n = G.number_of_nodes()
    if n == 0:
        return float("nan"), float("nan"), False
    if aproximar:
        k = min(K_MUESTRA_BETWEENNESS, n)
        bc = nx.betweenness_centrality(
            G, k=k, normalized=True, seed=SEMILLA_BETWEENNESS
        )
        media, mx = _media_y_max(bc)
        return media, mx, True
    bc = nx.betweenness_centrality(G, normalized=True)
    media, mx = _media_y_max(bc)
    return media, mx, False


def _closeness(G: nx.Graph, *, usar_muestreo: bool) -> tuple[float, float]:
    """
    Closeness (NetworkX): mean and maximum of per-node coefficients.

    If ``usar_muestreo`` is True (very dense graphs, typically NVG), mean and
    maximum are estimated with ``K_MUESTRA_CERCANIA`` randomly chosen nodes
    (``SEMILLA_CERCANIA``), computing ``closeness_centrality(G, u=...)`` per
    sampled node; avoids the $O(n(n+m))$ cost of the full version.

    Parameters
    ----------
    G : nx.Graph
        Undirected graph.
    usar_muestreo : bool
        If True, uses node sampling for closeness.

    Returns
    -------
    tuple[float, float]
        Estimated or exact mean and maximum.
    """
    if G.number_of_nodes() == 0:
        return float("nan"), float("nan")
    if not usar_muestreo:
        cc = nx.closeness_centrality(G)
        return _media_y_max(cc)
    rng = np.random.default_rng(SEMILLA_CERCANIA)
    nodos = list(G.nodes())
    k = min(K_MUESTRA_CERCANIA, len(nodos))
    muestra = rng.choice(nodos, size=k, replace=False)
    vals = [float(nx.closeness_centrality(G, u=u)) for u in muestra]
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.mean(arr)), float(np.max(arr))


def _eigenvector(G: nx.Graph) -> tuple[float, float]:
    """
    Eigenvector centrality (largest connected subgraph if needed).

    Parameters
    ----------
    G : nx.Graph
        Undirected graph.

    Returns
    -------
    tuple[float, float]
        Mean and maximum over nodes of the subgraph where it was computed.
    """
    if G.number_of_nodes() == 0:
        return float("nan"), float("nan")
    h: nx.Graph = G
    if not nx.is_connected(G):
        nodos = max(nx.connected_components(G), key=len)
        h = G.subgraph(nodos).copy()
    if h.number_of_nodes() <= N_MAX_EIGENVECTOR_NUMPY:
        try:
            ev = nx.eigenvector_centrality_numpy(h)
            return _media_y_max(ev)
        except (nx.NetworkXError, np.linalg.LinAlgError):
            pass
    try:
        ev = nx.eigenvector_centrality(h, max_iter=5000, tol=1e-06)
        return _media_y_max(ev)
    except nx.NetworkXError:
        return float("nan"), float("nan")


def _formatear_rango(r: dict[str, float], *, cuatro_decimales: bool) -> str:
    """
    Format ``min``--``max`` for CSV or LaTeX.

    Parameters
    ----------
    r : dict[str, float]
        Keys ``min`` and ``max``.
    cuatro_decimales : bool
        If True, four fixed decimal places.

    Returns
    -------
    str
        Formatted string.
    """
    if cuatro_decimales:
        return f"{r['min']:.4f} - {r['max']:.4f}"
    return f"{r['min']:.6g} - {r['max']:.6g}"


def _rango_columna(df: pd.DataFrame, nombre: str) -> dict[str, float]:
    """
    Minimum and maximum of a numeric column, ignoring NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Per-component table.
    nombre : str
        Column name.

    Returns
    -------
    dict[str, float]
        ``min`` y ``max``.
    """
    arr = np.asarray(pd.to_numeric(df[nombre], errors="coerce"), dtype=np.float64)
    return {"min": float(np.nanmin(arr)), "max": float(np.nanmax(arr))}


def main() -> None:
    """
    Write per-component centrality CSV and range summary (means).
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
        bh_m, bh_x, _ = _betweenness(Gh, aproximar=False)
        ch_m, ch_x = _closeness(Gh, usar_muestreo=False)
        eh_m, eh_x = _eigenvector(Gh)

        Gn = _grafo_nvg_desde_parquet_si_existe(nombre, len(x))
        if Gn is None:
            Gn = _grafo_visibilidad(x, NaturalVG)
        else:
            logger.info(
                "  NVG loaded from parquet (%s_nvg_edges.parquet).",
                nombre,
            )
        mn = Gn.number_of_edges()
        if mn > MAX_ARISTAS_NVG_CENTRALIDAD:
            logger.info(
                "  NVG with %s edges: centrality skipped (>%s).",
                mn,
                MAX_ARISTAS_NVG_CENTRALIDAD,
            )
            bn_m = bn_x = float("nan")
            cn_m = cn_x = float("nan")
            en_m = en_x = float("nan")
            nvg_bt_aprox = False
            nvg_cl_muestreo = False
        else:
            aprox_bt = mn > UMBRAL_ARISTAS_BETWEENNESS_APROX
            if aprox_bt:
                logger.info(
                    "  NVG: approximate betweenness (m=%s > %s).",
                    mn,
                    UMBRAL_ARISTAS_BETWEENNESS_APROX,
                )
            bn_m, bn_x, nvg_bt_aprox = _betweenness(Gn, aproximar=aprox_bt)
            nvg_cl_muestreo = mn > UMBRAL_ARISTAS_CERCANIA_EXACTA
            if nvg_cl_muestreo:
                logger.info(
                    "  NVG: closeness by sampling (m=%s > %s).",
                    mn,
                    UMBRAL_ARISTAS_CERCANIA_EXACTA,
                )
            cn_m, cn_x = _closeness(Gn, usar_muestreo=nvg_cl_muestreo)
            en_m, en_x = _eigenvector(Gn)

        Gr = _grafo_recurrencia(x)
        mr = Gr.number_of_edges()
        aprox_rec = mr > UMBRAL_ARISTAS_RECURRENCIA_APROX
        if aprox_rec:
            logger.info(
                "  Recurrence: approximate betweenness/closeness (m=%s > %s).",
                mr,
                UMBRAL_ARISTAS_RECURRENCIA_APROX,
            )
        br_m, br_x, rec_bt_aprox = _betweenness(Gr, aproximar=aprox_rec)
        cr_m, cr_x = _closeness(Gr, usar_muestreo=aprox_rec)
        er_m, er_x = _eigenvector(Gr)

        filas.append(
            {
                "componente": nombre,
                "hvg_betweenness_media": bh_m,
                "hvg_betweenness_max": bh_x,
                "hvg_closeness_media": ch_m,
                "hvg_closeness_max": ch_x,
                "hvg_eigenvector_media": eh_m,
                "hvg_eigenvector_max": eh_x,
                "nvg_m": mn,
                "nvg_centralidad_omitido": mn > MAX_ARISTAS_NVG_CENTRALIDAD,
                "nvg_betweenness_aprox": nvg_bt_aprox,
                "nvg_closeness_muestreado": nvg_cl_muestreo,
                "nvg_betweenness_media": bn_m,
                "nvg_betweenness_max": bn_x,
                "nvg_closeness_media": cn_m,
                "nvg_closeness_max": cn_x,
                "nvg_eigenvector_media": en_m,
                "nvg_eigenvector_max": en_x,
                "rec_betweenness_aprox": rec_bt_aprox,
                "rec_closeness_muestreado": aprox_rec,
                "rec_betweenness_media": br_m,
                "rec_betweenness_max": br_x,
                "rec_closeness_media": cr_m,
                "rec_closeness_max": cr_x,
                "rec_eigenvector_media": er_m,
                "rec_eigenvector_max": er_x,
            }
        )

    det = pd.DataFrame(filas)
    out_dir = _REPO_ROOT / "docs" / "20abr26" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    det_path = out_dir / "centralidad_por_componente_ceemdan.csv"
    det.to_csv(det_path, index=False)
    logger.info("Detail: %s", det_path)

    metricas = [
        ("betweenness_media", "Betweenness (mean range)"),
        ("closeness_media", "Closeness (mean range)"),
        ("eigenvector_media", "Eigenvector (mean range)"),
    ]
    filas_res: list[dict[str, str]] = []
    for suf, etiqueta in metricas:
        rh = _rango_columna(det, f"hvg_{suf}")
        rr = _rango_columna(det, f"rec_{suf}")
        rn = _rango_columna(det, f"nvg_{suf}")
        filas_res.append(
            {
                "metric": etiqueta,
                "HVG": _formatear_rango(rh, cuatro_decimales=True),
                "Recurrence": _formatear_rango(rr, cuatro_decimales=True),
                "NVG": _formatear_rango(rn, cuatro_decimales=True),
            }
        )
    res_df = pd.DataFrame(filas_res)
    res_path = out_dir / "resumen_centralidad_grafos_ceemdan.csv"
    res_df.to_csv(res_path, index=False)
    logger.info("Summary: %s", res_path)

    nota = (
        "\n\nResidual NVG: centrality skipped if "
        f"m > {MAX_ARISTAS_NVG_CENTRALIDAD}. "
        "NVG betweenness: sampling if "
        f"m > {UMBRAL_ARISTAS_BETWEENNESS_APROX} (k={K_MUESTRA_BETWEENNESS}, "
        f"seed={SEMILLA_BETWEENNESS}). "
        "NVG closeness: node sampling if "
        f"m > {UMBRAL_ARISTAS_CERCANIA_EXACTA} (k={K_MUESTRA_CERCANIA}, "
        f"seed={SEMILLA_CERCANIA}). "
        "Recurrence: approximate betweenness and closeness if "
        f"m > {UMBRAL_ARISTAS_RECURRENCIA_APROX}.\n"
    )
    md_path = out_dir / "resumen_centralidad_grafos_ceemdan.md"
    md_path.write_text(
        "# Centrality summary (CEEMDAN)\n\n"
        + res_df.to_string(index=False)
        + nota
        + "\nDetalle: `centralidad_por_componente_ceemdan.csv`.\n",
        encoding="utf-8",
    )
    logger.info("Markdown: %s", md_path)


if __name__ == "__main__":
    main()
