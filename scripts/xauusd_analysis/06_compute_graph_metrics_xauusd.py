"""
Script to obtain graph metrics from graph transformations applied to CEEMDAN IMF components of XAU/USD.

1. Load data required to obtain graph metrics from graph transformations applied to IMF components of XAU/USD.
2. Obtain density, edge count, connected components, diameter, and mean degree metrics of graphs from graph transformations applied to IMF components of XAU/USD.
3. Compute clustering coefficients (mean, median, and standard deviation) of graphs from graph transformations applied to IMF components of XAU/USD.
4. Compute centralities (degree, betweenness, closeness, and eigenvector) of graphs from graph transformations applied to IMF components of XAU/USD.
5. Document implemented logic and results in this script.
6. Do not change any existing code for now

Implemented logic
---------------------
- **Load**: graphs saved by ``05_graph_transformations_xauusd.py`` in
  ``data/GraphEMD/xauusd_analysis/grafos/{hvg,nvg,recurrencia}/{imf_k,residuo}/*.pt``
  (PyTorch Geometric → ``networkx.Graph``).
- **Structural** (per graph): density, edge count, connected components,
  diameter (double BFS on the largest connected component if the graph is connected; NaN otherwise),
  mean degree ``2m/n``.
- **Clustering**: mean (``average_clustering``), median and standard deviation of
  local coefficients; omitted for NVG with ``m > 4×10⁵`` (graphs too dense).
- **Centralities**: mean normalized degree, betweenness, closeness, and eigenvector;
  same thresholds as ``run_centrality_graphs_ceemdan_20abr26.py`` (sampling /
  omission on very dense NVG).

Results obtained (run 2026-05-17, 27 graphs = 9×3)
-------------------------------------------------------------

**HVG:** all connected (1 component); diameter grows in slow modes (IMF_8 ≈ 1306,
Residuo ≈ 2036); mean clustering ≈ 0.33–0.66.

**NVG IMF_1–IMF_6:** full metrics; density up to ≈ 0.035 (IMF_6). **NVG IMF_7+**:
clustering omitted (``m > 4×10⁵``); **IMF_8** and **Residual**: centrality omitted
(``m > 1.5×10⁶`` and ``> 2×10⁶`` respectively); structural metrics still computed
(Residual: density ≈ 0.65, mean degree ≈ 2316).

**Recurrence:** many connected components (except IMF_6 connected, diameter ≈ 332);
approximate betweenness/closeness if ``m > 1.2×10⁴``.

**Outputs:** ``xauusd_metricas_grafos_imf.csv``, ``xauusd_resumen_metricas_grafos.json``.
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=UserWarning)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DIR_DATOS = _REPO_ROOT / "data" / "GraphEMD" / "xauusd_analysis"
_DIR_GRAFOS = _DIR_DATOS / "grafos"
_RUTA_METRICAS_CSV = _DIR_DATOS / "xauusd_metricas_grafos_imf.csv"
_RUTA_RESUMEN_JSON = _DIR_DATOS / "xauusd_resumen_metricas_grafos.json"

TIPOS_GRAFO: tuple[str, ...] = ("hvg", "nvg", "recurrencia")

# Thresholds aligned with MSCI scripts (exploration/); NVG clustering limited
# due to cost on dense graphs (IMF_7+ in XAU/USD exceed ~4×10⁵ edges).
MAX_ARISTAS_NVG_CLUSTERING = 400_000
MAX_ARISTAS_NVG_CENTRALIDAD = 1_500_000
UMBRAL_ARISTAS_BETWEENNESS_APROX = 200_000
K_MUESTRA_BETWEENNESS = 600
SEMILLA_BETWEENNESS = 42
UMBRAL_ARISTAS_CERCANIA_EXACTA = 50_000
K_MUESTRA_CERCANIA = 500
SEMILLA_CERCANIA = 42
UMBRAL_ARISTAS_RECURRENCIA_APROX = 12_000
N_MAX_EIGENVECTOR_NUMPY = 4096

logger = logging.getLogger(__name__)


def id_carpeta_a_componente(id_carpeta: str) -> str:
    """
    Convert folder name (``imf_1``, ``residuo``) to component id.

    Parameters
    ----------
    id_carpeta : str
        Subdirectory name under ``grafos/<tipo>/``.

    Returns
    -------
    str
        ``IMF_1``, …, ``IMF_8``, or ``Residuo``.
    """
    if id_carpeta == "residuo":
        return "Residuo"
    if id_carpeta.startswith("imf_"):
        sufijo = id_carpeta.split("_", 1)[1]
        return f"IMF_{sufijo}"
    return id_carpeta


def listar_componentes_grafos(dir_grafos: Path = _DIR_GRAFOS) -> list[str]:
    """
    List available components from HVG folders.

    Parameters
    ----------
    dir_grafos : Path
        Base ``grafos/`` directory.

    Returns
    -------
    list[str]
        Sorted component names (``IMF_1``, …, ``Residuo``).
    """
    carpeta_hvg = dir_grafos / "hvg"
    if not carpeta_hvg.is_dir():
        raise FileNotFoundError(
            f"Not found: {carpeta_hvg}. Run 05_graph_transformations_xauusd.py first."
        )
    ids = sorted(p.name for p in carpeta_hvg.iterdir() if p.is_dir())
    return [id_carpeta_a_componente(n) for n in ids]


def ruta_archivo_grafo(
    dir_grafos: Path,
    tipo_grafo: str,
    componente: str,
) -> Path:
    """
    Return the path to the graph ``.pt`` file.

    Parameters
    ----------
    dir_grafos : Path
        Base graph directory.
    tipo_grafo : str
        ``hvg``, ``nvg``, or ``recurrencia``.
    componente : str
        ``IMF_k`` or ``Residuo``.

    Returns
    -------
    Path
        Path to the serialized Data object.
    """
    if componente == "Residuo":
        id_carpeta = "residuo"
    else:
        id_carpeta = componente.lower()
    return dir_grafos / tipo_grafo / id_carpeta / f"grafo_{tipo_grafo}_{id_carpeta}.pt"


def cargar_grafo_desde_pt(ruta_pt: Path) -> nx.Graph:
    """
    Load a PyG graph from ``.pt`` and convert it to an undirected NetworkX graph.

    Parameters
    ----------
    ruta_pt : Path
        ``grafo_<tipo>_<id>.pt`` file.

    Returns
    -------
    nx.Graph
        Simple graph with nodes ``0 … n-1``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not ruta_pt.is_file():
        raise FileNotFoundError(f"Graph not found: {ruta_pt}")
    grafo_data = torch.load(str(ruta_pt), map_location="cpu", weights_only=False)
    grafo = nx.Graph()
    grafo.add_nodes_from(range(int(grafo_data.num_nodes)))
    edge_index = grafo_data.edge_index.cpu().numpy()
    aristas = [
        (int(edge_index[0, i]), int(edge_index[1, i]))
        for i in range(edge_index.shape[1])
    ]
    grafo.add_edges_from(aristas)
    return grafo


def diametro_aproximado(grafo: nx.Graph) -> float:
    """
    Estimate diameter with double BFS on the largest connected component.

    Parameters
    ----------
    grafo : nx.Graph
        Undirected graph.

    Returns
    -------
    float
        Estimated diameter, 0 if no edges, NaN if multiple components.
    """
    num_nodos = grafo.number_of_nodes()
    if num_nodos == 0:
        return float("nan")
    if grafo.number_of_edges() == 0:
        return 0.0
    if nx.number_connected_components(grafo) != 1:
        return float("nan")
    nodo_inicial = next(iter(grafo.nodes))
    distancias_1 = nx.single_source_shortest_path_length(grafo, nodo_inicial)
    nodo_lejano = max(distancias_1, key=distancias_1.get)
    distancias_2 = nx.single_source_shortest_path_length(grafo, nodo_lejano)
    return float(max(distancias_2.values()))


def metricas_estructurales(grafo: nx.Graph) -> dict[str, Any]:
    """
    Compute density, edges, components, diameter, and mean degree.

    Parameters
    ----------
    grafo : nx.Graph
        Graph to analyze.

    Returns
    -------
    dict
        Structural graph metrics.
    """
    num_nodos = grafo.number_of_nodes()
    num_enlaces = grafo.number_of_edges()
    densidad = float(nx.density(grafo))
    num_componentes = int(nx.number_connected_components(grafo))
    diametro = diametro_aproximado(grafo)
    grado_medio = (2.0 * num_enlaces / num_nodos) if num_nodos else 0.0
    return {
        "num_nodos": num_nodos,
        "num_enlaces": num_enlaces,
        "densidad": densidad,
        "num_componentes_conexas": num_componentes,
        "diametro": diametro,
        "grado_medio": grado_medio,
    }


def metricas_clustering(
    grafo: nx.Graph,
    tipo_grafo: str,
) -> dict[str, Any]:
    """
    Compute statistics of the local clustering coefficient.

    Parameters
    ----------
    grafo : nx.Graph
        Undirected graph.
    tipo_grafo : str
        Graph type (to apply NVG threshold).

    Returns
    -------
    dict
        Mean, median, standard deviation, or omitted if the graph is too dense.
    """
    num_enlaces = grafo.number_of_edges()
    if tipo_grafo == "nvg" and num_enlaces > MAX_ARISTAS_NVG_CLUSTERING:
        return {
            "clustering_media": float("nan"),
            "clustering_mediana": float("nan"),
            "clustering_desviacion_tipica": float("nan"),
            "clustering_omitido": True,
        }
    if grafo.number_of_nodes() == 0:
        return {
            "clustering_media": float("nan"),
            "clustering_mediana": float("nan"),
            "clustering_desviacion_tipica": float("nan"),
            "clustering_omitido": False,
        }
    coeficientes = nx.clustering(grafo)
    valores = np.fromiter(coeficientes.values(), dtype=np.float64, count=len(coeficientes))
    return {
        "clustering_media": float(nx.average_clustering(grafo)),
        "clustering_mediana": float(np.median(valores)),
        "clustering_desviacion_tipica": float(np.std(valores, ddof=0)),
        "clustering_omitido": False,
    }


def _media_dict_escalar(mapa: dict[Any, float]) -> float:
    """
    Return the mean of a node → scalar dictionary.

    Parameters
    ----------
    mapa : dict
        Centrality values per node.

    Returns
    -------
    float
        Mean or NaN if the map is empty.
    """
    if not mapa:
        return float("nan")
    valores = np.fromiter(mapa.values(), dtype=np.float64, count=len(mapa))
    return float(np.mean(valores))


def _betweenness_media(grafo: nx.Graph, *, aproximar: bool) -> tuple[float, bool]:
    """
    Compute mean normalized betweenness centrality.

    Parameters
    ----------
    grafo : nx.Graph
        Undirected graph.
    aproximar : bool
        If True, use node sampling.

    Returns
    -------
    tuple[float, bool]
        Mean and approximation flag.
    """
    num_nodos = grafo.number_of_nodes()
    if num_nodos == 0:
        return float("nan"), False
    if aproximar:
        k = min(K_MUESTRA_BETWEENNESS, num_nodos)
        mapa = nx.betweenness_centrality(
            grafo, k=k, normalized=True, seed=SEMILLA_BETWEENNESS
        )
        return _media_dict_escalar(mapa), True
    mapa = nx.betweenness_centrality(grafo, normalized=True)
    return _media_dict_escalar(mapa), False


def _closeness_media(grafo: nx.Graph, *, usar_muestreo: bool) -> tuple[float, bool]:
    """
    Compute mean closeness centrality.

    Parameters
    ----------
    grafo : nx.Graph
        Undirected graph.
    usar_muestreo : bool
        If True, average over a sample of nodes.

    Returns
    -------
    tuple[float, bool]
        Mean and sampling flag.
    """
    if grafo.number_of_nodes() == 0:
        return float("nan"), False
    if not usar_muestreo:
        return _media_dict_escalar(nx.closeness_centrality(grafo)), False
    rng = np.random.default_rng(SEMILLA_CERCANIA)
    nodos = list(grafo.nodes())
    k = min(K_MUESTRA_CERCANIA, len(nodos))
    muestra = rng.choice(nodos, size=k, replace=False)
    valores: list[float] = []
    for u in muestra:
        cercania = nx.closeness_centrality(grafo, u=int(u))
        valores.append(
            float(cercania) if isinstance(cercania, (int, float)) else float(cercania[u])
        )
    return float(np.mean(valores)), True


def _eigenvector_media(grafo: nx.Graph) -> float:
    """
    Compute mean eigenvector centrality on the largest component.

    Parameters
    ----------
    grafo : nx.Graph
        Undirected graph.

    Returns
    -------
    float
        Mean eigenvector centrality or NaN if it does not converge.
    """
    if grafo.number_of_nodes() == 0:
        return float("nan")
    subgrafo: nx.Graph = grafo
    if not nx.is_connected(grafo):
        nodos = max(nx.connected_components(grafo), key=len)
        subgrafo = grafo.subgraph(nodos).copy()
    if subgrafo.number_of_nodes() <= N_MAX_EIGENVECTOR_NUMPY:
        try:
            mapa = nx.eigenvector_centrality_numpy(subgrafo)
            return _media_dict_escalar(mapa)
        except (nx.NetworkXError, np.linalg.LinAlgError):
            pass
    try:
        mapa = nx.eigenvector_centrality(subgrafo, max_iter=5000, tol=1e-06)
        return _media_dict_escalar(mapa)
    except nx.NetworkXError:
        return float("nan")


def metricas_centralidad(
    grafo: nx.Graph,
    tipo_grafo: str,
) -> dict[str, Any]:
    """
    Compute mean degree, betweenness, closeness, and eigenvector centralities.

    Parameters
    ----------
    grafo : nx.Graph
        Undirected graph.
    tipo_grafo : str
        ``hvg``, ``nvg``, or ``recurrencia`` (sets approximation thresholds).

    Returns
    -------
    dict
        Mean centralities and approximation or omission flags.
    """
    num_enlaces = grafo.number_of_edges()
    if tipo_grafo == "nvg" and num_enlaces > MAX_ARISTAS_NVG_CENTRALIDAD:
        return {
            "degree_centrality_media": float("nan"),
            "betweenness_centrality_media": float("nan"),
            "closeness_centrality_media": float("nan"),
            "eigenvector_centrality_media": float("nan"),
            "centralidad_omitida": True,
            "betweenness_aproximada": False,
            "closeness_muestreada": False,
        }

    grado_mean = _media_dict_escalar(nx.degree_centrality(grafo))

    if tipo_grafo == "recurrencia":
        aproximar_bt = num_enlaces > UMBRAL_ARISTAS_RECURRENCIA_APROX
        muestrear_cl = aproximar_bt
    elif tipo_grafo == "nvg":
        aproximar_bt = num_enlaces > UMBRAL_ARISTAS_BETWEENNESS_APROX
        muestrear_cl = num_enlaces > UMBRAL_ARISTAS_CERCANIA_EXACTA
    else:
        aproximar_bt = False
        muestrear_cl = False

    betweenness_media, bt_aprox = _betweenness_media(grafo, aproximar=aproximar_bt)
    closeness_media, cl_muest = _closeness_media(grafo, usar_muestreo=muestrear_cl)
    eigenvector_mean = _eigenvector_media(grafo)

    return {
        "degree_centrality_media": grado_media,
        "betweenness_centrality_media": betweenness_media,
        "closeness_centrality_media": closeness_media,
        "eigenvector_centrality_media": eigenvector_media,
        "centralidad_omitida": False,
        "betweenness_aproximada": bt_aprox,
        "closeness_muestreada": cl_muest,
    }


def calcular_metricas_grafo(
    grafo: nx.Graph,
    componente: str,
    tipo_grafo: str,
) -> dict[str, Any]:
    """
    Aggregate structural, clustering, and centrality metrics of a graph.

    Parameters
    ----------
    grafo : nx.Graph
        Graph loaded from disk.
    componente : str
        IMF or residual component name.
    tipo_grafo : str
        ``hvg``, ``nvg``, or ``recurrencia``.

    Returns
    -------
    dict
        Row ready for a DataFrame.
    """
    fila: dict[str, Any] = {
        "componente": componente,
        "tipo_grafo": tipo_grafo,
    }
    fila.update(metricas_estructurales(grafo))
    fila.update(metricas_clustering(grafo, tipo_grafo))
    fila.update(metricas_centralidad(grafo, tipo_grafo))
    return fila


def procesar_todos_los_grafos(
    dir_grafos: Path = _DIR_GRAFOS,
) -> pd.DataFrame:
    """
    Compute metrics for each component and available graph type.

    Parameters
    ----------
    dir_grafos : Path
        Base directory with ``hvg``, ``nvg``, ``recurrencia`` subfolders.

    Returns
    -------
    pd.DataFrame
        Table with one row per (component, tipo_grafo).
    """
    componentes = listar_componentes_grafos(dir_grafos)
    filas: list[dict[str, Any]] = []

    for componente in componentes:
        for tipo in TIPOS_GRAFO:
            ruta_pt = ruta_archivo_grafo(dir_grafos, tipo, componente)
            logger.info("Processing %s — %s ...", componente, tipo.upper())
            grafo = cargar_grafo_desde_pt(ruta_pt)
            filas.append(calcular_metricas_grafo(grafo, componente, tipo))

    return pd.DataFrame(filas)


def guardar_salidas(
    df_metricas: pd.DataFrame,
    ruta_csv: Path = _RUTA_METRICAS_CSV,
    ruta_json: Path = _RUTA_RESUMEN_JSON,
) -> None:
    """
    Persist detailed CSV and summary JSON.

    Parameters
    ----------
    df_metricas : pd.DataFrame
        Metrics per component and type.
    ruta_csv : Path
        CSV path.
    ruta_json : Path
        Summary JSON path.
    """
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    df_metricas.to_csv(ruta_csv, index=False)
    logger.info("Metrics saved: %s", ruta_csv)

    omitidos_clustering = df_metricas.loc[
        df_metricas["clustering_omitido"] == True,  # noqa: E712
        ["componente", "tipo_grafo"],
    ].to_dict(orient="records")
    omitidos_centralidad = df_metricas.loc[
        df_metricas["centralidad_omitida"] == True,  # noqa: E712
        ["componente", "tipo_grafo"],
    ].to_dict(orient="records")

    payload = {
        "num_filas": int(len(df_metricas)),
        "componentes": sorted(df_metricas["componente"].unique().tolist()),
        "tipos_grafo": sorted(df_metricas["tipo_grafo"].unique().tolist()),
        "clustering_omitido": omitidos_clustering,
        "centralidad_omitida": omitidos_centralidad,
        "metricas": df_metricas.to_dict(orient="records"),
    }
    with open(ruta_json, "w", encoding="utf-8") as archivo:
        json.dump(payload, archivo, indent=2, ensure_ascii=False)
    logger.info("JSON summary: %s", ruta_json)


def main(dir_grafos: Optional[Path] = None) -> dict[str, Any]:
    """
    Entry point: compute and save metrics for all XAU/USD.

    Parameters
    ----------
    dir_grafos : Path, optional
        Base graph directory. Default ``data/.../grafos``.

    Returns
    -------
    dict
        Output paths and metrics DataFrame.
    """
    carpeta = dir_grafos or _DIR_GRAFOS
    df_metricas = procesar_todos_los_grafos(carpeta)
    guardar_salidas(df_metricas)
    return {
        "ruta_metricas_csv": str(_RUTA_METRICAS_CSV),
        "ruta_resumen_json": str(_RUTA_RESUMEN_JSON),
        "metricas": df_metricas,
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Structural, clustering, and centrality metrics for XAU/USD."
    )
    parser.add_argument(
        "--dir-grafos",
        type=Path,
        default=None,
        help="Base directory with hvg/nvg/recurrencia (default data/.../grafos).",
    )
    args = parser.parse_args()
    try:
        salida = main(dir_grafos=args.dir_grafos)
        logger.info("Done: %s", salida["ruta_metricas_csv"])
    except Exception:
        logger.exception("Error computing graph metrics for XAU/USD")
        sys.exit(1)
