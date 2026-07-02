"""
Script to compute descriptive metrics for all graphs generated for the IMFs.

This script processes all generated graphs (NVG, HVG, recurrence) for all IMFs
of MSCI World, computes descriptive metrics and generates visualizations, saving both
the plots and a CSV with the final data in each data folder.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import torch
import networkx as nx

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def cargar_grafo_desde_archivo(archivo_grafo: Path) -> Optional[nx.Graph]:
    """
    Load a graph from a .pt file and convert it to NetworkX.

    Parameters
    ----------
    archivo_grafo : Path
        Path to the .pt file containing the graph.

    Returns
    -------
    Optional[nx.Graph]
        Graph in NetworkX format, or None if an error occurs.
    """
    try:
        grafo_data = torch.load(
            str(archivo_grafo), map_location="cpu", weights_only=False
        )

        G = nx.Graph()
        G.add_nodes_from(range(grafo_data.num_nodes))

        edge_index = grafo_data.edge_index.cpu().numpy()
        edges = [
            (int(edge_index[0, i]), int(edge_index[1, i]))
            for i in range(edge_index.shape[1])
        ]
        G.add_edges_from(edges)

        return G

    except Exception as e:
        logger.error(f"Error loading graph from {archivo_grafo}: {e}")
        return None


def calcular_metricas_basicas(grafo: nx.Graph, umbral_nodos: Optional[int] = None) -> Dict:
    """
    Compute basic graph metrics.

    For large graphs (more than umbral_nodos nodes), skips computing
    diameter, radius, and average eccentricity because they are
    computationally very expensive (O(n²) or worse).

    Parameters
    ----------
    grafo : nx.Graph
        NetworkX graph.
    umbral_nodos : Optional[int], optional
        Node threshold for skipping expensive computations. If None, always
        computes the expensive metrics. Default is None.

    Returns
    -------
    Dict
        Dictionary with the computed basic metrics.
    """
    num_nodos = grafo.number_of_nodes()
    num_enlaces = grafo.number_of_edges()
    densidad = nx.density(grafo)
    num_componentes = nx.number_connected_components(grafo)

    # Initialize default values
    diametro = None
    radio = None
    excentricidad_promedio = None

    # Only compute expensive metrics if the graph is relatively small
    # or if there is no threshold (umbral_nodos is None)
    if umbral_nodos is None or num_nodos <= umbral_nodos:
        if num_componentes > 1:
            componentes = list(nx.connected_components(grafo))
            componente_principal = max(componentes, key=len)
            grafo_principal = grafo.subgraph(componente_principal).copy()
        else:
            grafo_principal = grafo

        if nx.is_connected(grafo_principal):
            try:
                logger.info(
                    f"    Computing diameter, radius, and eccentricity "
                    f"(graph with {num_nodos} nodes)..."
                )
                diametro = nx.diameter(grafo_principal)
                radio = nx.radius(grafo_principal)
                excentricidad_promedio = np.mean(
                    list(nx.eccentricity(grafo_principal).values())
                )
            except Exception as e:
                logger.warning(
                    f"    Could not compute diameter/radius/eccentricity: {e}"
                )
                diametro = None
                radio = None
                excentricidad_promedio = None
    else:
        logger.info(
            f"    Skipping diameter/radius/eccentricity computation "
            f"(very large graph: {num_nodos} nodes, threshold: {umbral_nodos:,})"
        )

    return {
        "num_nodos": num_nodos,
        "num_enlaces": num_enlaces,
        "densidad": densidad,
        "num_componentes": num_componentes,
        "diametro": diametro,
        "radio": radio,
        "excentricidad_promedio": excentricidad_promedio,
    }


def calcular_metricas_grados(grafo: nx.Graph) -> Tuple[Dict, np.ndarray]:
    """
    Compute metrics related to node degrees.

    Parameters
    ----------
    grafo : nx.Graph
        NetworkX graph.

    Returns
    -------
    Tuple[Dict, np.ndarray]
        Tuple with (metrics dictionary, array of degree values).
    """
    # Get degrees of all nodes
    grados_dict = dict(grafo.degree)
    valores_grados = np.array(list(grados_dict.values()), dtype=int)

    metricas = {
        "grado_min": int(min(valores_grados)),
        "grado_max": int(max(valores_grados)),
        "grado_promedio": float(np.mean(valores_grados)),
        "grado_mediana": float(np.median(valores_grados)),
        "grado_std": float(np.std(valores_grados)),
    }

    return metricas, valores_grados


def calcular_metricas_clustering(grafo: nx.Graph) -> Tuple[Dict, List[float]]:
    """
    Compute metrics related to the clustering coefficient.

    Parameters
    ----------
    grafo : nx.Graph
        NetworkX graph.

    Returns
    -------
    Tuple[Dict, List[float]]
        Tuple with (metrics dictionary, list of clustering values).
    """
    clustering_nodos = nx.clustering(grafo)
    valores_clustering = list(clustering_nodos.values())

    metricas = {
        "clustering_promedio": float(nx.average_clustering(grafo)),
        "clustering_min": float(min(valores_clustering)),
        "clustering_max": float(max(valores_clustering)),
        "clustering_mediana": float(np.median(valores_clustering)),
        "clustering_std": float(np.std(valores_clustering)),
    }

    return metricas, valores_clustering


def calcular_centralidades(grafo: nx.Graph) -> Tuple[Dict, Dict]:
    """
    Compute different types of graph centrality.

    Parameters
    ----------
    grafo : nx.Graph
        NetworkX graph.

    Returns
    -------
    Tuple[Dict, Dict]
        Tuple with (metrics dictionary, dictionary with centrality values).
    """
    logger.info("  Computing degree centrality...")
    degree_centrality = nx.degree_centrality(grafo)
    valores_degree = list(degree_centrality.values())

    logger.info("  Computing betweenness centrality (this may take a while)...")
    betweenness_centrality = nx.betweenness_centrality(grafo)
    valores_betweenness = list(betweenness_centrality.values())

    logger.info("  Computing closeness centrality...")
    closeness_centrality = nx.closeness_centrality(grafo)
    valores_closeness = list(closeness_centrality.values())

    logger.info("  Computing eigenvector centrality...")
    try:
        eigenvector_centrality = nx.eigenvector_centrality(grafo, max_iter=1000)
        valores_eigenvector = list(eigenvector_centrality.values())
    except Exception:
        logger.warning("    Could not compute eigenvector centrality")
        valores_eigenvector = None

    metricas = {
        "degree_promedio": float(np.mean(valores_degree)),
        "degree_mediana": float(np.median(valores_degree)),
        "degree_max": float(np.max(valores_degree)),
        "degree_min": float(np.min(valores_degree)),
        "betweenness_promedio": float(np.mean(valores_betweenness)),
        "betweenness_mediana": float(np.median(valores_betweenness)),
        "betweenness_max": float(np.max(valores_betweenness)),
        "betweenness_min": float(np.min(valores_betweenness)),
        "closeness_promedio": float(np.mean(valores_closeness)),
        "closeness_mediana": float(np.median(valores_closeness)),
        "closeness_max": float(np.max(valores_closeness)),
        "closeness_min": float(np.min(valores_closeness)),
    }

    if valores_eigenvector is not None:
        metricas["eigenvector_promedio"] = float(np.mean(valores_eigenvector))
        metricas["eigenvector_mediana"] = float(np.median(valores_eigenvector))
        metricas["eigenvector_max"] = float(np.max(valores_eigenvector))
        metricas["eigenvector_min"] = float(np.min(valores_eigenvector))
    else:
        metricas["eigenvector_promedio"] = np.nan
        metricas["eigenvector_mediana"] = np.nan
        metricas["eigenvector_max"] = np.nan
        metricas["eigenvector_min"] = np.nan

    valores_centralidad = {
        "degree": valores_degree,
        "betweenness": valores_betweenness,
        "closeness": valores_closeness,
        "eigenvector": valores_eigenvector,
    }

    return metricas, valores_centralidad


def crear_grafica_grados(
    valores_grados: np.ndarray, metricas: Dict, titulo: str
) -> go.Figure:
    """
    Create a degree distribution plot.

    Parameters
    ----------
    valores_grados : np.ndarray
        Array with degree values.
    metricas : Dict
        Dictionary with degree metrics.
    titulo : str
        Plot title.

    Returns
    -------
    go.Figure
        Plotly figure with the degree distribution.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=valores_grados,
            nbinsx=50,
            name="Degree distribution",
            marker=dict(color="steelblue", line=dict(color="black", width=1)),
            opacity=0.7,
        )
    )

    fig.update_layout(
        title=dict(text=titulo, x=0.5, xanchor="center", font=dict(size=18)),
        xaxis_title="Node degree",
        yaxis_title="Frequency",
        height=500,
        showlegend=False,
        plot_bgcolor="white",
        annotations=[
            dict(
                text=(
                    f"Mean: {metricas['grado_promedio']:.2f} | "
                    f"Median: {metricas['grado_mediana']:.2f} | "
                    f"Std: {metricas['grado_std']:.2f}"
                ),
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.95,
                xanchor="center",
                yanchor="top",
                font=dict(size=12),
            )
        ],
    )

    return fig


def crear_grafica_clustering(
    valores_clustering: List[float], metricas: Dict, titulo: str
) -> go.Figure:
    """
    Create a clustering coefficient distribution plot.

    Parameters
    ----------
    valores_clustering : List[float]
        List with clustering values.
    metricas : Dict
        Dictionary with clustering metrics.
    titulo : str
        Plot title.

    Returns
    -------
    go.Figure
        Plotly figure with the clustering distribution.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=valores_clustering,
            nbinsx=50,
            name="Clustering distribution",
            marker=dict(color="coral", line=dict(color="black", width=1)),
            opacity=0.7,
        )
    )

    fig.update_layout(
        title=dict(text=titulo, x=0.5, xanchor="center", font=dict(size=18)),
        xaxis_title="Clustering coefficient",
        yaxis_title="Frequency",
        height=500,
        showlegend=False,
        plot_bgcolor="white",
        annotations=[
            dict(
                text=(
                    f"Mean: {metricas['clustering_promedio']:.6f} | "
                    f"Median: {metricas['clustering_mediana']:.6f}"
                ),
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.95,
                xanchor="center",
                yanchor="top",
                font=dict(size=12),
            )
        ],
    )

    return fig


def crear_grafica_centralidades(
    valores_centralidad: Dict, titulo: str
) -> go.Figure:
    """
    Create a plot with centrality distributions.

    Parameters
    ----------
    valores_centralidad : Dict
        Dictionary with values for each centrality type.
    titulo : str
        Plot title.

    Returns
    -------
    go.Figure
        Plotly figure with centrality distributions.
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Degree Centrality",
            "Betweenness Centrality",
            "Closeness Centrality",
            "Eigenvector Centrality",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    fig.add_trace(
        go.Histogram(
            x=valores_centralidad["degree"],
            nbinsx=50,
            name="Degree",
            marker=dict(color="steelblue"),
            opacity=0.7,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=valores_centralidad["betweenness"],
            nbinsx=50,
            name="Betweenness",
            marker=dict(color="coral"),
            opacity=0.7,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Histogram(
            x=valores_centralidad["closeness"],
            nbinsx=50,
            name="Closeness",
            marker=dict(color="lightgreen"),
            opacity=0.7,
        ),
        row=2,
        col=1,
    )

    if valores_centralidad["eigenvector"] is not None:
        fig.add_trace(
            go.Histogram(
                x=valores_centralidad["eigenvector"],
                nbinsx=50,
                name="Eigenvector",
                marker=dict(color="plum"),
                opacity=0.7,
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        title=dict(text=titulo, x=0.5, xanchor="center", font=dict(size=18)),
        height=800,
        showlegend=False,
        plot_bgcolor="white",
    )

    fig.update_xaxes(title_text="Value", row=2, col=1)
    fig.update_xaxes(title_text="Value", row=2, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)

    return fig


def procesar_grafo(
    archivo_grafo: Path,
    tipo_grafo: str,
    id_imf: str,
    carpeta_salida: Path,
) -> bool:
    """
    Process a graph, compute metrics, and generate visualizations.

    Parameters
    ----------
    archivo_grafo : Path
        Path to the graph .pt file.
    tipo_grafo : str
        Graph type (nvg, hvg, recurrencia).
    id_imf : str
        IMF identifier.
    carpeta_salida : Path
        Folder where results will be saved.

    Returns
    -------
    bool
        True if processing succeeded, False otherwise.
    """
    logger.info(f"Processing graph: {archivo_grafo.name}")

    # Load graph
    grafo = cargar_grafo_desde_archivo(archivo_grafo)
    if grafo is None:
        return False


    # Compute basic metrics
    logger.info("  Computing basic metrics...")
    metricas_basicas = calcular_metricas_basicas(grafo)

    # Compute degree metrics
    logger.info("  Computing degree metrics...")
    metricas_grados, valores_grados = calcular_metricas_grados(grafo)

    # Compute clustering metrics
    logger.info("  Computing clustering metrics...")
    metricas_clustering, valores_clustering = calcular_metricas_clustering(grafo)

    # Compute centralities
    logger.info("  Computing centralities...")
    metricas_centralidad, valores_centralidad = calcular_centralidades(grafo)

    # Combine all metrics
    todas_metricas = {
        **metricas_basicas,
        **metricas_grados,
        **metricas_clustering,
        **metricas_centralidad,
    }

    # Create DataFrame with metrics
    df_metricas = pd.DataFrame([todas_metricas])

    # Save CSV
    archivo_csv = carpeta_salida / "metricas_grafo.csv"
    df_metricas.to_csv(archivo_csv, index=False)
    logger.info(f"  Metrics saved to: {archivo_csv}")

    # Create and save plots
    logger.info("  Generating plots...")

    # Degree plot
    titulo_grados = f"Degree Distribution - {tipo_grafo.upper()} {id_imf}"
    fig_grados = crear_grafica_grados(valores_grados, metricas_grados, titulo_grados)
    archivo_grados = carpeta_salida / "distribucion_grados.html"
    fig_grados.write_html(str(archivo_grados))
    logger.info(f"  Degree plot saved to: {archivo_grados}")

    # Clustering plot
    titulo_clustering = (
        f"Clustering Distribution - {tipo_grafo.upper()} {id_imf}"
    )
    fig_clustering = crear_grafica_clustering(
        valores_clustering, metricas_clustering, titulo_clustering
    )
    archivo_clustering = carpeta_salida / "distribucion_clustering.html"
    fig_clustering.write_html(str(archivo_clustering))
    logger.info(f"  Clustering plot saved to: {archivo_clustering}")

    # Centrality plot
    titulo_centralidades = (
        f"Centrality Distributions - {tipo_grafo.upper()} {id_imf}"
    )
    fig_centralidades = crear_grafica_centralidades(
        valores_centralidad, titulo_centralidades
    )
    archivo_centralidades = carpeta_salida / "distribuciones_centralidades.html"
    fig_centralidades.write_html(str(archivo_centralidades))
    logger.info(f"  Centrality plot saved to: {archivo_centralidades}")

    return True


def get_available_imfs(carpeta_base: Path, tipo_grafo: str) -> List[str]:
    """
    Get the list of available IMFs for a graph type.

    Parameters
    ----------
    carpeta_base : Path
        Base folder where graphs are stored.
    tipo_grafo : str
        Graph type (nvg, hvg, recurrencia).

    Returns
    -------
    List[str]
        List of available IMF IDs.
    """
    carpeta_tipo = carpeta_base / tipo_grafo
    if not carpeta_tipo.exists():
        return []

    imfs = []
    for item in carpeta_tipo.iterdir():
        if item.is_dir():
            imfs.append(item.name)

    return sorted(imfs)


def main():
    """
    Main entry point for the script.

    Processes the selected graphs according to the boolean configuration,
    computes metrics, and generates visualizations.
    """
    # Selection configuration: boolean dictionary to select
    # transformations and IMFs to process
    configuracion_procesamiento = {
        "nvg": {
            "imf_1": False,
            "imf_2": False,
            "imf_3": False,
            "imf_4": False,
            "imf_5": False,
            "imf_6": False,
            "imf_7": False,
            "imf_8": False,
            "imf_9": True,
            "imf_10": True,
            "residuo": False,
        },
        "hvg": {
            "imf_1": False,
            "imf_2": False,
            "imf_3": False,
            "imf_4": False,
            "imf_5": False,
            "imf_6": False,
            "imf_7": False,
            "imf_8": False,
            "imf_9": False,
            "imf_10": False,
            "residuo": False,
        },
        "recurrencia": {
            "imf_1": False,
            "imf_2": False,
            "imf_3": False,
            "imf_4": False,
            "imf_5": False,
            "imf_6": False,
            "imf_7": False,
            "imf_8": False,
            "imf_9": False,
            "imf_10": False,
            "residuo": False,
        },
    }

    proyecto_root = Path(__file__).parent.parent
    carpeta_grafos = proyecto_root / "data" / "16dic25" / "grafos"

    logger.info("=" * 80)
    logger.info("GRAPH METRICS COMPUTATION FOR SELECTED IMFs")
    logger.info("=" * 80)

    for tipo_grafo, imfs_seleccionadas in configuracion_procesamiento.items():
        # Check whether any IMF is selected for this graph type
        if not any(imfs_seleccionadas.values()):
            logger.info(f"\n--- Skipping {tipo_grafo.upper()} (no IMF selected) ---")
            continue

        logger.info("\n" + "=" * 80)
        logger.info(f"PROCESSING GRAPHS OF TYPE: {tipo_grafo.upper()}")
        logger.info("=" * 80)

        imfs_disponibles = get_available_imfs(carpeta_grafos, tipo_grafo)

        if not imfs_disponibles:
            logger.warning(f"No IMFs found for {tipo_grafo}")
            continue

        # Filter only selected IMFs
        imfs_a_procesar = [
            imf for imf in imfs_disponibles
            if imfs_seleccionadas.get(imf, False)
        ]

        if not imfs_a_procesar:
            logger.warning(
                f"No selected IMFs found for {tipo_grafo} "
                f"(available: {imfs_disponibles})"
            )
            continue

        for idx, id_imf in enumerate(imfs_a_procesar, 1):
            logger.info(
                f"\n--- Processing {tipo_grafo.upper()} {id_imf} "
                f"({idx}/{len(imfs_a_procesar)}) ---"
            )

            carpeta_imf = carpeta_grafos / tipo_grafo / id_imf

            # Find .pt file
            archivos_pt = list(carpeta_imf.glob("*.pt"))
            if not archivos_pt:
                logger.warning(f"No .pt file found in {carpeta_imf}")
                continue

            archivo_grafo = archivos_pt[0]

            # Create graph_metrics folder if it does not exist
            carpeta_data = carpeta_imf / "graph_metrics"
            carpeta_data.mkdir(exist_ok=True)

            # Process graph
            exito = procesar_grafo(archivo_grafo, tipo_grafo, id_imf, carpeta_data)

            if exito:
                logger.info(f"✓ Processing completed for {tipo_grafo} {id_imf}")
            else:
                logger.error(f"✗ Error processing {tipo_grafo} {id_imf}")

    logger.info("\n" + "=" * 80)
    logger.info("PROCESS COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
