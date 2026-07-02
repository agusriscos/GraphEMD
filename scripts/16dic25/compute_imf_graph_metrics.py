"""
Script para calcular métricas descriptivas de todos los grafos generados para las IMFs.

Este script procesa todos los grafos generados (NVG, HVG, recurrencia) para todas las IMFs
del MSCI World, calcula métricas descriptivas y genera visualizaciones, guardando tanto
las gráficas como un CSV con los datos finales en cada carpeta data.
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

# Configurar logging
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
    Carga un grafo desde un archivo .pt y lo convierte a NetworkX.

    Parameters
    ----------
    archivo_grafo : Path
        Ruta al archivo .pt que contiene el grafo.

    Returns
    -------
    Optional[nx.Graph]
        Grafo en formato NetworkX, o None si hay un error.
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
        logger.error(f"Error al cargar grafo desde {archivo_grafo}: {e}")
        return None


def calcular_metricas_basicas(grafo: nx.Graph, umbral_nodos: Optional[int] = None) -> Dict:
    """
    Calcula métricas básicas del grafo.

    Para grafos grandes (más de umbral_nodos nodos), omite el cálculo de
    diámetro, radio y excentricidad promedio ya que son computacionalmente
    muy costosos (O(n²) o peor).

    Parameters
    ----------
    grafo : nx.Graph
        Grafo de NetworkX.
    umbral_nodos : Optional[int], optional
        Umbral de nodos para omitir cálculos costosos. Si es None, siempre
        calcula las métricas costosas. Por defecto es None.

    Returns
    -------
    Dict
        Diccionario con las métricas básicas calculadas.
    """
    num_nodos = grafo.number_of_nodes()
    num_enlaces = grafo.number_of_edges()
    densidad = nx.density(grafo)
    num_componentes = nx.number_connected_components(grafo)

    # Inicializar valores por defecto
    diametro = None
    radio = None
    excentricidad_promedio = None

    # Solo calcular métricas costosas si el grafo es relativamente pequeño
    # o si no hay umbral (umbral_nodos es None)
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
                    f"    Calculando diámetro, radio y excentricidad "
                    f"(grafo con {num_nodos} nodos)..."
                )
                diametro = nx.diameter(grafo_principal)
                radio = nx.radius(grafo_principal)
                excentricidad_promedio = np.mean(
                    list(nx.eccentricity(grafo_principal).values())
                )
            except Exception as e:
                logger.warning(
                    f"    No se pudieron calcular diámetro/radio/excentricidad: {e}"
                )
                diametro = None
                radio = None
                excentricidad_promedio = None
    else:
        logger.info(
            f"    Omitiendo cálculo de diámetro/radio/excentricidad "
            f"(grafo muy grande: {num_nodos} nodos, umbral: {umbral_nodos:,})"
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
    Calcula métricas relacionadas con los grados de los nodos.

    Parameters
    ----------
    grafo : nx.Graph
        Grafo de NetworkX.

    Returns
    -------
    Tuple[Dict, np.ndarray]
        Tupla con (diccionario de métricas, array de valores de grados).
    """
    # Obtener grados de todos los nodos
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
    Calcula métricas relacionadas con el coeficiente de clustering.

    Parameters
    ----------
    grafo : nx.Graph
        Grafo de NetworkX.

    Returns
    -------
    Tuple[Dict, List[float]]
        Tupla con (diccionario de métricas, lista de valores de clustering).
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
    Calcula diferentes tipos de centralidad del grafo.

    Parameters
    ----------
    grafo : nx.Graph
        Grafo de NetworkX.

    Returns
    -------
    Tuple[Dict, Dict]
        Tupla con (diccionario de métricas, diccionario con valores de centralidad).
    """
    logger.info("  Calculando degree centrality...")
    degree_centrality = nx.degree_centrality(grafo)
    valores_degree = list(degree_centrality.values())

    logger.info("  Calculando betweenness centrality (esto puede tardar)...")
    betweenness_centrality = nx.betweenness_centrality(grafo)
    valores_betweenness = list(betweenness_centrality.values())

    logger.info("  Calculando closeness centrality...")
    closeness_centrality = nx.closeness_centrality(grafo)
    valores_closeness = list(closeness_centrality.values())

    logger.info("  Calculando eigenvector centrality...")
    try:
        eigenvector_centrality = nx.eigenvector_centrality(grafo, max_iter=1000)
        valores_eigenvector = list(eigenvector_centrality.values())
    except Exception:
        logger.warning("    No se pudo calcular eigenvector centrality")
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
    Crea una gráfica de distribución de grados.

    Parameters
    ----------
    valores_grados : np.ndarray
        Array con los valores de grados.
    metricas : Dict
        Diccionario con las métricas de grados.
    titulo : str
        Título de la gráfica.

    Returns
    -------
    go.Figure
        Figura de Plotly con la distribución de grados.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=valores_grados,
            nbinsx=50,
            name="Distribución de grados",
            marker=dict(color="steelblue", line=dict(color="black", width=1)),
            opacity=0.7,
        )
    )

    fig.update_layout(
        title=dict(text=titulo, x=0.5, xanchor="center", font=dict(size=18)),
        xaxis_title="Grado del nodo",
        yaxis_title="Frecuencia",
        height=500,
        showlegend=False,
        plot_bgcolor="white",
        annotations=[
            dict(
                text=(
                    f"Promedio: {metricas['grado_promedio']:.2f} | "
                    f"Mediana: {metricas['grado_mediana']:.2f} | "
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
    Crea una gráfica de distribución del coeficiente de clustering.

    Parameters
    ----------
    valores_clustering : List[float]
        Lista con los valores de clustering.
    metricas : Dict
        Diccionario con las métricas de clustering.
    titulo : str
        Título de la gráfica.

    Returns
    -------
    go.Figure
        Figura de Plotly con la distribución de clustering.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=valores_clustering,
            nbinsx=50,
            name="Distribución de clustering",
            marker=dict(color="coral", line=dict(color="black", width=1)),
            opacity=0.7,
        )
    )

    fig.update_layout(
        title=dict(text=titulo, x=0.5, xanchor="center", font=dict(size=18)),
        xaxis_title="Coeficiente de Clustering",
        yaxis_title="Frecuencia",
        height=500,
        showlegend=False,
        plot_bgcolor="white",
        annotations=[
            dict(
                text=(
                    f"Promedio: {metricas['clustering_promedio']:.6f} | "
                    f"Mediana: {metricas['clustering_mediana']:.6f}"
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
    Crea una gráfica con las distribuciones de centralidades.

    Parameters
    ----------
    valores_centralidad : Dict
        Diccionario con los valores de cada tipo de centralidad.
    titulo : str
        Título de la gráfica.

    Returns
    -------
    go.Figure
        Figura de Plotly con las distribuciones de centralidades.
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

    fig.update_xaxes(title_text="Valor", row=2, col=1)
    fig.update_xaxes(title_text="Valor", row=2, col=2)
    fig.update_yaxes(title_text="Frecuencia", row=1, col=1)
    fig.update_yaxes(title_text="Frecuencia", row=2, col=1)

    return fig


def procesar_grafo(
    archivo_grafo: Path,
    tipo_grafo: str,
    id_imf: str,
    carpeta_salida: Path,
) -> bool:
    """
    Procesa un grafo, calcula métricas y genera visualizaciones.

    Parameters
    ----------
    archivo_grafo : Path
        Ruta al archivo .pt del grafo.
    tipo_grafo : str
        Tipo de grafo (nvg, hvg, recurrencia).
    id_imf : str
        Identificador de la IMF.
    carpeta_salida : Path
        Carpeta donde guardar los resultados.

    Returns
    -------
    bool
        True si el procesamiento fue exitoso, False en caso contrario.
    """
    logger.info(f"Procesando grafo: {archivo_grafo.name}")

    # Cargar grafo
    grafo = cargar_grafo_desde_archivo(archivo_grafo)
    if grafo is None:
        return False


    # Calcular métricas básicas
    logger.info("  Calculando métricas básicas...")
    metricas_basicas = calcular_metricas_basicas(grafo)

    # Calcular métricas de grados
    logger.info("  Calculando métricas de grados...")
    metricas_grados, valores_grados = calcular_metricas_grados(grafo)

    # Calcular métricas de clustering
    logger.info("  Calculando métricas de clustering...")
    metricas_clustering, valores_clustering = calcular_metricas_clustering(grafo)

    # Calcular centralidades
    logger.info("  Calculando centralidades...")
    metricas_centralidad, valores_centralidad = calcular_centralidades(grafo)

    # Combinar todas las métricas
    todas_metricas = {
        **metricas_basicas,
        **metricas_grados,
        **metricas_clustering,
        **metricas_centralidad,
    }

    # Crear DataFrame con métricas
    df_metricas = pd.DataFrame([todas_metricas])

    # Guardar CSV
    archivo_csv = carpeta_salida / "metricas_grafo.csv"
    df_metricas.to_csv(archivo_csv, index=False)
    logger.info(f"  Métricas guardadas en: {archivo_csv}")

    # Crear y guardar gráficas
    logger.info("  Generando gráficas...")

    # Gráfica de grados
    titulo_grados = f"Distribución de Grados - {tipo_grafo.upper()} {id_imf}"
    fig_grados = crear_grafica_grados(valores_grados, metricas_grados, titulo_grados)
    archivo_grados = carpeta_salida / "distribucion_grados.html"
    fig_grados.write_html(str(archivo_grados))
    logger.info(f"  Gráfica de grados guardada en: {archivo_grados}")

    # Gráfica de clustering
    titulo_clustering = (
        f"Distribución de Clustering - {tipo_grafo.upper()} {id_imf}"
    )
    fig_clustering = crear_grafica_clustering(
        valores_clustering, metricas_clustering, titulo_clustering
    )
    archivo_clustering = carpeta_salida / "distribucion_clustering.html"
    fig_clustering.write_html(str(archivo_clustering))
    logger.info(f"  Gráfica de clustering guardada en: {archivo_clustering}")

    # Gráfica de centralidades
    titulo_centralidades = (
        f"Distribuciones de Centralidades - {tipo_grafo.upper()} {id_imf}"
    )
    fig_centralidades = crear_grafica_centralidades(
        valores_centralidad, titulo_centralidades
    )
    archivo_centralidades = carpeta_salida / "distribuciones_centralidades.html"
    fig_centralidades.write_html(str(archivo_centralidades))
    logger.info(f"  Gráfica de centralidades guardada en: {archivo_centralidades}")

    return True


def obtener_imfs_disponibles(carpeta_base: Path, tipo_grafo: str) -> List[str]:
    """
    Obtiene la lista de IMFs disponibles para un tipo de grafo.

    Parameters
    ----------
    carpeta_base : Path
        Carpeta base donde están los grafos.
    tipo_grafo : str
        Tipo de grafo (nvg, hvg, recurrencia).

    Returns
    -------
    List[str]
        Lista de IDs de IMFs disponibles.
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
    Función principal del script.

    Procesa los grafos seleccionados según la configuración de booleanos,
    calcula métricas y genera visualizaciones.
    """
    # Configuración de selección: diccionario de booleanos para seleccionar
    # transformaciones e IMFs a procesar
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

    proyecto_root = Path(__file__).parent.parent.parent
    carpeta_grafos = proyecto_root / "data" / "16dic25" / "grafos"

    logger.info("=" * 80)
    logger.info("CÁLCULO DE MÉTRICAS DE GRAFOS PARA IMFS SELECCIONADAS")
    logger.info("=" * 80)

    for tipo_grafo, imfs_seleccionadas in configuracion_procesamiento.items():
        # Verificar si hay alguna IMF seleccionada para este tipo de grafo
        if not any(imfs_seleccionadas.values()):
            logger.info(f"\n--- Omitiendo {tipo_grafo.upper()} (ninguna IMF seleccionada) ---")
            continue

        logger.info("\n" + "=" * 80)
        logger.info(f"PROCESANDO GRAFOS DE TIPO: {tipo_grafo.upper()}")
        logger.info("=" * 80)

        imfs_disponibles = obtener_imfs_disponibles(carpeta_grafos, tipo_grafo)

        if not imfs_disponibles:
            logger.warning(f"No se encontraron IMFs para {tipo_grafo}")
            continue

        # Filtrar solo las IMFs seleccionadas
        imfs_a_procesar = [
            imf for imf in imfs_disponibles
            if imfs_seleccionadas.get(imf, False)
        ]

        if not imfs_a_procesar:
            logger.warning(
                f"No se encontraron IMFs seleccionadas para {tipo_grafo} "
                f"(disponibles: {imfs_disponibles})"
            )
            continue

        for idx, id_imf in enumerate(imfs_a_procesar, 1):
            logger.info(
                f"\n--- Procesando {tipo_grafo.upper()} {id_imf} "
                f"({idx}/{len(imfs_a_procesar)}) ---"
            )

            carpeta_imf = carpeta_grafos / tipo_grafo / id_imf

            # Buscar archivo .pt
            archivos_pt = list(carpeta_imf.glob("*.pt"))
            if not archivos_pt:
                logger.warning(f"No se encontró archivo .pt en {carpeta_imf}")
                continue

            archivo_grafo = archivos_pt[0]

            # Crear carpeta graph_metrics si no existe
            carpeta_data = carpeta_imf / "graph_metrics"
            carpeta_data.mkdir(exist_ok=True)

            # Procesar grafo
            exito = procesar_grafo(archivo_grafo, tipo_grafo, id_imf, carpeta_data)

            if exito:
                logger.info(f"✓ Procesamiento completado para {tipo_grafo} {id_imf}")
            else:
                logger.error(f"✗ Error al procesar {tipo_grafo} {id_imf}")

    logger.info("\n" + "=" * 80)
    logger.info("PROCESO COMPLETADO")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

