"""
Utilities for transforming IMFs into graphs as PyTorch Geometric Data objects.

This module contains functions to transform Intrinsic Mode Functions (IMFs) into different
graph types: Horizontal Visibility Graph (HVG), Natural Visibility Graph (NVG), and
recurrence graph.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from scipy.spatial import KDTree
from sklearn.metrics import mutual_info_score
from torch_geometric.data import Data
from ts2vg import HorizontalVG, NaturalVG

from GraphEMD.data.python_utils import save_graph_data


def build_hvg_imf_graph(
    archivo_imfs: str,
    id_imf: str,
) -> Data:
    """
    Transform an IMF into an HVG graph and return a PyTorch Geometric Data object.

    Loads a specific IMF from a parquet file and builds the Horizontal Visibility
    Graph (HVG). The Data object contains node features (IMF values) and graph
    edges (edge_index).

    Parameters
    ----------
    archivo_imfs : str
        Path to the parquet file with IMFs (must contain columns IMF_1, IMF_2, etc.).
    id_imf : str
        Identifier of the IMF to transform (e.g. "IMF_1", "IMF_2", "Residuo").

    Returns
    -------
    Data
        PyTorch Geometric Data object with the IMF HVG graph.

    Examples
    --------
    >>> from pathlib import Path
    >>> proyecto_root = Path(__file__).parent.parent.parent.parent.parent
    >>> archivo_imfs = proyecto_root / "data" / "16dic25" / "msci_world_imfs.parquet"
    >>> grafo = build_hvg_imf_graph(str(archivo_imfs), "IMF_1")
    >>> print(f"Nodes: {grafo.num_nodes}, Edges: {grafo.num_edges}")
    """
    # Load IMF data
    print(f"Loading IMFs from: {archivo_imfs}")
    df_imfs = pd.read_parquet(archivo_imfs, engine="pyarrow")
    print(f"DataFrame shape: {df_imfs.shape}")
    print(f"Available columns: {list(df_imfs.columns)}")

    # Verify that the specified IMF exists
    if id_imf not in df_imfs.columns:
        raise ValueError(
            f"IMF '{id_imf}' does not exist in the DataFrame. "
            f"Available columns: {list(df_imfs.columns)}"
        )

    # Extract the selected IMF
    imf_valores = np.array(df_imfs[id_imf].values)
    print(f"\nSelected IMF: {id_imf}")
    print(f"Shape of {id_imf}: {imf_valores.shape}")
    print(
        f"Valores - Min: {np.min(imf_valores):.4f}, Max: {np.max(imf_valores):.4f}, "
        f"Mean: {np.mean(imf_valores):.4f}"
    )

    # Build the HVG (Horizontal Visibility Graph)
    print("\nBuilding HVG graph...")
    hvg = HorizontalVG(directed="left_to_right")
    grafo_hvg = hvg.build(imf_valores)

    # Get nodes and edges
    nodos = np.arange(len(imf_valores))  # Nodes are temporal indices
    edges = np.array(grafo_hvg.edges)  # Edges are visibility connections

    print(f"Number of nodes: {len(nodos)}")
    print(f"Number of edges: {len(enlaces)}")

    # Convert the HVG graph to a PyTorch Geometric Data object
    print("Converting to PyTorch Geometric Data object...")
    # Convert edges to edge_index (format [2, num_edges])
    edge_index = torch.tensor(enlaces.T, dtype=torch.long)

    # Create node features from time series values
    # Each node has its time series value as its feature
    node_features = torch.tensor(imf_valores, dtype=torch.float).unsqueeze(1)

    # Create the Data object
    data = Data(x=node_features, edge_index=edge_index)

    print(f"Data object created:")
    print(f"  - Number of nodes: {data.num_nodes}")
    print(f"  - Number of edges: {data.num_edges}")
    if data.x is not None:
        print(f"  - Node features shape: {data.x.shape}")
    if data.edge_index is not None:
        print(f"  - Edge index shape: {data.edge_index.shape}")

    return data


def build_nvg_imf_graph(
    archivo_imfs: str,
    id_imf: str,
) -> Data:
    """
    Transform an IMF into an NVG graph and return a PyTorch Geometric Data object.

    Loads a specific IMF from a parquet file and builds the Natural Visibility
    Graph (NVG). The Data object contains node features (IMF values) and graph
    edges (edge_index).

    Parameters
    ----------
    archivo_imfs : str
        Path to the parquet file with IMFs (must contain columns IMF_1, IMF_2, etc.).
    id_imf : str
        Identifier of the IMF to transform (e.g. "IMF_1", "IMF_2", "Residuo").

    Returns
    -------
    Data
        PyTorch Geometric Data object with the IMF NVG graph.

    Examples
    --------
    >>> from pathlib import Path
    >>> proyecto_root = Path(__file__).parent.parent.parent.parent.parent
    >>> archivo_imfs = proyecto_root / "data" / "16dic25" / "msci_world_imfs.parquet"
    >>> grafo = build_nvg_imf_graph(str(archivo_imfs), "IMF_1")
    >>> print(f"Nodes: {grafo.num_nodes}, Edges: {grafo.num_edges}")
    """
    # Load IMF data
    print(f"Loading IMFs from: {archivo_imfs}")
    df_imfs = pd.read_parquet(archivo_imfs, engine="pyarrow")
    print(f"DataFrame shape: {df_imfs.shape}")
    print(f"Available columns: {list(df_imfs.columns)}")

    # Verify that the specified IMF exists
    if id_imf not in df_imfs.columns:
        raise ValueError(
            f"IMF '{id_imf}' does not exist in the DataFrame. "
            f"Available columns: {list(df_imfs.columns)}"
        )

    # Extract the selected IMF
    imf_valores = np.array(df_imfs[id_imf].values)
    print(f"\nSelected IMF: {id_imf}")
    print(f"Shape of {id_imf}: {imf_valores.shape}")
    print(
        f"Valores - Min: {np.min(imf_valores):.4f}, Max: {np.max(imf_valores):.4f}, "
        f"Mean: {np.mean(imf_valores):.4f}"
    )

    # Build the NVG (Natural Visibility Graph)
    print("\nBuilding NVG graph...")
    nvg = NaturalVG(directed="left_to_right")
    grafo_nvg = nvg.build(imf_valores)

    # Get nodes and edges
    nodos = np.arange(len(imf_valores))  # Nodes are temporal indices
    edges = np.array(grafo_nvg.edges)  # Edges are visibility connections

    print(f"Number of nodes: {len(nodos)}")
    print(f"Number of edges: {len(enlaces)}")

    # Convert the NVG graph to a PyTorch Geometric Data object
    print("Converting to PyTorch Geometric Data object...")
    # Convert edges to edge_index (format [2, num_edges])
    edge_index = torch.tensor(enlaces.T, dtype=torch.long)

    # Create node features from time series values
    # Each node has its time series value as its feature
    node_features = torch.tensor(imf_valores, dtype=torch.float).unsqueeze(1)

    # Create the Data object
    data = Data(x=node_features, edge_index=edge_index)

    print(f"Data object created:")
    print(f"  - Number of nodes: {data.num_nodes}")
    print(f"  - Number of edges: {data.num_edges}")
    if data.x is not None:
        print(f"  - Node features shape: {data.x.shape}")
    if data.edge_index is not None:
        print(f"  - Edge index shape: {data.edge_index.shape}")

    return data


def calcular_informacion_mutual(serie: np.ndarray, tau_max: int = 50) -> tuple:
    """
    Compute mutual information between the series and its time-shifted version.

    Parameters
    ----------
    serie : np.ndarray
        One-dimensional time series.
    tau_max : int, optional
        Maximum tau value to evaluate. Default is 50.

    Returns
    -------
    tuple
        Tuple (taus, valores_mi) where taus are the evaluated tau values
        and valores_mi are the corresponding mutual information values.
    """
    taus = np.arange(1, min(tau_max + 1, len(serie) // 2))
    valores_mi = []

    # Discretize the series for MI computation
    # Use a histogram for discretization
    n_bins = int(np.sqrt(len(serie)))
    serie_discreta = np.digitize(serie, bins=np.linspace(serie.min(), serie.max(), n_bins))

    for tau in taus:
        serie_desplazada = np.roll(serie_discreta, -tau)
        # Ensure both series have the same length
        serie_original = serie_discreta[:-tau] if tau > 0 else serie_discreta
        serie_desplazada = serie_desplazada[:-tau] if tau > 0 else serie_desplazada

        mi = mutual_info_score(serie_original, serie_desplazada)
        valores_mi.append(mi)

    return taus, np.array(valores_mi)


def seleccionar_tau(serie: np.ndarray, tau_max: int = 50) -> int:
    """
    Select the optimal tau value using the first mutual information minimum.

    Parameters
    ----------
    serie : np.ndarray
        One-dimensional time series.
    tau_max : int, optional
        Maximum tau value to evaluate. Default is 50.

    Returns
    -------
    int
        Optimal tau value (delay).
    """
    taus, valores_mi = calcular_informacion_mutual(serie, tau_max)

    # Find the first local minimum
    # If no clear minimum exists, use the first value where MI decreases significantly
    if len(valores_mi) > 2:
        # Find local minima
        minimos = []
        for i in range(1, len(valores_mi) - 1):
            if valores_mi[i] < valores_mi[i - 1] and valores_mi[i] < valores_mi[i + 1]:
                minimos.append(i)

        if minimos:
            tau_optimo = taus[minimos[0]]
        else:
            # If no clear minimum exists, use the first value where the derivative changes sign
            derivada = np.diff(valores_mi)
            cambios = np.where(derivada > 0)[0]
            if len(cambios) > 0:
                tau_optimo = taus[cambios[0] + 1] if cambios[0] + 1 < len(taus) else taus[1]
            else:
                tau_optimo = taus[1]  # Default value
    else:
        tau_optimo = taus[0] if len(taus) > 0 else 1

    return tau_optimo


def construir_espacio_embedding(serie: np.ndarray, dim: int, tau: int) -> np.ndarray:
    """
    Build the embedding space using delay embedding.

    Parameters
    ----------
    serie : np.ndarray
        One-dimensional time series.
    dim : int
        Embedding dimension.
    tau : int
        Delay for the embedding.

    Returns
    -------
    np.ndarray
        Embedding matrix of shape (N - (dim-1)*tau, dim) where N is the length
        of the original series.
    """
    n = len(serie)
    m = n - (dim - 1) * tau

    if m <= 0:
        raise ValueError(f"The combination of dim={dim} y tau={tau} yields m={m} <= 0")

    embedding = np.zeros((m, dim))
    for i in range(dim):
        embedding[:, i] = serie[i * tau : i * tau + m]

    return embedding


def calcular_false_nearest_neighbors(
    serie: np.ndarray,
    tau: int,
    dim_max: int = 10,
    umbral: float = 10.0,
    ratio_min: float = 0.1,
) -> int:
    """
    Compute the optimal dimension using the False Nearest Neighbors (FNN) method.

    Parameters
    ----------
    serie : np.ndarray
        One-dimensional time series.
    tau : int
        Pre-selected delay.
    dim_max : int, optional
        Maximum dimension to evaluate. Default is 10.
    umbral : float, optional
        Threshold for considering a neighbor as "false". Default is 10.0.
    ratio_min : float, optional
        Minimum FNN ratio to consider that the optimal dimension has been reached.
        Default is 0.1 (10%).

    Returns
    -------
    int
        Optimal embedding dimension.
    """
    ratios_fnn = []

    for dim in range(1, dim_max + 1):
        try:
            # Build embedding in dimension dim
            embedding_dim = construir_espacio_embedding(serie, dim, tau)

            if dim == 1:
                ratios_fnn.append(1.0)  # At dim=1, all are "false" by definition
                continue

            # Build embedding in dimension dim+1
            embedding_dim_plus = construir_espacio_embedding(serie, dim + 1, tau)

            # Build KDTree for neighbor search
            tree = KDTree(embedding_dim)

            # For each point, find the nearest neighbor
            n_puntos = len(embedding_dim)
            falsos_vecinos = 0

            for i in range(n_puntos):
                # Find the nearest neighbor in dimension dim
                # k=2 to get the point itself and its nearest neighbor
                distancias, indices = tree.query(embedding_dim[i], k=2)

                # Ensure they are arrays
                if not isinstance(distancias, np.ndarray):
                    distancias = np.array([distancias])
                if not isinstance(indices, np.ndarray):
                    indices = np.array([indices])

                # Find the nearest neighbor (excluding the point itself)
                if len(distancias) > 1:
                    if indices[0] == i:
                        vecino_idx = int(indices[1])
                        dist_dim = float(distancias[1])
                    else:
                        vecino_idx = int(indices[0])
                        dist_dim = float(distancias[0])
                else:
                    continue

                # Compute distance in dimension dim+1
                if vecino_idx < len(embedding_dim_plus) and i < len(embedding_dim_plus):
                    dist_dim_plus = np.linalg.norm(
                        embedding_dim_plus[i] - embedding_dim_plus[vecino_idx]
                    )

                    # Check whether it is a false neighbor
                    if dist_dim > 1e-10:  # Avoid division by zero
                        ratio = (
                            abs(embedding_dim_plus[i, -1] - embedding_dim_plus[vecino_idx, -1])
                            / dist_dim
                        )
                        if ratio > umbral:
                            falsos_vecinos += 1

            ratio_fnn = falsos_vecinos / n_puntos if n_puntos > 0 else 1.0
            ratios_fnn.append(ratio_fnn)

            # If the ratio is very low, the optimal dimension has likely been found
            if ratio_fnn < ratio_min:
                return dim

        except Exception as e:
            print(f"Error computing FNN for dim={dim}: {e}")
            ratios_fnn.append(1.0)

    # If no clear minimum was found, use the dimension with the lowest ratio
    if ratios_fnn:
        dim_optima = int(np.argmin(ratios_fnn)) + 1
        return dim_optima

    return 2  # Default value


def calcular_matriz_recurrencia(
    embedding: np.ndarray,
    umbral: Optional[float] = None,
    umbral_percentil: float = 10.0,
    random_state: Optional[int] = None,
) -> tuple:
    """
    Compute the recurrence matrix from the embedding space.

    Parameters
    ----------
    embedding : np.ndarray
        Embedding matrix of shape (N, dim).
    umbral : float, optional
        Absolute distance threshold. If None, umbral_percentil is used.
    umbral_percentil : float, optional
        Percentile used to determine the distance threshold. Default is 10.0.
    random_state : int, optional
        Seed for random number generation. Default is None.

    Returns
    -------
    tuple
        Tuple (matriz_recurrencia, umbral_utilizado) where matriz_recurrencia is
        a binary matrix of shape (N, N) and umbral_utilizado is the distance
        threshold used (float).
    """
    # Set random state for reproducibility
    if random_state is not None:
        np.random.seed(random_state)

    n = len(embedding)

    # Warning for very large matrices
    if n > 5000:
        print(
            f"Warning: The recurrence matrix will be very large ({n}x{n}). "
            f"This may take several minutes."
        )

    # Build KDTree
    tree = KDTree(embedding)

    # Determine threshold if not provided
    if umbral is None:
        # Compute threshold using a sample of distances
        # Take a random sample of points to estimate the percentile
        n_muestra = min(1000, n)
        indices_muestra = np.random.choice(n, size=n_muestra, replace=False)
        distancias_muestra = []

        for idx in indices_muestra:
            # Get distances to the k nearest neighbors
            distancias, _ = tree.query(embedding[idx], k=min(100, n))
            distancias_muestra.extend(distancias[1:])  # Exclude self-distance

        umbral = float(np.percentile(distancias_muestra, umbral_percentil))
        print(f"Computed threshold (percentile {umbral_percentil}): {umbral:.4f}")
    else:
        # Ensure the provided threshold is a float
        umbral = float(umbral)

    # Build recurrence matrix efficiently
    # Initialize zero matrix
    matriz_recurrencia = np.zeros((n, n), dtype=int)

    # For each point, find all points within the threshold
    for i in range(n):
        # Use query_ball_point to find all points within the radius
        indices_vecinos = tree.query_ball_point(embedding[i], r=umbral, p=2)
        # Convert to array and remove the point itself
        indices_vecinos = np.array([idx for idx in indices_vecinos if idx != i])
        if len(indices_vecinos) > 0:
            matriz_recurrencia[i, indices_vecinos] = 1

    print(f"Number of recurrences found: {np.sum(matriz_recurrencia)}")

    return matriz_recurrencia, umbral


def build_recurrence_imf_graph(
    archivo_imfs: str,
    id_imf: str,
    tau_max: int = 50,
    dim_max: int = 10,
    umbral_percentil: float = 10.0,
    umbral: Optional[float] = None,
    random_state: Optional[int] = None,
) -> Data:
    """
    Transform an IMF into a recurrence graph and return a PyTorch Geometric Data object.

    Loads a specific IMF from a parquet file and builds the recurrence graph
    using delay embedding. The Data object contains node features (embedding values)
    and graph edges (edge_index) based on the recurrence matrix.

    Process metadata (tau, embedding dimension, distance algorithm, and recurrence
    threshold) is stored as attributes of the Data object.

    Parameters
    ----------
    archivo_imfs : str
        Path to the parquet file with IMFs (must contain columns IMF_1, IMF_2, etc.).
    id_imf : str
        Identifier of the IMF to transform (e.g. "IMF_1", "IMF_2", "Residuo").
    tau_max : int, optional
        Maximum tau value to evaluate for delay selection. Default is 50.
    dim_max : int, optional
        Maximum dimension to evaluate for the embedding. Default is 10.
    umbral_percentil : float, optional
        Percentile used to determine the distance threshold in the recurrence matrix.
        Default is 10.0.
    umbral : float, optional
        Absolute distance threshold. If provided, this value is used instead of
        computing it with umbral_percentil. Default is None.
    random_state : int, optional
        Seed for random number generation. Used to ensure reproducibility in all
        calculations involving randomness. Default is None.

    Returns
    -------
    Data
        PyTorch Geometric Data object with the IMF recurrence graph.
        Metadata is stored in the attributes:
        - tau: delay value used
        - dim_embedding: embedding dimension used
        - algoritmo_distancia: distance algorithm used ("euclidean" with KDTree)
        - umbral_recurrencia: distance threshold used for the recurrence matrix

    Examples
    --------
    >>> from pathlib import Path
    >>> proyecto_root = Path(__file__).parent.parent.parent.parent.parent
    >>> archivo_imfs = proyecto_root / "data" / "16dic25" / "msci_world_imfs.parquet"
    >>> grafo = build_recurrence_imf_graph(str(archivo_imfs), "IMF_1", random_state=42)
    >>> print(f"Nodes: {grafo.num_nodes}, Edges: {grafo.num_edges}")
    >>> print(f"Tau: {grafo.tau}, Dim: {grafo.dim_embedding}")
    """
    # Set random state at the start to ensure reproducibility
    if random_state is not None:
        np.random.seed(random_state)

    # Load IMF data
    print(f"Loading IMFs from: {archivo_imfs}")
    df_imfs = pd.read_parquet(archivo_imfs, engine="pyarrow")
    print(f"DataFrame shape: {df_imfs.shape}")
    print(f"Available columns: {list(df_imfs.columns)}")

    # Verify that the specified IMF exists
    if id_imf not in df_imfs.columns:
        raise ValueError(
            f"IMF '{id_imf}' does not exist in the DataFrame. "
            f"Available columns: {list(df_imfs.columns)}"
        )

    # Extract the selected IMF
    imf_valores = np.array(df_imfs[id_imf].values)
    print(f"\nSelected IMF: {id_imf}")
    print(f"Shape of {id_imf}: {imf_valores.shape}")
    print(
        f"Valores - Min: {np.min(imf_valores):.4f}, Max: {np.max(imf_valores):.4f}, "
        f"Mean: {np.mean(imf_valores):.4f}"
    )

    # Step 1: Select tau (delay) using mutual information
    print("\nSelecting tau (delay) using mutual information...")
    tau_optimo = seleccionar_tau(imf_valores, tau_max=tau_max)
    print(f"Optimal tau selected: {tau_optimo}")

    # Step 2: Select dim using False Nearest Neighbors
    print("Selecting dim (embedding dimension) using False Nearest Neighbors...")
    dim_optima = calcular_false_nearest_neighbors(
        imf_valores, tau=tau_optimo, dim_max=dim_max
    )
    print(f"Optimal dimension selected: {dim_optima}")

    # Step 3: Build embedding space
    print(
        f"Building embedding space with dim={dim_optima} y tau={tau_optimo}..."
    )
    embedding = construir_espacio_embedding(imf_valores, dim=dim_optima, tau=tau_optimo)
    print(f"Embedding shape: {embedding.shape}")

    # Step 4: Compute recurrence matrix
    print("Computing recurrence matrix...")
    matriz_recurrencia, umbral_utilizado = calcular_matriz_recurrencia(
        embedding, umbral=umbral, umbral_percentil=umbral_percentil, random_state=random_state
    )
    print(f"Recurrence matrix shape: {matriz_recurrencia.shape}")
    print(f"Number of recurrences (edges): {np.sum(matriz_recurrencia)}")

    # Step 5: Convert recurrence matrix to edge_index format for PyTorch Geometric
    print("Converting recurrence matrix to graph format...")

    # Get edge indices (where matriz_recurrencia == 1)
    edge_indices = np.where(matriz_recurrencia == 1)

    # Create edge_index in format [2, num_edges]
    edge_index = np.array([edge_indices[0], edge_indices[1]])
    edge_index_torch = torch.tensor(edge_index, dtype=torch.long)

    print(f"Number of nodes: {len(embedding)}")
    print(f"Number of edges: {edge_index_torch.shape[1]}")

    # Create node features using the full embedding of each node
    node_features = torch.tensor(embedding, dtype=torch.float)

    # Create the PyTorch Geometric Data object
    data = Data(x=node_features, edge_index=edge_index_torch)

    # Store metadata in the Data object
    data.tau = int(tau_optimo)
    data.dim_embedding = int(dim_optima)
    data.algoritmo_distancia = "euclidean"  # KDTree with p=2 uses Euclidean distance
    data.umbral_recurrencia = umbral_utilizado

    print(f"\nData object created:")
    print(f"  - Number of nodes: {data.num_nodes}")
    print(f"  - Number of edges: {data.num_edges}")
    if data.x is not None:
        print(f"  - Node features shape: {data.x.shape}")
    if data.edge_index is not None:
        print(f"  - Edge index shape: {data.edge_index.shape}")
    print(f"\nMetadata saved:")
    print(f"  - Tau (delay): {data.tau}")
    print(f"  - Embedding dimension: {data.dim_embedding}")
    print(f"  - Distance algorithm: {data.algoritmo_distancia}")
    print(f"  - Recurrence threshold: {data.umbral_recurrencia:.4f}")

    return data


def build_all_imf_graphs(
    df_imfs: Union[pd.DataFrame, str],
    carpeta_salida_base: Union[str, Path],
    tau_max: int = 50,
    dim_max: int = 10,
    umbral_percentil: float = 10.0,
    umbral: Optional[float] = None,
    random_state: Optional[int] = None,
    columnas_imf: Optional[list] = None,
) -> dict:
    """
    Generate all graph types for every IMF in the dataframe.

    For each IMF in the dataframe, computes and saves the three implemented graph types:
    - Horizontal Visibility Graph (HVG)
    - Natural Visibility Graph (NVG)
    - Recurrence graph

    Graphs are saved in the following folder structure:
    - {carpeta_salida_base}/hvg/{id_imf}/grafo_hvg_{id_imf}
    - {carpeta_salida_base}/nvg/{id_imf}/grafo_nvg_{id_imf}
    - {carpeta_salida_base}/recurrencia/{id_imf}/grafo_recurrencia_{id_imf}

    Parameters
    ----------
    df_imfs : pd.DataFrame or str
        DataFrame with IMFs or path to the parquet file containing IMFs.
        Must contain columns named like "IMF_1", "IMF_2", etc.
    carpeta_salida_base : str or Path
        Base path where graphs will be saved. Subfolders will be created for each
        graph type and each IMF.
    tau_max : int, optional
        Maximum tau value to evaluate for delay selection in recurrence graphs.
        Default is 50.
    dim_max : int, optional
        Maximum dimension to evaluate for the embedding in recurrence graphs.
        Default is 10.
    umbral_percentil : float, optional
        Percentile used to determine the distance threshold in the recurrence matrix.
        Default is 10.0.
    umbral : float, optional
        Absolute distance threshold for recurrence graphs. If provided, this value
        is used instead of computing it with umbral_percentil. Default is None.
    random_state : int, optional
        Seed for random number generation. Used to ensure reproducibility in
        recurrence graph calculations. Default is None.
    columnas_imf : list, optional
        List of column names to process. If None, all columns starting with "IMF_"
        or named "Residuo" are processed. Default is None.

    Returns
    -------
    dict
        Dictionary with information about the generated graphs. Keys are IMF IDs
        and values are dictionaries with information about each generated graph type.

    Examples
    --------
    >>> from pathlib import Path
    >>> proyecto_root = Path(__file__).parent.parent.parent.parent.parent
    >>> archivo_imfs = proyecto_root / "data" / "16dic25" / "msci_world_imfs.parquet"
    >>> carpeta_salida = proyecto_root / "data" / "16dic25" / "grafos"
    >>> resultados = build_all_imf_graphs(
    ...     df_imfs=str(archivo_imfs),
    ...     carpeta_salida_base=str(carpeta_salida),
    ...     random_state=42
    ... )
    >>> print(f"Graphs generated for {len(resultados)} IMFs")
    """
    # Load dataframe if a path is provided
    if isinstance(df_imfs, str):
        archivo_imfs = df_imfs
        print(f"Loading IMFs from: {archivo_imfs}")
        if not Path(archivo_imfs).exists():
            raise FileNotFoundError(
                f"File {archivo_imfs} does not exist. "
                "Make sure the file is in the correct location."
            )
        df_imfs = pd.read_parquet(archivo_imfs, engine="pyarrow")
    else:
        archivo_imfs = None

    # Convert carpeta_salida_base to Path
    carpeta_salida_base = Path(carpeta_salida_base)
    carpeta_salida_base.mkdir(parents=True, exist_ok=True)

    # Determine which columns to process
    if columnas_imf is None:
        # Process all columns starting with "IMF_" or named "Residuo"
        columnas_imf = [
            col
            for col in df_imfs.columns
            if col.startswith("IMF_") or col == "Residuo"
        ]

    if not columnas_imf:
        raise ValueError(
            "No IMF columns found to process. "
            "Make sure the dataframe contains columns named like "
            '"IMF_1", "IMF_2", etc., o "Residuo".'
        )

    print(f"\nIMF columns to process: {columnas_imf}")
    print(f"Total IMFs: {len(columnas_imf)}")

    # If archivo_imfs is available, use it for the transformation functions
    if archivo_imfs is None:
        # If no file path is available, we would need to save the dataframe temporarily
        # or modify the functions to accept dataframes directly
        # For now, we assume a path is always provided
        raise ValueError(
            "A path to the parquet file must be provided, not a DataFrame directly. "
            "Transformation functions require a path to the file."
        )

    # Dictionary to store results
    resultados = {}

    # Process each IMF
    for idx, id_imf in enumerate(columnas_imf, 1):
        print("\n" + "=" * 80)
        print(f"PROCESSING IMF {idx}/{len(columnas_imf)}: {id_imf}")
        print("=" * 80)

        resultados_imf = {}

        # 1. HVG graph
        try:
            print(f"\n--- Generating HVG graph for {id_imf} ---")
            carpeta_hvg = carpeta_salida_base / "hvg" / id_imf.lower()
            carpeta_hvg.mkdir(parents=True, exist_ok=True)
            archivo_salida_hvg = str(carpeta_hvg / f"grafo_hvg_{id_imf.lower()}")

            grafo_hvg = build_hvg_imf_graph(archivo_imfs=archivo_imfs, id_imf=id_imf)
            save_graph_data(
                data=grafo_hvg, archivo_salida=archivo_salida_hvg, id_imf=id_imf
            )

            resultados_imf["hvg"] = {
                "archivo": archivo_salida_hvg,
                "num_nodes": grafo_hvg.num_nodes,
                "num_edges": grafo_hvg.num_edges,
                "exito": True,
            }
            print(f"✓ HVG graph generated successfully for {id_imf}")

        except Exception as e:
            print(f"✗ Error generating HVG graph for {id_imf}: {e}")
            resultados_imf["hvg"] = {"exito": False, "error": str(e)}

        # 2. NVG graph
        try:
            print(f"\n--- Generating NVG graph for {id_imf} ---")
            carpeta_nvg = carpeta_salida_base / "nvg" / id_imf.lower()
            carpeta_nvg.mkdir(parents=True, exist_ok=True)
            archivo_salida_nvg = str(carpeta_nvg / f"grafo_nvg_{id_imf.lower()}")

            grafo_nvg = build_nvg_imf_graph(archivo_imfs=archivo_imfs, id_imf=id_imf)
            save_graph_data(
                data=grafo_nvg, archivo_salida=archivo_salida_nvg, id_imf=id_imf
            )

            resultados_imf["nvg"] = {
                "archivo": archivo_salida_nvg,
                "num_nodes": grafo_nvg.num_nodes,
                "num_edges": grafo_nvg.num_edges,
                "exito": True,
            }
            print(f"✓ NVG graph generated successfully for {id_imf}")

        except Exception as e:
            print(f"✗ Error generating NVG graph for {id_imf}: {e}")
            resultados_imf["nvg"] = {"exito": False, "error": str(e)}

        # 3. Recurrence graph
        try:
            print(f"\n--- Generating recurrence graph for {id_imf} ---")
            carpeta_recurrencia = (
                carpeta_salida_base / "recurrencia" / id_imf.lower()
            )
            carpeta_recurrencia.mkdir(parents=True, exist_ok=True)
            archivo_salida_recurrencia = str(
                carpeta_recurrencia / f"grafo_recurrencia_{id_imf.lower()}"
            )

            grafo_recurrencia = build_recurrence_imf_graph(
                archivo_imfs=archivo_imfs,
                id_imf=id_imf,
                tau_max=tau_max,
                dim_max=dim_max,
                umbral_percentil=umbral_percentil,
                umbral=umbral,
                random_state=random_state,
            )
            save_graph_data(
                data=grafo_recurrencia,
                archivo_salida=archivo_salida_recurrencia,
                id_imf=id_imf,
            )

            resultados_imf["recurrencia"] = {
                "archivo": archivo_salida_recurrencia,
                "num_nodes": grafo_recurrencia.num_nodes,
                "num_edges": grafo_recurrencia.num_edges,
                "tau": grafo_recurrencia.tau,
                "dim_embedding": grafo_recurrencia.dim_embedding,
                "umbral_recurrencia": grafo_recurrencia.umbral_recurrencia,
                "exito": True,
            }
            print(f"✓ Recurrence graph generated successfully for {id_imf}")

        except Exception as e:
            print(f"✗ Error generating recurrence graph for {id_imf}: {e}")
            resultados_imf["recurrencia"] = {"exito": False, "error": str(e)}

        resultados[id_imf] = resultados_imf

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    total_imfs = len(resultados)
    total_exitosos = sum(
        1
        for imf_resultados in resultados.values()
        if all(
            grafo.get("exito", False)
            for grafo in imf_resultados.values()
            if isinstance(grafo, dict)
        )
    )

    print(f"Total IMFs processed: {total_imfs}")
    print(f"IMFs with all graphs generated successfully: {total_exitosos}")

    for id_imf, imf_resultados in resultados.items():
        print(f"\n{id_imf}:")
        for tipo_grafo, info_grafo in imf_resultados.items():
            if isinstance(info_grafo, dict) and info_grafo.get("exito", False):
                print(
                    f"  ✓ {tipo_grafo.upper()}: "
                    f"{info_grafo.get('num_nodes', 'N/A')} nodes, "
                    f"{info_grafo.get('num_edges', 'N/A')} edges"
                )
            else:
                error = info_grafo.get("error", "Unknown error") if isinstance(info_grafo, dict) else "Unknown error"
                print(f"  ✗ {tipo_grafo.upper()}: {error}")

    return resultados

