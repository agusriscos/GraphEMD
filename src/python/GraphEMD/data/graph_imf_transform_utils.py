"""
Utilidades para transformar IMFs a grafos como objetos Data de PyTorch Geometric.

Este módulo contiene funciones para transformar Intrinsic Mode Functions (IMFs) a diferentes
tipos de grafos: Horizontal Visibility Graph (HVG), Natural Visibility Graph (NVG) y
grafo de recurrencia.
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

from GraphEMD.data.python_utils import guardar_grafo_data


def obtener_grafo_hvg_imf(
    archivo_imfs: str,
    id_imf: str,
) -> Data:
    """
    Transforma una IMF a grafo HVG y retorna un objeto Data de PyTorch Geometric.

    Carga una IMF específica desde un archivo parquet y construye el grafo Horizontal Visibility
    Graph (HVG). El objeto Data contiene las features de nodos (valores de la IMF) y los enlaces
    del grafo (edge_index).

    Parameters
    ----------
    archivo_imfs : str
        Ruta al archivo parquet con las IMFs (debe contener columnas IMF_1, IMF_2, etc.).
    id_imf : str
        Identificador de la IMF a transformar (ej: "IMF_1", "IMF_2", "Residuo").

    Returns
    -------
    Data
        Objeto Data de PyTorch Geometric con el grafo HVG de la IMF.

    Examples
    --------
    >>> from pathlib import Path
    >>> proyecto_root = Path(__file__).parent.parent.parent.parent.parent
    >>> archivo_imfs = proyecto_root / "data" / "16dic25" / "msci_world_imfs.parquet"
    >>> grafo = obtener_grafo_hvg_imf(str(archivo_imfs), "IMF_1")
    >>> print(f"Nodos: {grafo.num_nodes}, Enlaces: {grafo.num_edges}")
    """
    # Cargar datos de IMFs
    print(f"Cargando IMFs desde: {archivo_imfs}")
    df_imfs = pd.read_parquet(archivo_imfs, engine="pyarrow")
    print(f"Shape del DataFrame: {df_imfs.shape}")
    print(f"Columnas disponibles: {list(df_imfs.columns)}")

    # Verificar que existe la IMF especificada
    if id_imf not in df_imfs.columns:
        raise ValueError(
            f"La IMF '{id_imf}' no existe en el DataFrame. "
            f"Columnas disponibles: {list(df_imfs.columns)}"
        )

    # Extraer la IMF seleccionada
    imf_valores = np.array(df_imfs[id_imf].values)
    print(f"\nIMF seleccionada: {id_imf}")
    print(f"Shape de {id_imf}: {imf_valores.shape}")
    print(
        f"Valores - Min: {np.min(imf_valores):.4f}, Max: {np.max(imf_valores):.4f}, "
        f"Mean: {np.mean(imf_valores):.4f}"
    )

    # Construir el grafo HVG (Horizontal Visibility Graph)
    print("\nConstruyendo grafo HVG...")
    hvg = HorizontalVG(directed="left_to_right")
    grafo_hvg = hvg.build(imf_valores)

    # Obtener nodos y enlaces
    nodos = np.arange(len(imf_valores))  # Los nodos son los índices temporales
    enlaces = np.array(grafo_hvg.edges)  # Los enlaces son las conexiones de visibilidad

    print(f"Número de nodos: {len(nodos)}")
    print(f"Número de enlaces: {len(enlaces)}")

    # Convertir el grafo HVG a un objeto Data de PyTorch Geometric
    print("Convirtiendo a objeto Data de PyTorch Geometric...")
    # Convertir enlaces a edge_index (formato [2, num_edges])
    edge_index = torch.tensor(enlaces.T, dtype=torch.long)

    # Crear features de nodos usando los valores de la serie temporal
    # Cada nodo tiene como feature su valor en la serie temporal
    node_features = torch.tensor(imf_valores, dtype=torch.float).unsqueeze(1)

    # Crear el objeto Data
    data = Data(x=node_features, edge_index=edge_index)

    print(f"Objeto Data creado:")
    print(f"  - Número de nodos: {data.num_nodes}")
    print(f"  - Número de enlaces: {data.num_edges}")
    if data.x is not None:
        print(f"  - Features de nodos shape: {data.x.shape}")
    if data.edge_index is not None:
        print(f"  - Edge index shape: {data.edge_index.shape}")

    return data


def obtener_grafo_nvg_imf(
    archivo_imfs: str,
    id_imf: str,
) -> Data:
    """
    Transforma una IMF a grafo NVG y retorna un objeto Data de PyTorch Geometric.

    Carga una IMF específica desde un archivo parquet y construye el grafo Natural Visibility
    Graph (NVG). El objeto Data contiene las features de nodos (valores de la IMF) y los enlaces
    del grafo (edge_index).

    Parameters
    ----------
    archivo_imfs : str
        Ruta al archivo parquet con las IMFs (debe contener columnas IMF_1, IMF_2, etc.).
    id_imf : str
        Identificador de la IMF a transformar (ej: "IMF_1", "IMF_2", "Residuo").

    Returns
    -------
    Data
        Objeto Data de PyTorch Geometric con el grafo NVG de la IMF.

    Examples
    --------
    >>> from pathlib import Path
    >>> proyecto_root = Path(__file__).parent.parent.parent.parent.parent
    >>> archivo_imfs = proyecto_root / "data" / "16dic25" / "msci_world_imfs.parquet"
    >>> grafo = obtener_grafo_nvg_imf(str(archivo_imfs), "IMF_1")
    >>> print(f"Nodos: {grafo.num_nodes}, Enlaces: {grafo.num_edges}")
    """
    # Cargar datos de IMFs
    print(f"Cargando IMFs desde: {archivo_imfs}")
    df_imfs = pd.read_parquet(archivo_imfs, engine="pyarrow")
    print(f"Shape del DataFrame: {df_imfs.shape}")
    print(f"Columnas disponibles: {list(df_imfs.columns)}")

    # Verificar que existe la IMF especificada
    if id_imf not in df_imfs.columns:
        raise ValueError(
            f"La IMF '{id_imf}' no existe en el DataFrame. "
            f"Columnas disponibles: {list(df_imfs.columns)}"
        )

    # Extraer la IMF seleccionada
    imf_valores = np.array(df_imfs[id_imf].values)
    print(f"\nIMF seleccionada: {id_imf}")
    print(f"Shape de {id_imf}: {imf_valores.shape}")
    print(
        f"Valores - Min: {np.min(imf_valores):.4f}, Max: {np.max(imf_valores):.4f}, "
        f"Mean: {np.mean(imf_valores):.4f}"
    )

    # Construir el grafo NVG (Natural Visibility Graph)
    print("\nConstruyendo grafo NVG...")
    nvg = NaturalVG(directed="left_to_right")
    grafo_nvg = nvg.build(imf_valores)

    # Obtener nodos y enlaces
    nodos = np.arange(len(imf_valores))  # Los nodos son los índices temporales
    enlaces = np.array(grafo_nvg.edges)  # Los enlaces son las conexiones de visibilidad

    print(f"Número de nodos: {len(nodos)}")
    print(f"Número de enlaces: {len(enlaces)}")

    # Convertir el grafo NVG a un objeto Data de PyTorch Geometric
    print("Convirtiendo a objeto Data de PyTorch Geometric...")
    # Convertir enlaces a edge_index (formato [2, num_edges])
    edge_index = torch.tensor(enlaces.T, dtype=torch.long)

    # Crear features de nodos usando los valores de la serie temporal
    # Cada nodo tiene como feature su valor en la serie temporal
    node_features = torch.tensor(imf_valores, dtype=torch.float).unsqueeze(1)

    # Crear el objeto Data
    data = Data(x=node_features, edge_index=edge_index)

    print(f"Objeto Data creado:")
    print(f"  - Número de nodos: {data.num_nodes}")
    print(f"  - Número de enlaces: {data.num_edges}")
    if data.x is not None:
        print(f"  - Features de nodos shape: {data.x.shape}")
    if data.edge_index is not None:
        print(f"  - Edge index shape: {data.edge_index.shape}")

    return data


def calcular_informacion_mutual(serie: np.ndarray, tau_max: int = 50) -> tuple:
    """
    Calcula la información mutua entre la serie y su versión desplazada.

    Parameters
    ----------
    serie : np.ndarray
        Serie temporal unidimensional.
    tau_max : int, optional
        Máximo valor de tau a evaluar. Por defecto es 50.

    Returns
    -------
    tuple
        Tupla con (taus, valores_mi) donde taus son los valores de tau evaluados
        y valores_mi son los valores de información mutua correspondientes.
    """
    taus = np.arange(1, min(tau_max + 1, len(serie) // 2))
    valores_mi = []

    # Discretizar la serie para el cálculo de MI
    # Usar histograma para discretizar
    n_bins = int(np.sqrt(len(serie)))
    serie_discreta = np.digitize(serie, bins=np.linspace(serie.min(), serie.max(), n_bins))

    for tau in taus:
        serie_desplazada = np.roll(serie_discreta, -tau)
        # Asegurar que ambas series tengan la misma longitud
        serie_original = serie_discreta[:-tau] if tau > 0 else serie_discreta
        serie_desplazada = serie_desplazada[:-tau] if tau > 0 else serie_desplazada

        mi = mutual_info_score(serie_original, serie_desplazada)
        valores_mi.append(mi)

    return taus, np.array(valores_mi)


def seleccionar_tau(serie: np.ndarray, tau_max: int = 50) -> int:
    """
    Selecciona el valor óptimo de tau usando el primer mínimo de información mutua.

    Parameters
    ----------
    serie : np.ndarray
        Serie temporal unidimensional.
    tau_max : int, optional
        Máximo valor de tau a evaluar. Por defecto es 50.

    Returns
    -------
    int
        Valor óptimo de tau (delay).
    """
    taus, valores_mi = calcular_informacion_mutual(serie, tau_max)

    # Buscar el primer mínimo local
    # Si no hay mínimo claro, usar el primer valor donde la MI disminuye significativamente
    if len(valores_mi) > 2:
        # Buscar mínimos locales
        minimos = []
        for i in range(1, len(valores_mi) - 1):
            if valores_mi[i] < valores_mi[i - 1] and valores_mi[i] < valores_mi[i + 1]:
                minimos.append(i)

        if minimos:
            tau_optimo = taus[minimos[0]]
        else:
            # Si no hay mínimo claro, usar el primer valor donde la derivada cambia de signo
            derivada = np.diff(valores_mi)
            cambios = np.where(derivada > 0)[0]
            if len(cambios) > 0:
                tau_optimo = taus[cambios[0] + 1] if cambios[0] + 1 < len(taus) else taus[1]
            else:
                tau_optimo = taus[1]  # Valor por defecto
    else:
        tau_optimo = taus[0] if len(taus) > 0 else 1

    return tau_optimo


def construir_espacio_embedding(serie: np.ndarray, dim: int, tau: int) -> np.ndarray:
    """
    Construye el espacio de embedding usando el método de delay embedding.

    Parameters
    ----------
    serie : np.ndarray
        Serie temporal unidimensional.
    dim : int
        Dimensión del embedding.
    tau : int
        Delay (retraso) para el embedding.

    Returns
    -------
    np.ndarray
        Matriz de embedding de forma (N - (dim-1)*tau, dim) donde N es la longitud
        de la serie original.
    """
    n = len(serie)
    m = n - (dim - 1) * tau

    if m <= 0:
        raise ValueError(f"La combinación de dim={dim} y tau={tau} resulta en m={m} <= 0")

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
    Calcula la dimensión óptima usando el método de False Nearest Neighbors (FNN).

    Parameters
    ----------
    serie : np.ndarray
        Serie temporal unidimensional.
    tau : int
        Delay (retraso) ya seleccionado.
    dim_max : int, optional
        Dimensión máxima a evaluar. Por defecto es 10.
    umbral : float, optional
        Umbral para considerar un vecino como "false". Por defecto es 10.0.
    ratio_min : float, optional
        Ratio mínimo de FNN para considerar que se ha alcanzado la dimensión óptima.
        Por defecto es 0.1 (10%).

    Returns
    -------
    int
        Dimensión óptima del embedding.
    """
    ratios_fnn = []

    for dim in range(1, dim_max + 1):
        try:
            # Construir embedding en dimensión dim
            embedding_dim = construir_espacio_embedding(serie, dim, tau)

            if dim == 1:
                ratios_fnn.append(1.0)  # En dim=1, todos son "false" por definición
                continue

            # Construir embedding en dimensión dim+1
            embedding_dim_plus = construir_espacio_embedding(serie, dim + 1, tau)

            # Construir KDTree para búsqueda de vecinos
            tree = KDTree(embedding_dim)

            # Para cada punto, encontrar el vecino más cercano
            n_puntos = len(embedding_dim)
            falsos_vecinos = 0

            for i in range(n_puntos):
                # Encontrar el vecino más cercano en dimensión dim
                # k=2 para obtener el punto mismo y su vecino más cercano
                distancias, indices = tree.query(embedding_dim[i], k=2)

                # Asegurar que sean arrays
                if not isinstance(distancias, np.ndarray):
                    distancias = np.array([distancias])
                if not isinstance(indices, np.ndarray):
                    indices = np.array([indices])

                # Encontrar el vecino más cercano (excluyendo el punto mismo)
                if len(distancias) > 1:
                    if indices[0] == i:
                        vecino_idx = int(indices[1])
                        dist_dim = float(distancias[1])
                    else:
                        vecino_idx = int(indices[0])
                        dist_dim = float(distancias[0])
                else:
                    continue

                # Calcular distancia en dimensión dim+1
                if vecino_idx < len(embedding_dim_plus) and i < len(embedding_dim_plus):
                    dist_dim_plus = np.linalg.norm(
                        embedding_dim_plus[i] - embedding_dim_plus[vecino_idx]
                    )

                    # Verificar si es un false neighbor
                    if dist_dim > 1e-10:  # Evitar división por cero
                        ratio = (
                            abs(embedding_dim_plus[i, -1] - embedding_dim_plus[vecino_idx, -1])
                            / dist_dim
                        )
                        if ratio > umbral:
                            falsos_vecinos += 1

            ratio_fnn = falsos_vecinos / n_puntos if n_puntos > 0 else 1.0
            ratios_fnn.append(ratio_fnn)

            # Si el ratio es muy bajo, probablemente hemos encontrado la dimensión óptima
            if ratio_fnn < ratio_min:
                return dim

        except Exception as e:
            print(f"Error al calcular FNN para dim={dim}: {e}")
            ratios_fnn.append(1.0)

    # Si no se encontró un mínimo claro, usar la dimensión con menor ratio
    if ratios_fnn:
        dim_optima = int(np.argmin(ratios_fnn)) + 1
        return dim_optima

    return 2  # Valor por defecto


def calcular_matriz_recurrencia(
    embedding: np.ndarray,
    umbral: Optional[float] = None,
    umbral_percentil: float = 10.0,
    random_state: Optional[int] = None,
) -> tuple:
    """
    Calcula la matriz de recurrencia a partir del espacio de embedding.

    Parameters
    ----------
    embedding : np.ndarray
        Matriz de embedding de forma (N, dim).
    umbral : float, optional
        Umbral absoluto para la distancia. Si es None, se usa umbral_percentil.
    umbral_percentil : float, optional
        Percentil para determinar el umbral de distancia. Por defecto es 10.0.
    random_state : int, optional
        Semilla para la generación de números aleatorios. Por defecto es None.

    Returns
    -------
    tuple
        Tupla con (matriz_recurrencia, umbral_utilizado) donde matriz_recurrencia es
        una matriz binaria de forma (N, N) y umbral_utilizado es el umbral de distancia
        utilizado (float).
    """
    # Configurar random state para reproducibilidad
    if random_state is not None:
        np.random.seed(random_state)

    n = len(embedding)

    # Advertencia para matrices muy grandes
    if n > 5000:
        print(
            f"Advertencia: La matriz de recurrencia será muy grande ({n}x{n}). "
            f"Esto puede tardar varios minutos."
        )

    # Construir KDTree
    tree = KDTree(embedding)

    # Determinar umbral si no se proporciona
    if umbral is None:
        # Calcular umbral usando una muestra de distancias
        # Tomar una muestra aleatoria de puntos para estimar el percentil
        n_muestra = min(1000, n)
        indices_muestra = np.random.choice(n, size=n_muestra, replace=False)
        distancias_muestra = []

        for idx in indices_muestra:
            # Obtener distancias a los k vecinos más cercanos
            distancias, _ = tree.query(embedding[idx], k=min(100, n))
            distancias_muestra.extend(distancias[1:])  # Excluir distancia a sí mismo

        umbral = float(np.percentile(distancias_muestra, umbral_percentil))
        print(f"Umbral calculado (percentil {umbral_percentil}): {umbral:.4f}")
    else:
        # Asegurar que el umbral proporcionado sea float
        umbral = float(umbral)

    # Construir matriz de recurrencia de manera eficiente
    # Inicializar matriz de ceros
    matriz_recurrencia = np.zeros((n, n), dtype=int)

    # Para cada punto, encontrar todos los puntos dentro del umbral
    for i in range(n):
        # Usar query_ball_point para encontrar todos los puntos dentro del radio
        indices_vecinos = tree.query_ball_point(embedding[i], r=umbral, p=2)
        # Convertir a array y eliminar el punto mismo
        indices_vecinos = np.array([idx for idx in indices_vecinos if idx != i])
        if len(indices_vecinos) > 0:
            matriz_recurrencia[i, indices_vecinos] = 1

    print(f"Número de recurrencias encontradas: {np.sum(matriz_recurrencia)}")

    return matriz_recurrencia, umbral


def obtener_grafo_recurrencia_imf(
    archivo_imfs: str,
    id_imf: str,
    tau_max: int = 50,
    dim_max: int = 10,
    umbral_percentil: float = 10.0,
    umbral: Optional[float] = None,
    random_state: Optional[int] = None,
) -> Data:
    """
    Transforma una IMF a grafo de recurrencia y retorna un objeto Data de PyTorch Geometric.

    Carga una IMF específica desde un archivo parquet y construye el grafo de recurrencia
    usando el método de delay embedding. El objeto Data contiene las features de nodos
    (valores del embedding) y los enlaces del grafo (edge_index) basados en la matriz
    de recurrencia.

    Los metadatos del proceso (tau, dimensión de embedding, algoritmo de distancia y
    umbral de recurrencia) se guardan como atributos del objeto Data.

    Parameters
    ----------
    archivo_imfs : str
        Ruta al archivo parquet con las IMFs (debe contener columnas IMF_1, IMF_2, etc.).
    id_imf : str
        Identificador de la IMF a transformar (ej: "IMF_1", "IMF_2", "Residuo").
    tau_max : int, optional
        Máximo valor de tau a evaluar para la selección del delay. Por defecto es 50.
    dim_max : int, optional
        Dimensión máxima a evaluar para el embedding. Por defecto es 10.
    umbral_percentil : float, optional
        Percentil para determinar el umbral de distancia en la matriz de recurrencia.
        Por defecto es 10.0.
    umbral : float, optional
        Umbral absoluto para la distancia. Si se proporciona, se usa este valor en lugar
        de calcularlo usando umbral_percentil. Por defecto es None.
    random_state : int, optional
        Semilla para la generación de números aleatorios. Se usa para garantizar
        reproducibilidad en todos los cálculos que involucran aleatoriedad. Por defecto es None.

    Returns
    -------
    Data
        Objeto Data de PyTorch Geometric con el grafo de recurrencia de la IMF.
        Los metadatos se guardan en los atributos:
        - tau: valor de delay (retraso) utilizado
        - dim_embedding: dimensión del embedding utilizado
        - algoritmo_distancia: algoritmo de distancia utilizado ("euclidean" con KDTree)
        - umbral_recurrencia: umbral de distancia utilizado para la matriz de recurrencia

    Examples
    --------
    >>> from pathlib import Path
    >>> proyecto_root = Path(__file__).parent.parent.parent.parent.parent
    >>> archivo_imfs = proyecto_root / "data" / "16dic25" / "msci_world_imfs.parquet"
    >>> grafo = obtener_grafo_recurrencia_imf(str(archivo_imfs), "IMF_1", random_state=42)
    >>> print(f"Nodos: {grafo.num_nodes}, Enlaces: {grafo.num_edges}")
    >>> print(f"Tau: {grafo.tau}, Dim: {grafo.dim_embedding}")
    """
    # Configurar random state al inicio para garantizar reproducibilidad
    if random_state is not None:
        np.random.seed(random_state)

    # Cargar datos de IMFs
    print(f"Cargando IMFs desde: {archivo_imfs}")
    df_imfs = pd.read_parquet(archivo_imfs, engine="pyarrow")
    print(f"Shape del DataFrame: {df_imfs.shape}")
    print(f"Columnas disponibles: {list(df_imfs.columns)}")

    # Verificar que existe la IMF especificada
    if id_imf not in df_imfs.columns:
        raise ValueError(
            f"La IMF '{id_imf}' no existe en el DataFrame. "
            f"Columnas disponibles: {list(df_imfs.columns)}"
        )

    # Extraer la IMF seleccionada
    imf_valores = np.array(df_imfs[id_imf].values)
    print(f"\nIMF seleccionada: {id_imf}")
    print(f"Shape de {id_imf}: {imf_valores.shape}")
    print(
        f"Valores - Min: {np.min(imf_valores):.4f}, Max: {np.max(imf_valores):.4f}, "
        f"Mean: {np.mean(imf_valores):.4f}"
    )

    # Paso 1: Seleccionar tau (delay) usando información mutua
    print("\nSeleccionando tau (delay) usando información mutua...")
    tau_optimo = seleccionar_tau(imf_valores, tau_max=tau_max)
    print(f"Tau óptimo seleccionado: {tau_optimo}")

    # Paso 2: Seleccionar dim usando False Nearest Neighbors
    print("Seleccionando dim (dimensión de embedding) usando False Nearest Neighbors...")
    dim_optima = calcular_false_nearest_neighbors(
        imf_valores, tau=tau_optimo, dim_max=dim_max
    )
    print(f"Dimensión óptima seleccionada: {dim_optima}")

    # Paso 3: Construir espacio de embedding
    print(
        f"Construyendo espacio de embedding con dim={dim_optima} y tau={tau_optimo}..."
    )
    embedding = construir_espacio_embedding(imf_valores, dim=dim_optima, tau=tau_optimo)
    print(f"Shape del embedding: {embedding.shape}")

    # Paso 4: Calcular matriz de recurrencia
    print("Calculando matriz de recurrencia...")
    matriz_recurrencia, umbral_utilizado = calcular_matriz_recurrencia(
        embedding, umbral=umbral, umbral_percentil=umbral_percentil, random_state=random_state
    )
    print(f"Shape de la matriz de recurrencia: {matriz_recurrencia.shape}")
    print(f"Número de recurrencias (enlaces): {np.sum(matriz_recurrencia)}")

    # Paso 5: Convertir matriz de recurrencia a formato edge_index para PyTorch Geometric
    print("Convirtiendo matriz de recurrencia a formato de grafo...")

    # Obtener índices de las aristas (donde matriz_recurrencia == 1)
    edge_indices = np.where(matriz_recurrencia == 1)

    # Crear edge_index en formato [2, num_edges]
    edge_index = np.array([edge_indices[0], edge_indices[1]])
    edge_index_torch = torch.tensor(edge_index, dtype=torch.long)

    print(f"Número de nodos: {len(embedding)}")
    print(f"Número de enlaces: {edge_index_torch.shape[1]}")

    # Crear features de nodos usando el embedding completo de cada nodo
    node_features = torch.tensor(embedding, dtype=torch.float)

    # Crear el objeto Data de PyTorch Geometric
    data = Data(x=node_features, edge_index=edge_index_torch)

    # Guardar metadatos en el objeto Data
    data.tau = int(tau_optimo)
    data.dim_embedding = int(dim_optima)
    data.algoritmo_distancia = "euclidean"  # KDTree con p=2 usa distancia euclidiana
    data.umbral_recurrencia = umbral_utilizado

    print(f"\nObjeto Data creado:")
    print(f"  - Número de nodos: {data.num_nodes}")
    print(f"  - Número de enlaces: {data.num_edges}")
    if data.x is not None:
        print(f"  - Features de nodos shape: {data.x.shape}")
    if data.edge_index is not None:
        print(f"  - Edge index shape: {data.edge_index.shape}")
    print(f"\nMetadatos guardados:")
    print(f"  - Tau (delay): {data.tau}")
    print(f"  - Dimensión de embedding: {data.dim_embedding}")
    print(f"  - Algoritmo de distancia: {data.algoritmo_distancia}")
    print(f"  - Umbral de recurrencia: {data.umbral_recurrencia:.4f}")

    return data


def obtener_grafos_all_imf(
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
    Genera todos los tipos de grafos para todas las IMFs en el dataframe.

    Para cada IMF en el dataframe, calcula y guarda los tres tipos de grafos implementados:
    - Horizontal Visibility Graph (HVG)
    - Natural Visibility Graph (NVG)
    - Grafo de recurrencia

    Los grafos se guardan en la estructura de carpetas:
    - {carpeta_salida_base}/hvg/{id_imf}/grafo_hvg_{id_imf}
    - {carpeta_salida_base}/nvg/{id_imf}/grafo_nvg_{id_imf}
    - {carpeta_salida_base}/recurrencia/{id_imf}/grafo_recurrencia_{id_imf}

    Parameters
    ----------
    df_imfs : pd.DataFrame o str
        DataFrame con las IMFs o ruta al archivo parquet que contiene las IMFs.
        Debe contener columnas con nombres como "IMF_1", "IMF_2", etc.
    carpeta_salida_base : str o Path
        Ruta base donde se guardarán los grafos. Se crearán subcarpetas para cada
        tipo de grafo y cada IMF.
    tau_max : int, optional
        Máximo valor de tau a evaluar para la selección del delay en grafos de recurrencia.
        Por defecto es 50.
    dim_max : int, optional
        Dimensión máxima a evaluar para el embedding en grafos de recurrencia.
        Por defecto es 10.
    umbral_percentil : float, optional
        Percentil para determinar el umbral de distancia en la matriz de recurrencia.
        Por defecto es 10.0.
    umbral : float, optional
        Umbral absoluto para la distancia en grafos de recurrencia. Si se proporciona,
        se usa este valor en lugar de calcularlo usando umbral_percentil. Por defecto es None.
    random_state : int, optional
        Semilla para la generación de números aleatorios. Se usa para garantizar
        reproducibilidad en los cálculos de grafos de recurrencia. Por defecto es None.
    columnas_imf : list, optional
        Lista de nombres de columnas a procesar. Si es None, se procesan todas las columnas
        que empiezan con "IMF_" o son "Residuo". Por defecto es None.

    Returns
    -------
    dict
        Diccionario con información sobre los grafos generados. Las claves son los IDs de las
        IMFs y los valores son diccionarios con información sobre cada tipo de grafo generado.

    Examples
    --------
    >>> from pathlib import Path
    >>> proyecto_root = Path(__file__).parent.parent.parent.parent.parent
    >>> archivo_imfs = proyecto_root / "data" / "16dic25" / "msci_world_imfs.parquet"
    >>> carpeta_salida = proyecto_root / "data" / "16dic25" / "grafos"
    >>> resultados = obtener_grafos_all_imf(
    ...     df_imfs=str(archivo_imfs),
    ...     carpeta_salida_base=str(carpeta_salida),
    ...     random_state=42
    ... )
    >>> print(f"Grafos generados para {len(resultados)} IMFs")
    """
    # Cargar dataframe si se proporciona una ruta
    if isinstance(df_imfs, str):
        archivo_imfs = df_imfs
        print(f"Cargando IMFs desde: {archivo_imfs}")
        if not Path(archivo_imfs).exists():
            raise FileNotFoundError(
                f"El archivo {archivo_imfs} no existe. "
                "Asegúrate de que el archivo esté en la ubicación correcta."
            )
        df_imfs = pd.read_parquet(archivo_imfs, engine="pyarrow")
    else:
        archivo_imfs = None

    # Convertir carpeta_salida_base a Path
    carpeta_salida_base = Path(carpeta_salida_base)
    carpeta_salida_base.mkdir(parents=True, exist_ok=True)

    # Determinar qué columnas procesar
    if columnas_imf is None:
        # Procesar todas las columnas que empiezan con "IMF_" o son "Residuo"
        columnas_imf = [
            col
            for col in df_imfs.columns
            if col.startswith("IMF_") or col == "Residuo"
        ]

    if not columnas_imf:
        raise ValueError(
            "No se encontraron columnas de IMFs para procesar. "
            "Asegúrate de que el dataframe contenga columnas con nombres como "
            '"IMF_1", "IMF_2", etc., o "Residuo".'
        )

    print(f"\nColumnas de IMFs a procesar: {columnas_imf}")
    print(f"Total de IMFs: {len(columnas_imf)}")

    # Si tenemos archivo_imfs, usarlo para las funciones de transformación
    if archivo_imfs is None:
        # Si no tenemos archivo, necesitamos guardar temporalmente el dataframe
        # o modificar las funciones para aceptar dataframes directamente
        # Por ahora, asumimos que siempre se proporciona una ruta
        raise ValueError(
            "Se debe proporcionar una ruta al archivo parquet, no un DataFrame directamente. "
            "Las funciones de transformación requieren una ruta al archivo."
        )

    # Diccionario para almacenar resultados
    resultados = {}

    # Procesar cada IMF
    for idx, id_imf in enumerate(columnas_imf, 1):
        print("\n" + "=" * 80)
        print(f"PROCESANDO IMF {idx}/{len(columnas_imf)}: {id_imf}")
        print("=" * 80)

        resultados_imf = {}

        # 1. Grafo HVG
        try:
            print(f"\n--- Generando grafo HVG para {id_imf} ---")
            carpeta_hvg = carpeta_salida_base / "hvg" / id_imf.lower()
            carpeta_hvg.mkdir(parents=True, exist_ok=True)
            archivo_salida_hvg = str(carpeta_hvg / f"grafo_hvg_{id_imf.lower()}")

            grafo_hvg = obtener_grafo_hvg_imf(archivo_imfs=archivo_imfs, id_imf=id_imf)
            guardar_grafo_data(
                data=grafo_hvg, archivo_salida=archivo_salida_hvg, id_imf=id_imf
            )

            resultados_imf["hvg"] = {
                "archivo": archivo_salida_hvg,
                "num_nodes": grafo_hvg.num_nodes,
                "num_edges": grafo_hvg.num_edges,
                "exito": True,
            }
            print(f"✓ Grafo HVG generado exitosamente para {id_imf}")

        except Exception as e:
            print(f"✗ Error al generar grafo HVG para {id_imf}: {e}")
            resultados_imf["hvg"] = {"exito": False, "error": str(e)}

        # 2. Grafo NVG
        try:
            print(f"\n--- Generando grafo NVG para {id_imf} ---")
            carpeta_nvg = carpeta_salida_base / "nvg" / id_imf.lower()
            carpeta_nvg.mkdir(parents=True, exist_ok=True)
            archivo_salida_nvg = str(carpeta_nvg / f"grafo_nvg_{id_imf.lower()}")

            grafo_nvg = obtener_grafo_nvg_imf(archivo_imfs=archivo_imfs, id_imf=id_imf)
            guardar_grafo_data(
                data=grafo_nvg, archivo_salida=archivo_salida_nvg, id_imf=id_imf
            )

            resultados_imf["nvg"] = {
                "archivo": archivo_salida_nvg,
                "num_nodes": grafo_nvg.num_nodes,
                "num_edges": grafo_nvg.num_edges,
                "exito": True,
            }
            print(f"✓ Grafo NVG generado exitosamente para {id_imf}")

        except Exception as e:
            print(f"✗ Error al generar grafo NVG para {id_imf}: {e}")
            resultados_imf["nvg"] = {"exito": False, "error": str(e)}

        # 3. Grafo de recurrencia
        try:
            print(f"\n--- Generando grafo de recurrencia para {id_imf} ---")
            carpeta_recurrencia = (
                carpeta_salida_base / "recurrencia" / id_imf.lower()
            )
            carpeta_recurrencia.mkdir(parents=True, exist_ok=True)
            archivo_salida_recurrencia = str(
                carpeta_recurrencia / f"grafo_recurrencia_{id_imf.lower()}"
            )

            grafo_recurrencia = obtener_grafo_recurrencia_imf(
                archivo_imfs=archivo_imfs,
                id_imf=id_imf,
                tau_max=tau_max,
                dim_max=dim_max,
                umbral_percentil=umbral_percentil,
                umbral=umbral,
                random_state=random_state,
            )
            guardar_grafo_data(
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
            print(f"✓ Grafo de recurrencia generado exitosamente para {id_imf}")

        except Exception as e:
            print(f"✗ Error al generar grafo de recurrencia para {id_imf}: {e}")
            resultados_imf["recurrencia"] = {"exito": False, "error": str(e)}

        resultados[id_imf] = resultados_imf

    # Resumen final
    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
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

    print(f"Total de IMFs procesadas: {total_imfs}")
    print(f"IMFs con todos los grafos generados exitosamente: {total_exitosos}")

    for id_imf, imf_resultados in resultados.items():
        print(f"\n{id_imf}:")
        for tipo_grafo, info_grafo in imf_resultados.items():
            if isinstance(info_grafo, dict) and info_grafo.get("exito", False):
                print(
                    f"  ✓ {tipo_grafo.upper()}: "
                    f"{info_grafo.get('num_nodes', 'N/A')} nodos, "
                    f"{info_grafo.get('num_edges', 'N/A')} enlaces"
                )
            else:
                error = info_grafo.get("error", "Error desconocido") if isinstance(info_grafo, dict) else "Error desconocido"
                print(f"  ✗ {tipo_grafo.upper()}: {error}")

    return resultados

