from enum import Enum
from pathlib import Path
from typing import Hashable

import pandas as pd
import torch
from torch_geometric.data import Data

class DictClass(Enum):

    def colname(self):
        return self.value

    @classmethod
    def values(cls):
        """
        set of values
        :return: (set)
        """
        return [v.value for v in cls._member_map_.values()]

    @classmethod
    def keys(cls):
        """
        set of keys
        :return: (list)
        """
        return [k for k, v in cls._member_map_.items()]

    @classmethod
    def to_dict(cls):
        """
        Class as dict
        :return: (dict)
        """
        return {k: v for k, v in zip(cls.keys(), cls.values())}

    @classmethod
    def get(cls, item: Hashable):
        """
        Get item
        :return: value in class
        """
        return cls.to_dict().get(item)


def guardar_grafo_data(
    data: Data,
    archivo_salida: str,
    id_imf: str,
) -> None:
    """
    Guarda un objeto Data de PyTorch Geometric en formato parquet y torch.

    Guarda el objeto Data separando las features de nodos, edge_index en archivos
    parquet, metadatos en formato CSV, además de una versión serializada completa
    en formato torch.

    Si el objeto Data representa un grafo de recurrencia (detectado por la presencia
    de los atributos tau, dim_embedding y algoritmo_distancia), también se guardan
    los parámetros utilizados en el cálculo de la matriz de recurrencia.

    Parameters
    ----------
    data : Data
        Objeto Data de PyTorch Geometric a guardar.
    archivo_salida : str
        Ruta base donde guardar los archivos (sin extensión).
    id_imf : str
        Identificador de la IMF para los metadatos.

    Examples
    --------
    >>> from torch_geometric.data import Data
    >>> import torch
    >>> grafo = Data(x=torch.randn(10, 1), edge_index=torch.randint(0, 10, (2, 20)))
    >>> guardar_grafo_data(grafo, "data/grafos/nvg/grafo_nvg_imf_1", "IMF_1")
    """
    # Verificar que los componentes existen
    if data.x is None:
        raise ValueError("El objeto Data no tiene features de nodos (x)")
    if data.edge_index is None:
        raise ValueError("El objeto Data no tiene edge_index")

    # Crear directorio de salida si no existe
    carpeta_salida = Path(archivo_salida).parent
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    print(f"\nGuardando grafo en: {archivo_salida}")

    # Convertir tensores a numpy arrays
    node_features_np = data.x.cpu().numpy()
    edge_index_np = data.edge_index.cpu().numpy().T

    # Guardar features de nodos
    num_features = int(node_features_np.shape[1])
    columnas_features = pd.Index([f"feature_{i}" for i in range(num_features)])
    df_node_features = pd.DataFrame(
        data=node_features_np,
        columns=columnas_features
    )

    # Guardar edge_index
    df_edges = pd.DataFrame(
        data=edge_index_np,
        columns=["source", "target"]  # type: ignore[arg-type]
    )

    # Guardar features de nodos
    archivo_features = str(Path(archivo_salida).with_suffix('')) + "_features.parquet"
    df_node_features.to_parquet(archivo_features, engine="pyarrow", index=False)

    # Guardar edge_index
    archivo_edges = str(Path(archivo_salida).with_suffix('')) + "_edges.parquet"
    df_edges.to_parquet(archivo_edges, engine="pyarrow", index=False)

    # Detectar si es un grafo de recurrencia
    # Un grafo de recurrencia tiene los atributos: tau, dim_embedding y algoritmo_distancia
    es_grafo_recurrencia = (
        hasattr(data, "tau")
        and hasattr(data, "dim_embedding")
        and hasattr(data, "algoritmo_distancia")
    )

    # Guardar metadatos
    num_nodes_int = int(data.num_nodes) if data.num_nodes is not None else 0
    num_edges_int = int(data.num_edges) if data.num_edges is not None else 0
    metadatos = {
        "id_imf": [id_imf],
        "num_nodes": [num_nodes_int],
        "num_edges": [num_edges_int],
        "num_features": [num_features],
        "archivo_features": [Path(archivo_features).name],
        "archivo_edges": [Path(archivo_edges).name],
    }

    # Si es un grafo de recurrencia, agregar los parámetros de la matriz de recurrencia
    if es_grafo_recurrencia:
        metadatos["tau"] = [int(data.tau)]
        metadatos["dim_embedding"] = [int(data.dim_embedding)]
        metadatos["algoritmo_distancia"] = [str(data.algoritmo_distancia)]
        if hasattr(data, "umbral_recurrencia"):
            metadatos["umbral_recurrencia"] = [float(data.umbral_recurrencia)]
        else:
            metadatos["umbral_recurrencia"] = [None]

    df_metadatos = pd.DataFrame(metadatos)
    archivo_metadatos = str(Path(archivo_salida).with_suffix('')) + "_metadata.csv"
    df_metadatos.to_csv(archivo_metadatos, index=False)

    # También guardamos una versión serializada del objeto Data completo
    # usando torch.save para poder cargarlo directamente después
    archivo_torch = str(Path(archivo_salida).with_suffix('')) + ".pt"
    torch.save(data, archivo_torch)

    print(f"✓ Grafo guardado exitosamente:")
    print(f"  - Features de nodos: {archivo_features}")
    print(f"  - Edge index: {archivo_edges}")
    print(f"  - Metadatos: {archivo_metadatos}")
    print(f"  - Objeto Data completo (torch): {archivo_torch}")

    # Tamaño de archivos
    tamaño_total = 0
    for archivo in [archivo_features, archivo_edges, archivo_metadatos, archivo_torch]:
        if Path(archivo).exists():
            tamaño = Path(archivo).stat().st_size / (1024 * 1024)
            tamaño_total += tamaño
            print(f"  - {Path(archivo).name}: {tamaño:.2f} MB")
    print(f"  - Tamaño total: {tamaño_total:.2f} MB")
