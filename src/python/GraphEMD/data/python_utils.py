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


def save_graph_data(
    data: Data,
    archivo_salida: str,
    id_imf: str,
) -> None:
    """
    Save a PyTorch Geometric Data object in parquet and torch formats.

    Saves the Data object by splitting node features and edge_index into
    parquet files, metadata as CSV, and a fully serialized version in torch format.

    If the Data object represents a recurrence graph (detected by the presence
    of the tau, dim_embedding, and algoritmo_distancia attributes), the parameters
    used to compute the recurrence matrix are also saved.

    Parameters
    ----------
    data : Data
        PyTorch Geometric Data object to save.
    archivo_salida : str
        Base path where files are saved (without extension).
    id_imf : str
        IMF identifier for metadata.

    Examples
    --------
    >>> from torch_geometric.data import Data
    >>> import torch
    >>> grafo = Data(x=torch.randn(10, 1), edge_index=torch.randint(0, 10, (2, 20)))
    >>> save_graph_data(grafo, "data/grafos/nvg/grafo_nvg_imf_1", "IMF_1")
    """
    # Verify that required components exist
    if data.x is None:
        raise ValueError("Data object has no node features (x)")
    if data.edge_index is None:
        raise ValueError("Data object has no edge_index")

    # Create output directory if it does not exist
    carpeta_salida = Path(archivo_salida).parent
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving graph to: {archivo_salida}")

    # Convert tensors to numpy arrays
    node_features_np = data.x.cpu().numpy()
    edge_index_np = data.edge_index.cpu().numpy().T

    # Save node features
    num_features = int(node_features_np.shape[1])
    columnas_features = pd.Index([f"feature_{i}" for i in range(num_features)])
    df_node_features = pd.DataFrame(
        data=node_features_np,
        columns=columnas_features
    )

    # Save edge_index
    df_edges = pd.DataFrame(
        data=edge_index_np,
        columns=["source", "target"]  # type: ignore[arg-type]
    )

    # Save node features
    archivo_features = str(Path(archivo_salida).with_suffix('')) + "_features.parquet"
    df_node_features.to_parquet(archivo_features, engine="pyarrow", index=False)

    # Save edge_index
    archivo_edges = str(Path(archivo_salida).with_suffix('')) + "_edges.parquet"
    df_edges.to_parquet(archivo_edges, engine="pyarrow", index=False)

    # Detect whether this is a recurrence graph
    # A recurrence graph has the attributes: tau, dim_embedding, and algoritmo_distancia
    es_grafo_recurrencia = (
        hasattr(data, "tau")
        and hasattr(data, "dim_embedding")
        and hasattr(data, "algoritmo_distancia")
    )

    # Save metadata
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

    # If this is a recurrence graph, add recurrence matrix parameters
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

    # Also save a serialized version of the full Data object
    # using torch.save so it can be loaded directly later
    archivo_torch = str(Path(archivo_salida).with_suffix('')) + ".pt"
    torch.save(data, archivo_torch)

    print(f"✓ Graph saved successfully:")
    print(f"  - Node features: {archivo_features}")
    print(f"  - Edge index: {archivo_edges}")
    print(f"  - Metadata: {archivo_metadatos}")
    print(f"  - Full Data object (torch): {archivo_torch}")

    # File sizes
    tamaño_total = 0
    for archivo in [archivo_features, archivo_edges, archivo_metadatos, archivo_torch]:
        if Path(archivo).exists():
            tamaño = Path(archivo).stat().st_size / (1024 * 1024)
            tamaño_total += tamaño
            print(f"  - {Path(archivo).name}: {tamaño:.2f} MB")
    print(f"  - Total size: {tamaño_total:.2f} MB")
