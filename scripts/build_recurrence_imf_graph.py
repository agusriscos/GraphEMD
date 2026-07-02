"""
Script to transform an IMF into a recurrence graph as a PyTorch Geometric Data object.

This script loads an IMF from a parquet file and transforms it into a recurrence graph
using delay embedding and a recurrence matrix as a PyTorch Geometric Data object.
"""

from pathlib import Path

from GraphEMD.data.graph_imf_transform_utils import build_recurrence_imf_graph
from GraphEMD.data.python_utils import save_graph_data


if __name__ == "__main__":

    # Configuration variables for testing the method
    proyecto_root = Path(__file__).parent.parent

    archivo_imfs = str(proyecto_root / "data" / "16dic25" / "msci_world_imfs.parquet")
    id_imf = "IMF_8"
    
    # Random state for reproducibility
    random_state = 42

    carpeta_salida = (
        proyecto_root / "data" / "16dic25" / "grafos" / "recurrencia" / id_imf.lower()
    )
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    archivo_salida = str(carpeta_salida / f"grafo_recurrencia_{id_imf.lower()}")

    # Verify that the file exists
    if not Path(archivo_imfs).exists():
        raise FileNotFoundError(
            f"File {archivo_imfs} does not exist. "
            "Make sure the file is in the correct location."
        )

    # Test the method with IMF_1
    print("=" * 60)
    print("IMF TO RECURRENCE GRAPH TRANSFORMATION")
    print("=" * 60)

    # Build the graph
    grafo = build_recurrence_imf_graph(
        archivo_imfs=archivo_imfs,
        id_imf=id_imf,
        random_state=random_state,
    )

    # Save the graph
    save_graph_data(
        data=grafo,
        archivo_salida=archivo_salida,
        id_imf=id_imf,
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Recurrence graph successfully created for {id_imf}")
    print(f"  - Nodes: {grafo.num_nodes}")
    print(f"  - Edges: {grafo.num_edges}")
    if grafo.x is not None:
        print(f"  - Node features: {grafo.x.shape}")
    print(f"  - Tau: {grafo.tau}")
    print(f"  - Embedding dimension: {grafo.dim_embedding}")
    print(f"  - Distance algorithm: {grafo.algoritmo_distancia}")
    print(f"  - Recurrence threshold: {grafo.umbral_recurrencia:.4f}")
