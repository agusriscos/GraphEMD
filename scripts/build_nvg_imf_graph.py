"""
Script to transform an IMF into an NVG graph as a PyTorch Geometric Data object.

This script loads an IMF from a parquet file and transforms it into a Natural Visibility
Graph (NVG) as a PyTorch Geometric Data object.
"""

from pathlib import Path

from GraphEMD.data.graph_imf_transform_utils import build_nvg_imf_graph
from GraphEMD.data.python_utils import save_graph_data


if __name__ == "__main__":
    
    # Configuration variables for testing the method
    proyecto_root = Path(__file__).parent.parent

    archivo_imfs = str(proyecto_root / "data" / "16dic25" / "msci_world_imfs.parquet")
    id_imf = "IMF_3"

    carpeta_salida = proyecto_root / "data" / "16dic25" / "grafos" / "nvg" / id_imf.lower()
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    archivo_salida = str(carpeta_salida / f"grafo_nvg_{id_imf.lower()}")

    # Verify that the file exists
    if not Path(archivo_imfs).exists():
        raise FileNotFoundError(
            f"File {archivo_imfs} does not exist. "
            "Make sure the file is in the correct location."
        )

    # Test the method with IMF_1
    print("=" * 60)
    print("IMF TO NVG GRAPH TRANSFORMATION")
    print("=" * 60)

    # Build the graph
    grafo = build_nvg_imf_graph(
        archivo_imfs=archivo_imfs,
        id_imf=id_imf,
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
    print(f"NVG graph successfully created for {id_imf}")
    print(f"  - Nodes: {grafo.num_nodes}")
    print(f"  - Edges: {grafo.num_edges}")
    if grafo.x is not None:
        print(f"  - Node features: {grafo.x.shape}")
