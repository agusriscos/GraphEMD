"""
Script para transformar una IMF a grafo HVG como objeto Data de PyTorch Geometric.

Este script carga una IMF desde un archivo parquet y la transforma a grafo Horizontal Visibility
Graph (HVG) como un objeto Data de PyTorch Geometric.
"""

from pathlib import Path

from GraphEMD.data.graph_imf_transform_utils import obtener_grafo_hvg_imf
from GraphEMD.data.python_utils import guardar_grafo_data


if __name__ == "__main__":
    
    # Variables de configuración para probar el método
    proyecto_root = Path(__file__).parent.parent.parent

    archivo_imfs = str(proyecto_root / "data" / "16dic25" / "msci_world_imfs.parquet")
    id_imf = "IMF_1"

    carpeta_salida = proyecto_root / "data" / "16dic25" / "grafos" / "hvg" / id_imf.lower()
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    archivo_salida = str(carpeta_salida / f"grafo_hvg_{id_imf.lower()}")

    # Verificar que el archivo existe
    if not Path(archivo_imfs).exists():
        raise FileNotFoundError(
            f"El archivo {archivo_imfs} no existe. "
            "Asegúrate de que el archivo esté en la ubicación correcta."
        )

    # Probar el método con IMF_1
    print("=" * 60)
    print("TRANSFORMACIÓN DE IMF A GRAFO HVG")
    print("=" * 60)

    # Construir el grafo
    grafo = obtener_grafo_hvg_imf(
        archivo_imfs=archivo_imfs,
        id_imf=id_imf,
    )

    # Guardar el grafo
    guardar_grafo_data(
        data=grafo,
        archivo_salida=archivo_salida,
        id_imf=id_imf,
    )

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Grafo HVG creado exitosamente para {id_imf}")
    print(f"  - Nodos: {grafo.num_nodes}")
    print(f"  - Enlaces: {grafo.num_edges}")
    if grafo.x is not None:
        print(f"  - Features de nodos: {grafo.x.shape}")

