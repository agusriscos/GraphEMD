"""
Script to transform all IMFs into all implemented graph types.

This script loads a dataframe with IMFs and, for each IMF, computes all implemented
graph representations: Horizontal Visibility Graph (HVG), Natural Visibility Graph (NVG),
and recurrence graph.
"""

from pathlib import Path

from GraphEMD.data.graph_imf_transform_utils import build_all_imf_graphs


if __name__ == "__main__":

    # Configuration variables
    proyecto_root = Path(__file__).parent.parent

    archivo_imfs = str(proyecto_root / "data" / "16dic25" / "msci_world_imfs.parquet")
    carpeta_salida = proyecto_root / "data" / "16dic25" / "grafos"

    # Parameters for recurrence graphs
    random_state = 42
    tau_max = 50
    dim_max = 10
    umbral_percentil = 10.0

    # Verify that the file exists
    if not Path(archivo_imfs).exists():
        raise FileNotFoundError(
            f"File {archivo_imfs} does not exist. "
            "Make sure the file is in the correct location."
        )

    # Generate all graphs for all IMFs
    print("=" * 80)
    print("GRAPH GENERATION FOR ALL IMFs")
    print("=" * 80)

    resultados = build_all_imf_graphs(
        df_imfs=archivo_imfs,
        carpeta_salida_base=str(carpeta_salida),
        tau_max=tau_max,
        dim_max=dim_max,
        umbral_percentil=umbral_percentil,
        random_state=random_state,
    )

    print("\n" + "=" * 80)
    print("PROCESS COMPLETED")
    print("=" * 80)
