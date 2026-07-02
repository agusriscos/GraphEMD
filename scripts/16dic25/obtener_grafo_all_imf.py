"""
Script para transformar todas las IMFs a todos los tipos de grafos implementados.

Este script carga un dataframe con IMFs y para cada IMF calcula todas las representaciones
de grafos implementadas: Horizontal Visibility Graph (HVG), Natural Visibility Graph (NVG)
y grafo de recurrencia.
"""

from pathlib import Path

from GraphEMD.data.graph_imf_transform_utils import obtener_grafos_all_imf


if __name__ == "__main__":

    # Variables de configuración
    proyecto_root = Path(__file__).parent.parent.parent

    archivo_imfs = str(proyecto_root / "data" / "16dic25" / "msci_world_imfs.parquet")
    carpeta_salida = proyecto_root / "data" / "16dic25" / "grafos"

    # Parámetros para grafos de recurrencia
    random_state = 42
    tau_max = 50
    dim_max = 10
    umbral_percentil = 10.0

    # Verificar que el archivo existe
    if not Path(archivo_imfs).exists():
        raise FileNotFoundError(
            f"El archivo {archivo_imfs} no existe. "
            "Asegúrate de que el archivo esté en la ubicación correcta."
        )

    # Generar todos los grafos para todas las IMFs
    print("=" * 80)
    print("GENERACIÓN DE GRAFOS PARA TODAS LAS IMFS")
    print("=" * 80)

    resultados = obtener_grafos_all_imf(
        df_imfs=archivo_imfs,
        carpeta_salida_base=str(carpeta_salida),
        tau_max=tau_max,
        dim_max=dim_max,
        umbral_percentil=umbral_percentil,
        random_state=random_state,
    )

    print("\n" + "=" * 80)
    print("PROCESO COMPLETADO")
    print("=" * 80)

