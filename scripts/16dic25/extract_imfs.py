"""
Script para obtener las IMFs (Intrinsic Mode Functions) de una serie temporal.

Este script realiza la descomposición de una serie temporal en Funciones de Modo
Intrínseco (IMFs) mediante Ensemble Empirical Mode Decomposition (EEMD) utilizando
los parámetros y configuración del análisis realizado en el notebook
03_analisis_emd_imfs.ipynb.
"""

import os
import sys
import subprocess
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Suprimir warnings
warnings.filterwarnings("ignore")

# Intentar importar PyEMD (EMD y EEMD)
try:
    from PyEMD import EEMD

    EEMD_AVAILABLE = True
except ImportError:
    EEMD_AVAILABLE = False
    print("⚠ PyEMD no está instalado. Intentando instalar EMD-signal...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "EMD-signal", "--quiet"]
        )
        from PyEMD import EEMD

        EEMD_AVAILABLE = True
        print("✓ PyEMD instalado e importado correctamente")
    except Exception as e:
        raise ImportError(
            f"Error al instalar PyEMD: {e}. "
            "Instalar manualmente con: pip install EMD-signal"
        ) from e


def obtener_imfs(
    archivo_entrada: str,
    columna_serie: str = "Close",
    archivo_salida: Optional[str] = None,
    max_imf: int = 14,
    sd_thresh: float = 0.25,
    s_number: int = 8,
    fixe_h: int = 5,
    trials: int = 100,
    noise_width: float = 0.05,
) -> pd.DataFrame:
    """
    Obtiene las IMFs de una serie temporal mediante EEMD.

    Realiza la descomposición de una serie temporal en Funciones de Modo Intrínseco
    (IMFs) usando Ensemble Empirical Mode Decomposition (EEMD) con los parámetros
    especificados.

    Parameters
    ----------
    archivo_entrada : str
        Ruta al archivo parquet con la serie temporal.
    columna_serie : str, optional
        Nombre de la columna que contiene la serie temporal a descomponer.
        Por defecto es "Close".
    archivo_salida : str, optional
        Ruta donde guardar el DataFrame con las IMFs. Si es None, se guarda en
        el mismo directorio que el archivo de entrada con el sufijo "_imfs.parquet".
    max_imf : int, optional
        Número máximo de IMFs a extraer. Por defecto es 14.
    sd_thresh : float, optional
        Umbral para el criterio de parada del sifting. Por defecto es 0.25.
    s_number : int, optional
        Número de iteraciones de sifting. Por defecto es 8.
    fixe_h : int, optional
        Número mínimo de iteraciones cuando se cumple la condición de IMF.
        Por defecto es 5.
    trials : int, optional
        Número de ensembles (añadir ruido y promediar). Por defecto es 100.
    noise_width : float, optional
        Nivel de ruido como porcentaje de la desviación estándar de la señal.
        Por defecto es 0.05 (5%).

    Returns
    -------
    pd.DataFrame
        DataFrame con las IMFs y el residuo. Las columnas son IMF_1, IMF_2, ...,
        IMF_N y Residuo. El índice corresponde a las fechas de la serie original.

    Examples
    --------
    >>> df_imfs = obtener_imfs("data/16nov25/msci_world.parquet")
    >>> print(df_imfs.head())
    """
    # Cargar datos
    print(f"Cargando datos desde: {archivo_entrada}")
    df = pd.read_parquet(archivo_entrada, engine="pyarrow")
    print(f"Shape del DataFrame: {df.shape}")

    # Verificar que existe la columna especificada
    if columna_serie not in df.columns:
        raise ValueError(
            f"La columna '{columna_serie}' no existe en el DataFrame. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    # Extraer la serie de precios
    serie_precios = np.array(df[columna_serie].values)
    fechas = df.index

    print(f"\nSerie temporal:")
    print(f"- Longitud: {len(serie_precios)} observaciones")
    print(f"- Rango de fechas: {fechas.min()} a {fechas.max()}")
    print(f"- Valor mínimo: {np.min(serie_precios):.2f}")
    print(f"- Valor máximo: {np.max(serie_precios):.2f}")
    print(f"- Valor medio: {np.mean(serie_precios):.2f}")

    # Configurar EEMD
    print("\n" + "=" * 60)
    print("Iniciando descomposición EEMD (Ensemble EMD)...")
    print("Parámetros configurados:")
    print(f"  - max_imf: {max_imf}")
    print(f"  - SD_thresh: {sd_thresh}")
    print(f"  - S_number: {s_number}")
    print(f"  - FIXE_H: {fixe_h}")
    print(f"  - trials (ensembles): {trials}")
    print(f"  - noise_width: {noise_width}")
    print("EEMD puede tardar significativamente más que EMD debido a los múltiples ensembles...")
    print("Esto puede tardar varios minutos dependiendo del tamaño de la serie...")
    print("=" * 60)

    # Crear instancia de EEMD
    eemd = EEMD(
        max_imf=max_imf,
        SD_thresh=sd_thresh,
        S_number=s_number,
        FIXE_H=fixe_h,
        trials=trials,
        noise_width=noise_width,
    )

    # Realizar descomposición
    imfs = eemd(serie_precios)

    # Procesar resultados
    if imfs is not None and len(imfs) > 0:
        todas_las_imfs = list(imfs[:-1])
        residuo_actual = imfs[-1].copy()
    else:
        todas_las_imfs = []
        residuo_actual = serie_precios.copy()

    print(f"\nDescomposición completada:")
    print(f"  - IMFs extraídas: {len(todas_las_imfs)}")
    print(f"  - Residuo obtenido: {'Sí' if residuo_actual is not None else 'No'}")

    # Crear DataFrame con las IMFs
    df_imfs = pd.DataFrame(index=fechas)
    for i, imf in enumerate(todas_las_imfs):
        df_imfs[f"IMF_{i+1}"] = imf

    if residuo_actual is not None:
        df_imfs["Residuo"] = residuo_actual

    print(f"\nShape del DataFrame de IMFs: {df_imfs.shape}")

    # Guardar resultados
    if archivo_salida is None:
        # Generar nombre de archivo automáticamente
        archivo_entrada_path = Path(archivo_entrada)
        archivo_salida = str(
            archivo_entrada_path.parent / (archivo_entrada_path.stem + "_imfs.parquet")
        )

    # Crear directorio si no existe
    archivo_salida_path = Path(archivo_salida)
    os.makedirs(archivo_salida_path.parent, exist_ok=True)

    # Guardar en formato parquet
    df_imfs.to_parquet(archivo_salida, engine="pyarrow", index=True)
    print(f"\n✓ IMFs guardadas exitosamente en: {archivo_salida}")
    print(f"  Tamaño del archivo: {archivo_salida_path.stat().st_size / (1024 * 1024):.2f} MB")

    return df_imfs


if __name__ == "__main__":
    # Configuración por defecto usando el ejemplo de msci_world.parquet
    proyecto_root = Path(__file__).parent.parent.parent
    archivo_entrada = proyecto_root / "data" / "16dic25" / "msci_world.parquet"

    if not archivo_entrada.exists():
        raise FileNotFoundError(
            f"El archivo {archivo_entrada} no existe. "
            "Asegúrate de que el archivo esté en la ubicación correcta."
        )

    # Obtener IMFs con los parámetros del notebook
    df_imfs = obtener_imfs(
        archivo_entrada=str(archivo_entrada),
        columna_serie="Close",
        max_imf=14,
        sd_thresh=0.25,
        s_number=8,
        fixe_h=5,
        trials=100,
        noise_width=0.05,
    )

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Total de IMFs extraídas: {len([col for col in df_imfs.columns if col.startswith('IMF_')])}")
    print(f"¿Incluye residuo?: {'Sí' if 'Residuo' in df_imfs.columns else 'No'}")
    print("\nPrimeras filas del DataFrame de IMFs:")
    print(df_imfs.head())
    print("\nInformación del DataFrame:")
    print(df_imfs.info())

