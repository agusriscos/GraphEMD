"""
Script para descargar el histórico completo de MSCI World y guardarlo en formato parquet.

Este script utiliza yfinance para descargar todos los datos históricos disponibles
del índice MSCI World y los guarda en formato parquet en la carpeta data/16nov25/.
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf


def descargar_msci_world(data_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Descarga el histórico completo de MSCI World desde Yahoo Finance.

    Parameters
    ----------
    data_dir : str, optional
        Directorio donde guardar los datos. Si es None, usa data/16nov25/ relativo
        al directorio del proyecto.

    Returns
    -------
    pd.DataFrame
        DataFrame con los datos históricos de MSCI World. Las columnas incluyen:
        Open, High, Low, Close, Volume, Dividends, Stock Splits.

    Examples
    --------
    >>> df = descargar_msci_world()
    >>> print(df.head())
    """
    # Determinar el directorio de datos
    if data_dir is None:
        # Obtener el directorio del proyecto (asumiendo que estamos en scripts/16nov25/)
        proyecto_root = Path(__file__).parent.parent.parent
        data_path = proyecto_root / "data" / "16nov25"
    else:
        data_path = Path(data_dir)

    # Crear el directorio si no existe
    os.makedirs(data_path, exist_ok=True)

    print(f"Descargando datos históricos de MSCI World...")
    print(f"Directorio de destino: {data_path}")

    # Intentar diferentes símbolos para MSCI World
    # ^MSWORLD es el símbolo del índice en Yahoo Finance
    # También podemos intentar con ETFs que replican el índice
    simbolos = ["^MSWORLD", "URTH", "ACWI"]

    df = None
    simbolo_usado = None

    for simbolo in simbolos:
        try:
            print(f"Intentando descargar con símbolo: {simbolo}")
            ticker = yf.Ticker(simbolo)
            # Descargar todo el histórico disponible
            df_temp = ticker.history(period="max")
            if df_temp is not None and not df_temp.empty:
                df = df_temp
                simbolo_usado = simbolo
                print(f"✓ Datos descargados exitosamente con símbolo: {simbolo}")
                print(f"  Rango de fechas: {df.index.min()} a {df.index.max()}")
                print(f"  Número de registros: {len(df)}")
                break
        except Exception as e:
            print(f"✗ Error al descargar con {simbolo}: {str(e)}")
            continue

    if df is None or df.empty:
        raise ValueError(
            "No se pudieron descargar los datos de MSCI World con ninguno de los símbolos intentados. "
            "Verifica tu conexión a internet y que los símbolos sean válidos."
        )

    # Guardar en formato parquet
    archivo_parquet = data_path / "msci_world.parquet"
    df.to_parquet(archivo_parquet, engine="pyarrow", index=True)
    print(f"\n✓ Datos guardados exitosamente en: {archivo_parquet}")
    print(f"  Símbolo usado: {simbolo_usado}")
    print(f"  Tamaño del archivo: {archivo_parquet.stat().st_size / (1024 * 1024):.2f} MB")

    return df


if __name__ == "__main__":
    df = descargar_msci_world()
    print("\nPrimeras filas del DataFrame:")
    print(df.head())
    print("\nInformación del DataFrame:")
    print(df.info())

