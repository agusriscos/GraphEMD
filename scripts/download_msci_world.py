"""
Script to download the full MSCI World history and save it in parquet format.

This script uses yfinance to download all available historical data
for the MSCI World index and saves it in parquet format in the data/16nov25/ folder.
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf


def descargar_msci_world(data_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Download the full MSCI World history from Yahoo Finance.

    Parameters
    ----------
    data_dir : str, optional
        Directory where data will be saved. If None, uses data/16nov25/
        relative to the project directory.

    Returns
    -------
    pd.DataFrame
        DataFrame with MSCI World historical data. Columns include:
        Open, High, Low, Close, Volume, Dividends, Stock Splits.

    Examples
    --------
    >>> df = descargar_msci_world()
    >>> print(df.head())
    """
    # Determine the data directory
    if data_dir is None:
        # Get the project directory (assuming we are in scripts/16nov25/)
        proyecto_root = Path(__file__).parent.parent
        data_path = proyecto_root / "data" / "16nov25"
    else:
        data_path = Path(data_dir)

    # Create the directory if it does not exist
    os.makedirs(data_path, exist_ok=True)

    print(f"Downloading MSCI World historical data...")
    print(f"Destination directory: {data_path}")

    # Try different symbols for MSCI World
    # ^MSWORLD is the index symbol on Yahoo Finance
    # We can also try ETFs that track the index
    simbolos = ["^MSWORLD", "URTH", "ACWI"]

    df = None
    simbolo_usado = None

    for simbolo in simbolos:
        try:
            print(f"Attempting download with symbol: {simbolo}")
            ticker = yf.Ticker(simbolo)
            # Download all available history
            df_temp = ticker.history(period="max")
            if df_temp is not None and not df_temp.empty:
                df = df_temp
                simbolo_usado = simbolo
                print(f"✓ Data successfully downloaded with symbol: {simbolo}")
                print(f"  Date range: {df.index.min()} to {df.index.max()}")
                print(f"  Number of records: {len(df)}")
                break
        except Exception as e:
            print(f"✗ Error downloading with {simbolo}: {str(e)}")
            continue

    if df is None or df.empty:
        raise ValueError(
            "Could not download MSCI World data with any of the attempted symbols. "
            "Check your internet connection and that the symbols are valid."
        )

    # Save in parquet format
    archivo_parquet = data_path / "msci_world.parquet"
    df.to_parquet(archivo_parquet, engine="pyarrow", index=True)
    print(f"\n✓ Data successfully saved to: {archivo_parquet}")
    print(f"  Symbol used: {simbolo_usado}")
    print(f"  File size: {archivo_parquet.stat().st_size / (1024 * 1024):.2f} MB")

    return df


if __name__ == "__main__":
    df = descargar_msci_world()
    print("\nFirst rows of the DataFrame:")
    print(df.head())
    print("\nDataFrame info:")
    print(df.info())
