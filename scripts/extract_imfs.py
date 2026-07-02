"""
Script to obtain IMFs (Intrinsic Mode Functions) from a time series.

This script decomposes a time series into Intrinsic Mode Functions (IMFs)
using Ensemble Empirical Mode Decomposition (EEMD) with the parameters
and configuration from the analysis in notebook 03_emd_imfs_analysis.ipynb.
"""

import os
import sys
import subprocess
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

# Try importing PyEMD (EMD and EEMD)
try:
    from PyEMD import EEMD

    EEMD_AVAILABLE = True
except ImportError:
    EEMD_AVAILABLE = False
    print("⚠ PyEMD is not installed. Attempting to install EMD-signal...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "EMD-signal", "--quiet"]
        )
        from PyEMD import EEMD

        EEMD_AVAILABLE = True
        print("✓ PyEMD installed and imported successfully")
    except Exception as e:
        raise ImportError(
            f"Error installing PyEMD: {e}. "
            "Install manually with: pip install EMD-signal"
        ) from e


def extract_imfs(
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
    Obtain IMFs from a time series using EEMD.

    Decomposes a time series into Intrinsic Mode Functions (IMFs)
    using Ensemble Empirical Mode Decomposition (EEMD) with the
    specified parameters.

    Parameters
    ----------
    archivo_entrada : str
        Path to the parquet file with the time series.
    columna_serie : str, optional
        Name of the column containing the time series to decompose.
        Default is "Close".
    archivo_salida : str, optional
        Path where the DataFrame with IMFs will be saved. If None, it is saved in
        the same directory as the input file with the "_imfs.parquet" suffix.
    max_imf : int, optional
        Maximum number of IMFs to extract. Default is 14.
    sd_thresh : float, optional
        Threshold for the sifting stopping criterion. Default is 0.25.
    s_number : int, optional
        Number of sifting iterations. Default is 8.
    fixe_h : int, optional
        Minimum number of iterations when the IMF condition is met.
        Default is 5.
    trials : int, optional
        Number of ensembles (add noise and average). Default is 100.
    noise_width : float, optional
        Noise level as a percentage of the signal standard deviation.
        Default is 0.05 (5%).

    Returns
    -------
    pd.DataFrame
        DataFrame with the IMFs and the residue. Columns are IMF_1, IMF_2, ...,
        IMF_N and Residuo. The index corresponds to the dates of the original series.

    Examples
    --------
    >>> df_imfs = extract_imfs("data/16nov25/msci_world.parquet")
    >>> print(df_imfs.head())
    """
    # Load data
    print(f"Loading data from: {archivo_entrada}")
    df = pd.read_parquet(archivo_entrada, engine="pyarrow")
    print(f"DataFrame shape: {df.shape}")

    # Verify that the specified column exists
    if columna_serie not in df.columns:
        raise ValueError(
            f"Column '{columna_serie}' does not exist in the DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    # Extract the price series
    serie_precios = np.array(df[columna_serie].values)
    fechas = df.index

    print(f"\nTime series:")
    print(f"- Length: {len(serie_precios)} observations")
    print(f"- Date range: {fechas.min()} to {fechas.max()}")
    print(f"- Minimum value: {np.min(serie_precios):.2f}")
    print(f"- Maximum value: {np.max(serie_precios):.2f}")
    print(f"- Mean value: {np.mean(serie_precios):.2f}")

    # Configure EEMD
    print("\n" + "=" * 60)
    print("Starting EEMD decomposition (Ensemble EMD)...")
    print("Configured parameters:")
    print(f"  - max_imf: {max_imf}")
    print(f"  - SD_thresh: {sd_thresh}")
    print(f"  - S_number: {s_number}")
    print(f"  - FIXE_H: {fixe_h}")
    print(f"  - trials (ensembles): {trials}")
    print(f"  - noise_width: {noise_width}")
    print("EEMD may take significantly longer than EMD due to multiple ensembles...")
    print("This may take several minutes depending on the series size...")
    print("=" * 60)

    # Create EEMD instance
    eemd = EEMD(
        max_imf=max_imf,
        SD_thresh=sd_thresh,
        S_number=s_number,
        FIXE_H=fixe_h,
        trials=trials,
        noise_width=noise_width,
    )

    # Perform decomposition
    imfs = eemd(serie_precios)

    # Process results
    if imfs is not None and len(imfs) > 0:
        todas_las_imfs = list(imfs[:-1])
        residuo_actual = imfs[-1].copy()
    else:
        todas_las_imfs = []
        residuo_actual = serie_precios.copy()

    print(f"\nDecomposition completed:")
    print(f"  - IMFs extracted: {len(todas_las_imfs)}")
    print(f"  - Residue obtained: {'Yes' if residuo_actual is not None else 'No'}")

    # Create DataFrame with IMFs
    df_imfs = pd.DataFrame(index=fechas)
    for i, imf in enumerate(todas_las_imfs):
        df_imfs[f"IMF_{i+1}"] = imf

    if residuo_actual is not None:
        df_imfs["Residuo"] = residuo_actual

    print(f"\nIMFs DataFrame shape: {df_imfs.shape}")

    # Save results
    if archivo_salida is None:
        # Generate filename automatically
        archivo_entrada_path = Path(archivo_entrada)
        archivo_salida = str(
            archivo_entrada_path.parent / (archivo_entrada_path.stem + "_imfs.parquet")
        )

    # Create directory if it does not exist
    archivo_salida_path = Path(archivo_salida)
    os.makedirs(archivo_salida_path.parent, exist_ok=True)

    # Save in parquet format
    df_imfs.to_parquet(archivo_salida, engine="pyarrow", index=True)
    print(f"\n✓ IMFs successfully saved to: {archivo_salida}")
    print(f"  File size: {archivo_salida_path.stat().st_size / (1024 * 1024):.2f} MB")

    return df_imfs


if __name__ == "__main__":
    # Default configuration using the msci_world.parquet example
    proyecto_root = Path(__file__).parent.parent
    archivo_entrada = proyecto_root / "data" / "16dic25" / "msci_world.parquet"

    if not archivo_entrada.exists():
        raise FileNotFoundError(
            f"File {archivo_entrada} does not exist. "
            "Make sure the file is in the correct location."
        )

    # Obtain IMFs with the notebook parameters
    df_imfs = extract_imfs(
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
    print("SUMMARY")
    print("=" * 60)
    print(f"Total IMFs extracted: {len([col for col in df_imfs.columns if col.startswith('IMF_')])}")
    print(f"Includes residue?: {'Yes' if 'Residuo' in df_imfs.columns else 'No'}")
    print("\nFirst rows of the IMFs DataFrame:")
    print(df_imfs.head())
    print("\nDataFrame info:")
    print(df_imfs.info())
