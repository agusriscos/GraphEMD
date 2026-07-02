"""
Script to obtain information about the downloaded MSCI World series and NVG/HVG graphs
built on the full series (no windowing).

Downloads MSCI World in the same style as download_data.py (using GraphEMD.conf DATA_PATH).
Includes on-disk parquet size of the saved PyTorch Geometric object (features + edges).
Decomposes the series via CEEMDAN (and optionally EEMD with parameters from the prior
work in ``docs/16dic25``) and computes the graph size (NVG and HVG) of each IMF.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from ts2vg import HorizontalVG, NaturalVG
import yfinance as yf

# Add project root to path for GraphEMD imports
_proyecto_root = Path(__file__).resolve().parent.parent.parent
if str(_proyecto_root) not in sys.path:
    sys.path.insert(0, str(_proyecto_root))

from GraphEMD.conf import DATA_PATH
from GraphEMD.data.python_utils import save_graph_data

# CEEMDAN parameters aligned with the technical document (docs/20abr26)
CEEMDAN_MAX_IMF: int = 14
CEEMDAN_TRIALS: int = 100
CEEMDAN_EPSILON: float = 0.05
CEEMDAN_SEED: int = 42

# EEMD parameters aligned with table ``tab:eemd_params`` in docs/16dic25/paper/main.tex
EEMD_MAX_IMF: int = 14
EEMD_TRIALS: int = 100
EEMD_NOISE_WIDTH: float = 0.05
EEMD_SD_THRESH: float = 0.25
EEMD_S_NUMBER: int = 8
EEMD_FIXE_H: int = 5
EEMD_SEED: int = 42

# CEEMDAN / EEMD (PyEMD / EMD-signal)
try:
    from PyEMD import CEEMDAN, EEMD

    CEEMDAN_AVAILABLE = True
    EEMD_AVAILABLE = True
except ImportError:
    CEEMDAN = None  # type: ignore[misc, assignment]
    EEMD = None  # type: ignore[misc, assignment]
    CEEMDAN_AVAILABLE = False
    EEMD_AVAILABLE = False


def descargar_msci_world(data_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Download the full MSCI World history from Yahoo Finance.

    Uses the same criterion as download_data.py: if data_dir is passed (e.g. DATA_PATH),
    data are saved there; otherwise data/16nov25 relative to the project root is used.

    Parameters
    ----------
    data_dir : str, optional
        Directory where data are saved. If None, uses data/16nov25 relative
        to the project root.

    Returns
    -------
    pd.DataFrame
        DataFrame with historical data (Open, High, Low, Close, Volume, etc.).

    Raises
    ------
    ValueError
        If data could not be downloaded with any symbol.
    """
    if data_dir is None:
        data_path = _proyecto_root / "data" / "16nov25"
    else:
        data_path = Path(data_dir)

    os.makedirs(data_path, exist_ok=True)
    print("Downloading MSCI World historical data...")
    print(f"Destination directory: {data_path}")

    simbolos = ["^MSWORLD", "URTH", "ACWI"]
    df = None
    simbolo_usado = None

    for simbolo in simbolos:
        try:
            print(f"Trying to download with symbol: {simbolo}")
            ticker = yf.Ticker(simbolo)
            df_temp = ticker.history(period="max")
            if df_temp is not None and not df_temp.empty:
                df = df_temp
                simbolo_usado = simbolo
                print(f"✓ Data downloaded with symbol: {simbolo}")
                print(
                    f"  Rango: {df.index.min()} a {df.index.max()}, registros: {len(df)}"
                )
                break
        except Exception as e:
            print(f"✗ Error with {simbolo}: {e}")
            continue

    if df is None or df.empty:
        raise ValueError(
            "Could not download MSCI World data. "
            "Check connection and symbols."
        )

    archivo_parquet = data_path / "msci_world.parquet"
    df.to_parquet(archivo_parquet, engine="pyarrow", index=True)
    print(f"✓ Saved to: {archivo_parquet}")
    print(f"  Size: {archivo_parquet.stat().st_size / (1024**2):.2f} MB")
    return df


def _buscar_archivo_msci_world(
    proyecto_root: Path, data_path: Optional[str] = None
) -> Path:
    """
    Search for msci_world.parquet: first in data_path (e.g. DATA_PATH), then
    in data/16dic25 and data/16nov25.

    Parameters
    ----------
    proyecto_root : Path
        Project root path.
    data_path : str, optional
        Data directory (e.g. DATA_PATH). If provided, searched here first.

    Returns
    -------
    Path
        Path to msci_world.parquet.

    Raises
    ------
    FileNotFoundError
        If the file is not found in any candidate location.
    """
    candidatos = []
    if data_path:
        candidatos.append(Path(data_path) / "msci_world.parquet")
    candidatos.extend(
        [
            proyecto_root / "data" / "20abr26" / "msci_world.parquet",
            proyecto_root / "data" / "16dic25" / "msci_world.parquet",
            proyecto_root / "data" / "16nov25" / "msci_world.parquet",
        ]
    )
    for ruta in candidatos:
        if ruta.is_file():
            return ruta
    raise FileNotFoundError(
        f"msci_world.parquet not found in {[str(p) for p in candidatos]}"
    )


def cargar_serie_msci_world(archivo_parquet: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load the MSCI World closing-price (Close) series from parquet.

    Parameters
    ----------
    archivo_parquet : Path
        Path to msci_world.parquet.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        Full DataFrame and 1D array of the Close column (writable copy for ts2vg).
    """
    df = pd.read_parquet(archivo_parquet, engine="pyarrow")
    if "Close" not in df.columns:
        raise ValueError(
            f"File does not contain column 'Close'. Columns: {list(df.columns)}"
        )
    serie = np.asarray(df["Close"].values, dtype=np.float64).copy()
    return df, serie


def serie_a_grafo_nvg(serie: np.ndarray) -> Data:
    """
    Build the Natural Visibility (NVG) graph on the full series.

    Parameters
    ----------
    serie : np.ndarray
        1D time series (all points, no windows).

    Returns
    -------
    Data
        PyTorch Geometric Data object with the NVG graph.
    """
    nvg = NaturalVG(directed="left_to_right")
    grafo = nvg.build(serie)
    edges = np.array(grafo.edges)
    edge_index = torch.tensor(enlaces.T, dtype=torch.long)
    node_features = torch.tensor(serie, dtype=torch.float).unsqueeze(1)
    return Data(x=node_features, edge_index=edge_index)


def serie_a_grafo_hvg(serie: np.ndarray) -> Data:
    """
    Build the Horizontal Visibility (HVG) graph on the full series.

    Parameters
    ----------
    serie : np.ndarray
        1D time series (all points, no windows).

    Returns
    -------
    Data
        PyTorch Geometric Data object with the HVG graph.
    """
    hvg = HorizontalVG(directed="left_to_right")
    grafo = hvg.build(serie)
    edges = np.array(grafo.edges)
    edge_index = torch.tensor(enlaces.T, dtype=torch.long)
    node_features = torch.tensor(serie, dtype=torch.float).unsqueeze(1)
    return Data(x=node_features, edge_index=edge_index)


def parquet_graph_size(ruta_base: Path) -> int:
    """
    Return the total size in bytes of the saved graph parquet files.

    Sums _features.parquet and _edges.parquet (format used by save_graph_data).

    Parameters
    ----------
    ruta_base : Path
        Base graph path (without _features or _edges suffix).

    Returns
    -------
    int
        Total size in bytes.
    """
    features = Path(str(ruta_base) + "_features.parquet")
    edges = Path(str(ruta_base) + "_edges.parquet")
    total = 0
    if features.exists():
        total += features.stat().st_size
    if edges.exists():
        total += edges.stat().st_size
    return total


def pt_graph_size(ruta_base: Path) -> int:
    """
    Return the size in bytes of the PyTorch Geometric Data .pt file.

    save_graph_data also writes a .pt (torch.save of the Data).

    Parameters
    ----------
    ruta_base : Path
        Base graph path (same as passed to save_graph_data without extension).

    Returns
    -------
    int
        Size in bytes of the .pt file, or 0 if missing.
    """
    pt_path = ruta_base.with_suffix(".pt")
    return pt_path.stat().st_size if pt_path.exists() else 0


def extract_ceemdan_imfs(
    serie: np.ndarray,
    max_imf: int = CEEMDAN_MAX_IMF,
    trials: int = CEEMDAN_TRIALS,
    epsilon: float = CEEMDAN_EPSILON,
    seed: Optional[int] = CEEMDAN_SEED,
    parallel: bool = False,
) -> pd.DataFrame:
    """
    Decompose the time series into IMFs via CEEMDAN.

    Parameters
    ----------
    serie : np.ndarray
        1D time series.
    max_imf : int, optional
        Maximum number of IMFs to extract. Default is 14.
    trials : int, optional
        Number of ensemble realizations. Default is 100.
    epsilon : float, optional
        Adaptive noise scale (fraction of signal std). Default is 0.05 (5 percent).
    seed : int, optional
        Seed for reproducibility. If None, not fixed. Default is 42.
    parallel : bool, optional
        If True, uses multiprocessing (less reproducible). Default is False.

    Returns
    -------
    pd.DataFrame
        Columns IMF_1, IMF_2, ..., IMF_N and Residuo. Same number of rows as serie.

    Raises
    ------
    ImportError
        If PyEMD is not installed.
    """
    if not CEEMDAN_AVAILABLE or CEEMDAN is None:
        raise ImportError(
            "CEEMDAN requires PyEMD. Install with: pip install EMD-signal"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ceemdan = CEEMDAN(trials=trials, epsilon=epsilon, parallel=parallel)
        if seed is not None:
            ceemdan.noise_seed(seed)
        componentes = ceemdan(serie, max_imf=max_imf)
    if componentes is None or len(componentes) == 0:
        return pd.DataFrame({"Residuo": serie})
    # components: each row is a mode (first row = IMF_1, last = residue)
    n_imfs = componentes.shape[0] - 1
    df = pd.DataFrame()
    for i in range(n_imfs):
        df[f"IMF_{i + 1}"] = componentes[i]
    df["Residuo"] = componentes[-1]
    return df


def extract_eemd_imfs(
    serie: np.ndarray,
    max_imf: int = EEMD_MAX_IMF,
    trials: int = EEMD_TRIALS,
    noise_width: float = EEMD_NOISE_WIDTH,
    sd_thresh: float = EEMD_SD_THRESH,
    s_number: int = EEMD_S_NUMBER,
    fixe_h: int = EEMD_FIXE_H,
    seed: Optional[int] = EEMD_SEED,
) -> pd.DataFrame:
    """
    Decompose the time series into IMFs via EEMD (noise ensemble).

    Default parameters reproduce the documented MSCI World configuration
    in ``docs/16dic25/paper/main.tex`` (EEMD parameter table).

    Parameters
    ----------
    serie : np.ndarray
        1D time series.
    max_imf : int, optional
        Maximum number of IMFs. Default is 14.
    trials : int, optional
        Ensemble realizations. Default is 100.
    noise_width : float, optional
        Noise amplitude (PyEMD ``noise_width`` definition). Default is 0.05.
    sd_thresh : float, optional
        SD threshold for the sifting stop criterion. Default is 0.25.
    s_number : int, optional
        Minimum sifting iterations. Default is 8.
    fixe_h : int, optional
        PyEMD ``FIXE_H``. Default is 5.
    seed : int, optional
        Ensemble noise seed. Default is 42.

    Returns
    -------
    pd.DataFrame
        Columns IMF_1, IMF_2, ..., IMF_N and Residuo. Same number of rows as ``serie``.

    Raises
    ------
    ImportError
        If PyEMD is not installed.
    """
    if not EEMD_AVAILABLE or EEMD is None:
        raise ImportError("EEMD requires PyEMD. Install with: pip install EMD-signal")
    x = np.asarray(serie, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eemd = EEMD(
            max_imf=max_imf,
            SD_thresh=sd_thresh,
            S_number=s_number,
            FIXE_H=fixe_h,
            trials=trials,
            noise_width=noise_width,
            parallel=False,
        )
        if seed is not None:
            eemd.noise_seed(seed)
        e_imf = np.asarray(eemd(x, max_imf=max_imf))
    residuo = np.asarray(eemd.residue, dtype=np.float64)
    if e_imf.size == 0:
        return pd.DataFrame({"Residuo": serie})
    n_imfs = e_imf.shape[0]
    df = pd.DataFrame()
    for i in range(n_imfs):
        df[f"IMF_{i + 1}"] = e_imf[i]
    df["Residuo"] = residuo
    return df


def _suma_columnas_imf_sin_residuo(df_imfs: pd.DataFrame) -> np.ndarray:
    """
    Sum IMF_* columns of a decomposition DataFrame (excludes ``Residuo``).

    Parameters
    ----------
    df_imfs : pd.DataFrame
        Columns ``IMF_1``, ... and optionally ``Residuo``.

    Returns
    -------
    np.ndarray
        Vector with the same length as each IMF column.
    """
    cols_imf = [c for c in df_imfs.columns if c.startswith("IMF_")]
    if not cols_imf:
        return np.zeros(len(df_imfs), dtype=np.float64)
    acc = np.zeros(len(df_imfs), dtype=np.float64)
    for c in cols_imf:
        acc += np.asarray(df_imfs[c].values, dtype=np.float64)
    return acc


def compute_graph_sizes_per_imf(
    df_imfs: pd.DataFrame,
    dir_salida: Path,
) -> pd.DataFrame:
    """
    For each column (IMF or Residuo), build NVG and HVG, save parquet and .pt,
    and return a DataFrame with nodes, edges, and on-disk size (parquet and .pt) per graph.

    Parameters
    ----------
    df_imfs : pd.DataFrame
        DataFrame with columns IMF_1, IMF_2, ..., Residuo.
    dir_salida : Path
        Directory where each graph parquet is saved (subfolders per id).

    Returns
    -------
    pd.DataFrame
        Columns: id_imf, nodes, edges_nvg, edges_hvg, tam_parquet_nvg_mb,
        tam_parquet_hvg_mb, tam_pt_nvg_mb, tam_pt_hvg_mb.
    """
    columnas_imf = [
        c for c in df_imfs.columns if c.startswith("IMF_") or c == "Residuo"
    ]
    filas = []
    dir_salida.mkdir(parents=True, exist_ok=True)
    for id_imf in columnas_imf:
        serie_imf = np.asarray(df_imfs[id_imf].values, dtype=np.float64).copy()
        # NVG
        data_nvg = serie_a_grafo_nvg(serie_imf)
        base_nvg = dir_salida / f"{id_imf}_nvg"
        save_graph_data(data_nvg, str(base_nvg), id_imf)
        tam_nvg = parquet_graph_size(base_nvg)
        tam_pt_nvg = pt_graph_size(base_nvg)
        # HVG
        data_hvg = serie_a_grafo_hvg(serie_imf)
        base_hvg = dir_salida / f"{id_imf}_hvg"
        save_graph_data(data_hvg, str(base_hvg), id_imf)
        tam_hvg = parquet_graph_size(base_hvg)
        tam_pt_hvg = pt_graph_size(base_hvg)
        filas.append(
            {
                "id_imf": id_imf,
                "nodos": data_nvg.num_nodes,
                "enlaces_nvg": data_nvg.num_edges,
                "enlaces_hvg": data_hvg.num_edges,
                "tam_parquet_nvg_mb": tam_nvg / (1024**2),
                "tam_parquet_hvg_mb": tam_hvg / (1024**2),
                "tam_pt_nvg_mb": tam_pt_nvg / (1024**2),
                "tam_pt_hvg_mb": tam_pt_hvg / (1024**2),
            }
        )
    return pd.DataFrame(filas)


def exportar_figuras_documento_20abr26(
    df_imfs: pd.DataFrame,
    serie_original: np.ndarray,
    directorio_imagenes: Path,
    df_imfs_eemd: Optional[pd.DataFrame] = None,
) -> None:
    """
    Generate English figures for the LaTeX document in ``docs/20abr26``.

    The top panel shows the original signal and the sum of CEEMDAN IMFs (no residue).
    The bottom panel compares ``|Original − Σ IMFs|`` between CEEMDAN and EEMD when EEMD
    data are available. The ``hvg_imf.png`` figure uses the highest-frequency CEEMDAN IMF.

    Parameters
    ----------
    df_imfs : pd.DataFrame
        CEEMDAN decomposition: columns ``IMF_1``, ..., ``IMF_N`` and ``Residuo``.
    serie_original : np.ndarray
        Price series (Close) aligned with ``df_imfs`` rows.
    directorio_imagenes : Path
        Destination directory (e.g. ``docs/20abr26/images/english``).
    df_imfs_eemd : pd.DataFrame, optional
        EEMD decomposition (same column conventions). Affects only the residue panel.
        If None and PyEMD exposes EEMD, computed with ``extract_eemd_imfs(serie_original)``.
    """
    import matplotlib  # pyright: ignore[reportMissingImports]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
    import networkx as nx

    directorio_imagenes.mkdir(parents=True, exist_ok=True)

    suma_imfs_ceemdan = _suma_columnas_imf_sin_residuo(df_imfs)
    error_abs_ceemdan = np.abs(serie_original - suma_imfs_ceemdan)

    df_eemd_usado: Optional[pd.DataFrame] = df_imfs_eemd
    if df_eemd_usado is None and EEMD_AVAILABLE:
        df_eemd_usado = extract_eemd_imfs(serie_original)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(serie_original, label="Original", linewidth=0.8, color="C0")
    axes[0].plot(
        suma_imfs_ceemdan,
        label="Sum of IMFs (CEEMDAN, no residue)",
        linewidth=0.8,
        alpha=0.85,
        color="C1",
    )
    axes[1].plot(
        error_abs_ceemdan,
        label="|Original − Σ IMFs| (CEEMDAN)",
        linewidth=0.7,
        color="C1",
    )
    if df_eemd_usado is not None:
        suma_imfs_eemd = _suma_columnas_imf_sin_residuo(df_eemd_usado)
        error_abs_eemd = np.abs(serie_original - suma_imfs_eemd)
        axes[1].plot(
            error_abs_eemd,
            label="|Original − Σ IMFs| (EEMD)",
            linewidth=0.7,
            color="C2",
            alpha=0.9,
        )
        axes[1].legend(loc="upper left", fontsize=7)
    axes[0].set_ylabel("Level")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("|Original − Σ IMFs|")
    axes[1].set_xlabel("Time index")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(directorio_imagenes / "imf_decomposition.png", dpi=200)
    plt.close(fig)

    imf1 = np.asarray(df_imfs["IMF_1"].values, dtype=np.float64).copy()
    hvg = HorizontalVG(directed="left_to_right")
    grafo_h = hvg.build(imf1)
    edges = np.array(grafo_h.edges)
    g_nx = nx.Graph()
    g_nx.add_edges_from(map(tuple, edges))
    n_nodos = g_nx.number_of_nodes()
    if n_nodos > 600:
        nodos_sel = np.linspace(0, n_nodos - 1, 500, dtype=int)
        conjunto = set(nodos_sel.tolist())
        aristas_sub = [
            (u, v) for u, v in g_nx.edges() if u in conjunto and v in conjunto
        ]
        g_plot = nx.Graph()
        g_plot.add_nodes_from(nodos_sel)
        g_plot.add_edges_from(aristas_sub)
    else:
        g_plot = g_nx
    pos = nx.spring_layout(g_plot, seed=42, k=0.25, iterations=50)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_nodes(
        g_plot, pos, node_size=8, node_color="#1f77b4", alpha=0.85, ax=ax2
    )
    nx.draw_networkx_edges(g_plot, pos, width=0.15, alpha=0.35, ax=ax2)
    ax2.axis("off")
    fig2.tight_layout()
    fig2.savefig(directorio_imagenes / "hvg_imf.png", dpi=200)
    plt.close(fig2)


def generar_doc_dimensionado_databricks(
    archivo_salida: Path,
    n_puntos_serie: int,
    tam_serie_parquet_mb: float,
    num_nodos_nvg: int,
    num_enlaces_nvg: int,
    tam_parquet_nvg_mb: float,
    tam_pt_nvg_mb: float,
    num_nodos_hvg: int,
    num_enlaces_hvg: int,
    tam_parquet_hvg_mb: float,
    tam_pt_hvg_mb: float,
    tabla_imfs: Optional[pd.DataFrame] = None,
    suma_parquet_nvg_imfs_mb: Optional[float] = None,
    suma_parquet_hvg_imfs_mb: Optional[float] = None,
    suma_pt_nvg_imfs_mb: Optional[float] = None,
    suma_pt_hvg_imfs_mb: Optional[float] = None,
) -> None:
    """
    Generate a Markdown documentation file to size the driver node in an Azure Databricks
    cluster from measured data sizes (MSCI World series, graph parquets, and Data .pt objects).

    Parameters
    ----------
    archivo_salida : Path
        Path of the .md file to generate.
    n_puntos_serie : int
        Time-series length (number of observations).
    tam_serie_parquet_mb : float
        On-disk size of the series parquet (MB).
    num_nodos_nvg : int
        Number of nodes in the NVG graph (full series).
    num_enlaces_nvg : int
        Number of edges in the NVG graph.
    tam_parquet_nvg_mb : float
        On-disk size of the NVG graph parquet (MB).
    tam_pt_nvg_mb : float
        On-disk size of the NVG graph .pt (PyTorch Geometric Data object) (MB).
    num_nodos_hvg : int
        Number of nodes in the HVG graph (full series).
    num_enlaces_hvg : int
        Number of edges in the HVG graph.
    tam_parquet_hvg_mb : float
        On-disk size of the HVG graph parquet (MB).
    tam_pt_hvg_mb : float
        On-disk size of the HVG graph .pt (MB).
    tabla_imfs : pd.DataFrame, optional
        Table with per-IMF sizes (includes tam_pt_nvg_mb, tam_pt_hvg_mb). If None, omitted.
    suma_parquet_nvg_imfs_mb : float, optional
        Sum of NVG parquet sizes over all IMFs (MB).
    suma_parquet_hvg_imfs_mb : float, optional
        Sum of HVG parquet sizes over all IMFs (MB).
    suma_pt_nvg_imfs_mb : float, optional
        Sum of NVG .pt sizes over all IMFs (MB).
    suma_pt_hvg_imfs_mb : float, optional
        Sum of HVG .pt sizes over all IMFs (MB).
    """
    lineas = []
    lineas.append("# Azure Databricks driver sizing")
    lineas.append("")
    lineas.append("## Objective")
    lineas.append("")
    lineas.append(
        "This document summarizes on-disk sizes of pipeline input data "
        "**MSCI World as graphs** (time series converted to NVG/HVG visibility graphs, "
        "and optionally CEEMDAN decomposition per IMF) to size the **"
        "driver disk** of each node in an Azure Databricks cluster."
    )
    lineas.append("")
    lineas.append("## Input data types")
    lineas.append("")
    lineas.append("| Type | Format | Description |")
    lineas.append("|------|---------|-------------|")
    lineas.append(
        "| MSCI World series | Parquet (1 Close column) | Closing prices, one point per day. |"
    )
    lineas.append(
        "| NVG graph (full series) | Parquet + .pt (PyG Data) | Natural Visibility Graph over the full series. |"
    )
    lineas.append(
        "| HVG graph (full series) | Parquet + .pt (PyG Data) | Horizontal Visibility Graph over the full series. |"
    )
    lineas.append(
        "| Graphs per IMF (CEEMDAN) | Parquet + .pt per IMF/Residual | NVG and HVG; each saved as parquet (features/edges) and as .pt (Data object). |"
    )
    lineas.append("")
    lineas.append("## Measured sizes (reference)")
    lineas.append("")
    lineas.append("### Full-series data and graphs")
    lineas.append("")
    lineas.append("| Item | Value |")
    lineas.append("|----------|-------|")
    lineas.append(f"| Series length (puntos) | {n_puntos_serie:,} |")
    lineas.append(
        f"| On-disk size — serie (parquet) | {tam_serie_parquet_mb:.2f} MB |"
    )
    lineas.append(f"| NVG graph — nodes | {num_nodos_nvg:,} |")
    lineas.append(f"| Grafo NVG — edges | {num_enlaces_nvg:,} |")
    lineas.append(
        f"| NVG graph — parquet (features + edges) | {tam_parquet_nvg_mb:.2f} MB |"
    )
    lineas.append(f"| NVG graph — .pt (PyG Data object) | {tam_pt_nvg_mb:.2f} MB |")
    lineas.append(f"| HVG graph — nodes | {num_nodos_hvg:,} |")
    lineas.append(f"| Grafo HVG — edges | {num_enlaces_hvg:,} |")
    lineas.append(
        f"| HVG graph — parquet (features + edges) | {tam_parquet_hvg_mb:.2f} MB |"
    )
    lineas.append(f"| HVG graph — .pt (PyG Data object) | {tam_pt_hvg_mb:.2f} MB |")
    lineas.append("")
    if tabla_imfs is not None and not tabla_imfs.empty:
        lineas.append("### Graphs per IMF (CEEMDAN)")
        lineas.append("")
        lineas.append(
            "| IMF | Nodos | Parquet NVG (MB) | .pt NVG (MB) | Parquet HVG (MB) | .pt HVG (MB) |"
        )
        lineas.append(
            "|-----|-------|------------------|--------------|------------------|--------------|"
        )
        for row in tabla_imfs.itertuples(index=False):
            nodos = int(getattr(row, "nodos", 0) or 0)
            pq_nvg = float(getattr(row, "tam_parquet_nvg_mb", 0.0) or 0.0)
            pt_nvg = float(getattr(row, "tam_pt_nvg_mb", 0.0) or 0.0)
            pq_hvg = float(getattr(row, "tam_parquet_hvg_mb", 0.0) or 0.0)
            pt_hvg = float(getattr(row, "tam_pt_hvg_mb", 0.0) or 0.0)
            id_imf = getattr(row, "id_imf", "")
            lineas.append(
                f"| {id_imf} | {nodos:,} | {pq_nvg:.2f} | {pt_nvg:.2f} | {pq_hvg:.2f} | {pt_hvg:.2f} |"
            )
        lineas.append("")
        if (
            suma_parquet_nvg_imfs_mb is not None
            and suma_parquet_hvg_imfs_mb is not None
            and suma_pt_nvg_imfs_mb is not None
            and suma_pt_hvg_imfs_mb is not None
        ):
            lineas.append(
                "| **Total (all IMFs)** | — | "
                f"**{suma_parquet_nvg_imfs_mb:.2f}** | **{suma_pt_nvg_imfs_mb:.2f}** | "
                f"**{suma_parquet_hvg_imfs_mb:.2f}** | **{suma_pt_hvg_imfs_mb:.2f}** |"
            )
            lineas.append("")
    lineas.append("## Recommended sizing for the Databricks driver")
    lineas.append("")
    lineas.append(
        "In Azure Databricks, the **driver** is the node that coordinates the cluster and may need "
    )
    lineas.append(
        "access to metadata and small data. Driver disk size should cover:"
    )
    lineas.append("")
    lineas.append(
        "1. **On-disk input data** (series and graph parquets, and .pt files for PyTorch Geometric Data objects) read or distributed from the driver."
    )
    lineas.append(
        "2. **System overhead** (OS, JVM/Spark, logs): typically 10–20 GB."
    )
    lineas.append(
        "3. **Safety margin**: multiplicative factor on **(data sum + overhead)**. "
        "This document uses **1.1×** (a tighter criterion than ~1.5–2× factors often "
        "cited for raw data alone)."
    )
    lineas.append("")
    tam_datos_mb = (
        tam_serie_parquet_mb
        + tam_parquet_nvg_mb
        + tam_parquet_hvg_mb
        + tam_pt_nvg_mb
        + tam_pt_hvg_mb
    )
    if suma_parquet_nvg_imfs_mb is not None and suma_parquet_hvg_imfs_mb is not None:
        tam_datos_mb += float(suma_parquet_nvg_imfs_mb) + float(
            suma_parquet_hvg_imfs_mb
        )
    if suma_pt_nvg_imfs_mb is not None and suma_pt_hvg_imfs_mb is not None:
        tam_datos_mb += float(suma_pt_nvg_imfs_mb) + float(suma_pt_hvg_imfs_mb)
    tam_datos_gb = tam_datos_mb / 1024
    overhead_gb = 15
    factor_margen = 1.1
    driver_min_gb = (tam_datos_gb + overhead_gb) * factor_margen
    lineas.append("### Example with measured sizes")
    lineas.append("")
    lineas.append("| Item | Value |")
    lineas.append("|----------|-------|")
    lineas.append(
        f"| Data sum (series + series-graph parquets/.pt + IMF graphs) | {tam_datos_mb:.2f} MB ({tam_datos_gb:.2f} GB) |"
    )
    lineas.append(f"| System overhead (reference) | {overhead_gb} GB |")
    lineas.append(f"| Margin factor | {factor_margen}× |")
    lineas.append(f"| **Minimum recommended driver** | **{driver_min_gb:.1f} GB** |")
    lineas.append("")
    lineas.append(
        "On Databricks, choose a node type (or cluster configuration) whose **disk size** "
        "is at least this value. If only the full series is used (without CEEMDAN), total "
        "data is smaller; if all per-IMF graphs are included, use the total above."
    )
    lineas.append("")
    lineas.append("---")
    lineas.append("*Document generated automatically by `info_msci_world_data.py`.*")
    archivo_salida.parent.mkdir(parents=True, exist_ok=True)
    archivo_salida.write_text("\n".join(lineas), encoding="utf-8")


def main() -> None:
    """
    Entry point: download MSCI World if missing (using DATA_PATH like
    download_data.py), then print series and NVG/HVG graph information
    (nodes, edges, on-disk parquet size).
    """
    proyecto_root = _proyecto_root

    descargar_msci_world(data_dir=DATA_PATH)
    archivo_msci = _buscar_archivo_msci_world(proyecto_root, data_path=DATA_PATH)
    print("=" * 60)
    print("SERIE MSCI WORLD")
    print("=" * 60)
    print(f"File: {archivo_msci}")
    print(
        f"On-disk size (parquet serie): {archivo_msci.stat().st_size / (1024**2):.2f} MB"
    )

    df, serie = cargar_serie_msci_world(archivo_msci)
    n_puntos = len(serie)
    if hasattr(df.index, "min") and hasattr(df.index, "max"):
        print(f"Date range: {df.index.min()} a {df.index.max()}")
    print(f"Number of points (series length): {n_puntos}")
    print(
        f"Close - min: {serie.min():.4f}, max: {serie.max():.4f}, mean: {serie.mean():.4f}"
    )

    # Output directory for graphs (parquet)
    dir_salida = Path(__file__).resolve().parent / "out_msci_world_grafos"
    dir_salida.mkdir(parents=True, exist_ok=True)
    base_nvg = dir_salida / "msci_world_nvg_serie_completa"
    base_hvg = dir_salida / "msci_world_hvg_serie_completa"

    # NVG (denser algorithm)
    print("\n" + "=" * 60)
    print("NVG GRAPH (Natural Visibility Graph) - Full series, no windows")
    print("=" * 60)
    data_nvg = serie_a_grafo_nvg(serie)
    print(f"Number of nodes: {data_nvg.num_nodes}")
    print(f"Number of edges: {data_nvg.num_edges}")
    save_graph_data(data_nvg, str(base_nvg), "msci_world_serie")
    tam_parquet_nvg = parquet_graph_size(base_nvg)
    tam_pt_nvg = pt_graph_size(base_nvg)
    print(
        f"On-disk size (solo parquet: features + edges): {tam_parquet_nvg / (1024**2):.2f} MB"
    )
    print(f"On-disk size (.pt Data object): {tam_pt_nvg / (1024**2):.2f} MB")

    # HVG
    print("\n" + "=" * 60)
    print("HVG GRAPH (Horizontal Visibility Graph) - Full series, no windows")
    print("=" * 60)
    data_hvg = serie_a_grafo_hvg(serie)
    print(f"Number of nodes: {data_hvg.num_nodes}")
    print(f"Number of edges: {data_hvg.num_edges}")
    save_graph_data(data_hvg, str(base_hvg), "msci_world_serie")
    tam_parquet_hvg = parquet_graph_size(base_hvg)
    tam_pt_hvg = pt_graph_size(base_hvg)
    print(
        f"On-disk size (solo parquet: features + edges): {tam_parquet_hvg / (1024**2):.2f} MB"
    )
    print(f"On-disk size (.pt Data object): {tam_pt_hvg / (1024**2):.2f} MB")

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(
        f"Serie: {n_puntos} puntos | NVG: {data_nvg.num_nodes} nodes, {data_nvg.num_edges} edges | HVG: {data_hvg.num_nodes} nodes, {data_hvg.num_edges} edges"
    )
    print(
        f"Parquet NVG: {tam_parquet_nvg / (1024**2):.2f} MB | Parquet HVG: {tam_parquet_hvg / (1024**2):.2f} MB"
    )

    # CEEMDAN: graph size per IMF
    tabla_imfs = None
    suma_parquet_nvg_imfs_mb = None
    suma_parquet_hvg_imfs_mb = None
    suma_pt_nvg_imfs_mb = None
    suma_pt_hvg_imfs_mb = None
    if CEEMDAN_AVAILABLE:
        print("\n" + "=" * 60)
        print("CEEMDAN - Graph size per IMF")
        print("=" * 60)
        print("Decomposing full series with CEEMDAN...")
        df_imfs = extract_ceemdan_imfs(
            serie,
            max_imf=CEEMDAN_MAX_IMF,
            trials=CEEMDAN_TRIALS,
            epsilon=CEEMDAN_EPSILON,
            seed=CEEMDAN_SEED,
            parallel=False,
        )
        print(f"IMFs obtenidas: {[c for c in df_imfs.columns if c.startswith('IMF_')]}")
        print(f"Residual: {'Yes' if 'Residuo' in df_imfs.columns else 'No'}")
        out_imfs_parquet = (
            proyecto_root / "data" / "20abr26" / "msci_world_imfs_ceemdan.parquet"
        )
        out_imfs_parquet.parent.mkdir(parents=True, exist_ok=True)
        df_imfs.to_parquet(out_imfs_parquet, index=False)
        print(f"✓ CEEMDAN IMFs saved to: {out_imfs_parquet}")
        docs_img = proyecto_root / "docs" / "20abr26" / "images" / "english"
        df_imfs_eemd: Optional[pd.DataFrame] = None
        if EEMD_AVAILABLE:
            print("Decomposing full series with EEMD (docs/16dic25 parameters)...")
            df_imfs_eemd = extract_eemd_imfs(serie)
            out_eemd_parquet = (
                proyecto_root / "data" / "20abr26" / "msci_world_imfs_eemd.parquet"
            )
            df_imfs_eemd.to_parquet(out_eemd_parquet, index=False)
            print(f"✓ EEMD IMFs saved to: {out_eemd_parquet}")
        else:
            print("(Skipped EEMD in comparison figure: PyEMD without EEMD.)")
        exportar_figuras_documento_20abr26(
            df_imfs, serie, docs_img, df_imfs_eemd=df_imfs_eemd
        )
        print(f"✓ LaTeX figures (docs/20abr26): {docs_img}")
        dir_imfs = dir_salida / "imfs_ceemdan"
        tabla_imfs = compute_graph_sizes_per_imf(df_imfs, dir_imfs)
        print("\nSize table (parquet) per IMF:")
        print(tabla_imfs.to_string(index=False))
        suma_parquet_nvg_imfs_mb = tabla_imfs["tam_parquet_nvg_mb"].sum()
        suma_parquet_hvg_imfs_mb = tabla_imfs["tam_parquet_hvg_mb"].sum()
        suma_pt_nvg_imfs_mb = tabla_imfs["tam_pt_nvg_mb"].sum()
        suma_pt_hvg_imfs_mb = tabla_imfs["tam_pt_hvg_mb"].sum()
        print(f"\nNVG parquet sum (all IMFs): {suma_parquet_nvg_imfs_mb:.2f} MB")
        print(f"HVG parquet sum (all IMFs): {suma_parquet_hvg_imfs_mb:.2f} MB")
        print(f"NVG .pt sum (all IMFs): {suma_pt_nvg_imfs_mb:.2f} MB")
        print(f"HVG .pt sum (all IMFs): {suma_pt_hvg_imfs_mb:.2f} MB")
        archivo_tabla = dir_imfs / "tamaño_grafo_por_imf.csv"
        tabla_imfs.to_csv(archivo_tabla, index=False)
        print(f"Table saved to: {archivo_tabla}")
    else:
        print("\n(CEEMDAN skipped: install EMD-signal to compute size per IMF.)")

    # # Markdown documentation for Azure Databricks driver sizing
    # tam_serie_mb = archivo_msci.stat().st_size / (1024**2)
    # archivo_md = dir_salida / "dimensionado_driver_azure_databricks.md"
    # num_nodos_nvg = data_nvg.num_nodes if data_nvg.num_nodes is not None else 0
    # num_nodos_hvg = data_hvg.num_nodes if data_hvg.num_nodes is not None else 0
    # sum_nvg_mb: Optional[float] = None
    # sum_hvg_mb: Optional[float] = None
    # sum_pt_nvg_mb: Optional[float] = None
    # sum_pt_hvg_mb: Optional[float] = None
    # if suma_parquet_nvg_imfs_mb is not None and isinstance(
    #     suma_parquet_nvg_imfs_mb, (int, float)
    # ):
    #     sum_nvg_mb = float(suma_parquet_nvg_imfs_mb)
    # if suma_parquet_hvg_imfs_mb is not None and isinstance(
    #     suma_parquet_hvg_imfs_mb, (int, float)
    # ):
    #     sum_hvg_mb = float(suma_parquet_hvg_imfs_mb)
    # if suma_pt_nvg_imfs_mb is not None and isinstance(
    #     suma_pt_nvg_imfs_mb, (int, float)
    # ):
    #     sum_pt_nvg_mb = float(suma_pt_nvg_imfs_mb)
    # if suma_pt_hvg_imfs_mb is not None and isinstance(
    #     suma_pt_hvg_imfs_mb, (int, float)
    # ):
    #     sum_pt_hvg_mb = float(suma_pt_hvg_imfs_mb)
    # generar_doc_dimensionado_databricks(
    #     archivo_salida=archivo_md,
    #     n_puntos_serie=n_puntos,
    #     tam_serie_parquet_mb=tam_serie_mb,
    #     num_nodos_nvg=num_nodos_nvg,
    #     num_enlaces_nvg=data_nvg.num_edges,
    #     tam_parquet_nvg_mb=tam_parquet_nvg / (1024**2),
    #     tam_pt_nvg_mb=tam_pt_nvg / (1024**2),
    #     num_nodos_hvg=num_nodos_hvg,
    #     num_enlaces_hvg=data_hvg.num_edges,
    #     tam_parquet_hvg_mb=tam_parquet_hvg / (1024**2),
    #     tam_pt_hvg_mb=tam_pt_hvg / (1024**2),
    #     tabla_imfs=tabla_imfs,
    #     suma_parquet_nvg_imfs_mb=sum_nvg_mb,
    #     suma_parquet_hvg_imfs_mb=sum_hvg_mb,
    #     suma_pt_nvg_imfs_mb=sum_pt_nvg_mb,
    #     suma_pt_hvg_imfs_mb=sum_pt_hvg_mb,
    # )
    # print(f"\n✓ Driver sizing documentation saved in: {archivo_md}")


if __name__ == "__main__":
    main()
