"""
Script para obtener información de la serie MSCI World descargada y de los grafos
NVG y HVG construidos sobre la serie completa (sin división por ventanas).

Descarga MSCI World en el mismo estilo que download_data.py (usando DATA_PATH de
GraphEMD.conf). Incluye el tamaño en disco en formato parquet del objeto torch
geometric guardado (features + edges). Descompone la serie vía CEEMDAN (y opcionalmente
EEMD con parámetros del trabajo previo en ``docs/16dic25``) y calcula el tamaño del
grafo (NVG y HVG) de cada IMF.
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

# Añadir raíz del proyecto al path para imports de GraphEMD
_proyecto_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_proyecto_root) not in sys.path:
    sys.path.insert(0, str(_proyecto_root))

from GraphEMD.conf import DATA_PATH
from GraphEMD.data.python_utils import guardar_grafo_data

# Parámetros CEEMDAN alineados con el documento técnico (docs/20abr26)
CEEMDAN_MAX_IMF: int = 14
CEEMDAN_TRIALS: int = 100
CEEMDAN_EPSILON: float = 0.05
CEEMDAN_SEED: int = 42

# Parámetros EEMD alineados con tabla ``tab:eemd_params`` en docs/16dic25/paper/main.tex
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
    Descarga el histórico completo de MSCI World desde Yahoo Finance.

    Usa el mismo criterio que download_data.py: si se pasa data_dir (p. ej. DATA_PATH),
    se guarda ahí; si no, se usa data/16nov25 relativo a la raíz del proyecto.

    Parameters
    ----------
    data_dir : str, optional
        Directorio donde guardar los datos. Si es None, usa data/16nov25 relativo
        a la raíz del proyecto.

    Returns
    -------
    pd.DataFrame
        DataFrame con los datos históricos (Open, High, Low, Close, Volume, etc.).

    Raises
    ------
    ValueError
        Si no se pudieron descargar los datos con ningún símbolo.
    """
    if data_dir is None:
        data_path = _proyecto_root / "data" / "16nov25"
    else:
        data_path = Path(data_dir)

    os.makedirs(data_path, exist_ok=True)
    print("Descargando datos históricos de MSCI World...")
    print(f"Directorio de destino: {data_path}")

    simbolos = ["^MSWORLD", "URTH", "ACWI"]
    df = None
    simbolo_usado = None

    for simbolo in simbolos:
        try:
            print(f"Intentando descargar con símbolo: {simbolo}")
            ticker = yf.Ticker(simbolo)
            df_temp = ticker.history(period="max")
            if df_temp is not None and not df_temp.empty:
                df = df_temp
                simbolo_usado = simbolo
                print(f"✓ Datos descargados con símbolo: {simbolo}")
                print(
                    f"  Rango: {df.index.min()} a {df.index.max()}, registros: {len(df)}"
                )
                break
        except Exception as e:
            print(f"✗ Error con {simbolo}: {e}")
            continue

    if df is None or df.empty:
        raise ValueError(
            "No se pudieron descargar los datos de MSCI World. "
            "Comprueba conexión y símbolos."
        )

    archivo_parquet = data_path / "msci_world.parquet"
    df.to_parquet(archivo_parquet, engine="pyarrow", index=True)
    print(f"✓ Guardado en: {archivo_parquet}")
    print(f"  Tamaño: {archivo_parquet.stat().st_size / (1024**2):.2f} MB")
    return df


def _buscar_archivo_msci_world(
    proyecto_root: Path, data_path: Optional[str] = None
) -> Path:
    """
    Busca el archivo msci_world.parquet: primero en data_path (ej. DATA_PATH), luego
    en data/16dic25 y data/16nov25.

    Parameters
    ----------
    proyecto_root : Path
        Ruta raíz del proyecto.
    data_path : str, optional
        Directorio de datos (ej. DATA_PATH). Si se indica, se busca aquí primero.

    Returns
    -------
    Path
        Ruta al archivo msci_world.parquet.

    Raises
    ------
    FileNotFoundError
        Si no se encuentra el archivo en ninguno de los candidatos.
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
        f"No se encontró msci_world.parquet en {[str(p) for p in candidatos]}"
    )


def cargar_serie_msci_world(archivo_parquet: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Carga la serie de precios de cierre (Close) del MSCI World desde parquet.

    Parameters
    ----------
    archivo_parquet : Path
        Ruta al archivo msci_world.parquet.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        DataFrame completo y array 1D de la columna Close (copia escribible para ts2vg).
    """
    df = pd.read_parquet(archivo_parquet, engine="pyarrow")
    if "Close" not in df.columns:
        raise ValueError(
            f"El archivo no contiene la columna 'Close'. Columnas: {list(df.columns)}"
        )
    serie = np.asarray(df["Close"].values, dtype=np.float64).copy()
    return df, serie


def serie_a_grafo_nvg(serie: np.ndarray) -> Data:
    """
    Construye el grafo Natural Visibility (NVG) sobre la serie completa.

    Parameters
    ----------
    serie : np.ndarray
        Serie temporal 1D (todos los puntos, sin ventanas).

    Returns
    -------
    Data
        Objeto Data de PyTorch Geometric con el grafo NVG.
    """
    nvg = NaturalVG(directed="left_to_right")
    grafo = nvg.build(serie)
    enlaces = np.array(grafo.edges)
    edge_index = torch.tensor(enlaces.T, dtype=torch.long)
    node_features = torch.tensor(serie, dtype=torch.float).unsqueeze(1)
    return Data(x=node_features, edge_index=edge_index)


def serie_a_grafo_hvg(serie: np.ndarray) -> Data:
    """
    Construye el grafo Horizontal Visibility (HVG) sobre la serie completa.

    Parameters
    ----------
    serie : np.ndarray
        Serie temporal 1D (todos los puntos, sin ventanas).

    Returns
    -------
    Data
        Objeto Data de PyTorch Geometric con el grafo HVG.
    """
    hvg = HorizontalVG(directed="left_to_right")
    grafo = hvg.build(serie)
    enlaces = np.array(grafo.edges)
    edge_index = torch.tensor(enlaces.T, dtype=torch.long)
    node_features = torch.tensor(serie, dtype=torch.float).unsqueeze(1)
    return Data(x=node_features, edge_index=edge_index)


def tamaño_parquet_grafo(ruta_base: Path) -> int:
    """
    Devuelve el tamaño total en bytes de los archivos parquet del grafo guardado.

    Se suman _features.parquet y _edges.parquet (formato usado por guardar_grafo_data).

    Parameters
    ----------
    ruta_base : Path
        Ruta base del grafo (sin sufijo _features ni _edges).

    Returns
    -------
    int
        Tamaño total en bytes.
    """
    features = Path(str(ruta_base) + "_features.parquet")
    edges = Path(str(ruta_base) + "_edges.parquet")
    total = 0
    if features.exists():
        total += features.stat().st_size
    if edges.exists():
        total += edges.stat().st_size
    return total


def tamaño_pt_grafo(ruta_base: Path) -> int:
    """
    Devuelve el tamaño en bytes del archivo .pt del objeto Data de PyTorch Geometric.

    guardar_grafo_data escribe además de parquet un .pt (torch.save del Data).

    Parameters
    ----------
    ruta_base : Path
        Ruta base del grafo (mismo que se pasa a guardar_grafo_data sin extensión).

    Returns
    -------
    int
        Tamaño en bytes del .pt, o 0 si no existe.
    """
    pt_path = ruta_base.with_suffix(".pt")
    return pt_path.stat().st_size if pt_path.exists() else 0


def obtener_imfs_ceemdan(
    serie: np.ndarray,
    max_imf: int = CEEMDAN_MAX_IMF,
    trials: int = CEEMDAN_TRIALS,
    epsilon: float = CEEMDAN_EPSILON,
    seed: Optional[int] = CEEMDAN_SEED,
    parallel: bool = False,
) -> pd.DataFrame:
    """
    Descompone la serie temporal en IMFs mediante CEEMDAN.

    Parameters
    ----------
    serie : np.ndarray
        Serie temporal 1D.
    max_imf : int, optional
        Número máximo de IMFs a extraer. Por defecto 14.
    trials : int, optional
        Número de realizaciones del ensemble. Por defecto 100.
    epsilon : float, optional
        Escala del ruido adaptativo (fracción de la std de la señal). Por defecto 0.05 (5 por ciento).
    seed : int, optional
        Semilla para reproducibilidad. Si es None, no se fija. Por defecto 42.
    parallel : bool, optional
        Si True, usa multiprocessing (menos reproducible). Por defecto False.

    Returns
    -------
    pd.DataFrame
        Columnas IMF_1, IMF_2, ..., IMF_N y Residuo. Mismo número de filas que serie.

    Raises
    ------
    ImportError
        Si PyEMD no está instalado.
    """
    if not CEEMDAN_AVAILABLE or CEEMDAN is None:
        raise ImportError(
            "CEEMDAN requiere PyEMD. Instalar con: pip install EMD-signal"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ceemdan = CEEMDAN(trials=trials, epsilon=epsilon, parallel=parallel)
        if seed is not None:
            ceemdan.noise_seed(seed)
        componentes = ceemdan(serie, max_imf=max_imf)
    if componentes is None or len(componentes) == 0:
        return pd.DataFrame({"Residuo": serie})
    # componentes: cada fila es un modo (primera fila = IMF_1, última = residuo)
    n_imfs = componentes.shape[0] - 1
    df = pd.DataFrame()
    for i in range(n_imfs):
        df[f"IMF_{i + 1}"] = componentes[i]
    df["Residuo"] = componentes[-1]
    return df


def obtener_imfs_eemd(
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
    Descompone la serie temporal en IMFs mediante EEMD (ensemble con ruido).

    Parámetros por defecto reproducen la configuración documentada para MSCI World
    en ``docs/16dic25/paper/main.tex`` (tabla de parámetros EEMD).

    Parameters
    ----------
    serie : np.ndarray
        Serie temporal 1D.
    max_imf : int, optional
        Número máximo de IMFs. Por defecto 14.
    trials : int, optional
        Realizaciones del ensemble. Por defecto 100.
    noise_width : float, optional
        Amplitud del ruido (definición PyEMD ``noise_width``). Por defecto 0.05.
    sd_thresh : float, optional
        Umbral SD del criterio de parada del sifting. Por defecto 0.25.
    s_number : int, optional
        Iteraciones mínimas de sifting. Por defecto 8.
    fixe_h : int, optional
        ``FIXE_H`` de PyEMD. Por defecto 5.
    seed : int, optional
        Semilla del ruido del ensemble. Por defecto 42.

    Returns
    -------
    pd.DataFrame
        Columnas IMF_1, IMF_2, ..., IMF_N y Residuo. Mismo número de filas que ``serie``.

    Raises
    ------
    ImportError
        Si PyEMD no está instalado.
    """
    if not EEMD_AVAILABLE or EEMD is None:
        raise ImportError("EEMD requiere PyEMD. Instalar con: pip install EMD-signal")
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
    Suma las columnas IMF_* de un DataFrame de descomposición (excluye ``Residuo``).

    Parameters
    ----------
    df_imfs : pd.DataFrame
        Columnas ``IMF_1``, ... y opcionalmente ``Residuo``.

    Returns
    -------
    np.ndarray
        Vector de la misma longitud que cada columna IMF.
    """
    cols_imf = [c for c in df_imfs.columns if c.startswith("IMF_")]
    if not cols_imf:
        return np.zeros(len(df_imfs), dtype=np.float64)
    acc = np.zeros(len(df_imfs), dtype=np.float64)
    for c in cols_imf:
        acc += np.asarray(df_imfs[c].values, dtype=np.float64)
    return acc


def calcular_tamaño_grafos_por_imf(
    df_imfs: pd.DataFrame,
    dir_salida: Path,
) -> pd.DataFrame:
    """
    Para cada columna (IMF o Residuo) construye NVG y HVG, guarda en parquet y .pt
    y devuelve un DataFrame con nodos, enlaces y tamaño en disco (parquet y .pt) por grafo.

    Parameters
    ----------
    df_imfs : pd.DataFrame
        DataFrame con columnas IMF_1, IMF_2, ..., Residuo.
    dir_salida : Path
        Directorio donde guardar los parquet de cada grafo (subcarpetas por id).

    Returns
    -------
    pd.DataFrame
        Columnas: id_imf, nodos, enlaces_nvg, enlaces_hvg, tam_parquet_nvg_mb,
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
        guardar_grafo_data(data_nvg, str(base_nvg), id_imf)
        tam_nvg = tamaño_parquet_grafo(base_nvg)
        tam_pt_nvg = tamaño_pt_grafo(base_nvg)
        # HVG
        data_hvg = serie_a_grafo_hvg(serie_imf)
        base_hvg = dir_salida / f"{id_imf}_hvg"
        guardar_grafo_data(data_hvg, str(base_hvg), id_imf)
        tam_hvg = tamaño_parquet_grafo(base_hvg)
        tam_pt_hvg = tamaño_pt_grafo(base_hvg)
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
    Genera figuras en inglés para el documento LaTeX en ``docs/20abr26``.

    El panel superior muestra la señal original y la suma de IMFs CEEMDAN (sin residuo).
    El panel inferior compara ``|Original − Σ IMFs|`` entre CEEMDAN y EEMD cuando hay
    datos EEMD. La figura ``hvg_imf.png`` usa la IMF de mayor frecuencia CEEMDAN.

    Parameters
    ----------
    df_imfs : pd.DataFrame
        Descomposición CEEMDAN: columnas ``IMF_1``, ..., ``IMF_N`` y ``Residuo``.
    serie_original : np.ndarray
        Serie de precios (Close) alineada con las filas de ``df_imfs``.
    directorio_imagenes : Path
        Directorio destino (p. ej. ``docs/20abr26/images/english``).
    df_imfs_eemd : pd.DataFrame, optional
        Descomposición EEMD (mismas convenciones de columnas). Solo afecta al panel
        de residuos. Si es None y PyEMD expone EEMD, se calcula con
        ``obtener_imfs_eemd(serie_original)``.
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
        df_eemd_usado = obtener_imfs_eemd(serie_original)

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
    enlaces = np.array(grafo_h.edges)
    g_nx = nx.Graph()
    g_nx.add_edges_from(map(tuple, enlaces))
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
    Genera un archivo Markdown de documentación para dimensionar el driver de
    nodos en un cluster de Azure Databricks según los tamaños de datos medidos
    (serie MSCI World, grafos en parquet y objetos Data en .pt).

    Parameters
    ----------
    archivo_salida : Path
        Ruta del archivo .md a generar.
    n_puntos_serie : int
        Longitud de la serie temporal (número de observaciones).
    tam_serie_parquet_mb : float
        Tamaño en disco del parquet de la serie (MB).
    num_nodos_nvg : int
        Número de nodos del grafo NVG (serie completa).
    num_enlaces_nvg : int
        Número de enlaces del grafo NVG.
    tam_parquet_nvg_mb : float
        Tamaño en disco del parquet del grafo NVG (MB).
    tam_pt_nvg_mb : float
        Tamaño en disco del .pt (objeto Data PyTorch Geometric) del grafo NVG (MB).
    num_nodos_hvg : int
        Número de nodos del grafo HVG (serie completa).
    num_enlaces_hvg : int
        Número de enlaces del grafo HVG.
    tam_parquet_hvg_mb : float
        Tamaño en disco del parquet del grafo HVG (MB).
    tam_pt_hvg_mb : float
        Tamaño en disco del .pt del grafo HVG (MB).
    tabla_imfs : pd.DataFrame, optional
        Tabla con tamaños por IMF (incluye tam_pt_nvg_mb, tam_pt_hvg_mb). Si es None, no se incluye.
    suma_parquet_nvg_imfs_mb : float, optional
        Suma de tamaños parquet NVG de todas las IMFs (MB).
    suma_parquet_hvg_imfs_mb : float, optional
        Suma de tamaños parquet HVG de todas las IMFs (MB).
    suma_pt_nvg_imfs_mb : float, optional
        Suma de tamaños .pt NVG de todas las IMFs (MB).
    suma_pt_hvg_imfs_mb : float, optional
        Suma de tamaños .pt HVG de todas las IMFs (MB).
    """
    lineas = []
    lineas.append("# Dimensionado de driver para Azure Databricks")
    lineas.append("")
    lineas.append("## Objetivo")
    lineas.append("")
    lineas.append(
        "Este documento resume los tamaños en disco de los datos de entrada del pipeline "
        "**MSCI World en grafos** (serie temporal convertida a grafos de visibilidad NVG/HVG, "
        "y opcionalmente descomposición CEEMDAN por IMF) para dimensionar el **tamaño del "
        "driver** de cada nodo del cluster en Azure Databricks."
    )
    lineas.append("")
    lineas.append("## Tipo de datos de entrada")
    lineas.append("")
    lineas.append("| Tipo | Formato | Descripción |")
    lineas.append("|------|---------|-------------|")
    lineas.append(
        "| Serie MSCI World | Parquet (1 columna Close) | Precios de cierre, un punto por día. |"
    )
    lineas.append(
        "| Grafo NVG (serie completa) | Parquet + .pt (Data PyG) | Natural Visibility Graph sobre toda la serie. |"
    )
    lineas.append(
        "| Grafo HVG (serie completa) | Parquet + .pt (Data PyG) | Horizontal Visibility Graph sobre toda la serie. |"
    )
    lineas.append(
        "| Grafos por IMF (CEEMDAN) | Parquet + .pt por cada IMF/Residuo | NVG y HVG; cada uno guardado como parquet (features/edges) y como .pt (objeto Data). |"
    )
    lineas.append("")
    lineas.append("## Tamaños medidos (referencia)")
    lineas.append("")
    lineas.append("### Serie y grafos de la serie completa")
    lineas.append("")
    lineas.append("| Concepto | Valor |")
    lineas.append("|----------|-------|")
    lineas.append(f"| Longitud de la serie (puntos) | {n_puntos_serie:,} |")
    lineas.append(
        f"| Tamaño en disco — serie (parquet) | {tam_serie_parquet_mb:.2f} MB |"
    )
    lineas.append(f"| Grafo NVG — nodos | {num_nodos_nvg:,} |")
    lineas.append(f"| Grafo NVG — enlaces | {num_enlaces_nvg:,} |")
    lineas.append(
        f"| Grafo NVG — parquet (features + edges) | {tam_parquet_nvg_mb:.2f} MB |"
    )
    lineas.append(f"| Grafo NVG — .pt (objeto Data PyG) | {tam_pt_nvg_mb:.2f} MB |")
    lineas.append(f"| Grafo HVG — nodos | {num_nodos_hvg:,} |")
    lineas.append(f"| Grafo HVG — enlaces | {num_enlaces_hvg:,} |")
    lineas.append(
        f"| Grafo HVG — parquet (features + edges) | {tam_parquet_hvg_mb:.2f} MB |"
    )
    lineas.append(f"| Grafo HVG — .pt (objeto Data PyG) | {tam_pt_hvg_mb:.2f} MB |")
    lineas.append("")
    if tabla_imfs is not None and not tabla_imfs.empty:
        lineas.append("### Grafos por IMF (CEEMDAN)")
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
                "| **Total (todas las IMFs)** | — | "
                f"**{suma_parquet_nvg_imfs_mb:.2f}** | **{suma_pt_nvg_imfs_mb:.2f}** | "
                f"**{suma_parquet_hvg_imfs_mb:.2f}** | **{suma_pt_hvg_imfs_mb:.2f}** |"
            )
            lineas.append("")
    lineas.append("## Cálculo recomendado para el driver en Databricks")
    lineas.append("")
    lineas.append(
        "En Azure Databricks, el **driver** es el nodo que coordina el cluster y puede necesitar "
    )
    lineas.append(
        "acceso a metadatos y datos pequeños. El tamaño del disco del driver debe cubrir:"
    )
    lineas.append("")
    lineas.append(
        "1. **Datos de entrada en disco** (parquet de serie y grafos, y archivos .pt de los objetos Data de PyTorch Geometric) que se lean o repartan desde el driver."
    )
    lineas.append(
        "2. **Overhead del sistema** (SO, JVM/Spark, logs): típicamente 10–20 GB."
    )
    lineas.append(
        "3. **Margen de seguridad**: factor multiplicativo sobre **(suma de datos + overhead)**. "
        "En este documento se usa **1,1×** (criterio más ajustado que factores ~1,5–2× a menudo "
        "citados solo sobre datos en bruto)."
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
    lineas.append("### Ejemplo con los tamaños medidos")
    lineas.append("")
    lineas.append("| Concepto | Valor |")
    lineas.append("|----------|-------|")
    lineas.append(
        f"| Suma datos (serie + parquet y .pt de grafos serie + grafos IMFs) | {tam_datos_mb:.2f} MB ({tam_datos_gb:.2f} GB) |"
    )
    lineas.append(f"| Overhead sistema (referencia) | {overhead_gb} GB |")
    lineas.append(f"| Factor margen | {factor_margen}× |")
    lineas.append(f"| **Driver mínimo recomendado** | **{driver_min_gb:.1f} GB** |")
    lineas.append("")
    lineas.append(
        "En Databricks, elegir un tipo de nodo (o configuración de cluster) cuyo **disk size** "
        "sea al menos este valor. Si solo se usa la serie completa (sin CEEMDAN), el total de "
        "datos es menor; si se incluyen todos los grafos por IMF, usar el total indicado arriba."
    )
    lineas.append("")
    lineas.append("---")
    lineas.append("*Documento generado automáticamente por `info_msci_world_data.py`.*")
    archivo_salida.parent.mkdir(parents=True, exist_ok=True)
    archivo_salida.write_text("\n".join(lineas), encoding="utf-8")


def main() -> None:
    """
    Punto de entrada: descarga MSCI World si no existe (usando DATA_PATH como
    download_data.py), luego imprime información de la serie y de los grafos
    NVG/HVG (nodos, enlaces, tamaño en disco en parquet).
    """
    proyecto_root = _proyecto_root

    descargar_msci_world(data_dir=DATA_PATH)
    archivo_msci = _buscar_archivo_msci_world(proyecto_root, data_path=DATA_PATH)
    print("=" * 60)
    print("SERIE MSCI WORLD")
    print("=" * 60)
    print(f"Archivo: {archivo_msci}")
    print(
        f"Tamaño en disco (parquet serie): {archivo_msci.stat().st_size / (1024**2):.2f} MB"
    )

    df, serie = cargar_serie_msci_world(archivo_msci)
    n_puntos = len(serie)
    if hasattr(df.index, "min") and hasattr(df.index, "max"):
        print(f"Rango fechas: {df.index.min()} a {df.index.max()}")
    print(f"Número de puntos (longitud serie): {n_puntos}")
    print(
        f"Close - min: {serie.min():.4f}, max: {serie.max():.4f}, mean: {serie.mean():.4f}"
    )

    # Directorio de salida para grafos (parquet)
    dir_salida = Path(__file__).resolve().parent / "out_msci_world_grafos"
    dir_salida.mkdir(parents=True, exist_ok=True)
    base_nvg = dir_salida / "msci_world_nvg_serie_completa"
    base_hvg = dir_salida / "msci_world_hvg_serie_completa"

    # NVG (algoritmo más denso)
    print("\n" + "=" * 60)
    print("GRAFO NVG (Natural Visibility Graph) - Serie completa, sin ventanas")
    print("=" * 60)
    data_nvg = serie_a_grafo_nvg(serie)
    print(f"Número de nodos: {data_nvg.num_nodes}")
    print(f"Número de enlaces: {data_nvg.num_edges}")
    guardar_grafo_data(data_nvg, str(base_nvg), "msci_world_serie")
    tam_parquet_nvg = tamaño_parquet_grafo(base_nvg)
    tam_pt_nvg = tamaño_pt_grafo(base_nvg)
    print(
        f"Tamaño en disco (solo parquet: features + edges): {tam_parquet_nvg / (1024**2):.2f} MB"
    )
    print(f"Tamaño en disco (.pt objeto Data): {tam_pt_nvg / (1024**2):.2f} MB")

    # HVG
    print("\n" + "=" * 60)
    print("GRAFO HVG (Horizontal Visibility Graph) - Serie completa, sin ventanas")
    print("=" * 60)
    data_hvg = serie_a_grafo_hvg(serie)
    print(f"Número de nodos: {data_hvg.num_nodes}")
    print(f"Número de enlaces: {data_hvg.num_edges}")
    guardar_grafo_data(data_hvg, str(base_hvg), "msci_world_serie")
    tam_parquet_hvg = tamaño_parquet_grafo(base_hvg)
    tam_pt_hvg = tamaño_pt_grafo(base_hvg)
    print(
        f"Tamaño en disco (solo parquet: features + edges): {tam_parquet_hvg / (1024**2):.2f} MB"
    )
    print(f"Tamaño en disco (.pt objeto Data): {tam_pt_hvg / (1024**2):.2f} MB")

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(
        f"Serie: {n_puntos} puntos | NVG: {data_nvg.num_nodes} nodos, {data_nvg.num_edges} enlaces | HVG: {data_hvg.num_nodes} nodos, {data_hvg.num_edges} enlaces"
    )
    print(
        f"Parquet NVG: {tam_parquet_nvg / (1024**2):.2f} MB | Parquet HVG: {tam_parquet_hvg / (1024**2):.2f} MB"
    )

    # CEEMDAN: tamaño del grafo de cada IMF
    tabla_imfs = None
    suma_parquet_nvg_imfs_mb = None
    suma_parquet_hvg_imfs_mb = None
    suma_pt_nvg_imfs_mb = None
    suma_pt_hvg_imfs_mb = None
    if CEEMDAN_AVAILABLE:
        print("\n" + "=" * 60)
        print("CEEMDAN - Tamaño del grafo por IMF")
        print("=" * 60)
        print("Descomponiendo serie completa con CEEMDAN...")
        df_imfs = obtener_imfs_ceemdan(
            serie,
            max_imf=CEEMDAN_MAX_IMF,
            trials=CEEMDAN_TRIALS,
            epsilon=CEEMDAN_EPSILON,
            seed=CEEMDAN_SEED,
            parallel=False,
        )
        print(f"IMFs obtenidas: {[c for c in df_imfs.columns if c.startswith('IMF_')]}")
        print(f"Residuo: {'Sí' if 'Residuo' in df_imfs.columns else 'No'}")
        out_imfs_parquet = (
            proyecto_root / "data" / "20abr26" / "msci_world_imfs_ceemdan.parquet"
        )
        out_imfs_parquet.parent.mkdir(parents=True, exist_ok=True)
        df_imfs.to_parquet(out_imfs_parquet, index=False)
        print(f"✓ IMFs CEEMDAN guardadas en: {out_imfs_parquet}")
        docs_img = proyecto_root / "docs" / "20abr26" / "images" / "english"
        df_imfs_eemd: Optional[pd.DataFrame] = None
        if EEMD_AVAILABLE:
            print("Descomponiendo serie completa con EEMD (parámetros docs/16dic25)...")
            df_imfs_eemd = obtener_imfs_eemd(serie)
            out_eemd_parquet = (
                proyecto_root / "data" / "20abr26" / "msci_world_imfs_eemd.parquet"
            )
            df_imfs_eemd.to_parquet(out_eemd_parquet, index=False)
            print(f"✓ IMFs EEMD guardadas en: {out_eemd_parquet}")
        else:
            print("(Omitido EEMD en figura comparada: PyEMD sin EEMD.)")
        exportar_figuras_documento_20abr26(
            df_imfs, serie, docs_img, df_imfs_eemd=df_imfs_eemd
        )
        print(f"✓ Figuras para LaTeX (docs/20abr26): {docs_img}")
        dir_imfs = dir_salida / "imfs_ceemdan"
        tabla_imfs = calcular_tamaño_grafos_por_imf(df_imfs, dir_imfs)
        print("\nTabla de tamaños (parquet) por IMF:")
        print(tabla_imfs.to_string(index=False))
        suma_parquet_nvg_imfs_mb = tabla_imfs["tam_parquet_nvg_mb"].sum()
        suma_parquet_hvg_imfs_mb = tabla_imfs["tam_parquet_hvg_mb"].sum()
        suma_pt_nvg_imfs_mb = tabla_imfs["tam_pt_nvg_mb"].sum()
        suma_pt_hvg_imfs_mb = tabla_imfs["tam_pt_hvg_mb"].sum()
        print(f"\nSuma parquet NVG (todas las IMFs): {suma_parquet_nvg_imfs_mb:.2f} MB")
        print(f"Suma parquet HVG (todas las IMFs): {suma_parquet_hvg_imfs_mb:.2f} MB")
        print(f"Suma .pt NVG (todas las IMFs): {suma_pt_nvg_imfs_mb:.2f} MB")
        print(f"Suma .pt HVG (todas las IMFs): {suma_pt_hvg_imfs_mb:.2f} MB")
        archivo_tabla = dir_imfs / "tamaño_grafo_por_imf.csv"
        tabla_imfs.to_csv(archivo_tabla, index=False)
        print(f"Tabla guardada en: {archivo_tabla}")
    else:
        print("\n(Omitido CEEMDAN: instalar EMD-signal para calcular tamaño por IMF.)")

    # # Documentación Markdown para dimensionado del driver en Azure Databricks
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
    # print(f"\n✓ Documentación de dimensionado guardada en: {archivo_md}")


if __name__ == "__main__":
    main()
