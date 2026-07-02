#!/usr/bin/env python3
"""
VMD pipeline for all financial signals in the GraphEMD project.

For each asset (MSCI World, XLE, XLP, XLV, XAU/USD), runs:

1. VMD hyperparameter calibration (K × alpha × DC grid).
2. Mode decomposition and persistence to parquet.
3. Validation (reconstruction, residual, per-IMF metrics, panel figure).
4. Graph transformations (HVG, NVG, recurrence).
5. FastICA with automatic selection of optimal k.

Example::

    PYTHONPATH=src/python python scripts/run_vmd_all_assets.py

    PYTHONPATH=src/python python scripts/run_vmd_all_assets.py \\
        --activos xle,xauusd --solo-paso ica
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_PYTHON = _REPO_ROOT / "src" / "python"
_EXPLORATION = _REPO_ROOT / "scripts" / "exploration"
_EMDSYNTH = _REPO_ROOT / "scripts" / "emdsynth"

N_OBSERVACIONES_ESPERADAS: int = 3587
UMBRAL_RMSE_RELATIVO: float = 1e-10

REJILLA_VMD_K: List[int] = [6, 8, 10, 12]
REJILLA_VMD_ALPHA: List[float] = [1000.0, 2000.0, 5000.0]
REJILLA_VMD_DC: List[int] = [0, 1]

TAU_MAX: int = 50
DIM_MAX: int = 10
UMBRAL_PERCENTIL_RECURRENCIA: float = 10.0
RANDOM_STATE_RECURRENCIA: int = 42

RANDOM_STATE_ICA: int = 42
K_MIN_CALIBRACION_ICA: int = 2
UMBRAL_CORR_Z: float = 0.01
FRACCION_R2_OBJETIVO: float = 0.72
FACTOR_ERROR_FROB_RODILLA: float = 1.15

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfigActivo:
    """
    Path and metadata configuration for a financial asset.

    Attributes
    ----------
    id_activo : str
        Short identifier (e.g. ``xle``).
    nombre : str
        Human-readable label.
    dir_datos : Path
        Asset data directory.
    ruta_precios : Path
        Parquet with closing prices.
    prefijo : str
        Output file prefix (e.g. ``xle`` or ``msci_world``).
    columna_precio : str
        Price column name.
    """

    id_activo: str
    nombre: str
    dir_datos: Path
    ruta_precios: Path
    prefijo: str
    columna_precio: str = "Close"


ACTIVOS: Tuple[ConfigActivo, ...] = (
    ConfigActivo(
        id_activo="msci_world",
        nombre="MSCI World",
        dir_datos=_REPO_ROOT / "data" / "20abr26",
        ruta_precios=_REPO_ROOT / "data" / "20abr26" / "msci_world.parquet",
        prefijo="msci_world",
    ),
    ConfigActivo(
        id_activo="xle",
        nombre="XLE",
        dir_datos=_REPO_ROOT / "data" / "GraphEMD" / "xle_etf_analysis",
        ruta_precios=_REPO_ROOT
        / "data"
        / "GraphEMD"
        / "xle_etf_analysis"
        / "xle.parquet",
        prefijo="xle",
    ),
    ConfigActivo(
        id_activo="xlp",
        nombre="XLP",
        dir_datos=_REPO_ROOT / "data" / "GraphEMD" / "xlp_analysis",
        ruta_precios=_REPO_ROOT / "data" / "GraphEMD" / "xlp_analysis" / "xlp.parquet",
        prefijo="xlp",
    ),
    ConfigActivo(
        id_activo="xlv",
        nombre="XLV",
        dir_datos=_REPO_ROOT / "data" / "GraphEMD" / "xlv_analysis",
        ruta_precios=_REPO_ROOT / "data" / "GraphEMD" / "xlv_analysis" / "xlv.parquet",
        prefijo="xlv",
    ),
    ConfigActivo(
        id_activo="xauusd",
        nombre="XAU/USD",
        dir_datos=_REPO_ROOT / "data" / "GraphEMD" / "xauusd_analysis",
        ruta_precios=_REPO_ROOT
        / "data"
        / "GraphEMD"
        / "xauusd_analysis"
        / "xauusd.parquet",
        prefijo="xauusd",
    ),
)


def _asegurar_paths() -> None:
    """
    Add project import paths to ``sys.path``.
    """
    for ruta in (_SRC_PYTHON, _EXPLORATION):
        s = str(ruta)
        if s not in sys.path:
            sys.path.insert(0, s)


def _cargar_modulo_emdsynth() -> Any:
    """
    Load the emdsynth pipeline with ``descomponer_vmd`` and ``calcular_metricas``.

    Returns
    -------
    module
        ``run_emdsynth_decompositions`` module.
    """
    ruta = _EMDSYNTH / "run_emdsynth_decompositions.py"
    spec = importlib.util.spec_from_file_location("emdsynth_vmd_activos", ruta)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {ruta}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cargar_modulo_validacion() -> Any:
    """
    Load validation functions from the XLE script 04.

    Returns
    -------
    module
        Module with ``validar_reconstruccion``, ``evaluar_residuo``, etc.
    """
    ruta = (
        _REPO_ROOT
        / "scripts"
        / "xle_etf_analysis"
        / "04_ceemdan_xle_validation_and_results.py"
    )
    spec = importlib.util.spec_from_file_location("validacion_vmd_activos", ruta)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {ruta}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cargar_modulo_ica() -> Any:
    """
    Load ICA utilities from ``reduce_ceemdan_imf_dimensionality``.

    Returns
    -------
    module
        Module with FastICA, k selection, and persistence.
    """
    _asegurar_paths()
    ruta = _EXPLORATION / "reduce_ceemdan_imf_dimensionality.py"
    spec = importlib.util.spec_from_file_location("ica_vmd_activos", ruta)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {ruta}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rutas_activo(cfg: ConfigActivo) -> Dict[str, Path]:
    """
    Return standard VMD output paths for an asset.

    Parameters
    ----------
    cfg : ConfigActivo
        Asset configuration.

    Returns
    -------
    dict
        Map of logical names to paths.
    """
    d = cfg.dir_datos
    p = cfg.prefijo
    return {
        "imfs": d / f"{p}_imfs_vmd.parquet",
        "parametros": d / f"{p}_vmd_parametros.json",
        "validacion": d / f"{p}_validacion_vmd.json",
        "metricas_imf_csv": d / f"{p}_imf_metricas_vmd.csv",
        "dir_grafos": d / "grafos_vmd",
        "resumen_grafos_csv": d / f"{p}_resumen_grafos_vmd.csv",
        "param_recurrencia_csv": d / f"{p}_parametros_recurrencia_vmd.csv",
        "manifest_grafos": d / f"{p}_grafos_vmd_manifest.json",
        "dir_ica": d / "ica_vmd" / "fastica",
        "parametros_ica": d / f"{p}_ica_vmd_parametros.json",
        "dir_figuras": d / "figures",
        "figura_panel": d / "figures" / f"{p}_imfs_panel_vmd.png",
        "figura_ica": d / "figures" / f"{p}_componentes_ica_vmd.png",
        "figura_comparativa_residuos": d
        / "figures"
        / f"{p}_imf_decomposition_ceemdan_eemd_vmd.png",
        "imfs_ceemdan": d / f"{p}_imfs_ceemdan.parquet",
        "imfs_eemd": d / f"{p}_imfs_eemd.parquet",
    }


def cargar_serie_precios(cfg: ConfigActivo) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load closing prices for the asset.

    Parameters
    ----------
    cfg : ConfigActivo
        Asset configuration.

    Returns
    -------
    tuple
        Price DataFrame and index, 1D closing vector.

    Raises
    ------
    FileNotFoundError
        If the price parquet does not exist.
    """
    if not cfg.ruta_precios.is_file():
        raise FileNotFoundError(
            f"Not found: {cfg.ruta_precios}. Run the download script."
        )
    df = pd.read_parquet(cfg.ruta_precios, engine="pyarrow")
    if cfg.columna_precio not in df.columns:
        raise ValueError(
            f"Missing column {cfg.columna_precio} in {cfg.ruta_precios}. "
            f"Columnas: {list(df.columns)}"
        )
    if len(df) != N_OBSERVACIONES_ESPERADAS:
        logger.warning(
            "%s: expected %s observations; found %s.",
            cfg.nombre,
            N_OBSERVACIONES_ESPERADAS,
            len(df),
        )
    serie = np.asarray(df[cfg.columna_precio].values, dtype=np.float64)
    logger.info(
        "%s: serie cargada n=%s, rango %s → %s",
        cfg.nombre,
        len(serie),
        df.index.min(),
        df.index.max(),
    )
    return df, serie


def imfs_array_a_dataframe(imfs: np.ndarray) -> pd.DataFrame:
    """
    Convert VMD modes (last row = residual) to IMF_* + Residuo DataFrame.

    Parameters
    ----------
    imfs : np.ndarray
        Matrix ``(n_modes, n_samples)``.

    Returns
    -------
    pd.DataFrame
        Columns ``IMF_1`` … ``IMF_K`` and ``Residuo``.
    """
    n_imfs = imfs.shape[0] - 1
    datos: Dict[str, np.ndarray] = {}
    for i in range(n_imfs):
        datos[f"IMF_{i + 1}"] = imfs[i]
    datos["Residuo"] = imfs[-1]
    return pd.DataFrame(datos)


def _columnas_imf_ordenadas(df: pd.DataFrame) -> List[str]:
    """
    Return ``IMF_*`` names sorted numerically.

    Parameters
    ----------
    df : pd.DataFrame
        Table with IMF columns.

    Returns
    -------
    list of str
        IMF columns in order.
    """
    cols = [c for c in df.columns if c.startswith("IMF_")]
    return sorted(cols, key=lambda x: int(x.split("_")[1]))


def _alinear_df_imfs_longitud(df_imfs: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Trim or validate an IMF DataFrame to match ``n`` samples.

    Parameters
    ----------
    df_imfs : pd.DataFrame
        CEEMDAN/EEMD/VMD IMFs.
    n : int
        Target length (e.g. closing prices).

    Returns
    -------
    pd.DataFrame
        Table aligned to ``n`` rows.

    Raises
    ------
    ValueError
        If there are fewer rows than ``n``.
    """
    if len(df_imfs) == n:
        return df_imfs
    if len(df_imfs) > n:
        logger.warning(
            "IMFs with %s rows; first %s rows trimmed to align with the signal "
            "(``_alinear_serie_e_imfs`` convention in CEEMDAN validation).",
            len(df_imfs),
            n,
        )
        return df_imfs.iloc[:n].copy()
    raise ValueError(
        f"IMFs with {len(df_imfs)} rows; at least {n} required for alignment."
    )


def _brecha_residuo_implicito(serie: np.ndarray, df_imfs: pd.DataFrame) -> np.ndarray:
    """
    Compute ``|x - Σ IMF_*|`` (same definition as CEEMDAN/EEMD in the residual panel).

    Parameters
    ----------
    serie : np.ndarray
        Original signal.
    df_imfs : pd.DataFrame
        Decomposition with ``IMF_*`` columns.

    Returns
    -------
    np.ndarray
        Reconstruction gap magnitude without ``Residuo`` column.
    """
    cols = _columnas_imf_ordenadas(df_imfs)
    suma = np.sum(
        [np.asarray(df_imfs[c].values, dtype=np.float64) for c in cols],
        axis=0,
    )
    return np.abs(serie - suma)


def _suma_imfs_sin_residuo(df_imfs: pd.DataFrame) -> np.ndarray:
    """
    Sum all ``IMF_*`` columns (excludes ``Residuo``).

    Parameters
    ----------
    df_imfs : pd.DataFrame
        Decomposition with IMF columns.

    Returns
    -------
    np.ndarray
        Sum of oscillatory modes excluding explicit residual.
    """
    cols = _columnas_imf_ordenadas(df_imfs)
    return np.sum(
        [np.asarray(df_imfs[c].values, dtype=np.float64) for c in cols],
        axis=0,
    )


def _suma_imfs_oscilatorias_vmd(df_imfs: pd.DataFrame, vmd_dc: int) -> np.ndarray:
    """
    Sum of IMFs excluding the trend mode in VMD.

    With ``DC=1``, skips ``IMF_1`` (DC mode); otherwise sums all ``IMF_*``.

    Parameters
    ----------
    df_imfs : pd.DataFrame
        VMD decomposition.
    vmd_dc : int
        ``DC`` value used in calibration.

    Returns
    -------
    np.ndarray
        Sum of oscillatory modes per VMD convention.
    """
    cols = _columnas_imf_ordenadas(df_imfs)
    if vmd_dc == 1 and len(cols) >= 2:
        cols_osc = cols[1:]
    else:
        cols_osc = cols
    return np.sum(
        [np.asarray(df_imfs[c].values, dtype=np.float64) for c in cols_osc],
        axis=0,
    )


def _brecha_tendencia_vmd_dc1(serie: np.ndarray, df_imfs: pd.DataFrame) -> np.ndarray:
    """
    VMD trend gap with ``DC=1``: ``|x - Σ_{i≥2} IMF_i|`` ≈ DC mode (``IMF_1``).

    Parameters
    ----------
    serie : np.ndarray
        Original signal.
    df_imfs : pd.DataFrame
        VMD decomposition with ``DC=1``.

    Returns
    -------
    np.ndarray
        Magnitude associated with the low-frequency / trend mode.
    """
    cols = _columnas_imf_ordenadas(df_imfs)
    if len(cols) < 2:
        return _brecha_residuo_implicito(serie, df_imfs)
    cols_osc = cols[1:]
    suma_osc = np.sum(
        [np.asarray(df_imfs[c].values, dtype=np.float64) for c in cols_osc],
        axis=0,
    )
    return np.abs(serie - suma_osc)


def calibrar_vmd_activo(
    serie: np.ndarray,
    mod_emdsynth: Any,
    dc_fijo: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Search VMD hyperparameters that minimize linear coupling between modes.

    Parameters
    ----------
    serie : np.ndarray
        1D closing prices.
    mod_emdsynth : module
        EmdSynth module with ``descomponer_vmd`` and ``calcular_metricas``.
    dc_fijo : int, optional
        If set (0 or 1), fixes ``DC`` and only sweeps ``K`` and ``alpha``.

    Returns
    -------
    dict
        ``mejor``, ``rejilla``, and ``evaluaciones`` blocks.
    """
    descomponer_vmd = mod_emdsynth.descomponer_vmd
    calcular_metricas = mod_emdsynth.calcular_metricas
    rejilla_dc = [int(dc_fijo)] if dc_fijo is not None else REJILLA_VMD_DC

    def _clave(mets: Dict[str, float]) -> Tuple[float, float]:
        return (
            float(mets["corr_promedio_pares"]),
            float(mets["frac_energia_residuo"]),
        )

    evaluaciones: List[Dict[str, Any]] = []
    mejor: Optional[Dict[str, Any]] = None
    mejor_clave: Optional[Tuple[float, float]] = None

    for k_vmd, alpha_vmd, dc_vmd in itertools.product(
        REJILLA_VMD_K, REJILLA_VMD_ALPHA, rejilla_dc
    ):
        imfs = descomponer_vmd(
            serie,
            k=k_vmd,
            alpha=alpha_vmd,
            dc=dc_vmd,
        )
        mets = calcular_metricas(serie, imfs)
        fila = {
            "K": k_vmd,
            "alpha": alpha_vmd,
            "DC": dc_vmd,
            "metricas": {k: float(v) for k, v in mets.items()},
        }
        evaluaciones.append(fila)
        if mets["rmse_relativo"] >= UMBRAL_RMSE_RELATIVO:
            continue
        cl = _clave(mets)
        if mejor_clave is None or cl < mejor_clave:
            mejor_clave = cl
            mejor = {
                "K": k_vmd,
                "alpha": alpha_vmd,
                "DC": dc_vmd,
                "metricas": {k: float(v) for k, v in mets.items()},
            }

    if mejor is None:
        raise RuntimeError("VMD calibration: no valid configuration for the signal.")

    criterio = "min_corr_promedio_pares_luego_frac_energia_residuo"
    if dc_fijo is not None:
        criterio = f"dc_fijo={dc_fijo}; " + criterio

    return {
        "criterio_seleccion": criterio,
        "dc_fijo": dc_fijo,
        "mejor": mejor,
        "rejilla": {
            "K": REJILLA_VMD_K,
            "alpha": REJILLA_VMD_ALPHA,
            "DC": rejilla_dc,
        },
        "evaluaciones": evaluaciones,
    }


def ejecutar_vmd_activo(
    cfg: ConfigActivo,
    mod_emdsynth: Any,
    calibrar: bool = True,
    parametros_fijos: Optional[Dict[str, Any]] = None,
    dc_fijo: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Optionally calibrate and decompose the signal with VMD.

    Parameters
    ----------
    cfg : ConfigActivo
        Asset to process.
    mod_emdsynth : module
        EmdSynth module.
    calibrar : bool
        If True, runs calibration grid.
    parametros_fijos : dict, optional
        VMD hyperparameters if ``calibrar=False``.
    dc_fijo : int, optional
        Fix ``DC`` during calibration (e.g. 1 for trend mode).

    Returns
    -------
    dict
        Calibration, IMF DataFrame, and written paths.
    """
    rutas = _rutas_activo(cfg)
    df_precios, serie = cargar_serie_precios(cfg)

    if calibrar:
        logger.info("=" * 60)
        logger.info("VMD CALIBRATION — %s (fixed DC=%s)", cfg.nombre, dc_fijo)
        logger.info("=" * 60)
        calibracion = calibrar_vmd_activo(serie, mod_emdsynth, dc_fijo=dc_fijo)
    else:
        dc_def = int(dc_fijo) if dc_fijo is not None else 1
        params = parametros_fijos or {"K": 6, "alpha": 5000.0, "DC": dc_def}
        calibracion = {"mejor": params, "criterio_seleccion": "parametros_fijos"}

    mejor = calibracion["mejor"]
    imfs = mod_emdsynth.descomponer_vmd(
        serie,
        k=int(mejor["K"]),
        alpha=float(mejor["alpha"]),
        dc=int(mejor["DC"]),
    )
    df_imfs = imfs_array_a_dataframe(imfs)
    df_imfs.index = df_precios.index

    cfg.dir_datos.mkdir(parents=True, exist_ok=True)
    df_imfs.to_parquet(rutas["imfs"], engine="pyarrow", index=True)

    var_serie = float(np.var(serie))
    resumen = {
        "n_imfs": int(sum(1 for c in df_imfs.columns if c.startswith("IMF_"))),
        "varianza_serie": var_serie,
        "frac_varianza_por_componente": {
            col: float(np.var(df_imfs[col].values) / (var_serie + 1e-15))
            for col in df_imfs.columns
        },
        "columnas": list(df_imfs.columns),
    }

    payload = {
        "activo": cfg.id_activo,
        "nombre": cfg.nombre,
        "metodo": "VMD",
        "calibracion": calibracion,
        "resumen_descomposicion": resumen,
    }
    with open(rutas["parametros"], "w", encoding="utf-8") as archivo:
        json.dump(payload, archivo, indent=2, ensure_ascii=False)

    m = mejor.get("metricas", mod_emdsynth.calcular_metricas(serie, imfs))
    logger.info(
        "%s VMD: K=%s alpha=%s DC=%s n_modos=%s rmse_rel=%.2e corr=%.4f",
        cfg.nombre,
        mejor["K"],
        mejor["alpha"],
        mejor["DC"],
        int(m["n_modos"]),
        m["rmse_relativo"],
        m["corr_promedio_pares"],
    )
    logger.info("IMFs saved: %s", rutas["imfs"])

    return {
        "cfg": cfg,
        "df_precios": df_precios,
        "serie": serie,
        "df_imfs": df_imfs,
        "calibracion": calibracion,
        "rutas": rutas,
    }


def generate_comparative_residual_figure(
    cfg: ConfigActivo,
    serie: np.ndarray,
    df_imfs_vmd: pd.DataFrame,
    vmd_dc: int,
    ruta_salida: Path,
) -> Optional[Path]:
    """
    CEEMDAN vs EEMD style figure adding the VMD trend gap (``DC=1``).

    Parameters
    ----------
    cfg : ConfigActivo
        Asset.
    serie : np.ndarray
        Closing prices.
    df_imfs_vmd : pd.DataFrame
        Newly computed VMD modes.
    vmd_dc : int
        ``DC`` value used in VMD (1 enables comparison via ``IMF_1``).
    ruta_salida : Path
        Output PNG.

    Returns
    -------
    Path or None
        Written path, or None if CEEMDAN/EEMD parquets are missing.
    """
    rutas = _rutas_activo(cfg)
    if not rutas["imfs_ceemdan"].is_file():
        logger.warning(
            "%s: missing %s; skipping comparative figure.",
            cfg.nombre,
            rutas["imfs_ceemdan"],
        )
        return None

    df_ceemdan = _alinear_df_imfs_longitud(
        pd.read_parquet(rutas["imfs_ceemdan"], engine="pyarrow"),
        len(serie),
    )
    brecha_ceemdan = _brecha_residuo_implicito(serie, df_ceemdan)
    suma_ceemdan = _suma_imfs_sin_residuo(df_ceemdan)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(serie, label="Original", linewidth=0.8, color="C0")
    axes[0].plot(
        suma_ceemdan,
        label="Σ IMFs (CEEMDAN, without residual)",
        linewidth=0.8,
        alpha=0.85,
        color="C1",
    )
    axes[1].plot(
        brecha_ceemdan,
        label="|Original − Σ IMFs| (CEEMDAN)",
        linewidth=0.7,
        color="C1",
    )

    if rutas["imfs_eemd"].is_file():
        df_eemd = _alinear_df_imfs_longitud(
            pd.read_parquet(rutas["imfs_eemd"], engine="pyarrow"),
            len(serie),
        )
        suma_eemd = _suma_imfs_sin_residuo(df_eemd)
        brecha_eemd = _brecha_residuo_implicito(serie, df_eemd)
        axes[0].plot(
            suma_eemd,
            label="Σ IMFs (EEMD, without residual)",
            linewidth=0.8,
            alpha=0.85,
            color="C2",
        )
        axes[1].plot(
            brecha_eemd,
            label="|Original − Σ IMFs| (EEMD)",
            linewidth=0.7,
            color="C2",
            alpha=0.9,
        )

    suma_vmd = _suma_imfs_oscilatorias_vmd(df_imfs_vmd, vmd_dc=vmd_dc)
    etiqueta_suma_vmd = (
        "Σ IMF_{2..K} (VMD, DC=1)" if vmd_dc == 1 else "Σ IMFs (VMD, without residual)"
    )
    axes[0].plot(
        suma_vmd,
        label=etiqueta_suma_vmd,
        linewidth=0.8,
        alpha=0.85,
        color="C3",
    )

    if vmd_dc == 1:
        brecha_vmd = _brecha_tendencia_vmd_dc1(serie, df_imfs_vmd)
        axes[1].plot(
            brecha_vmd,
            label="|Original − Σ IMF_{2..K}| (VMD, DC=1)",
            linewidth=0.7,
            color="C3",
            alpha=0.9,
        )
    else:
        brecha_vmd = _brecha_residuo_implicito(serie, df_imfs_vmd)
        axes[1].plot(
            brecha_vmd,
            label="|Original − Σ IMFs| (VMD)",
            linewidth=0.7,
            color="C3",
            alpha=0.9,
        )

    axes[0].set_ylabel("Level")
    axes[0].legend(loc="upper left", fontsize=7)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("|Original − Σ IMFs|")
    axes[1].set_xlabel("Time index")
    axes[1].legend(loc="upper left", fontsize=7)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Comparative residuals figure: %s", ruta_salida)
    return ruta_salida


def validar_vmd_activo(
    cfg: ConfigActivo,
    df_precios: pd.DataFrame,
    serie: np.ndarray,
    df_imfs: pd.DataFrame,
    mod_validacion: Any,
    vmd_dc: int = 1,
) -> Dict[str, Any]:
    """
    Validate VMD decomposition and generate metrics and panel figure.

    Parameters
    ----------
    cfg : ConfigActivo
        Asset.
    df_precios : pd.DataFrame
        Prices with time index.
    serie : np.ndarray
        Closing vector.
    df_imfs : pd.DataFrame
        VMD modes.
    mod_validacion : module
        Reused CEEMDAN validation functions.
    vmd_dc : int
        ``DC`` value used in VMD decomposition.

    Returns
    -------
    dict
        Validation report and paths.
    """
    rutas = _rutas_activo(cfg)
    recon = mod_validacion.validar_reconstruccion(serie, df_imfs)
    residuo = mod_validacion.evaluar_residuo(serie, df_imfs)
    if vmd_dc == 1:
        brecha_tendencia = _brecha_tendencia_vmd_dc1(serie, df_imfs)
        imf1 = np.asarray(df_imfs["IMF_1"].values, dtype=np.float64)
        residuo["brecha_tendencia_dc1_std"] = float(np.std(brecha_tendencia))
        residuo["brecha_tendencia_dc1_media"] = float(np.mean(brecha_tendencia))
        residuo["corr_brecha_tendencia_imf1"] = float(
            np.corrcoef(brecha_tendencia, imf1)[0, 1]
        )
        residuo["interpretacion_dc1"] = (
            "Con DC=1, IMF_1 captura la tendencia; la brecha |x-ΣIMF_{2..K}| es comparable "
            "to the implicit CEEMDAN/EEMD residual."
        )
    tabla = mod_validacion.tabla_metricas_imfs(df_imfs)

    rutas["metricas_imf_csv"].parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(rutas["metricas_imf_csv"])

    generar_panel_imfs_vmd(
        df_precios,
        df_imfs,
        serie,
        cfg.nombre,
        rutas["figura_panel"],
    )
    ruta_comparativa = generate_comparative_residual_figure(
        cfg,
        serie,
        df_imfs,
        vmd_dc=vmd_dc,
        ruta_salida=rutas["figura_comparativa_residuos"],
    )

    informe = {
        "activo": cfg.id_activo,
        "nombre": cfg.nombre,
        "metodo": "VMD",
        "n_observaciones": int(len(serie)),
        "n_imfs": int(sum(1 for c in df_imfs.columns if c.startswith("IMF_"))),
        "reconstruccion": recon,
        "residuo": residuo,
        "vmd_dc": int(vmd_dc),
        "figura_comparativa_residuos": (
            str(ruta_comparativa) if ruta_comparativa is not None else None
        ),
        "metricas_imf": tabla.reset_index().to_dict(orient="records"),
    }
    with open(rutas["validacion"], "w", encoding="utf-8") as archivo:
        json.dump(informe, archivo, indent=2, ensure_ascii=False)

    logger.info(
        "%s validation: rmse_rel=%.2e, residual R²=%.4f, Spearman=%.4f",
        cfg.nombre,
        recon["rmse_relativo"],
        residuo["r2_regresion_lineal"],
        residuo["spearman_tiempo"],
    )
    return informe


def generar_panel_imfs_vmd(
    df_precios: pd.DataFrame,
    df_imfs: pd.DataFrame,
    serie: np.ndarray,
    nombre_activo: str,
    ruta_salida: Path,
) -> None:
    """
    Generate stacked figure with closing price and VMD modes.

    Parameters
    ----------
    df_precios : pd.DataFrame
        Prices (time index).
    df_imfs : pd.DataFrame
        VMD modes.
    serie : np.ndarray
        Closing vector.
    nombre_activo : str
        Title label.
    ruta_salida : Path
        PNG path.
    """
    cols_imf = sorted(
        [c for c in df_imfs.columns if c.startswith("IMF_")],
        key=lambda x: int(x.split("_")[1]),
    )
    columnas = (
        ["Close"] + cols_imf + (["Residuo"] if "Residuo" in df_imfs.columns else [])
    )
    n = len(columnas)
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.6 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, nombre in zip(axes, columnas):
        y = serie if nombre == "Close" else df_imfs[nombre].values
        ax.plot(df_precios.index, y, linewidth=0.6)
        ax.set_ylabel(nombre, fontsize=8)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Fecha")
    fig.tight_layout()
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("VMD panel figure: %s", ruta_salida)


def _load_build_all_imf_graphs() -> Any:
    """
    Load ``build_all_imf_graphs`` without importing ``GraphEMD.data`` (depends on torch).

    Returns
    -------
    callable
        ``build_all_imf_graphs`` function.
    """
    ruta = _SRC_PYTHON / "GraphEMD" / "data" / "graph_imf_transform_utils.py"
    spec = importlib.util.spec_from_file_location("graph_imf_transform_utils_vmd", ruta)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {ruta}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_all_imf_graphs


def ejecutar_grafos_vmd_activo(
    cfg: ConfigActivo,
    df_imfs: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Apply HVG, NVG, and recurrence to each VMD mode.

    Parameters
    ----------
    cfg : ConfigActivo
        Asset.
    df_imfs : pd.DataFrame
        VMD modes.

    Returns
    -------
    dict
        Graph summary and paths.
    """
    _asegurar_paths()
    build_all_imf_graphs = _load_build_all_imf_graphs()
    from run_graph_subsection_outputs_ceemdan_20abr26 import (  # noqa: E402
        compute_recurrence_params_table,
    )

    rutas = _rutas_activo(cfg)
    rutas["dir_grafos"].mkdir(parents=True, exist_ok=True)

    df_params = compute_recurrence_params_table(
        df_imfs,
        umbral_percentil=UMBRAL_PERCENTIL_RECURRENCIA,
        random_state=RANDOM_STATE_RECURRENCIA,
    )
    df_params.to_csv(rutas["param_recurrencia_csv"], index=False)

    resultados = build_all_imf_graphs(
        df_imfs=str(rutas["imfs"]),
        carpeta_salida_base=str(rutas["dir_grafos"]),
        tau_max=TAU_MAX,
        dim_max=DIM_MAX,
        umbral_percentil=UMBRAL_PERCENTIL_RECURRENCIA,
        random_state=RANDOM_STATE_RECURRENCIA,
    )

    filas: List[Dict[str, Any]] = []
    for id_imf, tipos in resultados.items():
        fila: Dict[str, Any] = {"componente": id_imf}
        for tipo_grafo in ("hvg", "nvg", "recurrencia"):
            info = tipos.get(tipo_grafo, {})
            if isinstance(info, dict) and info.get("exito"):
                fila[f"nodos_{tipo_grafo}"] = info.get("num_nodes")
                fila[f"enlaces_{tipo_grafo}"] = info.get("num_edges")
                if tipo_grafo == "recurrencia":
                    fila["tau"] = info.get("tau")
                    fila["dim_embedding"] = info.get("dim_embedding")
                    fila["epsilon"] = info.get("umbral_recurrencia")
            else:
                fila[f"error_{tipo_grafo}"] = (
                    info.get("error", "failure") if isinstance(info, dict) else "failure"
                )
        filas.append(fila)

    df_resumen = pd.DataFrame(filas)
    df_resumen.to_csv(rutas["resumen_grafos_csv"], index=False)

    manifest = {
        "activo": cfg.id_activo,
        "metodo": "VMD",
        "ruta_imfs": str(rutas["imfs"]),
        "dir_grafos": str(rutas["dir_grafos"]),
        "parametros_recurrencia": df_params.to_dict(orient="records"),
        "resumen_por_componente": df_resumen.to_dict(orient="records"),
        "detalle_grafos": resultados,
    }
    with open(rutas["manifest_grafos"], "w", encoding="utf-8") as archivo:
        json.dump(manifest, archivo, indent=2, ensure_ascii=False)

    logger.info(
        "%s VMD graphs: %d components in %s",
        cfg.nombre,
        len(filas),
        rutas["dir_grafos"],
    )
    return {
        "resumen": df_resumen,
        "parametros_recurrencia": df_params,
        "manifest": manifest,
    }


def evaluar_k_ica(
    mod_ica: Any,
    x: np.ndarray,
    nombres_imf: List[str],
    k: int,
) -> Dict[str, Any]:
    """
    Evaluate one k value on the ICA grid.

    Parameters
    ----------
    mod_ica : module
        ICA utilities.
    x : np.ndarray
        IMF block ``(T, p)``.
    nombres_imf : list of str
        IMF column names.
    k : int
        Number of components.

    Returns
    -------
    dict
        Independence and reconstruction metrics.
    """
    ica, scaler, z = mod_ica.fit_fastica(x, k, RANDOM_STATE_ICA)
    x_hat = mod_ica.reconstruct_imfs_from_ica_z(z, ica, scaler)
    rec = mod_ica.imf_reconstruction_metrics(x, x_hat, nombres_imf)
    corr = mod_ica._correlation_metrics(z)
    return {
        "n_components": int(k),
        "max_abs_corr_Z": float(corr["max_abs_fuera_diagonal"]),
        "media_abs_corr_Z": float(corr["media_abs_fuera_diagonal"]),
        "error_relativo_frobenius": float(rec["error_relativo_frobenius"]),
        "rmse_global_reconstruccion": float(rec["rmse_global"]),
        "r2_medio_columnas": float(rec["r2_medio_columnas"]),
    }


def ejecutar_ica_vmd_activo(
    cfg: ConfigActivo,
    df_imfs: pd.DataFrame,
    mod_ica: Any,
) -> Dict[str, Any]:
    """
    Calibrate k and apply FastICA to oscillatory VMD modes.

    Parameters
    ----------
    cfg : ConfigActivo
        Asset.
    df_imfs : pd.DataFrame
        VMD modes.
    mod_ica : module
        ICA utilities.

    Returns
    -------
    dict
        Optimal k, calibration, and output paths.
    """
    rutas = _rutas_activo(cfg)
    x, nombres_imf, residuo = mod_ica.extract_imf_block_and_residual(df_imfs)
    p = int(x.shape[1])
    if p < 2:
        raise ValueError(f"{cfg.nombre}: at least 2 IMFs required for ICA.")

    rejilla: List[Dict[str, Any]] = []
    for k in range(K_MIN_CALIBRACION_ICA, p):
        logger.info("%s ICA: evaluating k=%d ...", cfg.nombre, k)
        rejilla.append(evaluar_k_ica(mod_ica, x, nombres_imf, k))

    df_rejilla = pd.DataFrame(rejilla)
    seleccion = mod_ica.select_optimal_k_from_ica_grid(
        df_rejilla,
        p=p,
        umbral_corr_z=UMBRAL_CORR_Z,
        fraccion_r2_objetivo=FRACCION_R2_OBJETIVO,
        factor_error_frobenius_rodilla=FACTOR_ERROR_FROB_RODILLA,
    )
    k_optimo = int(seleccion["n_components"])

    meta = {
        "activo": cfg.id_activo,
        "nombre": cfg.nombre,
        "metodo_descomposicion": "VMD",
        "parquet_entrada": str(rutas["imfs"].resolve()),
        "n_muestras_temporales": int(x.shape[0]),
        "columnas_imf_entrada": nombres_imf,
        "calibracion_mejor_k": seleccion,
    }

    rutas["dir_ica"].mkdir(parents=True, exist_ok=True)
    mod_ica.write_references(rutas["dir_ica"].parent)

    ica, scaler, z = mod_ica.fit_fastica(x, k_optimo, RANDOM_STATE_ICA)
    mod_ica.save_ica_outputs(
        rutas["dir_ica"],
        z,
        residuo,
        ica,
        scaler,
        nombres_imf,
        meta,
        x,
        guardar_parquet_recon=True,
    )

    calibracion = {
        "criterio_seleccion": seleccion["criterio_seleccion"],
        "p_imf_entrada": p,
        "k_min_evaluado": K_MIN_CALIBRACION_ICA,
        "k_max_evaluado": p - 1,
        "random_state": RANDOM_STATE_ICA,
        "mejor": seleccion,
        "detalle_seleccion": seleccion["detalle_seleccion"],
        "correlacion_imfs_nativas": mod_ica._correlation_metrics(x),
        "rejilla": rejilla,
    }
    aplicacion = {
        "n_components": k_optimo,
        "forma_Z": list(z.shape),
        "dir_salida": str(rutas["dir_ica"].resolve()),
        "parquet_reducido": str(
            (rutas["dir_ica"] / "imfs_reducidas.parquet").resolve()
        ),
        "modelo": str((rutas["dir_ica"] / "modelo_ica.npz").resolve()),
        "metricas": str((rutas["dir_ica"] / "metricas_reduccion.json").resolve()),
    }
    payload = {
        "parquet_entrada": str(rutas["imfs"].resolve()),
        "calibracion": calibracion,
        "aplicacion": aplicacion,
    }
    with open(rutas["parametros_ica"], "w", encoding="utf-8") as archivo:
        json.dump(payload, archivo, indent=2, ensure_ascii=False)

    df_z = pd.read_parquet(
        rutas["dir_ica"] / "imfs_reducidas.parquet", engine="pyarrow"
    )
    generar_panel_ica_vmd(
        df_z,
        df_imfs.index,
        cfg.nombre,
        k_optimo,
        rutas["figura_ica"],
    )

    logger.info(
        "%s ICA VMD: optimal k=%d saved to %s", cfg.nombre, k_optimo, rutas["dir_ica"]
    )
    return {
        "k_optimo": k_optimo,
        "calibracion": calibracion,
        "aplicacion": aplicacion,
    }


def generar_panel_ica_vmd(
    df_componentes: pd.DataFrame,
    indice: pd.Index,
    nombre_activo: str,
    k: int,
    ruta_salida: Path,
) -> None:
    """
    Generate temporal panel of ICA sources and VMD residual.

    Parameters
    ----------
    df_componentes : pd.DataFrame
        Columns ``Z_1``…``Z_k`` and optional ``Residuo``.
    indice : pd.Index
        Time axis.
    nombre_activo : str
        Asset label.
    k : int
        Number of ICA components.
    ruta_salida : Path
        PNG path.
    """
    cols_z = sorted(
        [c for c in df_componentes.columns if c.startswith("Z_")],
        key=lambda x: int(x.split("_")[1]),
    )
    if "Residuo" in df_componentes.columns:
        cols_z.append("Residuo")
    n = len(cols_z)
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, nombre in zip(axes, cols_z):
        ax.plot(indice, df_componentes[nombre].values, linewidth=0.6)
        ax.set_ylabel(nombre, fontsize=9)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Fecha")
    fig.tight_layout()
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("VMD ICA figure: %s", ruta_salida)


def procesar_activo(
    cfg: ConfigActivo,
    mod_emdsynth: Any,
    mod_validacion: Any,
    mod_ica: Any,
    pasos: Tuple[str, ...],
    calibrar: bool,
    dc_fijo: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run requested VMD pipeline steps for one asset.

    Parameters
    ----------
    cfg : ConfigActivo
        Asset.
    mod_emdsynth : module
        VMD module.
    mod_validacion : module
        Validation.
    mod_ica : module
        ICA.
    pasos : tuple of str
        Subset of ``vmd``, ``validacion``, ``grafos``, ``ica``.
    calibrar : bool
        Calibrate VMD hyperparameters.
    dc_fijo : int, optional
        Fix ``DC`` in calibration and decomposition.

    Returns
    -------
    dict
        Partial results per step.
    """
    resultado: Dict[str, Any] = {"activo": cfg.id_activo}
    rutas = _rutas_activo(cfg)

    if "vmd" in pasos:
        salida_vmd = ejecutar_vmd_activo(
            cfg, mod_emdsynth, calibrar=calibrar, dc_fijo=dc_fijo
        )
        resultado["vmd"] = salida_vmd
        df_precios = salida_vmd["df_precios"]
        serie = salida_vmd["serie"]
        df_imfs = salida_vmd["df_imfs"]
        vmd_dc_efectivo = int(salida_vmd["calibracion"]["mejor"]["DC"])
    else:
        df_precios, serie = cargar_serie_precios(cfg)
        if not rutas["imfs"].is_file():
            raise FileNotFoundError(
                f"Missing {rutas['imfs']}. Run the vmd step first."
            )
        df_imfs = pd.read_parquet(rutas["imfs"], engine="pyarrow")
        vmd_dc_efectivo = int(dc_fijo) if dc_fijo is not None else 1

    if "validacion" in pasos:
        resultado["validacion"] = validar_vmd_activo(
            cfg,
            df_precios,
            serie,
            df_imfs,
            mod_validacion,
            vmd_dc=vmd_dc_efectivo,
        )

    if "grafos" in pasos:
        resultado["grafos"] = ejecutar_grafos_vmd_activo(cfg, df_imfs)

    if "ica" in pasos:
        resultado["ica"] = ejecutar_ica_vmd_activo(cfg, df_imfs, mod_ica)

    return resultado


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Parameters
    ----------
    argv : list of str, optional
        Arguments (default ``sys.argv[1:]``).

    Returns
    -------
    argparse.Namespace
        CLI options.
    """
    parser = argparse.ArgumentParser(
        description="Full VMD pipeline for MSCI, ETFs, and gold."
    )
    parser.add_argument(
        "--activos",
        type=str,
        default=",".join(a.id_activo for a in ACTIVOS),
        help="Comma-separated IDs (msci_world,xle,xlp,xlv,xauusd).",
    )
    parser.add_argument(
        "--solo-paso",
        type=str,
        default="vmd,validacion,grafos,ica",
        help="Steps to run: vmd,validacion,grafos,ica.",
    )
    parser.add_argument(
        "--sin-calibracion",
        action="store_true",
        help="Use K=6, alpha=5000, DC=1 without per-asset grid.",
    )
    parser.add_argument(
        "--vmd-dc",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Fix DC in VMD (1 = trend mode in IMF_1). "
            "Default 1 to compare with CEEMDAN/EEMD."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Entry point: process all requested assets.

    Parameters
    ----------
    argv : list of str, optional
        CLI arguments.

    Returns
    -------
    dict
        Aggregated results per asset.
    """
    args = _parse_args(argv)
    ids = [s.strip() for s in args.activos.split(",") if s.strip()]
    pasos = tuple(s.strip() for s in args.solo_paso.split(",") if s.strip())
    mapa = {a.id_activo: a for a in ACTIVOS}

    mod_emdsynth = _cargar_modulo_emdsynth()
    mod_validacion = _cargar_modulo_validacion()
    mod_ica = _cargar_modulo_ica()

    resultados: Dict[str, Any] = {}
    for id_activo in ids:
        if id_activo not in mapa:
            raise ValueError(f"Unknown asset: {id_activo}. Valid: {list(mapa)}")
        cfg = mapa[id_activo]
        logger.info("=" * 70)
        logger.info("PROCESANDO %s (%s)", cfg.nombre, cfg.id_activo)
        logger.info("=" * 70)
        resultados[id_activo] = procesar_activo(
            cfg,
            mod_emdsynth,
            mod_validacion,
            mod_ica,
            pasos=pasos,
            calibrar=not args.sin_calibracion,
            dc_fijo=args.vmd_dc,
        )

    resumen_k = {
        id_a: resultados[id_a].get("ica", {}).get("k_optimo")
        for id_a in resultados
        if "ica" in resultados[id_a]
    }
    if resumen_k:
        logger.info("Optimal ICA k summary: %s", resumen_k)

    return resultados


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        main()
    except Exception:
        logger.exception("Error in VMD pipeline")
        sys.exit(1)
