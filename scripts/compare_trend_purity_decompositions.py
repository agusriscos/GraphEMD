#!/usr/bin/env python3
"""
Compare trend-component purity across CEEMDAN, EEMD, and VMD.

Loads persisted decompositions for each financial signal (and, optionally,
emdsynth synthetic scenarios) and computes, per method:

- ``r2_regresion_lineal``: goodness of fit of a temporal straight line.
- Fractions of increments/decrements (soft monotonicity).
- ``rugosidad_d2``: standard deviation of the normalized second difference
  (scientific notation, e.g. ``4.8333e-02``).

For CEEMDAN and EEMD, the evaluated component is the implicit residual
(``Residuo`` or ``|x - Σ IMF|``). For VMD with ``DC=1``, ``IMF_1`` is evaluated,
which captures the trend.

Example::

    python scripts/compare_trend_purity_decompositions.py

    python scripts/compare_trend_purity_decompositions.py \\
        --activos msci_world,xauusd --incluir-emdsynth
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EMDSYNTH_OUT = _REPO_ROOT / "scripts" / "emdsynth" / "out"
_OUT_DIR = _REPO_ROOT / "scripts" / "out"

TOL_DIFF_MONOTONA: float = 1e-12

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfigSenal:
    """
    Metadata for a signal with CEEMDAN, EEMD, and VMD decompositions.

    Parameters
    ----------
    id_senal : str
        Short identifier.
    nombre : str
        Human-readable label.
    dir_datos : Path
        Directory containing ``{prefijo}_imfs_*.parquet`` files.
    prefijo : str
        IMF file prefix.
    ruta_precios : Path | None
        Price parquet; if ``None``, the signal is reconstructed from IMFs.
    columna_precio : str
        Closing price column name.
    tipo : str
        ``empirica`` or ``sintetica``.
    """

    id_senal: str
    nombre: str
    dir_datos: Path
    prefijo: str
    ruta_precios: Path | None = None
    columna_precio: str = "Close"
    tipo: str = "empirica"


SENALES_EMPRICAS: tuple[ConfigSenal, ...] = (
    ConfigSenal(
        id_senal="msci_world",
        nombre="MSCI World",
        dir_datos=_REPO_ROOT / "data" / "20abr26",
        prefijo="msci_world",
        ruta_precios=_REPO_ROOT / "data" / "20abr26" / "msci_world.parquet",
    ),
    ConfigSenal(
        id_senal="xle",
        nombre="XLE",
        dir_datos=_REPO_ROOT / "data" / "GraphEMD" / "xle_etf_analysis",
        prefijo="xle",
        ruta_precios=_REPO_ROOT
        / "data"
        / "GraphEMD"
        / "xle_etf_analysis"
        / "xle.parquet",
    ),
    ConfigSenal(
        id_senal="xlp",
        nombre="XLP",
        dir_datos=_REPO_ROOT / "data" / "GraphEMD" / "xlp_analysis",
        prefijo="xlp",
        ruta_precios=_REPO_ROOT / "data" / "GraphEMD" / "xlp_analysis" / "xlp.parquet",
    ),
    ConfigSenal(
        id_senal="xlv",
        nombre="XLV",
        dir_datos=_REPO_ROOT / "data" / "GraphEMD" / "xlv_analysis",
        prefijo="xlv",
        ruta_precios=_REPO_ROOT / "data" / "GraphEMD" / "xlv_analysis" / "xlv.parquet",
    ),
    ConfigSenal(
        id_senal="xauusd",
        nombre="XAU/USD",
        dir_datos=_REPO_ROOT / "data" / "GraphEMD" / "xauusd_analysis",
        prefijo="xauusd",
        ruta_precios=_REPO_ROOT
        / "data"
        / "GraphEMD"
        / "xauusd_analysis"
        / "xauusd.parquet",
    ),
)


def _columnas_imf(df: pd.DataFrame) -> list[str]:
    """
    Return ``IMF_*`` columns sorted by index.

    Parameters
    ----------
    df : pd.DataFrame
        Decomposition table.

    Returns
    -------
    list[str]
        Sorted IMF column names.
    """
    return sorted(
        [c for c in df.columns if c.startswith("IMF_")],
        key=lambda nombre: int(nombre.split("_")[1]),
    )


def _alinear_longitud(df: pd.DataFrame, n_objetivo: int) -> pd.DataFrame:
    """
    Trim a DataFrame to the common number of observations.

    Parameters
    ----------
    df : pd.DataFrame
        Table to align.
    n_objetivo : int
        Target length.

    Returns
    -------
    pd.DataFrame
        First ``n_objetivo`` rows.
    """
    if len(df) == n_objetivo:
        return df
    if len(df) < n_objetivo:
        raise ValueError(
            f"DataFrame with {len(df)} rows; at least {n_objetivo} required."
        )
    return df.iloc[:n_objetivo].reset_index(drop=True)


def _cargar_serie_original(
    cfg: ConfigSenal,
    df_referencia: pd.DataFrame,
) -> np.ndarray:
    """
    Obtain the original signal from prices or IMF+residual reconstruction.

    Parameters
    ----------
    cfg : ConfigSenal
        Signal configuration.
    df_referencia : pd.DataFrame
        Reference decomposition (CEEMDAN) already aligned.

    Returns
    -------
    np.ndarray
        Original signal vector.
    """
    if cfg.ruta_precios is not None and cfg.ruta_precios.is_file():
        df_precios = pd.read_parquet(cfg.ruta_precios, engine="pyarrow")
        if cfg.columna_precio not in df_precios.columns:
            raise ValueError(
                f"Column {cfg.columna_precio!r} missing in {cfg.ruta_precios}"
            )
        serie = np.asarray(
            _alinear_longitud(df_precios, len(df_referencia))[
                cfg.columna_precio
            ].values,
            dtype=np.float64,
        )
        return serie

    cols_imf = _columnas_imf(df_referencia)
    suma = np.sum(
        [np.asarray(df_referencia[c].values, dtype=np.float64) for c in cols_imf],
        axis=0,
    )
    if "Residuo" in df_referencia.columns:
        suma += np.asarray(df_referencia["Residuo"].values, dtype=np.float64)
    return suma


def _brecha_sin_residuo(serie: np.ndarray, df_imfs: pd.DataFrame) -> np.ndarray:
    """
    Compute ``|x - Σ IMF|`` (trend gap without Residuo column).

    Parameters
    ----------
    serie : np.ndarray
        Original signal.
    df_imfs : pd.DataFrame
        Decomposition with ``IMF_*`` columns.

    Returns
    -------
    np.ndarray
        Gap magnitude.
    """
    cols = _columnas_imf(df_imfs)
    suma_imfs = np.sum(
        [np.asarray(df_imfs[c].values, dtype=np.float64) for c in cols],
        axis=0,
    )
    return np.abs(serie - suma_imfs)


def _leer_vmd_dc(cfg: ConfigSenal) -> int:
    """
    Read the ``DC`` value used in VMD calibration for the asset.

    Parameters
    ----------
    cfg : ConfigSenal
        Empirical signal.

    Returns
    -------
    int
        Calibrated ``DC``; defaults to 1 if JSON is missing.
    """
    ruta_json = cfg.dir_datos / f"{cfg.prefijo}_vmd_parametros.json"
    if not ruta_json.is_file():
        return 1
    with ruta_json.open(encoding="utf-8") as fichero:
        datos = json.load(fichero)
    return int(datos.get("calibracion", {}).get("mejor", {}).get("DC", 1))


def _extraer_componente_tendencia(
    metodo: str,
    serie: np.ndarray,
    df_imfs: pd.DataFrame,
    vmd_dc: int = 1,
) -> tuple[np.ndarray, str]:
    """
    Extract the slow component to evaluate for each method.

    Parameters
    ----------
    metodo : str
        ``ceemdan``, ``eemd``, or ``vmd``.
    serie : np.ndarray
        Original signal.
    df_imfs : pd.DataFrame
        Method decomposition.
    vmd_dc : int
        VMD ``DC`` parameter.

    Returns
    -------
    tuple[np.ndarray, str]
        Trend series and description of the definition used.
    """
    metodo_norm = metodo.lower()
    if metodo_norm == "ceemdan":
        if "Residuo" in df_imfs.columns:
            componente = np.asarray(df_imfs["Residuo"].values, dtype=np.float64)
            definicion = "columna Residuo"
        else:
            componente = _brecha_sin_residuo(serie, df_imfs)
            definicion = "|x - Σ IMF|"
        return componente, definicion

    if metodo_norm == "eemd":
        componente = _brecha_sin_residuo(serie, df_imfs)
        return componente, "|x - Σ IMF|"

    if metodo_norm == "vmd":
        cols = _columnas_imf(df_imfs)
        if vmd_dc == 1 and len(cols) >= 1:
            componente = np.asarray(df_imfs[cols[0]].values, dtype=np.float64)
            return componente, f"{cols[0]} (VMD DC=1)"
        if "Residuo" in df_imfs.columns:
            componente = np.asarray(df_imfs["Residuo"].values, dtype=np.float64)
            return componente, "columna Residuo"
        componente = _brecha_sin_residuo(serie, df_imfs)
        return componente, "|x - Σ IMF|"

    raise ValueError(f"Unsupported method: {metodo!r}")


def calcular_metricas_monotonia(serie: np.ndarray) -> dict[str, Any]:
    """
    Compute monotonicity fractions of the trend component.

    Parameters
    ----------
    serie : np.ndarray
        Slow component to evaluate.

    Returns
    -------
    dict
        Fractions of positive, negative, and same-sign differences.
    """
    y = np.asarray(serie, dtype=np.float64)
    diffs = np.diff(y)
    frac_creciente = float(np.mean(diffs > TOL_DIFF_MONOTONA))
    frac_decreciente = float(np.mean(diffs < -TOL_DIFF_MONOTONA))
    frac_mismo_signo = max(frac_creciente, frac_decreciente)
    return {
        "frac_diff_creciente": frac_creciente,
        "frac_diff_decreciente": frac_decreciente,
        "frac_diff_mismo_signo": frac_mismo_signo,
    }


def calcular_rugosidad_d2(serie: np.ndarray) -> float:
    """
    Measure roughness as ``std(Δ²y) / std(y)``.

    Parameters
    ----------
    serie : np.ndarray
        Slow component.

    Returns
    -------
    float
        Normalized roughness; lower values indicate a smoother trend.
    """
    y = np.asarray(serie, dtype=np.float64)
    if len(y) < 3:
        return float("nan")
    segunda_diff = np.diff(y, n=2)
    return float(np.std(segunda_diff) / (np.std(y) + 1e-15))


def formatear_notacion_cientifica(valor: float, decimales: int = 4) -> str:
    """
    Format a scalar in scientific notation (e.g. ``1.2786e-06``).

    Parameters
    ----------
    valor : float
        Number to format.
    decimales : int
        Decimal places in the mantissa.

    Returns
    -------
    str
        Scientific notation string, or ``"nan"`` if not finite.
    """
    if not np.isfinite(valor):
        return "nan"
    return f"{valor:.{decimales}e}"


def calcular_metricas_pureza_tendencia(
    componente: np.ndarray,
    serie_original: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Compute linear R², monotonicity, and roughness of a trend component.

    Parameters
    ----------
    componente : np.ndarray
        Slow-component series.
    serie_original : np.ndarray | None
        Price signal for optional correlation.

    Returns
    -------
    dict
        Trend purity metrics.
    """
    y = np.asarray(componente, dtype=np.float64)
    n = len(y)
    t = np.arange(n, dtype=np.float64)
    coef = np.polyfit(t, y, 1)
    ajuste_lineal = coef[0] * t + coef[1]
    ss_res = float(np.sum((y - ajuste_lineal) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-15
    r2_lineal = 1.0 - ss_res / ss_tot
    rho, p_spearman = spearmanr(t, y)

    rugosidad = calcular_rugosidad_d2(y)
    metricas = {
        "n_observaciones": int(n),
        "media": float(np.mean(y)),
        "std": float(np.std(y, ddof=1)),
        "r2_regresion_lineal": float(r2_lineal),
        "pendiente_regresion_temporal": float(coef[0]),
        "spearman_tiempo": float(rho),
        "p_valor_spearman": float(p_spearman),
        "rugosidad_d2": formatear_notacion_cientifica(rugosidad),
    }
    metricas.update(calcular_metricas_monotonia(y))

    if serie_original is not None:
        close = np.asarray(serie_original, dtype=np.float64)
        metricas["correlacion_con_senal_original"] = float(np.corrcoef(y, close)[0, 1])

    return metricas


def _ruta_parquet_imfs(cfg: ConfigSenal, metodo: str) -> Path:
    """
    Build the path to an IMF parquet for a method.

    Parameters
    ----------
    cfg : ConfigSenal
        Signal.
    metodo : str
        ``ceemdan``, ``eemd``, or ``vmd``.

    Returns
    -------
    Path
        Expected parquet path.
    """
    return cfg.dir_datos / f"{cfg.prefijo}_imfs_{metodo}.parquet"


def evaluar_senal(cfg: ConfigSenal) -> dict[str, Any]:
    """
    Evaluate trend purity for CEEMDAN, EEMD, and VMD of one signal.

    Parameters
    ----------
    cfg : ConfigSenal
        Signal configuration.

    Returns
    -------
    dict
        Report with metrics per method and metadata.
    """
    rutas = {m: _ruta_parquet_imfs(cfg, m) for m in ("ceemdan", "eemd", "vmd")}
    faltantes = [m for m, ruta in rutas.items() if not ruta.is_file()]
    if faltantes:
        raise FileNotFoundError(
            f"Missing decompositions {faltantes} for {cfg.id_senal} in {cfg.dir_datos}"
        )

    dataframes = {
        metodo: pd.read_parquet(ruta, engine="pyarrow")
        for metodo, ruta in rutas.items()
    }
    n_comun = min(len(df) for df in dataframes.values())
    dataframes = {
        metodo: _alinear_longitud(df, n_comun) for metodo, df in dataframes.items()
    }

    serie = _cargar_serie_original(cfg, dataframes["ceemdan"])
    if len(serie) != n_comun:
        serie = serie[:n_comun]

    vmd_dc = _leer_vmd_dc(cfg) if cfg.tipo == "empirica" else 1
    resultados_metodos: dict[str, Any] = {}

    for metodo, df_imfs in dataframes.items():
        componente, definicion = _extraer_componente_tendencia(
            metodo,
            serie,
            df_imfs,
            vmd_dc=vmd_dc,
        )
        metricas = calcular_metricas_pureza_tendencia(componente, serie)
        resultados_metodos[metodo] = {
            "definicion_componente": definicion,
            "metricas": metricas,
        }

    return {
        "id_senal": cfg.id_senal,
        "nombre": cfg.nombre,
        "tipo": cfg.tipo,
        "n_observaciones": n_comun,
        "vmd_dc": vmd_dc,
        "metodos": resultados_metodos,
    }


def _cargar_escenarios_emdsynth() -> list[ConfigSenal]:
    """
    Build configurations for synthetic scenarios with all three methods.

    Returns
    -------
    list[ConfigSenal]
        List of synthetic signals detected in ``emdsynth/out``.
    """
    manifest = _EMDSYNTH_OUT / "manifest_emdsynth.json"
    if not manifest.is_file():
        logger.warning(
            "emdsynth manifest not found; skipping synthetic signals."
        )
        return []

    with manifest.open(encoding="utf-8") as fichero:
        datos = json.load(fichero)

    escenarios: list[ConfigSenal] = []
    for item in datos.get("escenarios", []):
        id_escenario = item["id"]
        ruta_ceemdan = _EMDSYNTH_OUT / f"{id_escenario}_imfs_ceemdan.parquet"
        if not ruta_ceemdan.is_file():
            continue
        escenarios.append(
            ConfigSenal(
                id_senal=id_escenario,
                nombre=item.get("descripcion", id_escenario),
                dir_datos=_EMDSYNTH_OUT,
                prefijo=id_escenario,
                ruta_precios=None,
                tipo="sintetica",
            )
        )
    return escenarios


def _informe_a_filas(informe: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Flatten a per-signal report into tabular rows.

    Parameters
    ----------
    informe : dict
        Output of ``evaluar_senal``.

    Returns
    -------
    list[dict]
        Rows ready for DataFrame.
    """
    filas: list[dict[str, Any]] = []
    for metodo, bloque in informe["metodos"].items():
        fila = {
            "id_senal": informe["id_senal"],
            "nombre": informe["nombre"],
            "tipo": informe["tipo"],
            "metodo": metodo,
            "definicion_componente": bloque["definicion_componente"],
            "n_observaciones": informe["n_observaciones"],
            "vmd_dc": informe.get("vmd_dc"),
        }
        fila.update(bloque["metricas"])
        filas.append(fila)
    return filas


def ejecutar_comparativa(
    senales: list[ConfigSenal],
    ruta_json: Path,
    ruta_csv: Path,
) -> dict[str, Any]:
    """
    Run the comparison for all signals and persist JSON and CSV.

    Parameters
    ----------
    senales : list[ConfigSenal]
        Signals to evaluate.
    ruta_json : Path
        Aggregated report destination.
    ruta_csv : Path
        Flat table destination.

    Returns
    -------
    dict
        Full report.
    """
    informes: list[dict[str, Any]] = []
    filas: list[dict[str, Any]] = []
    errores: list[dict[str, str]] = []

    for cfg in senales:
        logger.info("Evaluating %s (%s)...", cfg.nombre, cfg.id_senal)
        try:
            informe = evaluar_senal(cfg)
            informes.append(informe)
            filas.extend(_informe_a_filas(informe))
        except Exception as exc:  # noqa: BLE001
            logger.error("Error in %s: %s", cfg.id_senal, exc)
            errores.append({"id_senal": cfg.id_senal, "error": str(exc)})

    salida = {
        "descripcion": (
            "Trend purity metrics: linear R², increment fraction, "
            "and rugosidad_d2 for CEEMDAN (Residuo), EEMD (|x-ΣIMF|), "
            "and VMD (IMF_1 if DC=1)."
        ),
        "n_senales_ok": len(informes),
        "n_errores": len(errores),
        "senales": informes,
        "errores": errores,
    }

    ruta_json.parent.mkdir(parents=True, exist_ok=True)
    with ruta_json.open("w", encoding="utf-8") as fichero:
        json.dump(salida, fichero, indent=2, ensure_ascii=False)

    if filas:
        pd.DataFrame(filas).to_csv(ruta_csv, index=False)
        logger.info("CSV saved: %s", ruta_csv)

    logger.info("JSON saved: %s", ruta_json)
    return salida


def _parsear_argumentos() -> argparse.Namespace:
    """
    Define and parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compare trend purity (R², increment fraction, roughness) across "
            "CEEMDAN, EEMD, and VMD for all project signals."
        )
    )
    parser.add_argument(
        "--activos",
        type=str,
        default="",
        help="Comma-separated IDs (e.g. msci_world,xauusd). Empty = all.",
    )
    parser.add_argument(
        "--incluir-emdsynth",
        action="store_true",
        help="Include synthetic scenarios from emdsynth/out.",
    )
    parser.add_argument(
        "--salida-json",
        type=Path,
        default=_OUT_DIR / "metricas_pureza_tendencia_descomposiciones.json",
        help="JSON report path.",
    )
    parser.add_argument(
        "--salida-csv",
        type=Path,
        default=_OUT_DIR / "metricas_pureza_tendencia_descomposiciones.csv",
        help="CSV table path.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry point: evaluate signals and write results.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parsear_argumentos()

    senales = list(SENALES_EMPRICAS)
    if args.activos.strip():
        ids = {s.strip() for s in args.activos.split(",") if s.strip()}
        senales = [s for s in senales if s.id_senal in ids]
        desconocidos = ids - {s.id_senal for s in senales}
        if desconocidos:
            raise ValueError(f"Unrecognized asset IDs: {sorted(desconocidos)}")

    if args.incluir_emdsynth:
        senales.extend(_cargar_escenarios_emdsynth())

    if not senales:
        raise ValueError("No signals to evaluate.")

    ejecutar_comparativa(senales, args.salida_json, args.salida_csv)


if __name__ == "__main__":
    main()
