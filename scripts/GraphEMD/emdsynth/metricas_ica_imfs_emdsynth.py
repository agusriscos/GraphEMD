#!/usr/bin/env python3
"""
Métricas de correlación entre fuentes FastICA aplicadas al bloque de IMF sintéticas.

Lee los parquets ``{escenario}_imfs_{metodo}.parquet`` generados por
``ejecutar_descomposiciones_emdsynth.py``, toma las primeras
``min(8, n_imf)`` columnas ``IMF_*`` (mismo criterio de tope que el resumen
nativo de pares), estandariza y ajusta FastICA con
``n_components = min(k_objetivo, p)`` con ``k_objetivo=4`` por defecto
(alineado al bloque MSCI con rank cuatro). El residuo EMD no
entra en el ajuste ni en los pares correlacionados. Se reutiliza
``_estadisticos_corr_pares_imfs`` del script de descomposición para
``\\bar{\\rho}`` y ``\\pi_{0.05}`` entre columnas de ``Z`` (equivalente a
correlacionar las filas de ``Z.T``).

Ejemplo::

    PYTHONPATH=src/python python3 \\
        scripts/GraphEMD/emdsynth/metricas_ica_imfs_emdsynth.py \\
        --entrada scripts/GraphEMD/emdsynth/out
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_ENTRADA = _REPO_ROOT / "scripts" / "GraphEMD" / "emdsynth" / "out"

logger = logging.getLogger(__name__)


def _cargar_modulo_descomposiciones() -> Any:
    """
    Carga ``ejecutar_descomposiciones_emdsynth`` como módulo para reutilizar métricas.

    Returns
    -------
    module
        Módulo con ``_estadisticos_corr_pares_imfs``.
    """
    ruta = Path(__file__).resolve().parent / "ejecutar_descomposiciones_emdsynth.py"
    spec = importlib.util.spec_from_file_location("emdsynth_decomposiciones", ruta)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar {ruta}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _columnas_imf_ordenadas(df: pd.DataFrame) -> List[str]:
    """
    Devuelve nombres ``IMF_*`` ordenados numéricamente.

    Parameters
    ----------
    df : pd.DataFrame
        Tabla con columnas ``IMF_1``, ...

    Returns
    -------
    list of str
        Lista ordenada.
    """
    cols = [c for c in df.columns if c.startswith("IMF_")]
    if not cols:
        raise ValueError("No hay columnas IMF_* en el parquet.")
    return sorted(cols, key=lambda x: int(x.split("_")[1]))


def extraer_bloque_imf_capado(
    df: pd.DataFrame, max_imf: int = 8
) -> Tuple[np.ndarray, bool, int]:
    """
    Extrae la matriz ``(T, p_use)`` con las primeras ``min(max_imf, n_imf)`` IMF.

    Parameters
    ----------
    df : pd.DataFrame
        Parquet de IMF + opcional ``Residuo``.
    max_imf : int
        Tope de modos intrínsecos alineado con el resumen nativo.

    Returns
    -------
    X : np.ndarray
        Bloque oscilatorio capado, forma ``(T, p_use)``.
    hay_residuo : bool
        True si existe columna ``Residuo``.
    n_imf_total : int
        Número de columnas ``IMF_*`` en el fichero (antes del capado).
    """
    nombres = _columnas_imf_ordenadas(df)
    n_total = len(nombres)
    use = min(max_imf, n_total)
    nombres_use = nombres[:use]
    X = np.asarray(df[nombres_use].values, dtype=np.float64)
    hay_residuo = "Residuo" in df.columns
    return X, hay_residuo, n_total


def ajustar_ica_fuentes(
    X: np.ndarray,
    random_state: int,
    n_components: int,
) -> np.ndarray:
    """
    Escala columnas y ajusta FastICA con ``n_components`` fijado (reducción de rango).

    Parameters
    ----------
    X : np.ndarray
        IMF capadas, forma ``(T, p)``.
    random_state : int
        Semilla para FastICA.
    n_components : int
        Dimensión de salida ``k`` (debe cumplir ``1 <= k <= p``).

    Returns
    -------
    np.ndarray
        Fuentes ``Z``, forma ``(T, k)``.
    """
    p = X.shape[1]
    if p < 1:
        raise ValueError("Se requiere al menos una columna IMF.")
    k = int(n_components)
    if k < 1 or k > p:
        raise ValueError(f"n_components debe estar en [1, p]; recibido k={k}, p={p}.")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        ica = FastICA(
            n_components=k,
            algorithm="parallel",
            whiten="unit-variance",
            max_iter=8000,
            random_state=random_state,
            tol=1e-5,
        )
        return np.asarray(ica.fit_transform(Xs), dtype=np.float64)


def metricas_corr_tras_ica(
    mod: Any,
    Z: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, int, float, int]:
    """
    Correlación media |r|, número de pares y fracción con p < alpha entre fuentes.

    Parameters
    ----------
    mod : module
        Módulo con ``_estadisticos_corr_pares_imfs``.
    Z : np.ndarray
        Fuentes ICA, forma ``(T, k)``.
    alpha : float
        Umbral de p-valor (sin corrección múltiple).

    Returns
    -------
    media_abs_r : float
        Media de |r_ij| en el triángulo superior.
    n_pares : int
        Número de pares válidos.
    frac_p : float
        Fracción con p < alpha.
    k : int
        Número de fuentes (columnas de ``Z``).
    """
    k = Z.shape[1]
    if k < 2:
        return 0.0, 0, float("nan"), k
    media_abs_r, n_pares, frac_p = mod._estadisticos_corr_pares_imfs(
        Z.T,
        max_filas=min(8, k),
        alpha=alpha,
    )
    return float(media_abs_r), int(n_pares), float(frac_p), k


def procesar_parquet(
    mod: Any,
    ruta: Path,
    random_state: int,
    max_imf: int,
    k_reduccion_objetivo: int,
) -> Dict[str, Any]:
    """
    Calcula métricas ICA para un parquet de IMF con reducción de rango.

    Parameters
    ----------
    mod : module
        Módulo de descomposiciones (función de correlación).
    ruta : Path
        Ruta al parquet ``*_imfs_*.parquet``.
    random_state : int
        Semilla FastICA.
    max_imf : int
        Máximo de IMF de entrada al ICA (primeras en orden).
    k_reduccion_objetivo : int
        Dimensión deseada de salida (p. ej. 4); efectivo ``k = min(k, p)``.

    Returns
    -------
    dict
        Metadatos de escenario, ``p_imf_entrada``, ``k_reduccion``, ``N`` reportado, métricas.
    """
    df = pd.read_parquet(ruta, engine="pyarrow")
    X, hay_residuo, n_imf_total = extraer_bloque_imf_capado(df, max_imf=max_imf)
    p_entrada = X.shape[1]
    k = min(int(k_reduccion_objetivo), p_entrada)
    Z = ajustar_ica_fuentes(X, random_state=random_state, n_components=k)
    media_abs_r, n_pares, frac_p, k_out = metricas_corr_tras_ica(mod, Z)
    if k_out != k:
        raise RuntimeError("Inconsistencia en dimensión de salida ICA.")
    nombre = ruta.name
    # "{escenario}_imfs_{metodo}.parquet"
    base = nombre.replace("_imfs_", "|").replace(".parquet", "")
    partes = base.split("|", 1)
    escenario = partes[0]
    metodo = partes[1] if len(partes) > 1 else "unknown"
    n_reporte = k + (1 if hay_residuo else 0)
    return {
        "escenario": escenario,
        "metodo": metodo,
        "ruta_parquet": str(ruta.resolve()),
        "n_imf_archivo": int(n_imf_total),
        "p_imf_entrada_ica": int(p_entrada),
        "k_reduccion_ica": int(k),
        "k_reduccion_objetivo": int(k_reduccion_objetivo),
        "n_componentes_reporte": int(n_reporte),
        "hay_residuo": bool(hay_residuo),
        "corr_promedio_pares_ica": media_abs_r,
        "n_pares_corr": n_pares,
        "frac_pares_p_lt_005": frac_p,
    }


def listar_parquets_imfs(dir_entrada: Path) -> List[Path]:
    """
    Lista parquets ``*_imfs_*.parquet`` bajo ``dir_entrada``.

    Parameters
    ----------
    dir_entrada : Path
        Directorio de salida de ``ejecutar_descomposiciones_emdsynth``.

    Returns
    -------
    list of Path
        Rutas ordenadas.
    """
    if not dir_entrada.is_dir():
        return []
    out = sorted(dir_entrada.glob("*_imfs_*.parquet"))
    return out


def main(argv: Optional[List[str]] = None) -> None:
    """
    CLI: procesa todos los parquets IMF y escribe JSON + resumen por consola.

    Parameters
    ----------
    argv : list of str, optional
        Argumentos (por defecto ``sys.argv[1:]``).
    """
    p = argparse.ArgumentParser(
        description="Correlación entre fuentes FastICA sobre IMF sintéticas (emdsynth)."
    )
    p.add_argument(
        "--entrada",
        type=Path,
        default=_DEFAULT_ENTRADA,
        help="Directorio con parquets *_imfs_*.parquet.",
    )
    p.add_argument(
        "--salida-json",
        type=Path,
        default=None,
        help="Ruta JSON (por defecto entrada/metricas_ica_emdsynth.json).",
    )
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--max-imf",
        type=int,
        default=8,
        help="Máximo de IMF (columnas) alimentadas al ICA; alineado con el tope nativo.",
    )
    p.add_argument(
        "--k-reduccion",
        type=int,
        default=4,
        help="Dimensión objetivo de salida FastICA (k); efectivo min(k, p) con p columnas IMF.",
    )
    p.add_argument(
        "--latex",
        action="store_true",
        help="Imprime filas tabular listas para pegar en LaTeX.",
    )
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    dir_in = args.entrada.resolve()
    rutas = listar_parquets_imfs(dir_in)
    if not rutas:
        logger.error(
            "No se encontraron parquets *_imfs_*.parquet en %s. "
            "Ejecute primero ejecutar_descomposiciones_emdsynth.py.",
            dir_in,
        )
        sys.exit(1)

    mod = _cargar_modulo_descomposiciones()
    filas: List[Dict[str, Any]] = []
    for ruta in rutas:
        try:
            fila = procesar_parquet(
                mod,
                ruta,
                random_state=args.random_state,
                max_imf=args.max_imf,
                k_reduccion_objetivo=args.k_reduccion,
            )
            filas.append(fila)
            logger.info(
                "%s %s: p=%s k=%s N=%s rho_bar=%.4e pi=%.2f",
                fila["escenario"],
                fila["metodo"],
                fila["p_imf_entrada_ica"],
                fila["k_reduccion_ica"],
                fila["n_componentes_reporte"],
                fila["corr_promedio_pares_ica"],
                fila["frac_pares_p_lt_005"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Fallo en %s: %s", ruta, exc)
            sys.exit(1)

    out_json = (
        args.salida_json
        if args.salida_json is not None
        else dir_in / "metricas_ica_emdsynth.json"
    )
    payload = {
        "random_state": args.random_state,
        "max_imf_entrada": args.max_imf,
        "k_reduccion_objetivo": args.k_reduccion,
        "filas": filas,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("JSON guardado en %s", out_json)

    if args.latex:
        _imprimir_latex_tabla(filas)


def _imprimir_latex_tabla(filas: List[Dict[str, Any]]) -> None:
    """
    Imprime filas LaTeX ordenadas como en ``tab:synthetic_emd_summary``.

    Parameters
    ----------
    filas : list of dict
        Salida de ``procesar_parquet`` para todos los escenarios y métodos.
    """
    orden_escenarios = [
        "frecuencias_cercanas",
        "chirp_lineal",
        "burst_sobre_portadora",
        "combinado_ruido",
        "superposicion_multicomponente",
    ]
    orden_metodos = ["emd", "eemd", "ceemdan", "vmd"]
    idx = {(f["escenario"], f["metodo"]): f for f in filas}
    etiquetas = [
        "Close tones",
        "Linear chirp",
        "Burst on carrier",
        "Close tones + chirp + noise",
        "MULTI",
    ]
    for esc, lab in zip(orden_escenarios, etiquetas):
        partes = []
        for met in orden_metodos:
            f = idx.get((esc, met))
            if f is None:
                partes.extend(["?", "?", "?"])
                continue
            rho = f["corr_promedio_pares_ica"]
            pi = f["frac_pares_p_lt_005"]
            nrep = f["n_componentes_reporte"]
            if np.isnan(pi):
                pi_s = "---"
            else:
                pi_s = f"{pi:.2f}"
            partes.append(str(nrep))
            partes.append(f"{rho:.3f}")
            partes.append(pi_s)
        print(f"{lab} & " + " & ".join(partes) + r" \\")


if __name__ == "__main__":
    main()
