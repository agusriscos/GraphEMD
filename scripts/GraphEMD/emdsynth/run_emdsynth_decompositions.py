#!/usr/bin/env python3
"""
Generación de señales sintéticas y descomposición con EMD, EEMD, CEEMDAN y VMD.

Las señales reutilizan los generadores de ``GraphEMD.data.emdsynth_utils``.
Los resultados (IMFs + métricas agregadas) se guardan bajo el directorio de
salida configurado (por defecto ``scripts/GraphEMD/emdsynth/out``).

Ejemplo
-------
Desde la raíz del repositorio::

    PYTHONPATH=src/python python3 scripts/GraphEMD/emdsynth/ejecutar_descomposiciones_emdsynth.py

Requiere ``EMD-signal`` (import ``PyEMD``) y ``vmdpy`` (VMD).

Hiperparámetros EEMD/CEEMDAN/VMD por defecto provienen de una rejilla sobre el escenario
``superposicion_multicomponente`` (``--calibrar-multicomponente`` regenera el JSON de salida).
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Raíz del repositorio GraphEMD
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Carga directa de ``emdsynth_utils`` para no ejecutar ``GraphEMD.data`` (depende de torch).
_ruta_emdsynth = (
    _REPO_ROOT / "src" / "python" / "GraphEMD" / "data" / "emdsynth_utils.py"
)
_spec_emdsynth = importlib.util.spec_from_file_location(
    "emdsynth_utils_emdsynth_script", _ruta_emdsynth
)
if _spec_emdsynth is None or _spec_emdsynth.loader is None:
    raise ImportError(f"No se pudo cargar {_ruta_emdsynth}")
_emdsynth = importlib.util.module_from_spec(_spec_emdsynth)
_spec_emdsynth.loader.exec_module(_emdsynth)
generar_senal_chirp = _emdsynth.generar_senal_chirp
generar_senal_frecuencia_cercana = _emdsynth.generar_senal_frecuencia_cercana
generar_senal_mode_mixing = _emdsynth.generar_senal_mode_mixing

logger = logging.getLogger(__name__)

try:
    from PyEMD import CEEMDAN, EEMD, EMD

    _PYEMD_OK = True
except ImportError:
    CEEMDAN = EEMD = EMD = None  # type: ignore[misc, assignment]
    _PYEMD_OK = False

try:
    from vmdpy import VMD

    _VMD_OK = True
except ImportError:
    VMD = None  # type: ignore[misc, assignment]
    _VMD_OK = False


def _asegurar_pyemd() -> None:
    """
    Comprueba que PyEMD esté disponible.

    Raises
    ------
    ImportError
        Si no se puede importar ``PyEMD``.
    """
    if not _PYEMD_OK or EMD is None:
        raise ImportError(
            "Se requiere PyEMD (pip install EMD-signal). "
            "Ejecutar en un entorno con esa dependencia instalada."
        )


def _asegurar_vmd() -> None:
    """
    Comprueba que ``vmdpy`` esté disponible.

    Raises
    ------
    ImportError
        Si no se puede importar ``vmdpy.VMD``.
    """
    if not _VMD_OK or VMD is None:
        raise ImportError(
            "Se requiere vmdpy (pip install vmdpy). "
            "Ejecutar en un entorno con esa dependencia instalada."
        )


def imfs_a_dataframe(imfs: np.ndarray) -> pd.DataFrame:
    """
    Convierte un array de IMFs (última fila = residuo) en un ``DataFrame``.

    Parameters
    ----------
    imfs : np.ndarray
        Forma ``(n_modos, n_muestras)``. La última fila se interpreta como residuo.

    Returns
    -------
    pd.DataFrame
        Columnas ``IMF_1`` … ``IMF_{K}`` y ``Residuo``.
    """
    imfs = np.asarray(imfs, dtype=np.float64)
    if imfs.ndim != 2:
        raise ValueError("imfs debe ser 2D (n_modos, n_muestras).")
    n = imfs.shape[0]
    if n < 2:
        return pd.DataFrame({"Residuo": imfs[0]})
    df = pd.DataFrame()
    for i in range(n - 1):
        df[f"IMF_{i + 1}"] = imfs[i]
    df["Residuo"] = imfs[-1]
    return df


def descomponer_emd(serie: np.ndarray, max_imf: int) -> np.ndarray:
    """
    Descompone la serie con EMD clásico.

    Parameters
    ----------
    serie : np.ndarray
        Serie 1D.
    max_imf : int
        Número máximo de modos intrínsecos (parámetro ``max_imf`` de PyEMD).

    Returns
    -------
    np.ndarray
        Array ``(n_modos, n_muestras)`` con IMFs y residuo en la última fila.
    """
    _asegurar_pyemd()
    emd = EMD()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(emd.emd(np.asarray(serie, dtype=np.float64), max_imf=max_imf))


def descomponer_eemd(
    serie: np.ndarray,
    max_imf: int,
    trials: int,
    noise_width: float,
    sd_thresh: float,
    s_number: int,
    fixe_h: int,
    semilla_ruido: int,
) -> np.ndarray:
    """
    Descompone la serie con EEMD (ensemble de EMD con ruido).

    PyEMD devuelve la media del ensemble de IMFs; el residuo que cierra la
    reconstrucción se expone en ``residue`` y aquí se concatena como última fila.

    Parameters
    ----------
    serie : np.ndarray
        Serie 1D.
    max_imf : int
        Número máximo de IMFs.
    trials : int
        Número de realizaciones del ensemble.
    noise_width : float
        Escala del ruido respecto al rango de la señal (definición PyEMD).
    sd_thresh : float
        Umbral SD del criterio de parada del sifting.
    s_number : int
        Número de iteraciones de sifting.
    fixe_h : int
        Iteraciones mínimas cuando se cumple la condición de IMF.
    semilla_ruido : int
        Semilla del generador de ruido del ensemble.

    Returns
    -------
    np.ndarray
        IMFs ensemble y residuo (última fila).
    """
    _asegurar_pyemd()
    eemd = EEMD(
        max_imf=max_imf,
        SD_thresh=sd_thresh,
        S_number=s_number,
        FIXE_H=fixe_h,
        trials=trials,
        noise_width=noise_width,
        parallel=False,
    )
    eemd.noise_seed(semilla_ruido)
    x = np.asarray(serie, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e_imf = np.asarray(eemd(x, max_imf=max_imf))
    residuo = np.asarray(eemd.residue, dtype=np.float64)
    return np.vstack([e_imf, residuo.reshape(1, -1)])


def descomponer_ceemdan(
    serie: np.ndarray,
    max_imf: int,
    trials: int,
    epsilon: float,
    seed: int,
) -> np.ndarray:
    """
    Descompone la serie con CEEMDAN.

    Parameters
    ----------
    serie : np.ndarray
        Serie 1D.
    max_imf : int
        Número máximo de IMFs.
    trials : int
        Número de realizaciones del ensemble.
    epsilon : float
        Escala del ruido adaptativo (fracción de la desviación típica).
    seed : int
        Semilla para el ruido.

    Returns
    -------
    np.ndarray
        IMFs y residuo (última fila).
    """
    _asegurar_pyemd()
    ceemdan = CEEMDAN(trials=trials, epsilon=epsilon, parallel=False)
    ceemdan.noise_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(ceemdan(np.asarray(serie, dtype=np.float64), max_imf=max_imf))


def descomponer_vmd(
    serie: np.ndarray,
    k: int,
    alpha: float,
    tau: float = 0.0,
    dc: int = 0,
    init: int = 1,
    tol: float = 1e-7,
) -> np.ndarray:
    """
    Descompone la serie con VMD (Variational Mode Decomposition).

    ``vmdpy`` devuelve ``K`` modos banda limitada; el residuo de cierre se
    define como ``x - sum(u_k)`` para alinear el contrato con EMD/EEMD/CEEMDAN
    (última fila = tendencia o error de reconstrucción).

    Parameters
    ----------
    serie : np.ndarray
        Serie 1D.
    k : int
        Número de modos VMD (``K``).
    alpha : float
        Penalización de ancho de banda (mayor ``alpha`` → modos más estrechos).
    tau : float, optional
        Tolerancia al ruido en la formulación dual (0 = sin término de ruido).
    dc : int, optional
        Si es 1, el primer modo puede capturar componente DC/tendencia.
    init : int, optional
        Esquema de inicialización de frecuencias centrales (1 = uniforme en [0, 0.5]).
    tol : float, optional
        Tolerancia de convergencia del ADMM.

    Returns
    -------
    np.ndarray
        Modos VMD y residuo (última fila), forma ``(K+1, n_muestras)``.
    """
    _asegurar_vmd()
    x = np.asarray(serie, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modos, _, _ = VMD(x, alpha, tau, k, dc, init, tol)
    modos = np.asarray(modos, dtype=np.float64)
    if modos.ndim == 1:
        modos = modos.reshape(1, -1)
    n = x.shape[0]
    if modos.shape[1] != n:
        # ``vmdpy`` puede devolver una muestra menos con longitudes impares (FFT).
        if modos.shape[1] < n:
            pad_ancho = n - modos.shape[1]
            modos = np.pad(modos, ((0, 0), (0, pad_ancho)), mode="edge")
        else:
            modos = modos[:, :n]
    residuo = x - np.sum(modos, axis=0)
    return np.vstack([modos, residuo.reshape(1, -1)])


def _energia_por_fila(imfs: np.ndarray) -> np.ndarray:
    """
    Varianza por fila (cada modo).

    Parameters
    ----------
    imfs : np.ndarray
        Forma ``(n_modos, n_muestras)``.

    Returns
    -------
    np.ndarray
        Varianza de cada modo.
    """
    return np.var(imfs, axis=1, dtype=np.float64)


def _estadisticos_corr_pares_imfs(
    imfs: np.ndarray,
    max_filas: int = 8,
    alpha: float = 0.05,
) -> Tuple[float, int, float]:
    """
    Correlación de Pearson entre pares de IMFs: media de |r| y p-valores (H0: ρ=0, bilateral).

    Para cada par (i,j) en el triángulo superior se usa ``scipy.stats.pearsonr``;
    con T muestras grandes, correlaciones distintas de cero suelen producir p muy
    pequeños (alta potencia).

    Parameters
    ----------
    imfs : np.ndarray
        Filas = modos (sin residuo), columnas = tiempo.
    max_filas : int
        Máximo de filas (IMFs) a incluir.
    alpha : float
        Umbral para contar pares con p-valor menor que ``alpha`` (sin corrección múltiple).

    Returns
    -------
    tuple
        ``(media_abs_r, n_pares, frac_p_lt_alpha)``.
    """
    m = min(imfs.shape[0], max_filas)
    if m < 2:
        return 0.0, 0, float("nan")
    sub = imfs[:m, :]
    norms = np.linalg.norm(sub, axis=1)
    mask = norms > 1e-12 * (np.max(norms) + 1e-12)
    sub = sub[mask]
    n_mod = sub.shape[0]
    if n_mod < 2:
        return 0.0, 0, float("nan")
    pvals: List[float] = []
    abs_rs: List[float] = []
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            xi = sub[i, :]
            xj = sub[j, :]
            if np.std(xi) < 1e-15 or np.std(xj) < 1e-15:
                continue
            r_ij, p_ij = pearsonr(xi, xj)
            if np.isfinite(r_ij) and np.isfinite(p_ij):
                abs_rs.append(float(abs(r_ij)))
                pvals.append(float(p_ij))
    if not pvals:
        return 0.0, 0, float("nan")
    p_arr = np.asarray(pvals, dtype=np.float64)
    frac = float(np.mean(p_arr < alpha))
    return (
        float(np.mean(abs_rs)),
        len(pvals),
        frac,
    )


def calcular_metricas(serie: np.ndarray, imfs: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas de diagnóstico para una descomposición.

    Parameters
    ----------
    serie : np.ndarray
        Señal original.
    imfs : np.ndarray
        Salida del método (IMFs + residuo).

    Returns
    -------
    dict
        Incluye ``corr_promedio_pares``, ``n_pares_corr``, ``frac_pares_p_lt_005``.
    """
    x = np.asarray(serie, dtype=np.float64)
    recon = np.sum(imfs, axis=0)
    nx = np.linalg.norm(x) + 1e-15
    rmse_rel = float(np.linalg.norm(recon - x) / nx)
    var_x = float(np.var(x)) + 1e-15
    e = _energia_por_fila(imfs)
    e_sum = float(np.sum(e)) + 1e-15
    imfs_sin_residuo = imfs[:-1, :] if imfs.shape[0] >= 2 else imfs
    media_abs_r, n_pares, frac_p = _estadisticos_corr_pares_imfs(imfs_sin_residuo)
    return {
        "n_modos": float(imfs.shape[0]),
        "rmse_relativo": rmse_rel,
        "energia_imf1_frac": float(e[0] / var_x) if e.size else 0.0,
        "corr_promedio_pares": media_abs_r,
        "n_pares_corr": float(n_pares),
        "frac_pares_p_lt_005": frac_p,
        "frac_energia_residuo": float(e[-1] / e_sum) if e.size else 0.0,
    }


def _senal_sinusoidal(
    t: np.ndarray,
    frecuencia: float,
    amplitud: float,
    fase: float,
    funcion: str,
) -> np.ndarray:
    """
    Onda sinusoidal ``A·sin(2πft+φ)`` o coseno según ``funcion``.

    Parameters
    ----------
    t : np.ndarray
        Tiempos.
    frecuencia : float
        Frecuencia en Hz.
    amplitud : float
        Amplitud.
    fase : float
        Fase en radianes.
    funcion : str
        ``sin`` o ``cos``.

    Returns
    -------
    np.ndarray
        Muestras de la sinusoide.
    """
    ang = 2.0 * np.pi * frecuencia * t + fase
    if funcion == "cos":
        return amplitud * np.cos(ang)
    return amplitud * np.sin(ang)


def construir_escenarios(
    duracion: float,
    frecuencia_muestreo: float,
    semilla_ruido: int,
) -> List[Tuple[str, np.ndarray, np.ndarray, Dict[str, Any]]]:
    """
    Construye la lista de escenarios sintéticos (nombre, tiempo, señal, metadatos).

    Parameters
    ----------
    duracion : float
        Duración en segundos.
    frecuencia_muestreo : float
        Frecuencia de muestreo (Hz).
    semilla_ruido : int
        Semilla para el ruido en el escenario combinado.

    Returns
    -------
    list of tuple
        Cada elemento: ``(nombre, t, x, meta_dict)``. El escenario
        ``superposicion_multicomponente`` incluye una rampa lineal ascendente
        adicional (tendencia lenta tipo índice) además de los generadores elementales.
    """
    num_muestras = int(duracion * frecuencia_muestreo)
    t_base = np.linspace(0.0, duracion, num_muestras, endpoint=False)

    out: List[Tuple[str, np.ndarray, np.ndarray, Dict[str, Any]]] = []

    _, x_close = generar_senal_frecuencia_cercana(
        f1=10.0,
        f2=11.3,
        amplitud1=1.0,
        amplitud2=1.0,
        duracion=duracion,
        frecuencia_muestreo=frecuencia_muestreo,
        t=t_base,
    )
    out.append(
        (
            "frecuencias_cercanas",
            t_base,
            x_close,
            {"descripcion": "Dos tonos cercanos (batido de amplitud)."},
        )
    )

    _, x_chirp = generar_senal_chirp(
        f0=1.5,
        k=1.2,
        amplitud=1.0,
        duracion=duracion,
        frecuencia_muestreo=frecuencia_muestreo,
        t=t_base,
    )
    out.append(
        (
            "chirp_lineal",
            t_base,
            x_chirp,
            {"descripcion": "Chirp con frecuencia instantánea lineal."},
        )
    )

    _, x_mm = generar_senal_mode_mixing(
        f_low=0.8,
        f_high=12.0,
        alpha=0.85,
        t1=1.2,
        t2=3.2,
        duracion=duracion,
        frecuencia_muestreo=frecuencia_muestreo,
        t=t_base,
    )
    out.append(
        (
            "burst_sobre_portadora",
            t_base,
            x_mm,
            {"descripcion": "Portadora baja + burst de alta frecuencia en ventana."},
        )
    )

    rng = np.random.default_rng(semilla_ruido)
    ruido = 0.03 * float(np.std(x_close)) * rng.standard_normal(num_muestras)
    x_comb = x_close + 0.35 * x_chirp + ruido
    out.append(
        (
            "combinado_ruido",
            t_base,
            x_comb,
            {
                "descripcion": "Superposición de tonos cercanos, chirp y ruido gaussiano."
            },
        )
    )

    # Multicomponente al estilo del notebook ``emdsynth_combinado`` (Ejemplo 4):
    # mode mixing + dos chirps + tonos cercanos + dos sinusoides + ruido.
    _, x_mm_m = generar_senal_mode_mixing(
        f_low=0.3,
        f_high=7.0,
        alpha=1.0,
        t1=duracion * 0.25,
        t2=duracion * 0.75,
        duracion=duracion,
        frecuencia_muestreo=frecuencia_muestreo,
        t=t_base,
    )
    _, x_c1_m = generar_senal_chirp(
        f0=0.8,
        k=0.2,
        amplitud=1.2,
        funcion="sin",
        duracion=duracion,
        frecuencia_muestreo=frecuencia_muestreo,
        t=t_base,
    )
    _, x_c2_m = generar_senal_chirp(
        f0=4.0,
        k=-0.15,
        amplitud=0.9,
        funcion="cos",
        fase=float(np.pi / 4),
        duracion=duracion,
        frecuencia_muestreo=frecuencia_muestreo,
        t=t_base,
    )
    _, x_fc_m = generar_senal_frecuencia_cercana(
        f1=1.5,
        f2=1.7,
        amplitud1=0.7,
        amplitud2=0.7,
        fase1=0.0,
        fase2=float(np.pi / 6),
        funcion="sin",
        duracion=duracion,
        frecuencia_muestreo=frecuencia_muestreo,
        t=t_base,
    )
    x_sin_lf = _senal_sinusoidal(t_base, 0.5, 1.0, 0.0, "sin")
    x_sin_hf = _senal_sinusoidal(t_base, 3.0, 0.6, float(np.pi / 3), "cos")
    ruido_m = rng.normal(0.0, 0.2, num_muestras)
    # Tendencia lenta creciente (tipo deriva de índice): rampa lineal 0 → escala en la ventana.
    # Pendiente mayor que en versiones previas para que la deriva sea visible junto al resto de componentes.
    escala_tendencia = 8.0
    x_tendencia = escala_tendencia * (t_base / float(duracion))
    x_multi = (
        x_mm_m
        + x_c1_m
        + x_c2_m
        + x_fc_m
        + x_sin_lf
        + x_sin_hf
        + ruido_m
        + x_tendencia
    )
    out.append(
        (
            "superposicion_multicomponente",
            t_base,
            x_multi,
            {
                "descripcion": (
                    "Suma de mode mixing, chirps, tonos cercanos, sinusoides 0.5 y 3 Hz, "
                    "ruido N(0,0.2) y tendencia lenta ascendente (rampa lineal)."
                ),
                "tendencia_lineal_amplitud": escala_tendencia,
            },
        )
    )

    return out


def _extraer_senal_multicomponente() -> np.ndarray:
    """
    Devuelve la serie del escenario ``superposicion_multicomponente``.

    Returns
    -------
    np.ndarray
        Muestras 1D de la señal sintética agregada.
    """
    for nombre, _t, x_vec, _meta in construir_escenarios(
        duracion=5.0,
        frecuencia_muestreo=500.0,
        semilla_ruido=42,
    ):
        if nombre == "superposicion_multicomponente":
            return np.asarray(x_vec, dtype=np.float64)
    raise RuntimeError("No se encontró superposicion_multicomponente en construir_escenarios.")


def calibrar_parametros_ensemble_multicomponente(
    max_imf: int = 12,
    semilla_ceemdan: int = 42,
    semilla_eemd: int = 42,
    eemd_s_number: int = 8,
    eemd_fixe_h: int = 5,
    umbral_rmse_relativo: float = 1e-10,
) -> Dict[str, Any]:
    """
    Busca parámetros de EEMD, CEEMDAN y VMD que minimicen el acoplamiento lineal
    entre modos (media de |ρ| en pares) en el escenario multicomponente.

    Se recorre una rejilla acotada; entre configuraciones válidas (reconstrucción
    numérica con RMSE relativo por debajo de ``umbral_rmse_relativo``) se elige
    la menor ``corr_promedio_pares`` y, en empate, la menor fracción de varianza
    en el residuo.

    Parameters
    ----------
    max_imf : int, optional
        Tope de modos intrínsecos (igual que en el pipeline). Por defecto 12.
    semilla_ceemdan : int, optional
        Semilla de ruido CEEMDAN. Por defecto 42.
    semilla_eemd : int, optional
        Semilla de ruido EEMD. Por defecto 42.
    eemd_s_number : int, optional
        ``S_number`` fijo durante la calibración. Por defecto 8.
    eemd_fixe_h : int, optional
        ``FIXE_H`` fijo durante la calibración. Por defecto 5.
    umbral_rmse_relativo : float, optional
        Cota superior aceptable para ``rmse_relativo``. Por defecto 1e-10.

    Returns
    -------
    dict
        Claves ``mejor_ceemdan``, ``mejor_eemd``, ``mejor_vmd`` (cada una con
        hiperparámetros y métricas), ``referencia_emd`` (métricas de EMD clásico
        en la misma señal) y ``rejilla`` (descripción corta de los valores explorados).
    """
    _asegurar_pyemd()
    _asegurar_vmd()
    x = _extraer_senal_multicomponente()
    imfs_ref = descomponer_emd(x, max_imf=max_imf)
    m_ref = calcular_metricas(x, imfs_ref)

    rejilla_ceemdan_eps = [0.03, 0.05, 0.07, 0.09, 0.12]
    rejilla_trials_ceemdan = [100, 140, 180]
    rejilla_eemd_nw = [0.035, 0.055, 0.075, 0.095]
    rejilla_trials_eemd = [100, 140, 180]
    rejilla_eemd_sd = [0.18, 0.22, 0.26, 0.30]
    rejilla_vmd_k = [6, 8, 10, 12]
    rejilla_vmd_alpha = [1000.0, 2000.0, 5000.0]
    rejilla_vmd_dc = [0, 1]

    def _clave(mets: Dict[str, float]) -> Tuple[float, float]:
        return (
            float(mets["corr_promedio_pares"]),
            float(mets["frac_energia_residuo"]),
        )

    mejor_c: Optional[Dict[str, Any]] = None
    mejor_clave_c: Optional[Tuple[float, float]] = None
    for eps, trials in itertools.product(rejilla_ceemdan_eps, rejilla_trials_ceemdan):
        imfs = descomponer_ceemdan(
            x,
            max_imf=max_imf,
            trials=trials,
            epsilon=eps,
            seed=semilla_ceemdan,
        )
        mets = calcular_metricas(x, imfs)
        if mets["rmse_relativo"] >= umbral_rmse_relativo:
            continue
        cl = _clave(mets)
        if mejor_clave_c is None or cl < mejor_clave_c:
            mejor_clave_c = cl
            mejor_c = {
                "epsilon": eps,
                "trials": trials,
                "metricas": mets,
            }

    mejor_e: Optional[Dict[str, Any]] = None
    mejor_clave_e: Optional[Tuple[float, float]] = None
    for nw, trials, sd in itertools.product(
        rejilla_eemd_nw, rejilla_trials_eemd, rejilla_eemd_sd
    ):
        imfs = descomponer_eemd(
            x,
            max_imf=max_imf,
            trials=trials,
            noise_width=nw,
            sd_thresh=sd,
            s_number=eemd_s_number,
            fixe_h=eemd_fixe_h,
            semilla_ruido=semilla_eemd,
        )
        mets = calcular_metricas(x, imfs)
        if mets["rmse_relativo"] >= umbral_rmse_relativo:
            continue
        cl = _clave(mets)
        if mejor_clave_e is None or cl < mejor_clave_e:
            mejor_clave_e = cl
            mejor_e = {
                "noise_width": nw,
                "trials": trials,
                "SD_thresh": sd,
                "metricas": mets,
            }

    if mejor_c is None:
        raise RuntimeError("Calibración CEEMDAN: ninguna configuración válida.")
    if mejor_e is None:
        raise RuntimeError("Calibración EEMD: ninguna configuración válida.")

    mejor_v: Optional[Dict[str, Any]] = None
    mejor_clave_v: Optional[Tuple[float, float]] = None
    for k_vmd, alpha_vmd, dc_vmd in itertools.product(
        rejilla_vmd_k, rejilla_vmd_alpha, rejilla_vmd_dc
    ):
        imfs = descomponer_vmd(
            x,
            k=k_vmd,
            alpha=alpha_vmd,
            dc=dc_vmd,
        )
        mets = calcular_metricas(x, imfs)
        if mets["rmse_relativo"] >= umbral_rmse_relativo:
            continue
        cl = _clave(mets)
        if mejor_clave_v is None or cl < mejor_clave_v:
            mejor_clave_v = cl
            mejor_v = {
                "K": k_vmd,
                "alpha": alpha_vmd,
                "DC": dc_vmd,
                "metricas": mets,
            }

    if mejor_v is None:
        raise RuntimeError("Calibración VMD: ninguna configuración válida.")

    return {
        "mejor_ceemdan": mejor_c,
        "mejor_eemd": mejor_e,
        "mejor_vmd": mejor_v,
        "referencia_emd": {"metricas": m_ref},
        "rejilla": {
            "ceemdan_epsilon": rejilla_ceemdan_eps,
            "ceemdan_trials": rejilla_trials_ceemdan,
            "eemd_noise_width": rejilla_eemd_nw,
            "eemd_trials": rejilla_trials_eemd,
            "eemd_SD_thresh": rejilla_eemd_sd,
            "vmd_K": rejilla_vmd_k,
            "vmd_alpha": rejilla_vmd_alpha,
            "vmd_DC": rejilla_vmd_dc,
        },
    }


def ejecutar_pipeline(
    dir_salida: Path,
    max_imf: int,
    trials_ensemble: int,
    epsilon_ceemdan: float,
    semilla_ceemdan: int,
    semilla_eemd: int,
    eemd_noise_width: float,
    eemd_sd_thresh: float,
    eemd_s_number: int,
    eemd_fixe_h: int,
    vmd_k: int,
    vmd_alpha: float,
    vmd_dc: int,
    vmd_tau: float,
    vmd_tol: float,
    vmd_init: int,
    trials_eemd: Optional[int] = None,
    trials_ceemdan: Optional[int] = None,
) -> pd.DataFrame:
    """
    Ejecuta todos los escenarios y métodos, guarda IMFs y devuelve tabla de métricas.

    Parameters
    ----------
    dir_salida : pathlib.Path
        Directorio de salida (se crea si no existe).
    max_imf : int
        Límite de modos para los tres algoritmos.
    trials_ensemble : int
        Ensayos para EEMD y CEEMDAN.
    epsilon_ceemdan : float
        Parámetro ``epsilon`` de CEEMDAN.
    semilla_ceemdan : int
        Semilla de ruido CEEMDAN.
    semilla_eemd : int
        Semilla del ruido en EEMD (PyEMD).
    eemd_noise_width : float
        Ancho de ruido EEMD.
    eemd_sd_thresh : float
        ``SD_thresh`` EEMD.
    eemd_s_number : int
        ``S_number`` EEMD.
    eemd_fixe_h : int
        ``FIXE_H`` EEMD.
    vmd_k : int
        Número de modos VMD (``K``).
    vmd_alpha : float
        Penalización de ancho de banda VMD.
    vmd_dc : int
        Modo DC en VMD (0 o 1).
    vmd_tau : float
        Tolerancia al ruido VMD.
    vmd_tol : float
        Tolerancia de convergencia VMD.
    vmd_init : int
        Inicialización de frecuencias centrales VMD.
    trials_eemd : int, optional
        Ensayos EEMD. Si es ``None``, se usa ``trials_ensemble``.
    trials_ceemdan : int, optional
        Ensayos CEEMDAN. Si es ``None``, se usa ``trials_ensemble``.

    Returns
    -------
    pd.DataFrame
        Una fila por (escenario, método) con métricas.
    """
    _asegurar_pyemd()
    _asegurar_vmd()
    dir_salida.mkdir(parents=True, exist_ok=True)
    trials_eemd_efectivo = int(
        trials_eemd if trials_eemd is not None else trials_ensemble
    )
    trials_ceemdan_efectivo = int(
        trials_ceemdan if trials_ceemdan is not None else trials_ensemble
    )

    escenarios = construir_escenarios(
        duracion=5.0,
        frecuencia_muestreo=500.0,
        semilla_ruido=42,
    )

    manifest: Dict[str, Any] = {
        "max_imf": max_imf,
        "trials_ensemble": trials_ensemble,
        "trials_eemd": trials_eemd_efectivo,
        "trials_ceemdan": trials_ceemdan_efectivo,
        "epsilon_ceemdan": epsilon_ceemdan,
        "semilla_ceemdan": semilla_ceemdan,
        "semilla_eemd": semilla_eemd,
        "eemd": {
            "noise_width": eemd_noise_width,
            "SD_thresh": eemd_sd_thresh,
            "S_number": eemd_s_number,
            "FIXE_H": eemd_fixe_h,
        },
        "vmd": {
            "K": vmd_k,
            "alpha": vmd_alpha,
            "DC": vmd_dc,
            "tau": vmd_tau,
            "tol": vmd_tol,
            "init": vmd_init,
        },
        "escenarios": [],
    }

    filas: List[Dict[str, Any]] = []

    for nombre, t_vec, x_vec, meta in escenarios:
        manifest["escenarios"].append({"id": nombre, **meta})

        df_sig = pd.DataFrame({"tiempo": t_vec, "senal": x_vec})
        ruta_senal = dir_salida / f"{nombre}_senal.parquet"
        df_sig.to_parquet(ruta_senal, index=False)
        logger.info("Guardada señal: %s", ruta_senal)

        descomposiciones: Dict[str, np.ndarray] = {
            "emd": descomponer_emd(x_vec, max_imf=max_imf),
            "eemd": descomponer_eemd(
                x_vec,
                max_imf=max_imf,
                trials=trials_eemd_efectivo,
                noise_width=eemd_noise_width,
                sd_thresh=eemd_sd_thresh,
                s_number=eemd_s_number,
                fixe_h=eemd_fixe_h,
                semilla_ruido=semilla_eemd,
            ),
            "ceemdan": descomponer_ceemdan(
                x_vec,
                max_imf=max_imf,
                trials=trials_ceemdan_efectivo,
                epsilon=epsilon_ceemdan,
                seed=semilla_ceemdan,
            ),
            "vmd": descomponer_vmd(
                x_vec,
                k=vmd_k,
                alpha=vmd_alpha,
                tau=vmd_tau,
                dc=vmd_dc,
                init=vmd_init,
                tol=vmd_tol,
            ),
        }

        for metodo, imfs in descomposiciones.items():
            df_imfs = imfs_a_dataframe(imfs)
            df_imfs.insert(0, "tiempo", t_vec)
            ruta_imfs = dir_salida / f"{nombre}_imfs_{metodo}.parquet"
            df_imfs.to_parquet(ruta_imfs, index=False)

            mets = calcular_metricas(x_vec, imfs)
            filas.append(
                {
                    "escenario": nombre,
                    "metodo": metodo,
                    **mets,
                }
            )
            logger.info(
                "%s %s: n_modos=%s rmse_rel=%.3e corr_pairs=%.4f frac_p<.05=%.2f",
                nombre,
                metodo,
                int(mets["n_modos"]),
                mets["rmse_relativo"],
                mets["corr_promedio_pares"],
                mets["frac_pares_p_lt_005"],
            )

    ruta_manifest = dir_salida / "manifest_emdsynth.json"
    with open(ruta_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    df_metricas = pd.DataFrame(filas)
    df_metricas.to_csv(dir_salida / "metricas_emdsynth_metodos.csv", index=False)

    # Resumen comparativo EMD vs CEEMDAN por escenario (para redacción)
    pivot = df_metricas.pivot_table(
        index="escenario",
        columns="metodo",
        values=[
            "n_modos",
            "corr_promedio_pares",
            "frac_pares_p_lt_005",
            "rmse_relativo",
        ],
    )
    pivot.to_csv(dir_salida / "resumen_pivot_emdsynth.csv")
    logger.info(
        "Métricas guardadas en %s", dir_salida / "metricas_emdsynth_metodos.csv"
    )
    return df_metricas


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parsea argumentos de línea de comandos.

    Parameters
    ----------
    argv : list of str, optional
        Argumentos (por defecto ``sys.argv[1:]``).

    Returns
    -------
    argparse.Namespace
        Namespace con ``salida``, ``max_imf``, ``trials``, etc.
    """
    p = argparse.ArgumentParser(
        description=(
            "Descomposiciones EMD / EEMD / CEEMDAN / VMD sobre señales sintéticas."
        )
    )
    p.add_argument(
        "--salida",
        type=Path,
        default=_REPO_ROOT / "scripts" / "GraphEMD" / "emdsynth" / "out",
        help="Directorio de salida para parquet, CSV y manifest.",
    )
    p.add_argument(
        "--max-imf", type=int, default=12, help="Máximo de IMFs (+ residuo)."
    )
    p.add_argument(
        "--trials",
        type=int,
        default=120,
        help="Ensayos de referencia (campo ``trials_ensemble`` del manifest).",
    )
    p.add_argument(
        "--trials-eemd",
        type=int,
        default=100,
        help="Ensayos EEMD (valor por defecto tras calibración en señal multicomponente).",
    )
    p.add_argument(
        "--trials-ceemdan",
        type=int,
        default=180,
        help="Ensayos CEEMDAN (calibrado en multicomponente).",
    )
    p.add_argument(
        "--epsilon-ceemdan",
        type=float,
        default=0.03,
        help="Epsilon relativo (CEEMDAN); valor por defecto calibrado en multicomponente.",
    )
    p.add_argument("--semilla-ceemdan", type=int, default=42)
    p.add_argument("--semilla-eemd", type=int, default=42)
    p.add_argument(
        "--eemd-noise-width",
        type=float,
        default=0.095,
        help="Ancho de ruido EEMD (calibrado en multicomponente).",
    )
    p.add_argument(
        "--eemd-sd-thresh",
        type=float,
        default=0.18,
        help="SD_thresh EEMD (calibrado en multicomponente).",
    )
    p.add_argument("--eemd-s-number", type=int, default=8)
    p.add_argument("--eemd-fixe-h", type=int, default=5)
    p.add_argument(
        "--vmd-k",
        type=int,
        default=6,
        help="Número de modos VMD (K); valor por defecto calibrado en multicomponente.",
    )
    p.add_argument(
        "--vmd-alpha",
        type=float,
        default=5000.0,
        help="Penalización de ancho de banda VMD (calibrado en multicomponente).",
    )
    p.add_argument(
        "--vmd-dc",
        type=int,
        choices=[0, 1],
        default=1,
        help="Modo DC en VMD (1 permite capturar tendencia lenta).",
    )
    p.add_argument("--vmd-tau", type=float, default=0.0)
    p.add_argument("--vmd-tol", type=float, default=1e-7)
    p.add_argument("--vmd-init", type=int, default=1)
    p.add_argument(
        "--calibrar-multicomponente",
        action="store_true",
        help=(
            "Solo ejecuta búsqueda por rejilla en la señal multicomponente y escribe "
            "parametros_calibrados_multicomponente.json bajo --salida."
        ),
    )
    return p.parse_args(argv)


def _metricas_a_json_serializable(obj: Any) -> Any:
    """
    Convierte métricas con tipos NumPy en tipos admitidos por ``json.dumps``.

    Parameters
    ----------
    obj : Any
        Valor anidado (dict, lista, escalar).

    Returns
    -------
    Any
        Estructura equivalente serializable.
    """
    if isinstance(obj, dict):
        return {k: _metricas_a_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_metricas_a_json_serializable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def main(argv: Optional[List[str]] = None) -> None:
    """
    Punto de entrada: configura logging y ejecuta el pipeline.

    Parameters
    ----------
    argv : list of str, optional
        Argumentos de línea de comandos.
    """
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    dir_salida = args.salida.resolve()
    if args.calibrar_multicomponente:
        resultado = calibrar_parametros_ensemble_multicomponente(
            max_imf=args.max_imf,
            semilla_ceemdan=args.semilla_ceemdan,
            semilla_eemd=args.semilla_eemd,
            eemd_s_number=args.eemd_s_number,
            eemd_fixe_h=args.eemd_fixe_h,
        )
        dir_salida.mkdir(parents=True, exist_ok=True)
        ruta_cal = dir_salida / "parametros_calibrados_multicomponente.json"
        with open(ruta_cal, "w", encoding="utf-8") as f:
            json.dump(
                _metricas_a_json_serializable(resultado),
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info("Calibración guardada en %s", ruta_cal)
        mc = resultado["mejor_ceemdan"]
        me = resultado["mejor_eemd"]
        logger.info(
            "Mejor CEEMDAN: epsilon=%s trials=%s corr=%.5f frac_res=%.4f",
            mc["epsilon"],
            mc["trials"],
            mc["metricas"]["corr_promedio_pares"],
            mc["metricas"]["frac_energia_residuo"],
        )
        logger.info(
            "Mejor EEMD: noise_width=%s trials=%s SD_thresh=%s corr=%.5f frac_res=%.4f",
            me["noise_width"],
            me["trials"],
            me["SD_thresh"],
            me["metricas"]["corr_promedio_pares"],
            me["metricas"]["frac_energia_residuo"],
        )
        mv = resultado["mejor_vmd"]
        logger.info(
            "Mejor VMD: K=%s alpha=%s DC=%s corr=%.5f frac_res=%.4f",
            mv["K"],
            mv["alpha"],
            mv["DC"],
            mv["metricas"]["corr_promedio_pares"],
            mv["metricas"]["frac_energia_residuo"],
        )
        return

    ejecutar_pipeline(
        dir_salida=dir_salida,
        max_imf=args.max_imf,
        trials_ensemble=args.trials,
        epsilon_ceemdan=args.epsilon_ceemdan,
        semilla_ceemdan=args.semilla_ceemdan,
        semilla_eemd=args.semilla_eemd,
        eemd_noise_width=args.eemd_noise_width,
        eemd_sd_thresh=args.eemd_sd_thresh,
        eemd_s_number=args.eemd_s_number,
        eemd_fixe_h=args.eemd_fixe_h,
        vmd_k=args.vmd_k,
        vmd_alpha=args.vmd_alpha,
        vmd_dc=args.vmd_dc,
        vmd_tau=args.vmd_tau,
        vmd_tol=args.vmd_tol,
        vmd_init=args.vmd_init,
        trials_eemd=args.trials_eemd,
        trials_ceemdan=args.trials_ceemdan,
    )


if __name__ == "__main__":
    main()
