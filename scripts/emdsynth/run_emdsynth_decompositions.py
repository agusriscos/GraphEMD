#!/usr/bin/env python3
"""
Synthetic signal generation and decomposition with EMD, EEMD, CEEMDAN, and VMD.

Signals reuse generators from ``GraphEMD.data.emdsynth_utils``. Results (IMFs +
aggregated metrics) are saved under the configured output directory (default
``scripts/emdsynth/out``).

Example
-------
From the repository root::

    PYTHONPATH=src/python python3 scripts/emdsynth/run_emdsynth_decompositions.py

Requires ``EMD-signal`` (import ``PyEMD``) and ``vmdpy`` (VMD).

Default EEMD/CEEMDAN/VMD hyperparameters come from a grid search on the
``superposicion_multicomponente`` scenario (``--calibrar-multicomponente`` regenerates
the output JSON).
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

# GraphEMD repository root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Direct load of ``emdsynth_utils`` to avoid executing ``GraphEMD.data`` (depends on torch).
_ruta_emdsynth = (
    _REPO_ROOT / "src" / "python" / "GraphEMD" / "data" / "emdsynth_utils.py"
)
_spec_emdsynth = importlib.util.spec_from_file_location(
    "emdsynth_utils_emdsynth_script", _ruta_emdsynth
)
if _spec_emdsynth is None or _spec_emdsynth.loader is None:
    raise ImportError(f"Could not load {_ruta_emdsynth}")
_emdsynth = importlib.util.module_from_spec(_spec_emdsynth)
_spec_emdsynth.loader.exec_module(_emdsynth)
generate_chirp_signal = _emdsynth.generate_chirp_signal
generate_close_frequency_signal = _emdsynth.generate_close_frequency_signal
generate_mode_mixing_signal = _emdsynth.generate_mode_mixing_signal

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
    Check that PyEMD is available.

    Raises
    ------
    ImportError
        If ``PyEMD`` cannot be imported.
    """
    if not _PYEMD_OK or EMD is None:
        raise ImportError(
            "PyEMD is required (pip install EMD-signal). "
            "Run in an environment with that dependency installed."
        )


def _asegurar_vmd() -> None:
    """
    Check that ``vmdpy`` is available.

    Raises
    ------
    ImportError
        If ``vmdpy.VMD`` cannot be imported.
    """
    if not _VMD_OK or VMD is None:
        raise ImportError(
            "vmdpy is required (pip install vmdpy). "
            "Run in an environment with that dependency installed."
        )


def imfs_a_dataframe(imfs: np.ndarray) -> pd.DataFrame:
    """
    Convert an IMF array (last row = residue) into a ``DataFrame``.

    Parameters
    ----------
    imfs : np.ndarray
        Shape ``(n_modos, n_muestras)``. The last row is interpreted as residue.

    Returns
    -------
    pd.DataFrame
        Columns ``IMF_1`` … ``IMF_{K}`` and ``Residuo``.
    """
    imfs = np.asarray(imfs, dtype=np.float64)
    if imfs.ndim != 2:
        raise ValueError("imfs must be 2D (n_modes, n_samples).")
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
    Decompose the time series with classical EMD.

    Parameters
    ----------
    serie : np.ndarray
        1D series.
    max_imf : int
        Maximum number of intrinsic modes (PyEMD ``max_imf`` parameter).

    Returns
    -------
    np.ndarray
        Array ``(n_modos, n_muestras)`` with IMFs and residue in the last row.
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
    Decompose the time series with EEMD (noise-assisted EMD ensemble).

    PyEMD returns the ensemble mean of IMFs; the residue that closes the
    reconstruction is exposed in ``residue`` and is concatenated here as the last row.

    Parameters
    ----------
    serie : np.ndarray
        1D series.
    max_imf : int
        Maximum number of IMFs.
    trials : int
        Number of ensemble realizations.
    noise_width : float
        Noise scale relative to the signal range (PyEMD definition).
    sd_thresh : float
        SD threshold for the sifting stop criterion.
    s_number : int
        Number of sifting iterations.
    fixe_h : int
        Minimum iterations when the IMF condition is met.
    semilla_ruido : int
        Seed for the ensemble noise generator.

    Returns
    -------
    np.ndarray
        Ensemble IMFs and residue (last row).
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
    Decompose the time series with CEEMDAN.

    Parameters
    ----------
    serie : np.ndarray
        1D series.
    max_imf : int
        Maximum number of IMFs.
    trials : int
        Number of ensemble realizations.
    epsilon : float
        Adaptive noise scale (fraction of the standard deviation).
    seed : int
        Noise seed.

    Returns
    -------
    np.ndarray
        IMFs and residue (last row).
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
    Decompose the time series with VMD (Variational Mode Decomposition).

    ``vmdpy`` returns ``K`` band-limited modes; the closing residue is defined as
    ``x - sum(u_k)`` to align the contract with EMD/EEMD/CEEMDAN
    (last row = trend or reconstruction error).

    Parameters
    ----------
    serie : np.ndarray
        1D series.
    k : int
        Number of VMD modes (``K``).
    alpha : float
        Bandwidth penalty (larger ``alpha`` → narrower modes).
    tau : float, optional
        Noise tolerance in the dual formulation (0 = no noise term).
    dc : int, optional
        If 1, the first mode may capture a DC/trend component.
    init : int, optional
        Central-frequency initialization scheme (1 = uniform in [0, 0.5]).
    tol : float, optional
        ADMM convergence tolerance.

    Returns
    -------
    np.ndarray
        VMD modes and residue (last row), shape ``(K+1, n_muestras)``.
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
        # ``vmdpy`` may return one fewer sample with odd lengths (FFT).
        if modos.shape[1] < n:
            pad_ancho = n - modos.shape[1]
            modos = np.pad(modos, ((0, 0), (0, pad_ancho)), mode="edge")
        else:
            modos = modos[:, :n]
    residuo = x - np.sum(modos, axis=0)
    return np.vstack([modos, residuo.reshape(1, -1)])


def _energia_por_fila(imfs: np.ndarray) -> np.ndarray:
    """
    Variance per row (each mode).

    Parameters
    ----------
    imfs : np.ndarray
        Shape ``(n_modos, n_muestras)``.

    Returns
    -------
    np.ndarray
        Variance of each mode.
    """
    return np.var(imfs, axis=1, dtype=np.float64)


def _estadisticos_corr_pares_imfs(
    imfs: np.ndarray,
    max_filas: int = 8,
    alpha: float = 0.05,
) -> Tuple[float, int, float]:
    """
    Pearson correlation between IMF pairs: mean |r| and p-values (H0: ρ=0, two-sided).

    For each pair (i,j) in the upper triangle, ``scipy.stats.pearsonr`` is used;
    with large T, nonzero correlations often yield very small p
    values (high power).

    Parameters
    ----------
    imfs : np.ndarray
        Rows = modes (excluding residue), columns = time.
    max_filas : int
        Maximum number of rows (IMFs) to include.
    alpha : float
        Threshold for counting pairs with p-value below ``alpha`` (without multiple-comparison correction).

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
    Compute diagnostic metrics for a decomposition.

    Parameters
    ----------
    serie : np.ndarray
        Original signal.
    imfs : np.ndarray
        Method output (IMFs + residuo).

    Returns
    -------
    dict
        Includes ``corr_promedio_pares``, ``n_pares_corr``, ``frac_pares_p_lt_005``.
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
    Sinusoidal wave ``A·sin(2πft+φ)`` or cosine according to ``funcion``.

    Parameters
    ----------
    t : np.ndarray
        Time samples.
    frecuencia : float
        Frequency in Hz.
    amplitud : float
        Amplitude.
    fase : float
        Phase in radians.
    funcion : str
        ``sin`` or ``cos``.

    Returns
    -------
    np.ndarray
        Sinusoid samples.
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
    Build the list of synthetic scenarios (name, time, signal, metadata).

    Parameters
    ----------
    duracion : float
        Duration in seconds.
    frecuencia_muestreo : float
        Sampling frequency (Hz).
    semilla_ruido : int
        Seed for noise in the combined scenario.

    Returns
    -------
    list of tuple
        Each element: ``(nombre, t, x, meta_dict)``. The
        ``superposicion_multicomponente`` scenario includes an additional ascending linear ramp
        (slow index-like trend) in addition to the elementary generators.
    """
    num_muestras = int(duracion * frecuencia_muestreo)
    t_base = np.linspace(0.0, duracion, num_muestras, endpoint=False)

    out: List[Tuple[str, np.ndarray, np.ndarray, Dict[str, Any]]] = []

    _, x_close = generate_close_frequency_signal(
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

    _, x_chirp = generate_chirp_signal(
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
            {"descripcion": "Chirp with linear instantaneous frequency."},
        )
    )

    _, x_mm = generate_mode_mixing_signal(
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
            {"descripcion": "Low carrier + high-frequency burst in a window."},
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
                "descripcion": "Superposition of close tones, chirp, and Gaussian noise."
            },
        )
    )

    # Multicomponent scenario in the style of the notebook ``emdsynth_combinado`` (Ejemplo 4):
    # mode mixing + dos chirps + tonos cercanos + dos sinusoides + ruido.
    _, x_mm_m = generate_mode_mixing_signal(
        f_low=0.3,
        f_high=7.0,
        alpha=1.0,
        t1=duracion * 0.25,
        t2=duracion * 0.75,
        duracion=duracion,
        frecuencia_muestreo=frecuencia_muestreo,
        t=t_base,
    )
    _, x_c1_m = generate_chirp_signal(
        f0=0.8,
        k=0.2,
        amplitud=1.2,
        funcion="sin",
        duracion=duracion,
        frecuencia_muestreo=frecuencia_muestreo,
        t=t_base,
    )
    _, x_c2_m = generate_chirp_signal(
        f0=4.0,
        k=-0.15,
        amplitud=0.9,
        funcion="cos",
        fase=float(np.pi / 4),
        duracion=duracion,
        frecuencia_muestreo=frecuencia_muestreo,
        t=t_base,
    )
    _, x_fc_m = generate_close_frequency_signal(
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
    # Slow ascending trend (index-like drift): linear ramp 0 → scale over the window.
    # Steeper slope than in previous versions so drift is visible alongside the other components.
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
                    "Sum of mode mixing, chirps, close tones, 0.5 and 3 Hz sinusoids, "
                    "N(0,0.2) noise and slow ascending trend (linear ramp)."
                ),
                "tendencia_lineal_amplitud": escala_tendencia,
            },
        )
    )

    return out


def _extraer_senal_multicomponente() -> np.ndarray:
    """
    Return the ``superposicion_multicomponente`` scenario series.

    Returns
    -------
    np.ndarray
        1D samples of the aggregated synthetic signal.
    """
    for nombre, _t, x_vec, _meta in construir_escenarios(
        duracion=5.0,
        frecuencia_muestreo=500.0,
        semilla_ruido=42,
    ):
        if nombre == "superposicion_multicomponente":
            return np.asarray(x_vec, dtype=np.float64)
    raise RuntimeError(
        "superposicion_multicomponente not found in construir_escenarios."
    )


def calibrar_parametros_ensemble_multicomponente(
    max_imf: int = 12,
    semilla_ceemdan: int = 42,
    semilla_eemd: int = 42,
    eemd_s_number: int = 8,
    eemd_fixe_h: int = 5,
    umbral_rmse_relativo: float = 1e-10,
) -> Dict[str, Any]:
    """
    Search for EEMD, CEEMDAN, and VMD parameters that minimize linear coupling
    between modes (mean |ρ| over pairs) on the multicomponent scenario.

    A bounded grid is explored; among valid configurations (numerical reconstruction
    with relative RMSE below ``umbral_rmse_relativo``) the smallest ``corr_promedio_pares``
    is chosen and, on ties, the smallest residue variance fraction.

    Parameters
    ----------
    max_imf : int, optional
        Intrinsic-mode cap (same as in the pipeline). Default is 12.
    semilla_ceemdan : int, optional
        CEEMDAN noise seed. Default is 42.
    semilla_eemd : int, optional
        EEMD noise seed. Default is 42.
    eemd_s_number : int, optional
        Fixed ``S_number`` during calibration. Default is 8.
    eemd_fixe_h : int, optional
        Fixed ``FIXE_H`` during calibration. Default is 5.
    umbral_rmse_relativo : float, optional
        Acceptable upper bound for ``rmse_relativo``. Default is 1e-10.

    Returns
    -------
    dict
        Keys ``mejor_ceemdan``, ``mejor_eemd``, ``mejor_vmd`` (each with
        hyperparameters and metrics), ``referencia_emd`` (classical EMD metrics
        on the same signal), and ``rejilla`` (short description of explored values).
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
        raise RuntimeError("CEEMDAN calibration: no valid configuration.")
    if mejor_e is None:
        raise RuntimeError("EEMD calibration: no valid configuration.")

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
        raise RuntimeError("VMD calibration: no valid configuration.")

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
    Run all scenarios and methods, save IMFs, and return a metrics table.

    Parameters
    ----------
    dir_salida : pathlib.Path
        Output directory (created if missing).
    max_imf : int
        Mode limit for all three algorithms.
    trials_ensemble : int
        Reference trial count for EEMD and CEEMDAN.
    epsilon_ceemdan : float
        CEEMDAN ``epsilon`` parameter.
    semilla_ceemdan : int
        CEEMDAN noise seed.
    semilla_eemd : int
        EEMD noise seed (PyEMD).
    eemd_noise_width : float
        EEMD noise width.
    eemd_sd_thresh : float
        EEMD ``SD_thresh``.
    eemd_s_number : int
        EEMD ``S_number``.
    eemd_fixe_h : int
        EEMD ``FIXE_H``.
    vmd_k : int
        Number of VMD modes (``K``).
    vmd_alpha : float
        VMD bandwidth penalty.
    vmd_dc : int
        VMD DC mode (0 or 1).
    vmd_tau : float
        VMD noise tolerance.
    vmd_tol : float
        VMD convergence tolerance.
    vmd_init : int
        VMD central-frequency initialization.
    trials_eemd : int, optional
        EEMD trials. If ``None``, ``trials_ensemble`` is used.
    trials_ceemdan : int, optional
        CEEMDAN trials. If ``None``, ``trials_ensemble`` is used.

    Returns
    -------
    pd.DataFrame
        One row per (scenario, method) with metrics.
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
        logger.info("Signal saved: %s", ruta_senal)

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

    # Comparative EMD vs CEEMDAN summary per scenario (for writing)
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
        "Metrics saved to %s", dir_salida / "metricas_emdsynth_metodos.csv"
    )
    return df_metricas


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
        Namespace with ``salida``, ``max_imf``, ``trials``, etc.
    """
    p = argparse.ArgumentParser(
        description=(
            "EMD / EEMD / CEEMDAN / VMD decompositions on synthetic signals."
        )
    )
    p.add_argument(
        "--salida",
        type=Path,
        default=_REPO_ROOT / "scripts" / "emdsynth" / "out",
        help="Output directory for parquet, CSV, and manifest.",
    )
    p.add_argument(
        "--max-imf", type=int, default=12, help="Maximum number of IMFs (+ residue)."
    )
    p.add_argument(
        "--trials",
        type=int,
        default=120,
        help="Reference trial count (``trials_ensemble`` field in the manifest).",
    )
    p.add_argument(
        "--trials-eemd",
        type=int,
        default=100,
        help="EEMD trials (default after multicomponent-signal calibration).",
    )
    p.add_argument(
        "--trials-ceemdan",
        type=int,
        default=180,
        help="CEEMDAN trials (calibrated on multicomponent signal).",
    )
    p.add_argument(
        "--epsilon-ceemdan",
        type=float,
        default=0.03,
        help="Relative epsilon (CEEMDAN); default calibrated on multicomponent signal.",
    )
    p.add_argument("--semilla-ceemdan", type=int, default=42)
    p.add_argument("--semilla-eemd", type=int, default=42)
    p.add_argument(
        "--eemd-noise-width",
        type=float,
        default=0.095,
        help="EEMD noise width (calibrated on multicomponent signal).",
    )
    p.add_argument(
        "--eemd-sd-thresh",
        type=float,
        default=0.18,
        help="EEMD SD_thresh (calibrated on multicomponent signal).",
    )
    p.add_argument("--eemd-s-number", type=int, default=8)
    p.add_argument("--eemd-fixe-h", type=int, default=5)
    p.add_argument(
        "--vmd-k",
        type=int,
        default=6,
        help="Number of VMD modes (K); default calibrated on multicomponent signal.",
    )
    p.add_argument(
        "--vmd-alpha",
        type=float,
        default=5000.0,
        help="VMD bandwidth penalty (calibrated on multicomponent signal).",
    )
    p.add_argument(
        "--vmd-dc",
        type=int,
        choices=[0, 1],
        default=1,
        help="VMD DC mode (1 allows capturing a slow trend).",
    )
    p.add_argument("--vmd-tau", type=float, default=0.0)
    p.add_argument("--vmd-tol", type=float, default=1e-7)
    p.add_argument("--vmd-init", type=int, default=1)
    p.add_argument(
        "--calibrar-multicomponente",
        action="store_true",
        help=(
            "Run grid search only on the multicomponent signal and write "
            "parametros_calibrados_multicomponente.json under --salida."
        ),
    )
    return p.parse_args(argv)


def _metricas_a_json_serializable(obj: Any) -> Any:
    """
    Convert metrics with NumPy types to ``json.dumps``-compatible types.

    Parameters
    ----------
    obj : Any
        Nested value (dict, list, scalar).

    Returns
    -------
    Any
        JSON-serializable equivalent structure.
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
    Entry point: configure logging and run the pipeline.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments.
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
        logger.info("Calibration saved to %s", ruta_cal)
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
