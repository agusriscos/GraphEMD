"""
Utilities for generating synthetic signals for EMD analysis.

This module contains functions to generate different types of synthetic signals
designed to evaluate the behavior of EMD, EEMD, and CEEMDAN.
"""

from typing import Tuple, Optional

import numpy as np


def generate_chirp_signal(
    f0: float,
    k: float,
    amplitud: float = 1.0,
    fase: float = 0.0,
    funcion: str = "sin",
    duracion: float = 10.0,
    frecuencia_muestreo: float = 1000.0,
    t: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a chirp-type synthetic signal (time-varying frequency).

    This signal is designed to evaluate the behavior of EMD, EEMD, and CEEMDAN
    on non-stationary signals where frequency changes over time. A chirp signal
    is ideal for testing whether EMD can correctly track a varying frequency
    or incorrectly decomposes it into multiple IMFs (over-decomposition).

    Parameters
    ----------
    f0 : float
        Initial signal frequency (Hz).
    k : float
        Frequency sweep rate (Hz/s). Indicates how quickly frequency changes
        over time. Positive values produce an ascending chirp (frequency increases),
        negative values produce a descending chirp (frequency decreases).
    amplitud : float, optional
        Signal amplitude (default: 1.0).
    fase : float, optional
        Initial signal phase in radians (default: 0.0).
    funcion : str, optional
        Trigonometric function to use: "sin" or "cos" (default: "sin").
    duracion : float, optional
        Total signal duration in seconds (default: 10.0).
    frecuencia_muestreo : float, optional
        Sampling frequency in Hz (default: 1000.0).
    t : np.ndarray, optional
        Custom time array. If provided, it is used instead of generating one
        based on duracion and frecuencia_muestreo.

    Returns
    -------
    t : np.ndarray
        Signal time array.
    x : np.ndarray
        Generated chirp-type synthetic signal.

    Examples
    --------
    >>> t, señal = generate_chirp_signal(
    ...     f0=1.0,
    ...     k=0.5,
    ...     duracion=10.0,
    ...     frecuencia_muestreo=1000.0
    ... )
    >>> plt.plot(t, señal)
    >>> plt.xlabel('Time (s)')
    >>> plt.ylabel('Amplitude')
    >>> plt.title('Chirp Signal')
    >>> plt.show()

    Notes
    -----
    The generated signal follows the formula:
    x(t) = A·func(2π·(f₀ + k·t)·t + φ)

    Where:
    - A is the amplitude
    - f₀ is the initial frequency
    - k is the frequency sweep rate
    - φ is the initial phase
    - func can be sin or cos depending on the 'funcion' parameter

    The instantaneous frequency of the signal at time t is:
    f(t) = f₀ + k·t

    Therefore:
    - If k > 0: frequency increases over time (ascending chirp)
    - If k < 0: frequency decreases over time (descending chirp)
    - If k = 0: the signal reduces to a sinusoid with constant frequency f₀

    This signal is particularly useful for evaluating EMD because:

    1. EMD is designed to handle non-stationary signals, so it should ideally
       produce a single IMF that tracks the changing frequency.

    2. If EMD decomposes the chirp signal into multiple IMFs, this indicates
       an over-decomposition problem, which is an algorithm error.

    3. Chirp signals are common in financial applications where volatility
       (equivalent to frequency) changes over time.

    The final frequency of the signal will be:
    f_final = f₀ + k·duracion

    It is important to ensure that the sampling frequency is at least twice
    the maximum expected frequency (Nyquist frequency):
    frecuencia_muestreo >= 2·max(|f₀|, |f₀ + k·duracion|)

    References
    ----------
    Huang, N. E., et al. (1998). The empirical mode decomposition and the
    Hilbert spectrum for nonlinear and non-stationary time series analysis.
    Proceedings of the Royal Society of London. Series A: Mathematical,
    Physical and Engineering Sciences, 454(1971), 903-995.
    """
    # Generate time array if not provided
    if t is None:
        num_muestras = int(duracion * frecuencia_muestreo)
        t = np.linspace(0, duracion, num_muestras)
    else:
        t = np.asarray(t)
        duracion = t[-1] - t[0]

    # Validate parameters
    if f0 < 0:
        raise ValueError("Initial frequency f0 must be non-negative")
    if amplitud < 0:
        raise ValueError("Amplitude must be non-negative")
    if funcion not in ["cos", "sin"]:
        raise ValueError("Function must be 'cos' or 'sin'")

    # Compute instantaneous frequency: f(t) = f0 + k*t
    frecuencia_instantanea = f0 + k * t

    # Verify that instantaneous frequency is not negative
    if np.any(frecuencia_instantanea < 0):
        raise ValueError(
            "Instantaneous frequency cannot be negative. "
            "Ensure that f0 + k*t >= 0 for all t in the range."
        )

    # Check Nyquist frequency
    frecuencia_maxima = np.max(frecuencia_instantanea)
    if frecuencia_muestreo < 2 * frecuencia_maxima:
        import warnings

        warnings.warn(
            f"Sampling frequency ({frecuencia_muestreo} Hz) may be "
            f"insufficient. At least {2 * frecuencia_maxima} Hz "
            f"is recommended to avoid aliasing (maximum frequency: {frecuencia_maxima} Hz).",
            UserWarning,
        )

    # Select trigonometric function
    if funcion == "cos":
        trig_func = np.cos
    else:
        trig_func = np.sin

    # Generate chirp signal: x(t) = A·func(2π·(f₀ + k·t)·t + φ)
    # Equivalent to: x(t) = A·func(2π·f₀·t + 2π·k·t² + φ)
    fase_instantanea = 2 * np.pi * frecuencia_instantanea * t + fase
    x = amplitud * trig_func(fase_instantanea)

    return t, x


def generate_close_frequency_signal(
    f1: float,
    f2: float,
    amplitud1: float = 1.0,
    amplitud2: float = 1.0,
    fase1: float = 0.0,
    fase2: float = 0.0,
    funcion: str = "cos",
    duracion: float = 10.0,
    frecuencia_muestreo: float = 1000.0,
    t: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic signal with two very close frequencies.

    This signal is designed to evaluate the ability of EMD, EEMD, and CEEMDAN
    to separate two components when their frequencies are close. When the ratio
    f1/f2 is near 1, the algorithms may struggle to produce two clean IMFs and
    may instead produce a single IMF with beating (amplitude modulation).

    Parameters
    ----------
    f1 : float
        Frequency of the first component (Hz).
    f2 : float
        Frequency of the second component (Hz).
    amplitud1 : float, optional
        Amplitude of the first component (default: 1.0).
    amplitud2 : float, optional
        Amplitude of the second component (default: 1.0).
    fase1 : float, optional
        Initial phase of the first component in radians (default: 0.0).
    fase2 : float, optional
        Initial phase of the second component in radians (default: 0.0).
    funcion : str, optional
        Trigonometric function to use: "cos" or "sin" (default: "cos").
    duracion : float, optional
        Total signal duration in seconds (default: 10.0).
    frecuencia_muestreo : float, optional
        Sampling frequency in Hz (default: 1000.0).
    t : np.ndarray, optional
        Custom time array. If provided, it is used instead of generating one
        based on duracion and frecuencia_muestreo.

    Returns
    -------
    t : np.ndarray
        Signal time array.
    x : np.ndarray
        Generated synthetic signal.

    Examples
    --------
    >>> t, señal = generate_close_frequency_signal(
    ...     f1=10.0,
    ...     f2=12.0,
    ...     duracion=5.0,
    ...     frecuencia_muestreo=1000.0
    ... )
    >>> plt.plot(t, señal)
    >>> plt.xlabel('Time (s)')
    >>> plt.ylabel('Amplitude')
    >>> plt.title('Close-Frequency Signal')
    >>> plt.show()

    Notes
    -----
    The generated signal follows the formula:
    x(t) = A₁·func(2π·f₁·t + φ₁) + A₂·func(2π·f₂·t + φ₂)

    Where func can be cos or sin depending on the 'funcion' parameter.

    When f₁ and f₂ are very close (ratio f₁/f₂ ≈ 1), the resulting signal
    shows a beating pattern, where the effective amplitude varies periodically.
    This is an important test case for EMD because:

    1. If the algorithm works correctly, it should produce two separate IMFs,
       one for each frequency.
    2. If the algorithm struggles, it may produce a single IMF that captures
       the beating, which is incorrect.

    The Rilling-Flandrin bound states that EMD can separate two components if
    the frequency ratio is greater than approximately 2. For smaller ratios,
    especially near 1, problems are expected.

    References
    ----------
    Rilling, G., & Flandrin, P. (2008). One or two frequencies? The
    empirical mode decomposition answers. IEEE Transactions on Signal
    Processing, 56(1), 85-95.
    """
    # Generate time array if not provided
    if t is None:
        num_muestras = int(duracion * frecuencia_muestreo)
        t = np.linspace(0, duracion, num_muestras)
    else:
        t = np.asarray(t)
        duracion = t[-1] - t[0]

    # Validate parameters
    if f1 <= 0 or f2 <= 0:
        raise ValueError("Las frecuencias deben ser positivas")
    if amplitud1 < 0 or amplitud2 < 0:
        raise ValueError("Amplitudes must be non-negative")
    if funcion not in ["cos", "sin"]:
        raise ValueError("Function must be 'cos' or 'sin'")

    # Select trigonometric function
    if funcion == "cos":
        trig_func = np.cos
    else:
        trig_func = np.sin

    # Generate first component
    componente1 = amplitud1 * trig_func(2 * np.pi * f1 * t + fase1)

    # Generate second component
    componente2 = amplitud2 * trig_func(2 * np.pi * f2 * t + fase2)

    # Full signal
    x = componente1 + componente2

    return t, x


def generate_mode_mixing_signal(
    f_low: float,
    f_high: float,
    alpha: float,
    t1: float,
    t2: float,
    duracion: float = 10.0,
    frecuencia_muestreo: float = 1000.0,
    t: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic signal designed to detect mode mixing in EMD.

    The signal consists of a continuous low-frequency component and a
    high-frequency component that exists only in the interval [t1, t2].
    This setup is ideal for testing how EMD, EEMD, and CEEMDAN handle
    intermittent signals that cause mode mixing.

    Parameters
    ----------
    f_low : float
        Low-frequency signal frequency (Hz).
    f_high : float
        High-frequency signal frequency (Hz).
    alpha : float
        Amplitude of the high-frequency component.
    t1 : float
        Start time when the high-frequency signal appears.
    t2 : float
        End time when the high-frequency signal disappears.
    duracion : float, optional
        Total signal duration in seconds (default: 10.0).
    frecuencia_muestreo : float, optional
        Sampling frequency in Hz (default: 1000.0).
    t : np.ndarray, optional
        Custom time array. If provided, it is used instead of generating one
        based on duracion and frecuencia_muestreo.

    Returns
    -------
    t : np.ndarray
        Signal time array.
    x : np.ndarray
        Generated synthetic signal.

    Examples
    --------
    >>> t, señal = generate_mode_mixing_signal(
    ...     f_low=1.0,
    ...     f_high=10.0,
    ...     alpha=0.5,
    ...     t1=3.0,
    ...     t2=7.0,
    ...     duracion=10.0,
    ...     frecuencia_muestreo=1000.0
    ... )
    >>> plt.plot(t, señal)
    >>> plt.xlabel('Time (s)')
    >>> plt.ylabel('Amplitude')
    >>> plt.title('Mode Mixing Signal')
    >>> plt.show()

    Notes
    -----
    The generated signal follows the formula:
    x(t) = sin(2π·f_low·t) + α·sin(2π·f_high·t)·I_{[t1, t2]}(t)

    Where I_{[t1, t2]}(t) is an indicator function that equals 1 only in the
    interval [t1, t2].

    This signal is especially useful for detecting mode mixing because classical
    EMD often fails to separate the intermittent fast signal from the continuous
    slow signal, mixing both into the same IMF.
    """
    # Generate time array if not provided
    if t is None:
        num_muestras = int(duracion * frecuencia_muestreo)
        t = np.linspace(0, duracion, num_muestras)
    else:
        t = np.asarray(t)
        duracion = t[-1] - t[0]

    # Validate parameters
    if t1 >= t2:
        raise ValueError("t1 must be less than t2")
    if t1 < t[0] or t2 > t[-1]:
        raise ValueError("Interval [t1, t2] must lie within the range of t")
    if f_low <= 0 or f_high <= 0:
        raise ValueError("Las frecuencias deben ser positivas")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    # Low-frequency component (always present)
    componente_baja = np.sin(2 * np.pi * f_low * t)

    # Indicator function for the interval [t1, t2]
    indicador = (t >= t1) & (t <= t2)

    # High-frequency component (only in [t1, t2])
    componente_alta = alpha * np.sin(2 * np.pi * f_high * t) * indicador

    # Full signal
    x = componente_baja + componente_alta

    return t, x
