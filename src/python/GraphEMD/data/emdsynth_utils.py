"""
Utilidades para generar señales sintéticas para análisis EMD.

Este módulo contiene funciones para generar diferentes tipos de señales sintéticas
diseñadas para evaluar el comportamiento de EMD, EEMD y CEEMDAN.
"""

from typing import Tuple, Optional

import numpy as np


def generar_senal_chirp(
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
    Genera una señal sintética tipo chirp (frecuencia variable en el tiempo).

    Esta señal está diseñada para evaluar el comportamiento de EMD, EEMD y CEEMDAN
    ante señales no estacionarias donde la frecuencia cambia con el tiempo. Una señal
    chirp es ideal para probar si EMD puede rastrear correctamente una frecuencia
    variable o si la descompone incorrectamente en múltiples IMFs (sobre-descomposición).

    Parameters
    ----------
    f0 : float
        Frecuencia inicial de la señal (Hz).
    k : float
        Tasa de barrido de frecuencia (Hz/s). Indica qué tan rápido cambia la
        frecuencia con el tiempo. Valores positivos generan un chirp ascendente
        (frecuencia aumenta), valores negativos generan un chirp descendente
        (frecuencia disminuye).
    amplitud : float, optional
        Amplitud de la señal (por defecto: 1.0).
    fase : float, optional
        Fase inicial de la señal en radianes (por defecto: 0.0).
    funcion : str, optional
        Función trigonométrica a usar: "sin" o "cos" (por defecto: "sin").
    duracion : float, optional
        Duración total de la señal en segundos (por defecto: 10.0).
    frecuencia_muestreo : float, optional
        Frecuencia de muestreo en Hz (por defecto: 1000.0).
    t : np.ndarray, optional
        Array de tiempos personalizado. Si se proporciona, se usa en lugar
        de generar uno basado en duracion y frecuencia_muestreo.

    Returns
    -------
    t : np.ndarray
        Array de tiempos de la señal.
    x : np.ndarray
        Señal sintética tipo chirp generada.

    Examples
    --------
    >>> t, señal = generar_senal_chirp(
    ...     f0=1.0,
    ...     k=0.5,
    ...     duracion=10.0,
    ...     frecuencia_muestreo=1000.0
    ... )
    >>> plt.plot(t, señal)
    >>> plt.xlabel('Tiempo (s)')
    >>> plt.ylabel('Amplitud')
    >>> plt.title('Señal Chirp')
    >>> plt.show()

    Notes
    -----
    La señal generada sigue la fórmula:
    x(t) = A·func(2π·(f₀ + k·t)·t + φ)

    Donde:
    - A es la amplitud
    - f₀ es la frecuencia inicial
    - k es la tasa de barrido de frecuencia
    - φ es la fase inicial
    - func puede ser sin o cos según el parámetro 'funcion'

    La frecuencia instantánea de la señal en el tiempo t es:
    f(t) = f₀ + k·t

    Por lo tanto:
    - Si k > 0: la frecuencia aumenta con el tiempo (chirp ascendente)
    - Si k < 0: la frecuencia disminuye con el tiempo (chirp descendente)
    - Si k = 0: la señal se reduce a una sinusoide de frecuencia constante f₀

    Esta señal es particularmente útil para evaluar EMD porque:

    1. EMD está diseñado para manejar señales no estacionarias, por lo que
       idealmente debería generar un solo IMF que rastrea la frecuencia
       cambiante.

    2. Si EMD descompone la señal chirp en múltiples IMFs, esto indica un
       problema de sobre-descomposición, lo cual es un error del algoritmo.

    3. Las señales chirp son comunes en aplicaciones financieras donde la
       volatilidad (equivalente a frecuencia) cambia con el tiempo.

    La frecuencia final de la señal será:
    f_final = f₀ + k·duracion

    Es importante asegurarse de que la frecuencia de muestreo sea al menos
    el doble de la frecuencia máxima esperada (frecuencia de Nyquist):
    frecuencia_muestreo >= 2·max(|f₀|, |f₀ + k·duracion|)

    References
    ----------
    Huang, N. E., et al. (1998). The empirical mode decomposition and the
    Hilbert spectrum for nonlinear and non-stationary time series analysis.
    Proceedings of the Royal Society of London. Series A: Mathematical,
    Physical and Engineering Sciences, 454(1971), 903-995.
    """
    # Generar array de tiempos si no se proporciona
    if t is None:
        num_muestras = int(duracion * frecuencia_muestreo)
        t = np.linspace(0, duracion, num_muestras)
    else:
        t = np.asarray(t)
        duracion = t[-1] - t[0]

    # Validar parámetros
    if f0 < 0:
        raise ValueError("La frecuencia inicial f0 debe ser no negativa")
    if amplitud < 0:
        raise ValueError("La amplitud debe ser no negativa")
    if funcion not in ["cos", "sin"]:
        raise ValueError("La función debe ser 'cos' o 'sin'")

    # Calcular frecuencia instantánea: f(t) = f0 + k*t
    frecuencia_instantanea = f0 + k * t

    # Verificar que la frecuencia instantánea no sea negativa
    if np.any(frecuencia_instantanea < 0):
        raise ValueError(
            "La frecuencia instantánea no puede ser negativa. "
            "Asegúrese de que f0 + k*t >= 0 para todo t en el rango."
        )

    # Verificar frecuencia de Nyquist
    frecuencia_maxima = np.max(frecuencia_instantanea)
    if frecuencia_muestreo < 2 * frecuencia_maxima:
        import warnings

        warnings.warn(
            f"La frecuencia de muestreo ({frecuencia_muestreo} Hz) puede ser "
            f"insuficiente. Se recomienda al menos {2 * frecuencia_maxima} Hz "
            f"para evitar aliasing (frecuencia máxima: {frecuencia_maxima} Hz).",
            UserWarning,
        )

    # Seleccionar función trigonométrica
    if funcion == "cos":
        trig_func = np.cos
    else:
        trig_func = np.sin

    # Generar señal chirp: x(t) = A·func(2π·(f₀ + k·t)·t + φ)
    # Esto es equivalente a: x(t) = A·func(2π·f₀·t + 2π·k·t² + φ)
    fase_instantanea = 2 * np.pi * frecuencia_instantanea * t + fase
    x = amplitud * trig_func(fase_instantanea)

    return t, x


def generar_senal_frecuencia_cercana(
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
    Genera una señal sintética con dos frecuencias muy cercanas.

    Esta señal está diseñada para evaluar la capacidad de EMD, EEMD y CEEMDAN
    para separar dos componentes cuando sus frecuencias están próximas. Cuando
    la razón f1/f2 es cercana a 1, los algoritmos pueden tener dificultades
    para generar dos IMFs limpios y pueden producir un solo IMF con batimiento
    (modulación de amplitud).

    Parameters
    ----------
    f1 : float
        Frecuencia de la primera componente (Hz).
    f2 : float
        Frecuencia de la segunda componente (Hz).
    amplitud1 : float, optional
        Amplitud de la primera componente (por defecto: 1.0).
    amplitud2 : float, optional
        Amplitud de la segunda componente (por defecto: 1.0).
    fase1 : float, optional
        Fase inicial de la primera componente en radianes (por defecto: 0.0).
    fase2 : float, optional
        Fase inicial de la segunda componente en radianes (por defecto: 0.0).
    funcion : str, optional
        Función trigonométrica a usar: "cos" o "sin" (por defecto: "cos").
    duracion : float, optional
        Duración total de la señal en segundos (por defecto: 10.0).
    frecuencia_muestreo : float, optional
        Frecuencia de muestreo en Hz (por defecto: 1000.0).
    t : np.ndarray, optional
        Array de tiempos personalizado. Si se proporciona, se usa en lugar
        de generar uno basado en duracion y frecuencia_muestreo.

    Returns
    -------
    t : np.ndarray
        Array de tiempos de la señal.
    x : np.ndarray
        Señal sintética generada.

    Examples
    --------
    >>> t, señal = generar_senal_frecuencia_cercana(
    ...     f1=10.0,
    ...     f2=12.0,
    ...     duracion=5.0,
    ...     frecuencia_muestreo=1000.0
    ... )
    >>> plt.plot(t, señal)
    >>> plt.xlabel('Tiempo (s)')
    >>> plt.ylabel('Amplitud')
    >>> plt.title('Señal con Frecuencias Cercanas')
    >>> plt.show()

    Notes
    -----
    La señal generada sigue la fórmula:
    x(t) = A₁·func(2π·f₁·t + φ₁) + A₂·func(2π·f₂·t + φ₂)

    Donde func puede ser cos o sin según el parámetro 'funcion'.

    Cuando f₁ y f₂ están muy cerca (razón f₁/f₂ ≈ 1), la señal resultante
    muestra un patrón de batimiento, donde la amplitud efectiva varía
    periódicamente. Este es un caso de prueba importante para EMD porque:

    1. Si el algoritmo funciona correctamente, debería generar dos IMFs
       separados, uno para cada frecuencia.
    2. Si el algoritmo tiene dificultades, puede generar un solo IMF que
       captura el batimiento, lo cual es incorrecto.

    El límite de Rilling-Flandrin establece que EMD puede separar dos
    componentes si la razón de frecuencias es mayor que aproximadamente 2.
    Para razones menores, especialmente cercanas a 1, se esperan problemas.

    References
    ----------
    Rilling, G., & Flandrin, P. (2008). One or two frequencies? The
    empirical mode decomposition answers. IEEE Transactions on Signal
    Processing, 56(1), 85-95.
    """
    # Generar array de tiempos si no se proporciona
    if t is None:
        num_muestras = int(duracion * frecuencia_muestreo)
        t = np.linspace(0, duracion, num_muestras)
    else:
        t = np.asarray(t)
        duracion = t[-1] - t[0]

    # Validar parámetros
    if f1 <= 0 or f2 <= 0:
        raise ValueError("Las frecuencias deben ser positivas")
    if amplitud1 < 0 or amplitud2 < 0:
        raise ValueError("Las amplitudes deben ser no negativas")
    if funcion not in ["cos", "sin"]:
        raise ValueError("La función debe ser 'cos' o 'sin'")

    # Seleccionar función trigonométrica
    if funcion == "cos":
        trig_func = np.cos
    else:
        trig_func = np.sin

    # Generar primera componente
    componente1 = amplitud1 * trig_func(2 * np.pi * f1 * t + fase1)

    # Generar segunda componente
    componente2 = amplitud2 * trig_func(2 * np.pi * f2 * t + fase2)

    # Señal completa
    x = componente1 + componente2

    return t, x


def generar_senal_mode_mixing(
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
    Genera una señal sintética diseñada para detectar mode mixing en EMD.

    La señal consiste en una componente de baja frecuencia continua y una
    componente de alta frecuencia que solo existe en el intervalo [t1, t2].
    Esta configuración es ideal para probar cómo EMD, EEMD y CEEMDAN manejan
    señales intermitentes que causan mode mixing.

    Parameters
    ----------
    f_low : float
        Frecuencia de la señal de baja frecuencia (Hz).
    f_high : float
        Frecuencia de la señal de alta frecuencia (Hz).
    alpha : float
        Amplitud de la componente de alta frecuencia.
    t1 : float
        Tiempo inicial donde aparece la señal de alta frecuencia.
    t2 : float
        Tiempo final donde desaparece la señal de alta frecuencia.
    duracion : float, optional
        Duración total de la señal en segundos (por defecto: 10.0).
    frecuencia_muestreo : float, optional
        Frecuencia de muestreo en Hz (por defecto: 1000.0).
    t : np.ndarray, optional
        Array de tiempos personalizado. Si se proporciona, se usa en lugar
        de generar uno basado en duracion y frecuencia_muestreo.

    Returns
    -------
    t : np.ndarray
        Array de tiempos de la señal.
    x : np.ndarray
        Señal sintética generada.

    Examples
    --------
    >>> t, señal = generar_senal_mode_mixing(
    ...     f_low=1.0,
    ...     f_high=10.0,
    ...     alpha=0.5,
    ...     t1=3.0,
    ...     t2=7.0,
    ...     duracion=10.0,
    ...     frecuencia_muestreo=1000.0
    ... )
    >>> plt.plot(t, señal)
    >>> plt.xlabel('Tiempo (s)')
    >>> plt.ylabel('Amplitud')
    >>> plt.title('Señal Mode Mixing')
    >>> plt.show()

    Notes
    -----
    La señal generada sigue la fórmula:
    x(t) = sin(2π·f_low·t) + α·sin(2π·f_high·t)·I_{[t1, t2]}(t)

    Donde I_{[t1, t2]}(t) es una función indicador que vale 1 solo en el
    intervalo [t1, t2].

    Esta señal es especialmente útil para detectar mode mixing porque el EMD
    clásico suele fallar al separar la señal rápida intermitente de la señal
    lenta continua, mezclando ambas en el mismo IMF.
    """
    # Generar array de tiempos si no se proporciona
    if t is None:
        num_muestras = int(duracion * frecuencia_muestreo)
        t = np.linspace(0, duracion, num_muestras)
    else:
        t = np.asarray(t)
        duracion = t[-1] - t[0]

    # Validar parámetros
    if t1 >= t2:
        raise ValueError("t1 debe ser menor que t2")
    if t1 < t[0] or t2 > t[-1]:
        raise ValueError("El intervalo [t1, t2] debe estar dentro del rango de t")
    if f_low <= 0 or f_high <= 0:
        raise ValueError("Las frecuencias deben ser positivas")
    if alpha < 0:
        raise ValueError("alpha debe ser no negativo")

    # Componente de baja frecuencia (siempre presente)
    componente_baja = np.sin(2 * np.pi * f_low * t)

    # Función indicador para el intervalo [t1, t2]
    indicador = (t >= t1) & (t <= t2)

    # Componente de alta frecuencia (solo en [t1, t2])
    componente_alta = alpha * np.sin(2 * np.pi * f_high * t) * indicador

    # Señal completa
    x = componente_baja + componente_alta

    return t, x
