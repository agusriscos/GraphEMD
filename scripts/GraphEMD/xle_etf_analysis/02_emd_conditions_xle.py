"""
Script para analizar las condiciones de la ETF XLE (Energy Select Sector SPDR Fund) para su analisis con EMD (Empirical Mode Decomposition).

1. Carga los datos de la ETF XLE desde el archivo parquet data/GraphEMD/xle_etf_analysis/xle.parquet.
2. Evalua la estacionaridad de la serie aplicando la prueba de Dickey-Fuller aumentada (ADF) y la prueba de Kwiatkowski-Phillips-Schmidt-Shin (KPSS) al igual que se ha realizado para el MSCI World en este mismo repositorio
3. Evalua la no linealidad de la serie aplicando la prueba de Brock-Dechert-Scheinkman (BDS) sobre los residuos de un modelo AR(1).
4. Evalua la volatilidad en amplitud y frecuencia aplicando ARCH/GARCH y la prueba de Ljung-Box sobre el cuadrado de los rendimientos del indice.
5. Documenta en este mismo script la logica implementada y los resultados obtenidos.
6. No cambies por el momento ningun codigo que ya exista

Lógica implementada
-------------------
Réplica de ``analysis/16dic25/02_analisis_emd_condiciones.ipynb`` (MSCI World):

- **Estacionaridad**: ADF (``autolag='AIC'``) y KPSS (``regression='ct'``, ``nlags='auto'``) sobre
  ``Close`` y sobre retornos diarios en % (``pct_change×100``).
- **No linealidad**: BDS sobre residuos de AR(1) (``statsmodels.tsa.ar_model.AutoReg``,
  ``lags=1``) en dimensiones de embedding 2–5; autocorrelación de cuadrados; Ljung-Box
  sobre cuadrados (10 lags); asimetría, curtosis y Jarque-Bera.
- **Volatilidad (amplitud/frecuencia)**: ajuste ARCH(1) y GARCH(1,1) con ``arch`` sobre
  retornos en %; Ljung-Box sobre el cuadrado de los rendimientos; variabilidad en amplitud
  (volatilidad móvil 252 días) y en frecuencia (cruces por cero en ventanas móviles), como
  en el notebook de referencia.

Resultados obtenidos (ejecución 2026-05-16, n=3587)
-------------------------------------------------

**Estacionaridad — Close:** ADF estadístico=-0.0783, p=0.9516 (no estacionaria);
KPSS estadístico=1.6104, p=0.0100 (no estacionaria).

**Estacionaridad — Returns (%):** ADF=-11.6799, p<0.0001 (estacionaria);
KPSS=0.0458, p=0.1000 (estacionaria).

**BDS (residuos AR(1) sobre Close):** dim 3: 17.28 (p<0.0001); dim 4: 20.75;
dim 5: 23.42 — no linealidad en todas las dimensiones reportadas.

**Ljung-Box (rendimientos²):** estadístico=2267.29, p<0.0001 — efectos ARCH/GARCH.

**ARCH(1):** AIC=13446.82, BIC=13459.19. **GARCH(1,1):** AIC=12612.98,
BIC=12631.53 (mejor ajuste que ARCH).

**Variabilidad:** amplitud (vol. móvil 252d) CV=0.4619 — alta; frecuencia
(cruces/ventana media=12.67) — alta.

**Conclusión EMD:** precio no estacionario, no linealidad y agrupación de
volatilidad — condiciones favorables para EMD (análogo a MSCI World).
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf, adfuller, bds, kpss

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from arch import arch_model

    ARCH_DISPONIBLE = True
except ImportError:
    arch_model = None  # type: ignore[misc, assignment]
    ARCH_DISPONIBLE = False

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_RUTA_PARQUET_XLE = _REPO_ROOT / "data" / "GraphEMD" / "xle_etf_analysis" / "xle.parquet"
_VENTANA_VOLATILIDAD_ANUAL: int = 252
_LAGS_LJUNG_BOX: int = 10
_LAGS_ACF: int = 20
_DIMENSIONES_BDS: list[int] = [2, 3, 4, 5]
_NIVEL_SIGNIFICANCIA: float = 0.05

logger = logging.getLogger(__name__)


def cargar_datos_xle(ruta_parquet: Path = _RUTA_PARQUET_XLE) -> pd.DataFrame:
    """
    Carga la serie XLE y calcula retornos diarios en porcentaje.

    Parameters
    ----------
    ruta_parquet : Path
        Ruta al archivo ``xle.parquet``.

    Returns
    -------
    pd.DataFrame
        Datos OHLCV con columna ``Returns`` (%).

    Raises
    ------
    FileNotFoundError
        Si no existe el parquet.
    ValueError
        Si falta la columna ``Close``.
    """
    if not ruta_parquet.is_file():
        raise FileNotFoundError(f"No se encontró: {ruta_parquet}")
    df = pd.read_parquet(ruta_parquet, engine="pyarrow")
    if "Close" not in df.columns:
        raise ValueError(f"Falta columna Close. Columnas: {list(df.columns)}")
    df = df.copy()
    df["Returns"] = df["Close"].pct_change() * 100.0
    logger.info(
        "Datos cargados: %s filas, %s → %s",
        len(df),
        df.index.min(),
        df.index.max(),
    )
    return df


def realizar_tests_estacionaridad(serie: pd.Series) -> dict[str, Any]:
    """
    Aplica ADF y KPSS a una serie temporal (misma lógica que MSCI World).

    Parameters
    ----------
    serie : pd.Series
        Serie a analizar.

    Returns
    -------
    dict
        Resultados de ADF y KPSS con estadísticos, p-valores e interpretación.
    """
    resultados: dict[str, Any] = {}
    serie_limpia = serie.dropna()

    try:
        resultado_adf = adfuller(serie_limpia, autolag="AIC")
        valores_criticos = resultado_adf[4] if len(resultado_adf) > 4 else {}
        p_valor = float(resultado_adf[1])
        resultados["ADF"] = {
            "estadistico": float(resultado_adf[0]),
            "p_valor": p_valor,
            "valores_criticos": valores_criticos,
            "es_estacionaria": p_valor < _NIVEL_SIGNIFICANCIA,
            "interpretacion": (
                "Estacionaria" if p_valor < _NIVEL_SIGNIFICANCIA else "No estacionaria"
            ),
        }
    except Exception as exc:
        resultados["ADF"] = {"error": str(exc)}

    try:
        resultado_kpss = kpss(serie_limpia, regression="ct", nlags="auto")
        p_valor = float(resultado_kpss[1])
        resultados["KPSS"] = {
            "estadistico": float(resultado_kpss[0]),
            "p_valor": p_valor,
            "valores_criticos": resultado_kpss[3],
            "es_estacionaria": p_valor > _NIVEL_SIGNIFICANCIA,
            "interpretacion": (
                "Estacionaria" if p_valor > _NIVEL_SIGNIFICANCIA else "No estacionaria"
            ),
        }
    except Exception as exc:
        resultados["KPSS"] = {"error": str(exc)}

    return resultados


def _ajustar_ar1_residuos(serie: pd.Series) -> np.ndarray:
    """
    Ajusta AR(1) y devuelve los residuos para el test BDS.

    Parameters
    ----------
    serie : pd.Series
        Serie de precios o retornos.

    Returns
    -------
    np.ndarray
        Residuos del modelo AR(1).
    """
    serie_limpia = serie.dropna()
    modelo_ar = AutoReg(serie_limpia, lags=1).fit()
    return np.asarray(modelo_ar.resid, dtype=np.float64)


def realizar_test_bds(residuos: np.ndarray) -> dict[str, Any]:
    """
    Ejecuta BDS sobre residuos AR(1) en varias dimensiones de embedding.

    Parameters
    ----------
    residuos : np.ndarray
        Residuos del ajuste AR(1).

    Returns
    -------
    dict
        Estadísticos y p-valores por dimensión, o clave ``error``.
    """
    resultados_bds: dict[str, Any] = {}
    for dim in _DIMENSIONES_BDS:
        try:
            estadistico, p_valor = bds(residuos, max_dim=dim)
            stat = (
                float(estadistico[-1])
                if isinstance(estadistico, (list, np.ndarray))
                else float(estadistico)
            )
            pval = (
                float(p_valor[-1])
                if isinstance(p_valor, (list, np.ndarray))
                else float(p_valor)
            )
            resultados_bds[dim] = {
                "estadistico": stat,
                "p_valor": pval,
                "es_no_lineal": pval < _NIVEL_SIGNIFICANCIA,
            }
        except Exception as exc:
            resultados_bds[dim] = {"error": str(exc)}
    return resultados_bds


def analizar_no_linealidad(serie: pd.Series) -> dict[str, Any]:
    """
    Evalúa no linealidad: BDS (residuos AR(1)), ACF de cuadrados y Ljung-Box.

    Parameters
    ----------
    serie : pd.Series
        Serie temporal (precio o retornos).

    Returns
    -------
    dict
        Resultados agregados del análisis de no linealidad.
    """
    resultados: dict[str, Any] = {}
    serie_limpia = serie.dropna()

    try:
        residuos = _ajustar_ar1_residuos(serie_limpia)
        resultados["BDS"] = realizar_test_bds(residuos)
    except Exception as exc:
        resultados["BDS"] = {"error": str(exc)}

    try:
        serie_cuadrada = serie_limpia**2
        umbral = 1.96 / np.sqrt(len(serie_limpia))
        acf_cuadrados = acf(serie_cuadrada, nlags=_LAGS_ACF, fft=True)
        lags_sig = sum(abs(x) > umbral for x in acf_cuadrados[1:])
        resultados["ACF_cuadrados"] = {
            "lags_significativos": int(lags_sig),
            "max_autocorr": float(np.max(np.abs(acf_cuadrados[1:]))),
            "interpretacion": (
                "Presencia de dependencia no lineal"
                if lags_sig > 0
                else "Sin dependencia no lineal evidente"
            ),
        }
    except Exception as exc:
        resultados["ACF_cuadrados"] = {"error": str(exc)}

    try:
        lb_test = acorr_ljungbox(
            serie_limpia**2, lags=_LAGS_LJUNG_BOX, return_df=True
        )
        p_lb = float(lb_test["lb_pvalue"].iloc[-1])
        resultados["Ljung_Box_cuadrados"] = {
            "estadistico": float(lb_test["lb_stat"].iloc[-1]),
            "p_valor": p_lb,
            "es_no_lineal": p_lb < _NIVEL_SIGNIFICANCIA,
            "interpretacion": (
                "Presencia de efectos ARCH/GARCH (no linealidad)"
                if p_lb < _NIVEL_SIGNIFICANCIA
                else "Sin efectos ARCH/GARCH evidentes"
            ),
        }
    except Exception as exc:
        resultados["Ljung_Box_cuadrados"] = {"error": str(exc)}

    try:
        skew = float(serie_limpia.skew())
        kurt = float(serie_limpia.kurtosis())
        jb_stat, jb_pval = jarque_bera(serie_limpia)
        resultados["distribucion"] = {
            "skewness": skew,
            "kurtosis": kurt,
            "interpretacion": (
                f"Distribución {'normal' if abs(skew) < 0.5 and abs(kurt) < 0.5 else 'no normal'} "
                f"(skew={skew:.4f}, kurt={kurt:.4f})"
            ),
        }
        resultados["Jarque_Bera"] = {
            "estadistico": float(jb_stat),
            "p_valor": float(jb_pval),
            "es_normal": float(jb_pval) > _NIVEL_SIGNIFICANCIA,
            "interpretacion": (
                "Distribución normal"
                if float(jb_pval) > _NIVEL_SIGNIFICANCIA
                else "Distribución no normal"
            ),
        }
    except Exception as exc:
        resultados["distribucion"] = {"error": str(exc)}

    return resultados


def ajustar_modelos_arch_garch(retornos_pct: pd.Series) -> dict[str, Any]:
    """
    Ajusta ARCH(1) y GARCH(1,1) sobre retornos diarios en %.

    Parameters
    ----------
    retornos_pct : pd.Series
        Retornos diarios en porcentaje.

    Returns
    -------
    dict
        Parámetros, AIC/BIC y log-verosimilitud de cada modelo, o error si falta ``arch``.
    """
    if not ARCH_DISPONIBLE:
        return {
            "error": "Paquete 'arch' no instalado. Instale con: pip install arch"
        }

    serie = retornos_pct.dropna()
    resultados: dict[str, Any] = {}

    especificaciones = {
        "ARCH_1": {"vol": "ARCH", "p": 1, "q": 0},
        "GARCH_1_1": {"vol": "Garch", "p": 1, "q": 1},
    }

    for nombre, kwargs in especificaciones.items():
        try:
            modelo = arch_model(
                serie,
                mean="Zero",
                **kwargs,
                rescale=False,
            )
            ajuste = modelo.fit(disp="off", show_warning=False)
            resultados[nombre] = {
                "aic": float(ajuste.aic),
                "bic": float(ajuste.bic),
                "loglik": float(ajuste.loglikelihood),
                "parametros": {k: float(v) for k, v in ajuste.params.items()},
            }
        except Exception as exc:
            resultados[nombre] = {"error": str(exc)}

    return resultados


def ljung_box_rendimientos_cuadrado(retornos_pct: pd.Series) -> dict[str, Any]:
    """
    Prueba de Ljung-Box sobre el cuadrado de los rendimientos del índice.

    Parameters
    ----------
    retornos_pct : pd.Series
        Retornos diarios en %.

    Returns
    -------
    dict
        Estadístico, p-valor e interpretación ARCH/GARCH.
    """
    serie_limpia = retornos_pct.dropna()
    cuadrados = serie_limpia**2
    lb_test = acorr_ljungbox(cuadrados, lags=_LAGS_LJUNG_BOX, return_df=True)
    p_lb = float(lb_test["lb_pvalue"].iloc[-1])
    return {
        "estadistico": float(lb_test["lb_stat"].iloc[-1]),
        "p_valor": p_lb,
        "es_arch_garch": p_lb < _NIVEL_SIGNIFICANCIA,
        "interpretacion": (
            "Presencia de efectos ARCH/GARCH (agrupación de volatilidad)"
            if p_lb < _NIVEL_SIGNIFICANCIA
            else "Sin evidencia de efectos ARCH/GARCH"
        ),
    }


def analizar_variabilidad_amplitud_frecuencia(
    serie: pd.Series,
    ventana: int = _VENTANA_VOLATILIDAD_ANUAL,
) -> dict[str, Any]:
    """
    Mide variabilidad en amplitud (volatilidad móvil) y frecuencia (cruces por cero).

    Parameters
    ----------
    serie : pd.Series
        Serie de precios de cierre.
    ventana : int
        Ventana móvil en días (252 ≈ un año de negociación).

    Returns
    -------
    dict
        Métricas de amplitud, frecuencia y dispersión (IQR móvil).
    """
    resultados: dict[str, Any] = {}
    serie_limpia = serie.dropna()

    volatilidad_movil = serie_limpia.rolling(window=ventana).std()
    vol_media = float(volatilidad_movil.mean())
    vol_std = float(volatilidad_movil.std())
    resultados["variabilidad_amplitud"] = {
        "volatilidad_promedio": vol_media,
        "volatilidad_std": vol_std,
        "coeficiente_variacion": vol_std / vol_media if vol_media > 0 else 0.0,
        "ratio_max_min": (
            float(volatilidad_movil.max() / volatilidad_movil.min())
            if volatilidad_movil.min() > 0
            else 0.0
        ),
        "interpretacion": (
            "Alta variabilidad en amplitud"
            if vol_std / vol_media > 0.3
            else "Variabilidad moderada en amplitud"
        ),
    }

    cambios_signo = np.diff(
        np.sign(serie_limpia - serie_limpia.rolling(window=ventana).mean())
    )
    cruces_cero = int(np.sum(cambios_signo != 0))

    frecuencias_moviles: list[float] = []
    for i in range(ventana, len(serie_limpia)):
        ventana_serie = serie_limpia.iloc[i - ventana : i]
        media_ventana = ventana_serie.mean()
        cambios = np.diff(np.sign(ventana_serie - media_ventana))
        frecuencias_moviles.append(float(np.sum(cambios != 0)))

    if frecuencias_moviles:
        freq_media = float(np.mean(frecuencias_moviles))
        freq_std = float(np.std(frecuencias_moviles))
        resultados["variabilidad_frecuencia"] = {
            "cruces_cero_totales": cruces_cero,
            "frecuencia_promedio": freq_media,
            "frecuencia_std": freq_std,
            "coeficiente_variacion_freq": (
                freq_std / freq_media if freq_media > 0 else 0.0
            ),
            "interpretacion": (
                "Alta variabilidad en frecuencia"
                if freq_std / freq_media > 0.3
                else "Variabilidad moderada en frecuencia"
            ),
        }

    iqr_movil = serie_limpia.rolling(window=ventana).quantile(0.75) - serie_limpia.rolling(
        window=ventana
    ).quantile(0.25)
    iqr_media = float(iqr_movil.mean())
    iqr_std = float(iqr_movil.std())
    resultados["variabilidad_dispersion"] = {
        "iqr_promedio": iqr_media,
        "iqr_std": iqr_std,
        "coeficiente_variacion_iqr": iqr_std / iqr_media if iqr_media > 0 else 0.0,
    }

    return resultados


def _imprimir_estacionaridad(
    titulo: str, resultados: dict[str, Any]
) -> None:
    """
    Imprime resultados ADF/KPSS por consola.

    Parameters
    ----------
    titulo : str
        Etiqueta de la serie.
    resultados : dict
        Salida de ``realizar_tests_estacionaridad``.
    """
    logger.info("\n%s", titulo)
    logger.info("-" * 70)
    if "ADF" in resultados and "error" not in resultados["ADF"]:
        adf = resultados["ADF"]
        logger.info(
            "ADF: estadístico=%.4f, p-valor=%.6f → %s",
            adf["estadistico"],
            adf["p_valor"],
            adf["interpretacion"],
        )
    if "KPSS" in resultados and "error" not in resultados["KPSS"]:
        kpss_res = resultados["KPSS"]
        logger.info(
            "KPSS: estadístico=%.4f, p-valor=%.6f → %s",
            kpss_res["estadistico"],
            kpss_res["p_valor"],
            kpss_res["interpretacion"],
        )


def _imprimir_bds(resultados_bds: dict[str, Any]) -> None:
    """
    Imprime estadísticos BDS por dimensión.

    Parameters
    ----------
    resultados_bds : dict
        Salida de ``realizar_test_bds`` o clave BDS anidada.
    """
    if "error" in resultados_bds:
        logger.warning("BDS: %s", resultados_bds["error"])
        return
    for dim, res in sorted(resultados_bds.items()):
        if isinstance(res, dict) and "error" not in res:
            logger.info(
                "BDS dim %s: estadístico=%.4f, p-valor=%.6f → %s",
                dim,
                res["estadistico"],
                res["p_valor"],
                "No linealidad detectada" if res["es_no_lineal"] else "Sin evidencia",
            )


def main() -> dict[str, Any]:
    """
    Ejecuta el pipeline completo de condiciones EMD para XLE.

    Returns
    -------
    dict
        Resultados estructurados de todos los bloques de análisis.
    """
    df = cargar_datos_xle()

    logger.info("=" * 70)
    logger.info("TESTS DE ESTACIONARIDAD — XLE")
    logger.info("=" * 70)

    est_precio = realizar_tests_estacionaridad(df["Close"])
    est_returns = realizar_tests_estacionaridad(df["Returns"].dropna())
    _imprimir_estacionaridad("1. PRECIO DE CIERRE (Close)", est_precio)
    _imprimir_estacionaridad("2. RETURNS (%)", est_returns)

    logger.info("=" * 70)
    logger.info("NO LINEALIDAD — BDS (residuos AR(1))")
    logger.info("=" * 70)

    no_lin_precio = analizar_no_linealidad(df["Close"])
    no_lin_returns = analizar_no_linealidad(df["Returns"].dropna())

    _imprimir_bds(no_lin_precio.get("BDS", {}))
    if "Ljung_Box_cuadrados" in no_lin_precio:
        lb = no_lin_precio["Ljung_Box_cuadrados"]
        if "error" not in lb:
            logger.info(
                "Ljung-Box (cuadrados, Close): p-valor=%.6f → %s",
                lb["p_valor"],
                lb["interpretacion"],
            )

    logger.info("=" * 70)
    logger.info("VOLATILIDAD — ARCH/GARCH Y LJUNG-BOX (rendimientos²)")
    logger.info("=" * 70)

    lb_returns = ljung_box_rendimientos_cuadrado(df["Returns"])
    logger.info(
        "Ljung-Box rendimientos²: estadístico=%.4f, p-valor=%.6f → %s",
        lb_returns["estadistico"],
        lb_returns["p_valor"],
        lb_returns["interpretacion"],
    )

    arch_garch = ajustar_modelos_arch_garch(df["Returns"])
    for nombre, res in arch_garch.items():
        if isinstance(res, dict) and "error" not in res:
            logger.info(
                "%s: AIC=%.2f, BIC=%.2f, loglik=%.2f",
                nombre,
                res["aic"],
                res["bic"],
                res["loglik"],
            )
        elif isinstance(res, dict):
            logger.warning("%s: %s", nombre, res.get("error"))

    variabilidad = analizar_variabilidad_amplitud_frecuencia(df["Close"])
    if "variabilidad_amplitud" in variabilidad:
        va = variabilidad["variabilidad_amplitud"]
        logger.info(
            "Amplitud (vol. móvil %sd): CV=%.4f → %s",
            _VENTANA_VOLATILIDAD_ANUAL,
            va["coeficiente_variacion"],
            va["interpretacion"],
        )
    if "variabilidad_frecuencia" in variabilidad:
        vf = variabilidad["variabilidad_frecuencia"]
        logger.info(
            "Frecuencia: cruces/ventana media=%.2f → %s",
            vf["frecuencia_promedio"],
            vf["interpretacion"],
        )

    logger.info("=" * 70)
    logger.info("CONCLUSIÓN PARA EMD")
    logger.info("=" * 70)
    precio_no_est = (
        "ADF" in est_precio
        and "error" not in est_precio["ADF"]
        and not est_precio["ADF"]["es_estacionaria"]
    )
    hay_no_linealidad = lb_returns.get("es_arch_garch", False)
    if precio_no_est:
        logger.info("✓ Precio NO estacionario — favorable para EMD")
    if hay_no_linealidad:
        logger.info("✓ No linealidad / ARCH-GARCH detectados — favorable para EMD")

    return {
        "estacionaridad_precio": est_precio,
        "estacionaridad_returns": est_returns,
        "no_linealidad_precio": no_lin_precio,
        "no_linealidad_returns": no_lin_returns,
        "ljung_box_rendimientos_cuadrado": lb_returns,
        "arch_garch": arch_garch,
        "variabilidad_amplitud_frecuencia": variabilidad,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        main()
    except Exception:
        logger.exception("Error en el análisis de condiciones EMD de XLE")
        sys.exit(1)
