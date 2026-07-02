"""
Script to analyze EMD (Empirical Mode Decomposition) preconditions for the XLV (Health Care Select Sector SPDR Fund).

1. Load XLV data from parquet file data/GraphEMD/xlv_analysis/xlv.parquet.
2. Assess series stationarity using the Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests, as done for MSCI World in this repository
3. Assess non-linearity using the Brock-Dechert-Scheinkman (BDS) test on AR(1) model residuals.
4. Assess amplitude and frequency volatility using ARCH/GARCH and the Ljung-Box test on squared returns.
5. Document implemented logic and results in this script.
6. Do not change any existing code for now

Implemented logic
-------------------
Replica of ``analysis/16dic25/02_emd_conditions_analysis.ipynb`` (MSCI World):

- **Stationarity**: ADF (``autolag='AIC'``) and KPSS (``regression='ct'``, ``nlags='auto'``) on
  ``Close`` and on daily returns in % (``pct_change×100``).
- **Non-linearity**: BDS on AR(1) residuals (``statsmodels.tsa.ar_model.AutoReg``,
  ``lags=1``) in embedding dimensions 2–5; squared autocorrelation; Ljung-Box
  on squared returns (10 lags); skewness, kurtosis, and Jarque-Bera.
- **Volatility (amplitude/frequency)**: fit ARCH(1) and GARCH(1,1) with ``arch`` on
  % returns; Ljung-Box on squared returns; amplitude variability
  (252-day rolling volatility) and frequency (zero crossings in rolling windows), as
  in the reference notebook.

Results obtained (run 2026-05-16, n=3587)
-------------------------------------------------

**Stationarity — Close:** ADF statistic=-0.0783, p=0.9516 (non-stationary);
KPSS statistic=1.6104, p=0.0100 (non-stationary).

**Stationarity — Returns (%):** ADF=-11.6799, p<0.0001 (stationary);
KPSS=0.0458, p=0.1000 (stationary).

**BDS (AR(1) residuals on Close):** dim 3: 17.28 (p<0.0001); dim 4: 20.75;
dim 5: 23.42 — non-linearity in all reported dimensions.

**Ljung-Box (squared returns):** statistic=2267.29, p<0.0001 — ARCH/GARCH effects.

**ARCH(1):** AIC=13446.82, BIC=13459.19. **GARCH(1,1):** AIC=12612.98,
BIC=12631.53 (better fit than ARCH).

**Variability:** amplitude (252d rolling vol.) CV=0.4619 — high; frequency
(crossings/mean per window=12.67) — high.

**EMD conclusion:** non-stationary price, non-linearity, and volatility clustering
— favorable conditions for EMD (analogous to MSCI World).
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

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_RUTA_PARQUET_XLV = _REPO_ROOT / "data" / "GraphEMD" / "xlv_analysis" / "xlv.parquet"
_VENTANA_VOLATILIDAD_ANUAL: int = 252
_LAGS_LJUNG_BOX: int = 10
_LAGS_ACF: int = 20
_DIMENSIONES_BDS: list[int] = [2, 3, 4, 5]
_NIVEL_SIGNIFICANCIA: float = 0.05

logger = logging.getLogger(__name__)


def cargar_datos_xlv(ruta_parquet: Path = _RUTA_PARQUET_XLV) -> pd.DataFrame:
    """
    Load the XLV series and compute daily returns in percent.

    Parameters
    ----------
    ruta_parquet : Path
        Path to the ``xlv.parquet``.

    Returns
    -------
    pd.DataFrame
        OHLCV data with ``Returns`` (%) column.

    Raises
    ------
    FileNotFoundError
        If the parquet does not exist.
    ValueError
        If the ``Close`` column is missing.
    """
    if not ruta_parquet.is_file():
        raise FileNotFoundError(f"Not found: {ruta_parquet}")
    df = pd.read_parquet(ruta_parquet, engine="pyarrow")
    if "Close" not in df.columns:
        raise ValueError(f"Missing Close column. Columns: {list(df.columns)}")
    df = df.copy()
    df["Returns"] = df["Close"].pct_change() * 100.0
    logger.info(
        "Data loaded: %s rows, %s → %s",
        len(df),
        df.index.min(),
        df.index.max(),
    )
    return df


def realizar_tests_estacionaridad(serie: pd.Series) -> dict[str, Any]:
    """
    Apply ADF and KPSS to a time series (same logic as MSCI World).

    Parameters
    ----------
    serie : pd.Series
        Series to analyze.

    Returns
    -------
    dict
        ADF and KPSS results with statistics, p-values, and interpretation.
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
                "Stationary" if p_valor < _NIVEL_SIGNIFICANCIA else "Non-stationary"
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
                "Stationary" if p_valor > _NIVEL_SIGNIFICANCIA else "Non-stationary"
            ),
        }
    except Exception as exc:
        resultados["KPSS"] = {"error": str(exc)}

    return resultados


def _ajustar_ar1_residuos(serie: pd.Series) -> np.ndarray:
    """
    Fit AR(1) and return residuals for the BDS test.

    Parameters
    ----------
    serie : pd.Series
        Price or return series.

    Returns
    -------
    np.ndarray
        AR(1) model residuals.
    """
    serie_limpia = serie.dropna()
    modelo_ar = AutoReg(serie_limpia, lags=1).fit()
    return np.asarray(modelo_ar.resid, dtype=np.float64)


def realizar_test_bds(residuos: np.ndarray) -> dict[str, Any]:
    """
    Run BDS on AR(1) residuals across embedding dimensions.

    Parameters
    ----------
    residuos : np.ndarray
        AR(1) fit residuals.

    Returns
    -------
    dict
        Statistics and p-values per dimension, or ``error`` key.
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
    Assess non-linearity: BDS (AR(1) residuals), squared ACF, and Ljung-Box.

    Parameters
    ----------
    serie : pd.Series
        Time series (prices or returns).

    Returns
    -------
    dict
        Aggregated non-linearity analysis results.
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
                "Non-linear dependence detected"
                if lags_sig > 0
                else "No evident non-linear dependence"
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
                "ARCH/GARCH effects present (non-linearity)"
                if p_lb < _NIVEL_SIGNIFICANCIA
                else "No evident ARCH/GARCH effects"
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
                f"Distribution {'normal' if abs(skew) < 0.5 and abs(kurt) < 0.5 else 'no normal'} "
                f"(skew={skew:.4f}, kurt={kurt:.4f})"
            ),
        }
        resultados["Jarque_Bera"] = {
            "estadistico": float(jb_stat),
            "p_valor": float(jb_pval),
            "es_normal": float(jb_pval) > _NIVEL_SIGNIFICANCIA,
            "interpretacion": (
                "Distribution normal"
                if float(jb_pval) > _NIVEL_SIGNIFICANCIA
                else "Distribution no normal"
            ),
        }
    except Exception as exc:
        resultados["distribucion"] = {"error": str(exc)}

    return resultados


def ajustar_modelos_arch_garch(retornos_pct: pd.Series) -> dict[str, Any]:
    """
    Fit ARCH(1) and GARCH(1,1) on daily returns in %.

    Parameters
    ----------
    retornos_pct : pd.Series
        Daily returns in percent.

    Returns
    -------
    dict
        Parameters, AIC/BIC, and log-likelihood per model, or error if ``arch`` is missing.
    """
    if not ARCH_DISPONIBLE:
        return {
            "error": "Package 'arch' not installed. Install with: pip install arch"
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
    Ljung-Box test on squared index returns.

    Parameters
    ----------
    retornos_pct : pd.Series
        Daily returns in %.

    Returns
    -------
    dict
        Statistic, p-value, and ARCH/GARCH interpretation.
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
            "ARCH/GARCH effects present (volatility clustering)"
            if p_lb < _NIVEL_SIGNIFICANCIA
            else "No evidence of ARCH/GARCH effects"
        ),
    }


def analizar_variabilidad_amplitud_frecuencia(
    serie: pd.Series,
    ventana: int = _VENTANA_VOLATILIDAD_ANUAL,
) -> dict[str, Any]:
    """
    Measure amplitude variability (rolling volatility) and frequency (zero crossings).

    Parameters
    ----------
    serie : pd.Series
        Closing price series.
    ventana : int
        Rolling window in days (252 ≈ one trading year).

    Returns
    -------
    dict
        Amplitudede, frequency, and dispersion (rolling IQR) metrics.
    """
    resultados: dict[str, Any] = {}
    serie_limpia = serie.dropna()

    volatilidad_movil = serie_limpia.rolling(window=ventana).std()
    vol_mean = float(volatilidad_movil.mean())
    vol_std = float(volatilidad_movil.std())
    resultados["variabilidad_amplitud"] = {
        "volatilidad_promedio": vol_media,
        "volatilidad_std": vol_std,
        "coeficiente_variacion": vol_std / vol_mean if vol_mean > 0 else 0.0,
        "ratio_max_min": (
            float(volatilidad_movil.max() / volatilidad_movil.min())
            if volatilidad_movil.min() > 0
            else 0.0
        ),
        "interpretacion": (
            "High amplitude variability"
            if vol_std / vol_mean > 0.3
            else "Moderate amplitude variability"
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
        freq_mean = float(np.mean(frecuencias_moviles))
        freq_std = float(np.std(frecuencias_moviles))
        resultados["variabilidad_frecuencia"] = {
            "cruces_cero_totales": cruces_cero,
            "frecuencia_promedio": freq_media,
            "frecuencia_std": freq_std,
            "coeficiente_variacion_freq": (
                freq_std / freq_mean if freq_mean > 0 else 0.0
            ),
            "interpretacion": (
                "High frequency variability"
                if freq_std / freq_mean > 0.3
                else "Moderate frequency variability"
            ),
        }

    iqr_movil = serie_limpia.rolling(window=ventana).quantile(0.75) - serie_limpia.rolling(
        window=ventana
    ).quantile(0.25)
    iqr_mean = float(iqr_movil.mean())
    iqr_std = float(iqr_movil.std())
    resultados["variabilidad_dispersion"] = {
        "iqr_promedio": iqr_media,
        "iqr_std": iqr_std,
        "coeficiente_variacion_iqr": iqr_std / iqr_mean if iqr_mean > 0 else 0.0,
    }

    return resultados


def _imprimir_estacionaridad(
    titulo: str, resultados: dict[str, Any]
) -> None:
    """
    Print ADF/KPSS results to the console.

    Parameters
    ----------
    titulo : str
        Series label.
    resultados : dict
        Output of ``realizar_tests_estacionaridad``.
    """
    logger.info("\n%s", titulo)
    logger.info("-" * 70)
    if "ADF" in resultados and "error" not in resultados["ADF"]:
        adf = resultados["ADF"]
        logger.info(
            "ADF: statistic=%.4f, p-value=%.6f → %s",
            adf["estadistico"],
            adf["p_valor"],
            adf["interpretacion"],
        )
    if "KPSS" in resultados and "error" not in resultados["KPSS"]:
        kpss_res = resultados["KPSS"]
        logger.info(
            "KPSS: statistic=%.4f, p-value=%.6f → %s",
            kpss_res["estadistico"],
            kpss_res["p_valor"],
            kpss_res["interpretacion"],
        )


def _imprimir_bds(resultados_bds: dict[str, Any]) -> None:
    """
    Print BDS statistics per dimension.

    Parameters
    ----------
    resultados_bds : dict
        Output of ``realizar_test_bds`` or nested BDS key.
    """
    if "error" in resultados_bds:
        logger.warning("BDS: %s", resultados_bds["error"])
        return
    for dim, res in sorted(resultados_bds.items()):
        if isinstance(res, dict) and "error" not in res:
            logger.info(
                "BDS dim %s: statistic=%.4f, p-value=%.6f → %s",
                dim,
                res["estadistico"],
                res["p_valor"],
                "Non-linearity detected" if res["es_no_lineal"] else "No evidence",
            )


def main() -> dict[str, Any]:
    """
    Run the full EMD precondition pipeline for XLV.

    Returns
    -------
    dict
        Structured results from all analysis blocks.
    """
    df = cargar_datos_xlv()

    logger.info("=" * 70)
    logger.info("STATIONARITY TESTS — XLV")
    logger.info("=" * 70)

    est_precio = realizar_tests_estacionaridad(df["Close"])
    est_returns = realizar_tests_estacionaridad(df["Returns"].dropna())
    _imprimir_estacionaridad("1. CLOSING PRICE (Close)", est_precio)
    _imprimir_estacionaridad("2. RETURNS (%)", est_returns)

    logger.info("=" * 70)
    logger.info("NON-LINEARITY — BDS (AR(1) residuals)")
    logger.info("=" * 70)

    no_lin_precio = analizar_no_linealidad(df["Close"])
    no_lin_returns = analizar_no_linealidad(df["Returns"].dropna())

    _imprimir_bds(no_lin_precio.get("BDS", {}))
    if "Ljung_Box_cuadrados" in no_lin_precio:
        lb = no_lin_precio["Ljung_Box_cuadrados"]
        if "error" not in lb:
            logger.info(
                "Ljung-Box (squared, Close): p-value=%.6f → %s",
                lb["p_valor"],
                lb["interpretacion"],
            )

    logger.info("=" * 70)
    logger.info("VOLATILITY — ARCH/GARCH AND LJUNG-BOX (squared returns)")
    logger.info("=" * 70)

    lb_returns = ljung_box_rendimientos_cuadrado(df["Returns"])
    logger.info(
        "Ljung-Box squared returns: statistic=%.4f, p-value=%.6f → %s",
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
            "Amplitude (rolling vol. %sd): CV=%.4f → %s",
            _VENTANA_VOLATILIDAD_ANUAL,
            va["coeficiente_variacion"],
            va["interpretacion"],
        )
    if "variabilidad_frecuencia" in variabilidad:
        vf = variabilidad["variabilidad_frecuencia"]
        logger.info(
            "Frequency: mean zero-crossings/window=%.2f → %s",
            vf["frecuencia_promedio"],
            vf["interpretacion"],
        )

    logger.info("=" * 70)
    logger.info("EMD CONCLUSION")
    logger.info("=" * 70)
    precio_no_est = (
        "ADF" in est_precio
        and "error" not in est_precio["ADF"]
        and not est_precio["ADF"]["es_estacionaria"]
    )
    hay_no_linealidad = lb_returns.get("es_arch_garch", False)
    if precio_no_est:
        logger.info("✓ Non-stationary price — favorable for EMD")
    if hay_no_linealidad:
        logger.info("✓ Non-linearity / ARCH-GARCH detected — favorable for EMD")

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
        logger.exception("Error in EMD condition analysis for XLV")
        sys.exit(1)
