"""
Reducción de dimensionalidad de las IMFs CEEMDAN (MSCI World) antes de grafos.

Objetivo
--------
Reducir el número de **variables por instante** (p. ej. 8 IMFs oscilatorias) a
``n_components < p`` **sin** reducir el número de muestras temporales. El residuo
CEEMDAN, si está presente, se **conserva sin mezclar** en una columna aparte.

Método recomendado (SOTA práctico para independencia estadística)
-----------------------------------------------------------------
**FastICA** (``sklearn.decomposition.FastICA``) busca componentes que maximicen
la no gaussianidad y, bajo el modelo de mezcla lineal instantánea, aproximan
fuentes **estadísticamente independientes** frente a PCA, que solo garantiza
**incorrelación** (ortogonalidad) de los scores. Referencias clave:

- Hyvärinen, A., & Oja, E. (2000). Independent component analysis: algorithms
  and applications. *Neural Networks*, 13(4-5), 411-430.
  https://doi.org/10.1016/S0893-6080(00)00026-5
- Hyvärinen, A. (1999). Fast and robust fixed-point algorithms for independent
  component analysis. *IEEE Transactions on Neural Networks*, 10(3), 626-634.

Limitación conceptual
---------------------
Cualquier proyección lineal ``R^p -> R^k`` con ``k < p`` **mezcla** las IMFs
originales en el sentido físico de EMD (bandas de frecuencia). FastICA mitiga la
**dependencia estadística entre canales de salida**; no preserva la separación
espectral garantizada por CEEMDAN en el espacio de las 8 IMFs. Para auditar el
grado de mezcla, el script guarda la matriz de mezcla y la correlación cruzada
entre componentes reducidas.

Reconstrucción de las IMF
-------------------------
PCA y FastICA (scikit-learn) exponen ``inverse_transform`` sobre el espacio
estandarizado: a partir de ``Z`` se obtiene ``\\hat{X}`` en la escala original
de las IMF mediante ``scaler.inverse_transform(modelo.inverse_transform(Z))``.
Si ``k < p``, la reconstrucción es **aproximada** (subespacio de rango ``k``);
con ``k = p``, PCA recupera los datos estandarizados salvo errores numéricos.
Los archivos ``modelo_*.npz`` incluyen el escalado y, en ICA, la media interna
de FastICA necesaria para invertir la mezcla.

Salidas típicas (por defecto en ``data/20abr26/imfs_ceemdan_dim_red/``)
-----------------------------------------------------------------------
- ``imfs_reducidas.parquet``: series temporales ``Z_1..Z_k`` (+ ``Residuo``).
- ``imfs_reconstruidas_aprox.parquet``: IMF reconstruidas desde ``Z`` (mismos nombres que la entrada).
- ``metricas_reduccion.json``: correlaciones entre ``Z``, varianza PCA, error de reconstrucción, etc.
- ``modelo_ica.npz`` / ``modelo_pca.npz``: matrices, escalado y parámetros para invertir.
- ``REFERENCIAS_METODO.md``: citas breves verificables.

Ejemplo::

    PYTHONPATH=src/python python \\
        scripts/GraphEMD/exploracion/reducir_dimensionalidad_imfs_ceemdan.py \\
        --n-components 4 --metodo ica
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DATA_DEFAULT = _REPO_ROOT / "data" / "20abr26" / "msci_world_imfs_ceemdan.parquet"
_OUT_DEFAULT = _REPO_ROOT / "data" / "20abr26" / "imfs_ceemdan_dim_red"

logger = logging.getLogger(__name__)


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
        Lista ordenada de columnas IMF.
    """
    cols = [c for c in df.columns if c.startswith("IMF_")]
    if not cols:
        raise ValueError("No se encontraron columnas IMF_* en el DataFrame.")
    return sorted(cols, key=lambda x: int(x.split("_")[1]))


def extraer_bloque_imf_y_residuo(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, List[str], Optional[np.ndarray]]:
    """
    Extrae matriz (T, p) de IMFs y opcionalmente el vector residuo.

    Parameters
    ----------
    df : pd.DataFrame
        Debe contener ``IMF_1``, ...; ``Residuo`` es opcional.

    Returns
    -------
    X : np.ndarray
        Forma ``(n_muestras, n_imfs)``.
    nombres_imf : list of str
        Nombres de columnas IMF en orden.
    residuo : np.ndarray or None
        Mismo largo temporal que ``X``, o None si no hay columna.
    """
    nombres = _columnas_imf_ordenadas(df)
    X = np.asarray(df[nombres].values, dtype=np.float64)
    residuo = None
    if "Residuo" in df.columns:
        residuo = np.asarray(df["Residuo"].values, dtype=np.float64)
    return X, nombres, residuo


def _metricas_correlacion(Z: np.ndarray) -> Dict[str, Any]:
    """
    Correlación de Pearson entre columnas de ``Z`` (excluye diagonal).

    Parameters
    ----------
    Z : np.ndarray
        Forma ``(n_muestras, n_components)``.

    Returns
    -------
    dict
        ``matriz_corr``, ``max_abs_fuera_diagonal``, ``media_abs_fuera_diagonal``.
    """
    if Z.shape[1] < 2:
        return {
            "matriz_corr": np.array([[1.0]]).tolist(),
            "max_abs_fuera_diagonal": 0.0,
            "media_abs_fuera_diagonal": 0.0,
        }
    c = np.corrcoef(Z.T)
    n = c.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off = c[mask]
    return {
        "matriz_corr": np.round(c, 6).tolist(),
        "max_abs_fuera_diagonal": float(np.max(np.abs(off))),
        "media_abs_fuera_diagonal": float(np.mean(np.abs(off))),
    }


def reconstruir_imfs_desde_z_ica(
    Z: np.ndarray,
    ica: FastICA,
    scaler: StandardScaler,
) -> np.ndarray:
    """
    Reconstruye la matriz de IMF en escala original a partir de fuentes ICA.

    Parameters
    ----------
    Z : np.ndarray
        Fuentes, forma ``(T, k)``.
    ica : FastICA
        Modelo ajustado sobre datos estandarizados.
    scaler : StandardScaler
        Escalador ajustado sobre ``X`` original.

    Returns
    -------
    np.ndarray
        ``\\hat{X}`` forma ``(T, p)``, aproximación de las IMF de entrada.
    """
    Xs_hat = ica.inverse_transform(Z)
    return scaler.inverse_transform(Xs_hat)


def reconstruir_imfs_desde_z_pca(
    Z: np.ndarray,
    pca: PCA,
    scaler: StandardScaler,
) -> np.ndarray:
    """
    Reconstruye la matriz de IMF en escala original a partir de scores PCA.

    Parameters
    ----------
    Z : np.ndarray
        Scores PCA, forma ``(T, k)``.
    pca : PCA
        Modelo ajustado sobre datos estandarizados.
    scaler : StandardScaler
        Escalador ajustado sobre ``X`` original.

    Returns
    -------
    np.ndarray
        ``\\hat{X}`` forma ``(T, p)``; si ``k=p`` es (casi) exacta en espacio estandarizado.
    """
    Xs_hat = pca.inverse_transform(Z)
    return scaler.inverse_transform(Xs_hat)


def metricas_reconstruccion_imfs(
    X: np.ndarray,
    X_hat: np.ndarray,
    nombres_imf: List[str],
) -> Dict[str, Any]:
    """
    Errores entre IMF originales y reconstruidas.

    Parameters
    ----------
    X : np.ndarray
        IMF originales ``(T, p)``.
    X_hat : np.ndarray
        Reconstrucción ``(T, p)``.
    nombres_imf : list of str
        Etiquetas por columna.

    Returns
    -------
    dict
        RMSE global, error relativo de Frobenius, $R^2$ por columna y media.
    """
    diff = X - X_hat
    rmse_global = float(np.sqrt(np.mean(diff**2)))
    norm_x = np.linalg.norm(X, ord="fro")
    rel_frob = float(np.linalg.norm(diff, ord="fro") / norm_x) if norm_x > 0 else 0.0
    r2_por_columna = []
    for j in range(X.shape[1]):
        y = X[:, j]
        yh = X_hat[:, j]
        ss_res = float(np.sum((y - yh) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
        r2_por_columna.append(
            {"columna": nombres_imf[j], "r2": float(r2), "rmse": float(np.sqrt(np.mean((y - yh) ** 2)))}
        )
    return {
        "rmse_global": rmse_global,
        "error_relativo_frobenius": rel_frob,
        "r2_medio_columnas": float(np.mean([c["r2"] for c in r2_por_columna])),
        "por_columna": r2_por_columna,
    }


def detectar_k_rodilla(
    k_vals: np.ndarray,
    metrica: np.ndarray,
) -> int:
    """
    Detecta el punto de rodilla en ``metrica`` frente a ``k`` (Kneedle simplificado).

    Tras ordenar por ``k``, normaliza ambos ejes a ``[0, 1]`` y devuelve el ``k`` con
    mayor distancia perpendicular a la recta entre los extremos (Satopaa et al., 2011).

    Parameters
    ----------
    k_vals : np.ndarray
        Valores de ``k`` evaluados.
    metrica : np.ndarray
        Métrica creciente con ``k`` (p. ej. ``R²`` medio de reconstrucción).

    Returns
    -------
    int
        ``k`` en el codo de la curva.

    Examples
    --------
    >>> detectar_k_rodilla(np.array([2, 3, 4]), np.array([0.3, 0.5, 0.9]))
    3
    """
    k_arr = np.asarray(k_vals, dtype=float)
    y_arr = np.asarray(metrica, dtype=float)
    if k_arr.shape != y_arr.shape:
        raise ValueError("k_vals y metrica deben tener la misma longitud.")
    if len(k_arr) == 0:
        raise ValueError("k_vals vacío.")
    if len(k_arr) == 1:
        return int(k_arr[0])

    orden = np.argsort(k_arr)
    k = k_arr[orden]
    y = y_arr[orden]
    k_norm = (k - k[0]) / (k[-1] - k[0] + 1e-12)
    y_norm = (y - y[0]) / (y[-1] - y[0] + 1e-12)
    dist = np.abs(y_norm - k_norm) / np.sqrt(2.0)
    return int(k[int(np.argmax(dist))])


def _score_compuesto_k(
    r2: float,
    error_frobenius: float,
    max_abs_corr_z: float,
    k: int,
    p: int,
    r2_max: float,
    err_max: float,
    umbral_corr_z: float,
) -> float:
    """
    Puntuación adimensional para desempatar candidatos de ``k`` (mayor es mejor).

    Combina reconstrucción (``R²``), error de Frobenius, independencia de ``Z`` y
  parsimonia ``1 - k/p``.
    """
    r2_norm = r2 / r2_max if r2_max > 1e-12 else 0.0
    err_norm = error_frobenius / err_max if err_max > 1e-12 else 0.0
    if max_abs_corr_z < umbral_corr_z:
        indep = 1.0
    else:
        indep = max(0.0, 1.0 - max_abs_corr_z / max(umbral_corr_z, 1e-12))
    parsimonia = 1.0 - float(k) / float(p)
    return (
        0.45 * r2_norm
        + 0.25 * (1.0 - err_norm)
        + 0.20 * indep
        + 0.10 * parsimonia
    )


def seleccionar_k_optimo_desde_rejilla_ica(
    df_rejilla: pd.DataFrame,
    p: int,
    umbral_corr_z: float = 0.01,
    fraccion_r2_objetivo: float = 0.72,
    factor_error_frobenius_rodilla: float = 1.15,
) -> Dict[str, Any]:
    """
    Elige el número óptimo de componentes ICA a partir de la rejilla de calibración.

    Criterio parsimonioso (solo datos de la rejilla):

    1. **Candidatos**: ``max|corr(Z)| < umbral_corr_z``; si no hay ninguno, el 50 %
       con menor correlación cruzada entre fuentes.
    2. **Rodilla en** ``R²``: ``k_rodilla`` (Kneedle) y error de Frobenius de referencia
       ``err_ref = error(k_rodilla)``.
    3. **Menor** ``k`` entre candidatos que cumple simultáneamente:

       - ``R² ≥ fraccion_r2_objetivo × max(R²)`` en la rejilla viable;
       - ``error_frobenius ≤ factor_error_frobenius_rodilla × err_ref``.

       Así se prioriza dimensionalidad baja sin alejarse mucho del error en el codo de ``R²``.
    4. Si ningún ``k`` cumple ambas condiciones, se usa ``k_rodilla``.

    Parameters
    ----------
    df_rejilla : pd.DataFrame
        Filas con ``n_components``, ``r2_medio_columnas``, ``error_relativo_frobenius``,
        ``max_abs_corr_Z``.
    p : int
        Número de IMF de entrada (para parsimonia en métricas auxiliares).
    umbral_corr_z : float
        Umbral de independencia práctica entre fuentes ``Z``.
    fraccion_r2_objetivo : float
        Fracción mínima del ``R²`` máximo entre candidatos (p. ej. 0.72).
    factor_error_frobenius_rodilla : float
        Múltiplo máximo del error en ``k_rodilla`` (p. ej. 1.15 = +15 %).

    Returns
    -------
    dict
        ``n_components``, métricas de la fila elegida, ``criterio_seleccion`` y
        ``detalle_seleccion``.
    """
    columnas_req = {
        "n_components",
        "r2_medio_columnas",
        "error_relativo_frobenius",
        "max_abs_corr_Z",
    }
    faltantes = columnas_req - set(df_rejilla.columns)
    if faltantes:
        raise ValueError(f"Rejilla ICA incompleta; faltan columnas: {sorted(faltantes)}")

    df = df_rejilla.sort_values("n_components").reset_index(drop=True)
    viables = df[df["max_abs_corr_Z"] < umbral_corr_z].copy()
    if viables.empty:
        n_cand = max(1, len(df) // 2)
        viables = df.nsmallest(n_cand, "max_abs_corr_Z").copy()
        filtro = f"sin k con max|corr|<{umbral_corr_z}; {n_cand} candidatos con menor |corr|"
    else:
        filtro = f"candidatos con max|corr(Z)|<{umbral_corr_z}"

    r2_max = float(viables["r2_medio_columnas"].max())
    err_max = float(viables["error_relativo_frobenius"].max())
    k_rodilla = detectar_k_rodilla(
        viables["n_components"].to_numpy(),
        viables["r2_medio_columnas"].to_numpy(),
    )
    fila_rodilla = viables[viables["n_components"] == k_rodilla].iloc[0]
    err_ref = float(fila_rodilla["error_relativo_frobenius"])
    umbral_r2 = fraccion_r2_objetivo * r2_max
    umbral_err = factor_error_frobenius_rodilla * err_ref

    aceptables = viables[
        (viables["r2_medio_columnas"] >= umbral_r2)
        & (viables["error_relativo_frobenius"] <= umbral_err)
    ].sort_values("n_components")

    if not aceptables.empty:
        mejor_k = int(aceptables["n_components"].iloc[0])
        nota_sel = (
            f"menor k con R²≥{fraccion_r2_objetivo:.0%} del máximo ({umbral_r2:.4f}) "
            f"y error≤{factor_error_frobenius_rodilla:.2f}×err(k_rodilla)={umbral_err:.4f}"
        )
    else:
        mejor_k = int(k_rodilla)
        nota_sel = (
            f"sin k que cumpla R² y Frobenius → k_rodilla={k_rodilla} "
            f"(err_ref={err_ref:.4f})"
        )

    fila_mejor = df[df["n_components"] == mejor_k].iloc[0]
    score_mejor = _score_compuesto_k(
        float(fila_mejor["r2_medio_columnas"]),
        float(fila_mejor["error_relativo_frobenius"]),
        float(fila_mejor["max_abs_corr_Z"]),
        mejor_k,
        p,
        r2_max,
        err_max,
        umbral_corr_z,
    )

    criterio = (
        f"{filtro}; rodilla R²→k={k_rodilla} (err_ref={err_ref:.4f}); "
        f"{nota_sel}→k={mejor_k}"
    )

    return {
        "n_components": mejor_k,
        "criterio_seleccion": criterio,
        "detalle_seleccion": {
            "k_rodilla_r2": int(k_rodilla),
            "error_frobenius_referencia_rodilla": float(err_ref),
            "r2_max_candidatos": r2_max,
            "umbral_r2_reconstruccion": float(umbral_r2),
            "umbral_error_frobenius": float(umbral_err),
            "fraccion_r2_objetivo": float(fraccion_r2_objetivo),
            "factor_error_frobenius_rodilla": float(factor_error_frobenius_rodilla),
            "score_compuesto_k_elegido": score_mejor,
        },
        **{
            k: float(v) if isinstance(v, (np.floating, float)) else v
            for k, v in fila_mejor.items()
            if k != "n_components"
        },
    }


def ajustar_fastica(
    X: np.ndarray,
    n_components: int,
    random_state: int,
) -> Tuple[FastICA, StandardScaler, np.ndarray]:
    """
    Escala ``X`` y ajusta FastICA; devuelve fuentes estimadas ``Z``.

    Parameters
    ----------
    X : np.ndarray
        Forma ``(T, p)``.
    n_components : int
        Dimensión de salida ``k``.
    random_state : int
        Semilla.

    Returns
    -------
    ica : FastICA
        Modelo ajustado.
    scaler : StandardScaler
        Escalado estándar aplicado antes de ICA.
    Z : np.ndarray
        Fuentes estimadas, forma ``(T, k)``.
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    ica = FastICA(
        n_components=n_components,
        algorithm="parallel",
        whiten="unit-variance",
        max_iter=3000,
        random_state=random_state,
        tol=1e-4,
    )
    Z = ica.fit_transform(Xs)
    return ica, scaler, Z


def ajustar_pca(
    X: np.ndarray,
    n_components: int,
) -> Tuple[PCA, StandardScaler, np.ndarray]:
    """
    Escala ``X`` y ajusta PCA; devuelve scores ``Z``.

    Parameters
    ----------
    X : np.ndarray
        Forma ``(T, p)``.
    n_components : int
        Número de componentes principales.

    Returns
    -------
    pca : PCA
        Modelo ajustado.
    scaler : StandardScaler
        Escalado estándar.
    Z : np.ndarray
        Scores, forma ``(T, k)``.
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, svd_solver="full", random_state=0)
    Z = pca.fit_transform(Xs)
    return pca, scaler, Z


def guardar_salidas_ica(
    out_dir: Path,
    Z: np.ndarray,
    residuo: Optional[np.ndarray],
    ica: FastICA,
    scaler: StandardScaler,
    nombres_imf: List[str],
    metricas: Dict[str, Any],
    X: np.ndarray,
    guardar_parquet_recon: bool = True,
) -> None:
    """
    Escribe parquet reducido, reconstrucción aproximada de IMF, métricas y ``modelo_ica.npz``.

    Parameters
    ----------
    out_dir : Path
        Directorio de salida.
    Z : np.ndarray
        Componentes, forma ``(T, k)``.
    residuo : np.ndarray or None
        Columna opcional sin transformar.
    ica : FastICA
        Modelo ajustado.
    scaler : StandardScaler
        Escalador usado antes de ICA.
    nombres_imf : list of str
        Nombres de las IMF de entrada.
    metricas : dict
        Diccionario serializable a JSON (se fusiona con diagnósticos propios).
    X : np.ndarray
        Bloque original de IMF ``(T, p)`` para medir error de reconstrucción.
    guardar_parquet_recon : bool, optional
        Si True, escribe ``imfs_reconstruidas_aprox.parquet``. Por defecto True.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    k = Z.shape[1]
    data: Dict[str, Any] = {f"Z_{j + 1}": Z[:, j] for j in range(k)}
    if residuo is not None:
        data["Residuo"] = residuo
    df_out = pd.DataFrame(data)
    parquet_path = out_dir / "imfs_reducidas.parquet"
    df_out.to_parquet(parquet_path, index=False, engine="pyarrow")
    logger.info("Parquet: %s", parquet_path)

    X_hat = reconstruir_imfs_desde_z_ica(Z, ica, scaler)
    rec_metrics = metricas_reconstruccion_imfs(X, X_hat, nombres_imf)
    if guardar_parquet_recon:
        df_rec = pd.DataFrame(
            {nombres_imf[j]: X_hat[:, j] for j in range(len(nombres_imf))}
        )
        path_rec = out_dir / "imfs_reconstruidas_aprox.parquet"
        df_rec.to_parquet(path_rec, index=False, engine="pyarrow")
        logger.info("Parquet reconstrucción: %s", path_rec)

    np.savez_compressed(
        out_dir / "modelo_ica.npz",
        mixing_=getattr(ica, "mixing_", np.array([])),
        components_=ica.components_,
        fastica_mean_=ica.mean_,
        mean_=scaler.mean_,
        scale_=scaler.scale_,
        nombres_imf=np.array(nombres_imf, dtype=object),
    )

    metricas_ica = {
        "metodo": "FastICA",
        "n_components": int(k),
        "n_features": int(len(nombres_imf)),
        "correlacion_entre_Z": _metricas_correlacion(Z),
        "reconstruccion_imfs": rec_metrics,
        **metricas,
    }
    (out_dir / "metricas_reduccion.json").write_text(
        json.dumps(metricas_ica, indent=2), encoding="utf-8"
    )
    logger.info(
        "Reconstrucción IMF: error rel. Frobenius=%.4f, R² medio columnas=%.4f",
        rec_metrics["error_relativo_frobenius"],
        rec_metrics["r2_medio_columnas"],
    )


def guardar_salidas_pca(
    out_dir: Path,
    Z: np.ndarray,
    residuo: Optional[np.ndarray],
    pca: PCA,
    scaler: StandardScaler,
    nombres_imf: List[str],
    metricas: Dict[str, Any],
    X: np.ndarray,
    guardar_parquet_recon: bool = True,
) -> None:
    """
    Igual que ``guardar_salidas_ica`` pero para PCA (scores como ``Z_j``).

    Parameters
    ----------
    out_dir : Path
        Directorio de salida.
    Z : np.ndarray
        Scores PCA.
    residuo : np.ndarray or None
        Residuo opcional.
    pca : PCA
        Modelo PCA ajustado.
    scaler : StandardScaler
        Escalador.
    nombres_imf : list of str
        IMF de entrada.
    metricas : dict
        Metadatos adicionales.
    X : np.ndarray
        Bloque original de IMF para error de reconstrucción.
    guardar_parquet_recon : bool, optional
        Si True, escribe ``imfs_reconstruidas_aprox.parquet``. Por defecto True.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    k = Z.shape[1]
    data = {f"Z_{j + 1}": Z[:, j] for j in range(k)}
    if residuo is not None:
        data["Residuo"] = residuo
    df_out = pd.DataFrame(data)
    parquet_path = out_dir / "imfs_reducidas.parquet"
    df_out.to_parquet(parquet_path, index=False, engine="pyarrow")
    logger.info("Parquet: %s", parquet_path)

    X_hat = reconstruir_imfs_desde_z_pca(Z, pca, scaler)
    rec_metrics = metricas_reconstruccion_imfs(X, X_hat, nombres_imf)
    if guardar_parquet_recon:
        df_rec = pd.DataFrame(
            {nombres_imf[j]: X_hat[:, j] for j in range(len(nombres_imf))}
        )
        path_rec = out_dir / "imfs_reconstruidas_aprox.parquet"
        df_rec.to_parquet(path_rec, index=False, engine="pyarrow")
        logger.info("Parquet reconstrucción: %s", path_rec)

    np.savez_compressed(
        out_dir / "modelo_pca.npz",
        components_=pca.components_,
        explained_variance_ratio_=pca.explained_variance_ratio_,
        pca_mean_=pca.mean_,
        mean_=scaler.mean_,
        scale_=scaler.scale_,
        nombres_imf=np.array(nombres_imf, dtype=object),
    )

    metricas_pca = {
        "metodo": "PCA",
        "n_components": int(k),
        "n_features": int(len(nombres_imf)),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "explained_variance_cumulative": float(
            np.sum(pca.explained_variance_ratio_)
        ),
        "correlacion_entre_Z": _metricas_correlacion(Z),
        "reconstruccion_imfs": rec_metrics,
        **metricas,
    }
    (out_dir / "metricas_reduccion.json").write_text(
        json.dumps(metricas_pca, indent=2), encoding="utf-8"
    )
    logger.info(
        "Reconstrucción IMF: error rel. Frobenius=%.4f, R² medio columnas=%.4f",
        rec_metrics["error_relativo_frobenius"],
        rec_metrics["r2_medio_columnas"],
    )


def escribir_referencias(out_dir: Path) -> None:
    """
    Escribe un markdown con referencias verificables del método.

    Parameters
    ----------
    out_dir : Path
        Directorio de salida.
    """
    texto = """# Referencias (reducción de dimensionalidad con independencia aproximada)

## FastICA (recomendado en este script para minimizar dependencia entre salidas)

1. Hyvärinen, A., & Oja, E. (2000). Independent component analysis: algorithms and applications.
   *Neural Networks*, 13(4-5), 411-430. DOI: 10.1016/S0893-6080(00)00026-5
   https://doi.org/10.1016/S0893-6080(00)00026-5

2. Hyvärinen, A. (1999). Fast and robust fixed-point algorithms for independent component analysis.
   *IEEE Transactions on Neural Networks*, 10(3), 626-634.
   https://ieeexplore.ieee.org/document/761722

3. Documentación scikit-learn (implementación usada): `sklearn.decomposition.FastICA`
   https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html

## PCA (línea base: varianza explicada, scores incorrelacionados)

4. Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: a review and recent developments.
   *Philosophical Transactions of the Royal Society A*, 374(2065), 20150202.
   https://doi.org/10.1098/rsta.2015.0202

## Nota sobre «mode mixing» en EMD

FastICA y PCA operan en el espacio de las IMF como variables instantáneas.
Reducir de *p* a *k* implica combinaciones lineales de IMF; eso no coincide con
la noción de mode mixing en EMD (separación tiempo-frecuencia). Para auditar,
revisar `metricas_reduccion.json` (correlación entre `Z`, error de reconstrucción) y `modelo_*.npz`
(matriz de mezcla / componentes, medias para `inverse_transform`).
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "REFERENCIAS_METODO.md").write_text(texto, encoding="utf-8")


def main() -> None:
    """Punto de entrada CLI."""
    parser = argparse.ArgumentParser(
        description="Reduce la dimensión de las IMF CEEMDAN (variables por tiempo, no muestras)."
    )
    parser.add_argument(
        "--parquet-imfs",
        type=Path,
        default=_DATA_DEFAULT,
        help="Parquet con IMF_1,...,IMF_p y opcionalmente Residuo.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_OUT_DEFAULT,
        help="Directorio de salida (parquet, json, npz, referencias).",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=4,
        help="Número de dimensiones de salida k (k < número de IMF).",
    )
    parser.add_argument(
        "--metodo",
        choices=("ica", "pca", "ambos"),
        default="ica",
        help="FastICA (independencia aproximada), PCA (incorrelación + varianza), o ambos.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--sin-parquet-reconstruccion",
        action="store_true",
        help="No escribir imfs_reconstruidas_aprox.parquet (las métricas siguen en JSON).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if not args.parquet_imfs.is_file():
        logger.error("No existe el parquet: %s", args.parquet_imfs)
        sys.exit(1)

    df = pd.read_parquet(args.parquet_imfs, engine="pyarrow")
    X, nombres_imf, residuo = extraer_bloque_imf_y_residuo(df)
    p = X.shape[1]
    if args.n_components >= p:
        logger.error("n_components (%d) debe ser < número de IMF (%d).", args.n_components, p)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    escribir_referencias(args.out_dir)

    meta_comun = {
        "parquet_entrada": str(args.parquet_imfs.resolve()),
        "n_muestras_temporales": int(X.shape[0]),
        "columnas_imf_entrada": nombres_imf,
    }

    if args.metodo in ("ica", "ambos"):
        sub = args.out_dir if args.metodo == "ica" else args.out_dir / "fastica"
        ica, scaler_i, Z_i = ajustar_fastica(
            X, args.n_components, args.random_state
        )
        guardar_salidas_ica(
            sub,
            Z_i,
            residuo,
            ica,
            scaler_i,
            nombres_imf,
            meta_comun,
            X,
            guardar_parquet_recon=not args.sin_parquet_reconstruccion,
        )
        logger.info(
            "FastICA: max|corr(Z_i,Z_j)| fuera diagonal = %.4f",
            _metricas_correlacion(Z_i)["max_abs_fuera_diagonal"],
        )

    if args.metodo in ("pca", "ambos"):
        sub = args.out_dir if args.metodo == "pca" else args.out_dir / "pca"
        pca, scaler_p, Z_p = ajustar_pca(X, args.n_components)
        guardar_salidas_pca(
            sub,
            Z_p,
            residuo,
            pca,
            scaler_p,
            nombres_imf,
            meta_comun,
            X,
            guardar_parquet_recon=not args.sin_parquet_reconstruccion,
        )
        logger.info(
            "PCA: varianza acumulada primeras %d CP = %.4f",
            args.n_components,
            float(np.sum(pca.explained_variance_ratio_)),
        )
        logger.info(
            "PCA: max|corr(Z_i,Z_j)| fuera diagonal = %.4f",
            _metricas_correlacion(Z_p)["max_abs_fuera_diagonal"],
        )

    logger.info("Listo. Salidas en %s", args.out_dir.resolve())


if __name__ == "__main__":
    main()
