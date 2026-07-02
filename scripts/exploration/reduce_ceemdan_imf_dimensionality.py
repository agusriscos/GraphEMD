"""
Dimensionality reduction of CEEMDAN IMFs (MSCI World) before graph construction.

Objective
---------
Reduce the number of **variables per time step** (e.g. 8 oscillatory IMFs) to
``n_components < p`` **without** reducing the number of temporal samples. If present,
the CEEMDAN residue is **kept unmixed** in a separate column.

Recommended method (practical SOTA for statistical independence)
----------------------------------------------------------------
**FastICA** (``sklearn.decomposition.FastICA``) seeks components that maximize
non-Gaussianity and, under the instantaneous linear mixing model, approximate
**statistically independent** sources compared with PCA, which only guarantees
**uncorrelated** (orthogonal) scores. Key references:

- Hyvärinen, A., & Oja, E. (2000). Independent component analysis: algorithms
  and applications. *Neural Networks*, 13(4-5), 411-430.
  https://doi.org/10.1016/S0893-6080(00)00026-5
- Hyvärinen, A. (1999). Fast and robust fixed-point algorithms for independent
  component analysis. *IEEE Transactions on Neural Networks*, 10(3), 626-634.

Conceptual limitation
---------------------
Any linear projection ``R^p -> R^k`` with ``k < p`` **mixes** the original IMFs
in the physical EMD sense (frequency bands). FastICA mitigates **statistical
dependence among output channels**; it does not preserve the spectral separation
guaranteed by CEEMDAN in the space of 8 IMFs. To audit the degree of mixing,
the script saves the mixing matrix and cross-correlation between reduced components.

IMF reconstruction
------------------
PCA and FastICA (scikit-learn) expose ``inverse_transform`` on the standardized
space: from ``Z`` one obtains ``\\hat{X}`` on the original IMF scale via
``scaler.inverse_transform(model.inverse_transform(Z))``. If ``k < p``, reconstruction
is **approximate** (rank-``k`` subspace); with ``k = p``, PCA recovers the
standardized data up to numerical error. The ``modelo_*.npz`` files include scaling
and, for ICA, the internal FastICA mean needed to invert the mixing.

Typical outputs (default under ``data/20abr26/imfs_ceemdan_dim_red/``)
-----------------------------------------------------------------------
- ``imfs_reducidas.parquet``: time series ``Z_1..Z_k`` (+ ``Residuo``).
- ``imfs_reconstruidas_aprox.parquet``: IMFs reconstructed from ``Z`` (same names as input).
- ``metricas_reduccion.json``: correlations among ``Z``, PCA variance, reconstruction error, etc.
- ``modelo_ica.npz`` / ``modelo_pca.npz``: matrices, scaling, and parameters for inversion.
- ``REFERENCIAS_METODO.md``: brief verifiable citations.

Example::

    PYTHONPATH=src/python python \\
        scripts/exploration/reduce_ceemdan_imf_dimensionality.py \\
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

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_DEFAULT = _REPO_ROOT / "data" / "20abr26" / "msci_world_imfs_ceemdan.parquet"
_OUT_DEFAULT = _REPO_ROOT / "data" / "20abr26" / "imfs_ceemdan_dim_red"

logger = logging.getLogger(__name__)


def _columnas_imf_ordenadas(df: pd.DataFrame) -> List[str]:
    """
    Return ``IMF_*`` names sorted numerically.

    Parameters
    ----------
    df : pd.DataFrame
        Table with columns ``IMF_1``, ...

    Returns
    -------
    list of str
        Sorted list of IMF columns.
    """
    cols = [c for c in df.columns if c.startswith("IMF_")]
    if not cols:
        raise ValueError("No IMF_* columns found in the DataFrame.")
    return sorted(cols, key=lambda x: int(x.split("_")[1]))


def extract_imf_block_and_residual(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, List[str], Optional[np.ndarray]]:
    """
    Extract matrix ``(T, p)`` of IMFs and optionally the residue vector.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``IMF_1``, ...; ``Residuo`` is optional.

    Returns
    -------
    X : np.ndarray
        Shape ``(n_samples, n_imfs)``.
    nombres_imf : list of str
        IMF column names in order.
    residuo : np.ndarray or None
        Same temporal length as ``X``, or None if the column is missing.
    """
    nombres = _columnas_imf_ordenadas(df)
    X = np.asarray(df[nombres].values, dtype=np.float64)
    residuo = None
    if "Residuo" in df.columns:
        residuo = np.asarray(df["Residuo"].values, dtype=np.float64)
    return X, nombres, residuo


def _correlation_metrics(Z: np.ndarray) -> Dict[str, Any]:
    """
    Pearson correlation between columns of ``Z`` (diagonal excluded).

    Parameters
    ----------
    Z : np.ndarray
        Shape ``(n_samples, n_components)``.

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


def reconstruct_imfs_from_ica_z(
    Z: np.ndarray,
    ica: FastICA,
    scaler: StandardScaler,
) -> np.ndarray:
    """
    Reconstruct the IMF matrix on the original scale from ICA sources.

    Parameters
    ----------
    Z : np.ndarray
        Sources, shape ``(T, k)``.
    ica : FastICA
        Fitted model on standardized data.
    scaler : StandardScaler
        Scaler fitted on the original ``X``.

    Returns
    -------
    np.ndarray
        ``\\hat{X}`` with shape ``(T, p)``, an approximation of the input IMFs.
    """
    Xs_hat = ica.inverse_transform(Z)
    return scaler.inverse_transform(Xs_hat)


def reconstruct_imfs_from_pca_z(
    Z: np.ndarray,
    pca: PCA,
    scaler: StandardScaler,
) -> np.ndarray:
    """
    Reconstruct the IMF matrix on the original scale from PCA scores.

    Parameters
    ----------
    Z : np.ndarray
        PCA scores, shape ``(T, k)``.
    pca : PCA
        Fitted model on standardized data.
    scaler : StandardScaler
        Scaler fitted on the original ``X``.

    Returns
    -------
    np.ndarray
        ``\\hat{X}`` with shape ``(T, p)``; if ``k=p`` it is (nearly) exact in standardized space.
    """
    Xs_hat = pca.inverse_transform(Z)
    return scaler.inverse_transform(Xs_hat)


def imf_reconstruction_metrics(
    X: np.ndarray,
    X_hat: np.ndarray,
    nombres_imf: List[str],
) -> Dict[str, Any]:
    """
    Errors between original and reconstructed IMFs.

    Parameters
    ----------
    X : np.ndarray
        Original IMFs ``(T, p)``.
    X_hat : np.ndarray
        Reconstruction ``(T, p)``.
    nombres_imf : list of str
        Labels per column.

    Returns
    -------
    dict
        Global RMSE, relative Frobenius error, per-column $R^2$, and mean $R^2$.
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
    Detect the knee point in ``metrica`` versus ``k`` (simplified Kneedle).

    After sorting by ``k``, normalize both axes to ``[0, 1]`` and return the ``k`` with
    the largest perpendicular distance to the line between the endpoints (Satopaa et al., 2011).

    Parameters
    ----------
    k_vals : np.ndarray
        Evaluated ``k`` values.
    metrica : np.ndarray
        Metric increasing with ``k`` (e.g. mean reconstruction ``R²``).

    Returns
    -------
    int
        ``k`` at the curve elbow.

    Examples
    --------
    >>> detectar_k_rodilla(np.array([2, 3, 4]), np.array([0.3, 0.5, 0.9]))
    3
    """
    k_arr = np.asarray(k_vals, dtype=float)
    y_arr = np.asarray(metrica, dtype=float)
    if k_arr.shape != y_arr.shape:
        raise ValueError("k_vals and metrica must have the same length.")
    if len(k_arr) == 0:
        raise ValueError("k_vals is empty.")
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
    Dimensionless score to break ties among ``k`` candidates (higher is better).

    Combines reconstruction (``R²``), Frobenius error, independence of ``Z``, and
    parsimony ``1 - k/p``.
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


def select_optimal_k_from_ica_grid(
    df_rejilla: pd.DataFrame,
    p: int,
    umbral_corr_z: float = 0.01,
    fraccion_r2_objetivo: float = 0.72,
    factor_error_frobenius_rodilla: float = 1.15,
) -> Dict[str, Any]:
    """
    Choose the optimal number of ICA components from the calibration grid.

    Parsimonious criterion (grid data only):

    1. **Candidates**: ``max|corr(Z)| < umbral_corr_z``; if none qualify, the 50 %
       with the lowest cross-correlation between sources.
    2. **Knee on** ``R²``: ``k_rodilla`` (Kneedle) and reference Frobenius error
       ``err_ref = error(k_rodilla)``.
    3. **Smallest** ``k`` among candidates that simultaneously satisfy:

       - ``R² ≥ fraccion_r2_objetivo × max(R²)`` on the viable grid;
       - ``error_frobenius ≤ factor_error_frobenius_rodilla × err_ref``.

       This favors low dimensionality without straying far from the ``R²`` knee error.
    4. If no ``k`` satisfies both conditions, use ``k_rodilla``.

    Parameters
    ----------
    df_rejilla : pd.DataFrame
        Rows with ``n_components``, ``r2_medio_columnas``, ``error_relativo_frobenius``,
        ``max_abs_corr_Z``.
    p : int
        Number of input IMFs (for parsimony in auxiliary metrics).
    umbral_corr_z : float
        Practical independence threshold among ``Z`` sources.
    fraccion_r2_objetivo : float
        Minimum fraction of maximum ``R²`` among candidates (e.g. 0.72).
    factor_error_frobenius_rodilla : float
        Maximum multiple of the error at ``k_rodilla`` (e.g. 1.15 = +15 %).

    Returns
    -------
    dict
        ``n_components``, metrics for the chosen row, ``criterio_seleccion``, and
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
        raise ValueError(f"Incomplete ICA grid; missing columns: {sorted(faltantes)}")

    df = df_rejilla.sort_values("n_components").reset_index(drop=True)
    viables = df[df["max_abs_corr_Z"] < umbral_corr_z].copy()
    if viables.empty:
        n_cand = max(1, len(df) // 2)
        viables = df.nsmallest(n_cand, "max_abs_corr_Z").copy()
        filtro = f"no k with max|corr|<{umbral_corr_z}; {n_cand} candidates with smallest |corr|"
    else:
        filtro = f"candidates with max|corr(Z)|<{umbral_corr_z}"

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
            f"smallest k with R²≥{fraccion_r2_objetivo:.0%} of the maximum ({umbral_r2:.4f}) "
            f"y error≤{factor_error_frobenius_rodilla:.2f}×err(k_rodilla)={umbral_err:.4f}"
        )
    else:
        mejor_k = int(k_rodilla)
        nota_sel = (
            f"no k meeting R² and Frobenius → knee k={k_rodilla} "
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
        f"{filtro}; knee R²→k={k_rodilla} (err_ref={err_ref:.4f}); "
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


def fit_fastica(
    X: np.ndarray,
    n_components: int,
    random_state: int,
) -> Tuple[FastICA, StandardScaler, np.ndarray]:
    """
    Scale ``X`` and fit FastICA; return estimated sources ``Z``.

    Parameters
    ----------
    X : np.ndarray
        Shape ``(T, p)``.
    n_components : int
        Output dimension ``k``.
    random_state : int
        Seed.

    Returns
    -------
    ica : FastICA
        Fitted model.
    scaler : StandardScaler
        Standard scaling applied before ICA.
    Z : np.ndarray
        Estimated sources, shape ``(T, k)``.
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


def fit_pca(
    X: np.ndarray,
    n_components: int,
) -> Tuple[PCA, StandardScaler, np.ndarray]:
    """
    Scale ``X`` and fit PCA; return scores ``Z``.

    Parameters
    ----------
    X : np.ndarray
        Shape ``(T, p)``.
    n_components : int
        Number of principal components.

    Returns
    -------
    pca : PCA
        Fitted model.
    scaler : StandardScaler
        Standard scaling.
    Z : np.ndarray
        Scores, shape ``(T, k)``.
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, svd_solver="full", random_state=0)
    Z = pca.fit_transform(Xs)
    return pca, scaler, Z


def save_ica_outputs(
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
    Write reduced parquet, approximate IMF reconstruction, metrics, and ``modelo_ica.npz``.

    Parameters
    ----------
    out_dir : Path
        Output directory.
    Z : np.ndarray
        Components, shape ``(T, k)``.
    residuo : np.ndarray or None
        Optional column left untransformed.
    ica : FastICA
        Fitted model.
    scaler : StandardScaler
        Scaler used before ICA.
    nombres_imf : list of str
        Input IMF names.
    metricas : dict
        JSON-serializable dictionary merged with local diagnostics.
    X : np.ndarray
        Original IMF block ``(T, p)`` for reconstruction error.
    guardar_parquet_recon : bool, optional
        If True, write ``imfs_reconstruidas_aprox.parquet``. Default is True.
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

    X_hat = reconstruct_imfs_from_ica_z(Z, ica, scaler)
    rec_metrics = imf_reconstruction_metrics(X, X_hat, nombres_imf)
    if guardar_parquet_recon:
        df_rec = pd.DataFrame(
            {nombres_imf[j]: X_hat[:, j] for j in range(len(nombres_imf))}
        )
        path_rec = out_dir / "imfs_reconstruidas_aprox.parquet"
        df_rec.to_parquet(path_rec, index=False, engine="pyarrow")
        logger.info("Reconstruction parquet: %s", path_rec)

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
        "correlacion_entre_Z": _correlation_metrics(Z),
        "reconstruccion_imfs": rec_metrics,
        **metricas,
    }
    (out_dir / "metricas_reduccion.json").write_text(
        json.dumps(metricas_ica, indent=2), encoding="utf-8"
    )
    logger.info(
        "IMF reconstruction: relative Frobenius error=%.4f, mean column R²=%.4f",
        rec_metrics["error_relativo_frobenius"],
        rec_metrics["r2_medio_columnas"],
    )


def save_pca_outputs(
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
    Same as ``save_ica_outputs`` but for PCA (scores stored as ``Z_j``).

    Parameters
    ----------
    out_dir : Path
        Output directory.
    Z : np.ndarray
        PCA scores.
    residuo : np.ndarray or None
        Optional residue.
    pca : PCA
        Fitted PCA model.
    scaler : StandardScaler
        Scaler.
    nombres_imf : list of str
        Input IMFs.
    metricas : dict
        Additional metadata.
    X : np.ndarray
        Original IMF block for reconstruction error.
    guardar_parquet_recon : bool, optional
        If True, write ``imfs_reconstruidas_aprox.parquet``. Default is True.
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

    X_hat = reconstruct_imfs_from_pca_z(Z, pca, scaler)
    rec_metrics = imf_reconstruction_metrics(X, X_hat, nombres_imf)
    if guardar_parquet_recon:
        df_rec = pd.DataFrame(
            {nombres_imf[j]: X_hat[:, j] for j in range(len(nombres_imf))}
        )
        path_rec = out_dir / "imfs_reconstruidas_aprox.parquet"
        df_rec.to_parquet(path_rec, index=False, engine="pyarrow")
        logger.info("Reconstruction parquet: %s", path_rec)

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
        "correlacion_entre_Z": _correlation_metrics(Z),
        "reconstruccion_imfs": rec_metrics,
        **metricas,
    }
    (out_dir / "metricas_reduccion.json").write_text(
        json.dumps(metricas_pca, indent=2), encoding="utf-8"
    )
    logger.info(
        "IMF reconstruction: relative Frobenius error=%.4f, mean column R²=%.4f",
        rec_metrics["error_relativo_frobenius"],
        rec_metrics["r2_medio_columnas"],
    )


def write_references(out_dir: Path) -> None:
    """
    Write a markdown file with verifiable method references.

    Parameters
    ----------
    out_dir : Path
        Output directory.
    """
    texto = """# References (dimensionality reduction with approximate independence)

## FastICA (recommended in this script to minimize dependence between outputs)

1. Hyvärinen, A., & Oja, E. (2000). Independent component analysis: algorithms and applications.
   *Neural Networks*, 13(4-5), 411-430. DOI: 10.1016/S0893-6080(00)00026-5
   https://doi.org/10.1016/S0893-6080(00)00026-5

2. Hyvärinen, A. (1999). Fast and robust fixed-point algorithms for independent component analysis.
   *IEEE Transactions on Neural Networks*, 10(3), 626-634.
   https://ieeexplore.ieee.org/document/761722

3. scikit-learn documentation (implementation used): `sklearn.decomposition.FastICA`
   https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html

## PCA (baseline: explained variance, uncorrelated scores)

4. Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: a review and recent developments.
   *Philosophical Transactions of the Royal Society A*, 374(2065), 20150202.
   https://doi.org/10.1098/rsta.2015.0202

## Note on «mode mixing» in EMD

FastICA and PCA operate in the IMF space as instantaneous variables.
Reducing from *p* to *k* implies linear combinations of IMFs; that does not match
the notion of mode mixing in EMD (time-frequency separation). To audit,
review `metricas_reduccion.json` (correlation between `Z`, reconstruction error) and `modelo_*.npz`
(mixing matrix / components, means for `inverse_transform`).
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "REFERENCIAS_METODO.md").write_text(texto, encoding="utf-8")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Reduce CEEMDAN IMF dimensionality (variables per time step, not samples)."
    )
    parser.add_argument(
        "--parquet-imfs",
        type=Path,
        default=_DATA_DEFAULT,
        help="Parquet with IMF_1,...,IMF_p and optionally Residuo.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_OUT_DEFAULT,
        help="Output directory (parquet, json, npz, references).",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=4,
        help="Output dimension k (k < number of IMFs).",
    )
    parser.add_argument(
        "--metodo",
        choices=("ica", "pca", "ambos"),
        default="ica",
        help="FastICA (approximate independence), PCA (uncorrelation + variance), or both.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--sin-parquet-reconstruccion",
        action="store_true",
        help="Do not write imfs_reconstruidas_aprox.parquet (metrics remain in JSON).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if not args.parquet_imfs.is_file():
        logger.error("Parquet does not exist: %s", args.parquet_imfs)
        sys.exit(1)

    df = pd.read_parquet(args.parquet_imfs, engine="pyarrow")
    X, nombres_imf, residuo = extract_imf_block_and_residual(df)
    p = X.shape[1]
    if args.n_components >= p:
        logger.error("n_components (%d) must be < number of IMFs (%d).", args.n_components, p)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_references(args.out_dir)

    meta_comun = {
        "parquet_entrada": str(args.parquet_imfs.resolve()),
        "n_muestras_temporales": int(X.shape[0]),
        "columnas_imf_entrada": nombres_imf,
    }

    if args.metodo in ("ica", "ambos"):
        sub = args.out_dir if args.metodo == "ica" else args.out_dir / "fastica"
        ica, scaler_i, Z_i = fit_fastica(
            X, args.n_components, args.random_state
        )
        save_ica_outputs(
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
            _correlation_metrics(Z_i)["max_abs_fuera_diagonal"],
        )

    if args.metodo in ("pca", "ambos"):
        sub = args.out_dir if args.metodo == "pca" else args.out_dir / "pca"
        pca, scaler_p, Z_p = fit_pca(X, args.n_components)
        save_pca_outputs(
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
            "PCA: cumulative variance first %d PCs = %.4f",
            args.n_components,
            float(np.sum(pca.explained_variance_ratio_)),
        )
        logger.info(
            "PCA: max|corr(Z_i,Z_j)| fuera diagonal = %.4f",
            _correlation_metrics(Z_p)["max_abs_fuera_diagonal"],
        )

    logger.info("Done. Outputs in %s", args.out_dir.resolve())


if __name__ == "__main__":
    main()
