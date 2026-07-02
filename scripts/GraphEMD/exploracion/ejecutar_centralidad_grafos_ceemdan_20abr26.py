"""
Calcula medias (y máximos por nodo) de centralidad de intermediación, cercanía y
vector propio para la subsubsección *Centrality analysis* y la tabla
``tab:centralidad`` en ``docs/20abr26/main.tex``.

Para cada componente (IMF$_1$--IMF$_8$ y ``Residuo``) se construyen HVG, NVG y
grafo de recurrencia con los mismos criterios que en
``ejecutar_metricas_estructurales_grafos_ceemdan_20abr26.py``.

Si el NVG supera ``MAX_ARISTAS_NVG_CENTRALIDAD`` aristas, no se calculan
centralidades (residuo). Si ``m`` supera ``UMBRAL_ARISTAS_BETWEENNESS_APROX``, la intermediación usa
muestreo ``k`` (NetworkX). Si ``m`` supera ``UMBRAL_ARISTAS_CERCANIA_EXACTA``,
la cercanía NVG usa muestreo de nodos (``closeness_centrality(G, u=...)``).
Para grafos de recurrencia con ``m`` grande, intermediación y cercanía usan
los mismos criterios de aproximación (véase ``UMBRAL_ARISTAS_RECURRENCIA_APROX``).
Para ``n <= N_MAX_EIGENVECTOR_NUMPY`` se usa ``eigenvector_centrality_numpy``
(matriz densa ``n x n``; en este estudio ``n`` es el tamaño muestral diario).

Si existen ``*_nvg_edges.parquet`` en ``out_msci_world_grafos/imfs_ceemdan_20abr26``,
el NVG se carga desde disco en lugar de ``ts2vg``.

Salida
------
- ``docs/20abr26/out/centralidad_por_componente_ceemdan.csv``
- ``docs/20abr26/out/resumen_centralidad_grafos_ceemdan.csv``
- ``docs/20abr26/out/resumen_centralidad_grafos_ceemdan.md``

Dependencias: mismas que ``ejecutar_metricas_estructurales_grafos_ceemdan_20abr26.py``.

Ejecución::

    PYTHONPATH=src/python python scripts/GraphEMD/exploracion/ejecutar_centralidad_grafos_ceemdan_20abr26.py
"""

from __future__ import annotations

import contextlib
import io
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from ts2vg import HorizontalVG, NaturalVG

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SRC_PYTHON = _REPO_ROOT / "src" / "python"
if str(_SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(_SRC_PYTHON))

from GraphEMD.data.graph_imf_transform_utils import (
    calcular_false_nearest_neighbors,
    calcular_matriz_recurrencia,
    construir_espacio_embedding,
    seleccionar_tau,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
_BUF = io.StringIO()

MAX_ARISTAS_NVG_CENTRALIDAD = 2_000_000
UMBRAL_ARISTAS_BETWEENNESS_APROX = 200_000
K_MUESTRA_BETWEENNESS = 600
SEMILLA_BETWEENNESS = 42
UMBRAL_ARISTAS_CERCANIA_EXACTA = 50_000
K_MUESTRA_CERCANIA = 500
SEMILLA_CERCANIA = 42
UMBRAL_ARISTAS_RECURRENCIA_APROX = 12_000
N_MAX_EIGENVECTOR_NUMPY = 4096

_DIR_NVG_PRECALC = (
    _REPO_ROOT
    / "scripts"
    / "GraphEMD"
    / "exploracion"
    / "out_msci_world_grafos"
    / "imfs_ceemdan_20abr26"
)


def _grafo_nvg_desde_parquet_si_existe(
    nombre_componente: str, n_nodos: int
) -> nx.Graph | None:
    """
    Carga aristas NVG desde parquet precalculado si existe.

    Evita ``NaturalVG.build`` en grafos grandes cuando el equipo ya exportó
    ``{componente}_nvg_edges.parquet`` en ``_DIR_NVG_PRECALC``.

    Parameters
    ----------
    nombre_componente : str
        ``IMF_1``, ..., ``IMF_8`` o ``Residuo``.
    n_nodos : int
        Número de nodos esperado (longitud de la serie).

    Returns
    -------
    nx.Graph | None
        Grafo no dirigido o None si no hay archivo.
    """
    ruta = _DIR_NVG_PRECALC / f"{nombre_componente}_nvg_edges.parquet"
    if not ruta.is_file():
        return None
    ed = pd.read_parquet(ruta, columns=["source", "target"])
    g = nx.Graph()
    g.add_edges_from(
        zip(
            ed["source"].to_numpy(dtype=np.int64), ed["target"].to_numpy(dtype=np.int64)
        )
    )
    g.add_nodes_from(range(n_nodos))
    return g


def _grafo_visibilidad(
    x: np.ndarray, constructor: type[HorizontalVG] | type[NaturalVG]
) -> nx.Graph:
    """
    Construye un grafo no dirigido simple a partir de aristas devueltas por ts2vg.

    Parameters
    ----------
    x : np.ndarray
        Serie 1D.
    constructor : type
        ``HorizontalVG`` o ``NaturalVG``.

    Returns
    -------
    nx.Graph
        Grafo con aristas deduplicadas.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    g = constructor(directed="left_to_right").build(x)
    G = nx.Graph()
    G.add_edges_from(map(tuple, np.array(g.edges)))
    return G


def _grafo_recurrencia(
    x: np.ndarray,
    umbral_percentil: float = 10.0,
    random_state: int = 42,
) -> nx.Graph:
    """
    Construye el grafo de recurrencia como ``Graph`` de NetworkX.

    Parameters
    ----------
    x : np.ndarray
        Serie de una IMF o del residuo.
    umbral_percentil : float, optional
        Percentil para el umbral de distancia.
    random_state : int, optional
        Semilla para el umbral basado en muestreo.

    Returns
    -------
    nx.Graph
        Grafo no ponderado sin bucles.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    tau = int(seleccionar_tau(x, tau_max=50))
    d = int(calcular_false_nearest_neighbors(x, tau=tau, dim_max=10))
    emb = construir_espacio_embedding(x, d, tau)
    with contextlib.redirect_stdout(_BUF):
        mat, _eps = calcular_matriz_recurrencia(
            emb, umbral_percentil=umbral_percentil, random_state=random_state
        )
    sym = np.maximum(mat, mat.T).astype(np.uint8)
    np.fill_diagonal(sym, 0)
    return nx.from_numpy_array(sym)


def _media_y_max(d: dict[Any, float]) -> tuple[float, float]:
    """
    Devuelve media y máximo de un diccionario nodo $\\to$ escalar.

    Parameters
    ----------
    d : dict
        Mapa de nodos a coeficientes.

    Returns
    -------
    tuple[float, float]
        Media y máximo; NaN si ``d`` está vacío.
    """
    if not d:
        return float("nan"), float("nan")
    vals = np.fromiter(d.values(), dtype=np.float64, count=len(d))
    return float(np.mean(vals)), float(np.max(vals))


def _betweenness(G: nx.Graph, *, aproximar: bool) -> tuple[float, float, bool]:
    """
    Intermediación normalizada: media y máximo sobre nodos.

    Parameters
    ----------
    G : nx.Graph
        Grafo no dirigido.
    aproximar : bool
        Si es True, usa muestreo aleatorio de ``K_MUESTRA_BETWEENNESS`` nodos.

    Returns
    -------
    tuple[float, float, bool]
        Media, máximo y si se usó aproximación.
    """
    n = G.number_of_nodes()
    if n == 0:
        return float("nan"), float("nan"), False
    if aproximar:
        k = min(K_MUESTRA_BETWEENNESS, n)
        bc = nx.betweenness_centrality(
            G, k=k, normalized=True, seed=SEMILLA_BETWEENNESS
        )
        media, mx = _media_y_max(bc)
        return media, mx, True
    bc = nx.betweenness_centrality(G, normalized=True)
    media, mx = _media_y_max(bc)
    return media, mx, False


def _closeness(G: nx.Graph, *, usar_muestreo: bool) -> tuple[float, float]:
    """
    Cercanía (NetworkX): media y máximo de coeficientes por nodo.

    Si ``usar_muestreo`` es True (grafos muy densos, típicamente NVG), la media y
    el máximo se estiman con ``K_MUESTRA_CERCANIA`` nodos elegidos al azar
    (``SEMILLA_CERCANIA``), calculando ``closeness_centrality(G, u=...)`` por
    nodo muestreado; evita el coste $O(n(n+m))$ de la versión completa.

    Parameters
    ----------
    G : nx.Graph
        Grafo no dirigido.
    usar_muestreo : bool
        Si es True, usa muestreo de nodos para la cercanía.

    Returns
    -------
    tuple[float, float]
        Media y máximo estimados o exactos.
    """
    if G.number_of_nodes() == 0:
        return float("nan"), float("nan")
    if not usar_muestreo:
        cc = nx.closeness_centrality(G)
        return _media_y_max(cc)
    rng = np.random.default_rng(SEMILLA_CERCANIA)
    nodos = list(G.nodes())
    k = min(K_MUESTRA_CERCANIA, len(nodos))
    muestra = rng.choice(nodos, size=k, replace=False)
    vals = [float(nx.closeness_centrality(G, u=u)) for u in muestra]
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.mean(arr)), float(np.max(arr))


def _eigenvector(G: nx.Graph) -> tuple[float, float]:
    """
    Centralidad de vector propio (subgrafo conexo más grande si hace falta).

    Parameters
    ----------
    G : nx.Graph
        Grafo no dirigido.

    Returns
    -------
    tuple[float, float]
        Media y máximo sobre los nodos del subgrafo donde se calculó.
    """
    if G.number_of_nodes() == 0:
        return float("nan"), float("nan")
    h: nx.Graph = G
    if not nx.is_connected(G):
        nodos = max(nx.connected_components(G), key=len)
        h = G.subgraph(nodos).copy()
    if h.number_of_nodes() <= N_MAX_EIGENVECTOR_NUMPY:
        try:
            ev = nx.eigenvector_centrality_numpy(h)
            return _media_y_max(ev)
        except (nx.NetworkXError, np.linalg.LinAlgError):
            pass
    try:
        ev = nx.eigenvector_centrality(h, max_iter=5000, tol=1e-06)
        return _media_y_max(ev)
    except nx.NetworkXError:
        return float("nan"), float("nan")


def _formatear_rango(r: dict[str, float], *, cuatro_decimales: bool) -> str:
    """
    Formatea ``min``--``max`` para CSV o LaTeX.

    Parameters
    ----------
    r : dict[str, float]
        Claves ``min`` y ``max``.
    cuatro_decimales : bool
        Si es True, cuatro decimales fijos.

    Returns
    -------
    str
        Cadena formateada.
    """
    if cuatro_decimales:
        return f"{r['min']:.4f} - {r['max']:.4f}"
    return f"{r['min']:.6g} - {r['max']:.6g}"


def _rango_columna(df: pd.DataFrame, nombre: str) -> dict[str, float]:
    """
    Mínimo y máximo de una columna numérica ignorando NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Tabla por componente.
    nombre : str
        Nombre de columna.

    Returns
    -------
    dict[str, float]
        ``min`` y ``max``.
    """
    arr = np.asarray(pd.to_numeric(df[nombre], errors="coerce"), dtype=np.float64)
    return {"min": float(np.nanmin(arr)), "max": float(np.nanmax(arr))}


def main() -> None:
    """
    Escribe CSV de centralidad por componente y resumen de rangos (medias).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ruta_imfs = _REPO_ROOT / "data" / "20abr26" / "msci_world_imfs_ceemdan.parquet"
    if not ruta_imfs.is_file():
        raise FileNotFoundError(f"No se encuentra {ruta_imfs}")

    df_imfs = pd.read_parquet(ruta_imfs, engine="pyarrow")
    columnas = [c for c in df_imfs.columns if c.startswith("IMF_") or c == "Residuo"]

    filas: list[dict[str, Any]] = []
    for nombre in columnas:
        x = np.asarray(df_imfs[nombre].to_numpy(), dtype=np.float64)
        logger.info("Procesando %s...", nombre)

        Gh = _grafo_visibilidad(x, HorizontalVG)
        bh_m, bh_x, _ = _betweenness(Gh, aproximar=False)
        ch_m, ch_x = _closeness(Gh, usar_muestreo=False)
        eh_m, eh_x = _eigenvector(Gh)

        Gn = _grafo_nvg_desde_parquet_si_existe(nombre, len(x))
        if Gn is None:
            Gn = _grafo_visibilidad(x, NaturalVG)
        else:
            logger.info(
                "  NVG cargado desde parquet (%s_nvg_edges.parquet).",
                nombre,
            )
        mn = Gn.number_of_edges()
        if mn > MAX_ARISTAS_NVG_CENTRALIDAD:
            logger.info(
                "  NVG con %s aristas: centralidad omitida (>%s).",
                mn,
                MAX_ARISTAS_NVG_CENTRALIDAD,
            )
            bn_m = bn_x = float("nan")
            cn_m = cn_x = float("nan")
            en_m = en_x = float("nan")
            nvg_bt_aprox = False
            nvg_cl_muestreo = False
        else:
            aprox_bt = mn > UMBRAL_ARISTAS_BETWEENNESS_APROX
            if aprox_bt:
                logger.info(
                    "  NVG: intermediación aproximada (m=%s > %s).",
                    mn,
                    UMBRAL_ARISTAS_BETWEENNESS_APROX,
                )
            bn_m, bn_x, nvg_bt_aprox = _betweenness(Gn, aproximar=aprox_bt)
            nvg_cl_muestreo = mn > UMBRAL_ARISTAS_CERCANIA_EXACTA
            if nvg_cl_muestreo:
                logger.info(
                    "  NVG: cercanía por muestreo (m=%s > %s).",
                    mn,
                    UMBRAL_ARISTAS_CERCANIA_EXACTA,
                )
            cn_m, cn_x = _closeness(Gn, usar_muestreo=nvg_cl_muestreo)
            en_m, en_x = _eigenvector(Gn)

        Gr = _grafo_recurrencia(x)
        mr = Gr.number_of_edges()
        aprox_rec = mr > UMBRAL_ARISTAS_RECURRENCIA_APROX
        if aprox_rec:
            logger.info(
                "  Recurrencia: intermediación/cercanía aprox. (m=%s > %s).",
                mr,
                UMBRAL_ARISTAS_RECURRENCIA_APROX,
            )
        br_m, br_x, rec_bt_aprox = _betweenness(Gr, aproximar=aprox_rec)
        cr_m, cr_x = _closeness(Gr, usar_muestreo=aprox_rec)
        er_m, er_x = _eigenvector(Gr)

        filas.append(
            {
                "componente": nombre,
                "hvg_betweenness_media": bh_m,
                "hvg_betweenness_max": bh_x,
                "hvg_closeness_media": ch_m,
                "hvg_closeness_max": ch_x,
                "hvg_eigenvector_media": eh_m,
                "hvg_eigenvector_max": eh_x,
                "nvg_m": mn,
                "nvg_centralidad_omitido": mn > MAX_ARISTAS_NVG_CENTRALIDAD,
                "nvg_betweenness_aprox": nvg_bt_aprox,
                "nvg_closeness_muestreado": nvg_cl_muestreo,
                "nvg_betweenness_media": bn_m,
                "nvg_betweenness_max": bn_x,
                "nvg_closeness_media": cn_m,
                "nvg_closeness_max": cn_x,
                "nvg_eigenvector_media": en_m,
                "nvg_eigenvector_max": en_x,
                "rec_betweenness_aprox": rec_bt_aprox,
                "rec_closeness_muestreado": aprox_rec,
                "rec_betweenness_media": br_m,
                "rec_betweenness_max": br_x,
                "rec_closeness_media": cr_m,
                "rec_closeness_max": cr_x,
                "rec_eigenvector_media": er_m,
                "rec_eigenvector_max": er_x,
            }
        )

    det = pd.DataFrame(filas)
    out_dir = _REPO_ROOT / "docs" / "20abr26" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    det_path = out_dir / "centralidad_por_componente_ceemdan.csv"
    det.to_csv(det_path, index=False)
    logger.info("Detalle: %s", det_path)

    metricas = [
        ("betweenness_media", "Betweenness (mean range)"),
        ("closeness_media", "Closeness (mean range)"),
        ("eigenvector_media", "Eigenvector (mean range)"),
    ]
    filas_res: list[dict[str, str]] = []
    for suf, etiqueta in metricas:
        rh = _rango_columna(det, f"hvg_{suf}")
        rr = _rango_columna(det, f"rec_{suf}")
        rn = _rango_columna(det, f"nvg_{suf}")
        filas_res.append(
            {
                "metric": etiqueta,
                "HVG": _formatear_rango(rh, cuatro_decimales=True),
                "Recurrence": _formatear_rango(rr, cuatro_decimales=True),
                "NVG": _formatear_rango(rn, cuatro_decimales=True),
            }
        )
    res_df = pd.DataFrame(filas_res)
    res_path = out_dir / "resumen_centralidad_grafos_ceemdan.csv"
    res_df.to_csv(res_path, index=False)
    logger.info("Resumen: %s", res_path)

    nota = (
        "\n\nNVG del residuo: centralidad omitida si "
        f"m > {MAX_ARISTAS_NVG_CENTRALIDAD}. "
        "Betweenness NVG: muestreo si "
        f"m > {UMBRAL_ARISTAS_BETWEENNESS_APROX} (k={K_MUESTRA_BETWEENNESS}, "
        f"seed={SEMILLA_BETWEENNESS}). "
        "Closeness NVG: muestreo de nodos si "
        f"m > {UMBRAL_ARISTAS_CERCANIA_EXACTA} (k={K_MUESTRA_CERCANIA}, "
        f"seed={SEMILLA_CERCANIA}). "
        "Recurrencia: intermediación y cercanía aproximadas si "
        f"m > {UMBRAL_ARISTAS_RECURRENCIA_APROX}.\n"
    )
    md_path = out_dir / "resumen_centralidad_grafos_ceemdan.md"
    md_path.write_text(
        "# Resumen centralidad (CEEMDAN)\n\n"
        + res_df.to_string(index=False)
        + nota
        + "\nDetalle: `centralidad_por_componente_ceemdan.csv`.\n",
        encoding="utf-8",
    )
    logger.info("Markdown: %s", md_path)


if __name__ == "__main__":
    main()
