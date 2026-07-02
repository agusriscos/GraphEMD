"""
Calcula las magnitudes resumidas en ``docs/20abr26/main.tex`` (subsubsección
*General structural properties. Connectivity and distance*) y la tabla
``tab:propiedades_estructurales``.

Para cada componente (IMF$_1$--IMF$_8$ y ``Residuo``) del parquet CEEMDAN se
obtienen grafos HVG, NVG y de recurrencia (mismos criterios que en el documento:
percentil 10 para $\\varepsilon$, FNN, MI para $\\tau$).

Salida
------
- ``docs/20abr26/out/metricas_estructurales_por_componente_ceemdan.csv``
- ``docs/20abr26/out/resumen_estructural_grafos_ceemdan.csv`` (rangos min--max)
- ``docs/20abr26/out/resumen_estructural_grafos_ceemdan.md`` (tabla LaTeX sugerida)

Dependencias: ``networkx``, ``ts2vg``, ``pandas``, ``numpy``, ``scipy``, ``scikit-learn``,
y el paquete ``GraphEMD`` (vía ``PYTHONPATH=src/python``), que importa ``torch`` y
``torch_geometric`` al cargar ``graph_imf_transform_utils``.

Ejecución::

    PYTHONPATH=src/python python scripts/GraphEMD/exploracion/ejecutar_metricas_estructurales_grafos_ceemdan_20abr26.py
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


def _diametro_aproximado(G: nx.Graph) -> float:
    """
    Estima el diámetro con dos BFS (peor caso acotado en grafos no ponderados).

    Parameters
    ----------
    G : nx.Graph
        Grafo conexo no vacío.

    Returns
    -------
    float
        Diámetro estimado (exacto si el grafo es un árbol o cumple propiedades
        del algoritmo de doble barrido).
    """
    if G.number_of_nodes() == 0:
        return float("nan")
    if G.number_of_edges() == 0:
        return 0.0
    u = next(iter(G.nodes))
    d1 = nx.single_source_shortest_path_length(G, u)
    v = max(d1, key=d1.get)
    d2 = nx.single_source_shortest_path_length(G, v)
    return float(max(d2.values()))


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


def metricas_hvg(x: np.ndarray) -> dict[str, Any]:
    """
    Métricas estructurales del HVG sobre la serie ``x``.

    Parameters
    ----------
    x : np.ndarray
        Serie de una IMF o del residuo.

    Returns
    -------
    dict[str, Any]
        n, m, densidad, componentes, diámetro (aprox.), grado medio.
    """
    G = _grafo_visibilidad(x, HorizontalVG)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    dens = nx.density(G)
    comp = nx.number_connected_components(G)
    if comp != 1:
        diam = float("nan")
    else:
        diam = _diametro_aproximado(G)
    avg_deg = (2.0 * m / n) if n else 0.0
    return {
        "n": n,
        "m": m,
        "densidad": dens,
        "componentes": comp,
        "diametro": diam,
        "grado_medio": avg_deg,
    }


def metricas_nvg(x: np.ndarray) -> dict[str, Any]:
    """
    Métricas estructurales del NVG sobre la serie ``x``.

    Parameters
    ----------
    x : np.ndarray
        Serie de una IMF o del residuo.

    Returns
    -------
    dict[str, Any]
        n, m, densidad, componentes, diámetro (aprox.), grado medio.
    """
    G = _grafo_visibilidad(x, NaturalVG)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    dens = nx.density(G)
    comp = nx.number_connected_components(G)
    if comp != 1:
        diam = float("nan")
    else:
        diam = _diametro_aproximado(G)
    avg_deg = (2.0 * m / n) if n else 0.0
    return {
        "n": n,
        "m": m,
        "densidad": dens,
        "componentes": comp,
        "diametro": diam,
        "grado_medio": avg_deg,
    }


def metricas_recurrencia(
    x: np.ndarray,
    umbral_percentil: float = 10.0,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Métricas del grafo de recurrencia (matriz simétrica, sin bucles).

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
    dict[str, Any]
        n (nodos embedding), m (aristas no dirigidas), densidad, componentes,
        diámetro si es conexo, grado medio.
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
    G = nx.from_numpy_array(sym)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    dens = nx.density(G)
    comp = nx.number_connected_components(G)
    if comp != 1:
        diam = float("nan")
    else:
        diam = _diametro_aproximado(G)
    avg_deg = (2.0 * m / n) if n else 0.0
    return {
        "n": n,
        "m": m,
        "densidad": dens,
        "componentes": comp,
        "diametro": diam,
        "grado_medio": avg_deg,
    }


def _formatear_rango_resumen(r: dict[str, float], *, entero: bool) -> str:
    """
    Formatea un par min--max para la tabla de resumen exportada.

    Parameters
    ----------
    r : dict[str, float]
        Diccionario con claves ``min`` y ``max``.
    entero : bool
        Si es True, redondea a enteros (p. ej. número de aristas).

    Returns
    -------
    str
        Cadena ``min - max`` lista para CSV o LaTeX.
    """
    if entero:
        return f"{int(round(r['min']))} - {int(round(r['max']))}"
    return f"{r['min']:.6g} - {r['max']:.6g}"


def _rango_columna(df: pd.DataFrame, nombre_columna: str) -> dict[str, float]:
    """
    Devuelve mínimo y máximo de una columna, ignorando NaN en métricas de diámetro.

    Parameters
    ----------
    df : pd.DataFrame
        Tabla detallada por componente.
    nombre_columna : str
        Nombre de la columna (p. ej. ``hvg_densidad``).

    Returns
    -------
    dict[str, float]
        Claves ``min`` y ``max``.
    """
    arr = np.asarray(
        pd.to_numeric(df[nombre_columna], errors="coerce"), dtype=np.float64
    )
    return {
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
    }


def main() -> None:
    """
    Punto de entrada: calcula CSV de métricas y resumen para el documento LaTeX.
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
        mh = metricas_hvg(x)
        mn = metricas_nvg(x)
        mr = metricas_recurrencia(x)
        filas.append(
            {
                "componente": nombre,
                "hvg_n": mh["n"],
                "hvg_m": mh["m"],
                "hvg_densidad": mh["densidad"],
                "hvg_componentes": mh["componentes"],
                "hvg_diametro": mh["diametro"],
                "hvg_grado_medio": mh["grado_medio"],
                "nvg_n": mn["n"],
                "nvg_m": mn["m"],
                "nvg_densidad": mn["densidad"],
                "nvg_componentes": mn["componentes"],
                "nvg_diametro": mn["diametro"],
                "nvg_grado_medio": mn["grado_medio"],
                "rec_n": mr["n"],
                "rec_m": mr["m"],
                "rec_densidad": mr["densidad"],
                "rec_componentes": mr["componentes"],
                "rec_diametro": mr["diametro"],
                "rec_grado_medio": mr["grado_medio"],
            }
        )

    det = pd.DataFrame(filas)
    out_dir = _REPO_ROOT / "docs" / "20abr26" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    det_path = out_dir / "metricas_estructurales_por_componente_ceemdan.csv"
    det.to_csv(det_path, index=False)
    logger.info("Detalle: %s", det_path)

    sufijos = [
        ("densidad", "Density (range)"),
        ("m", "Number of links (range)"),
        ("componentes", "Connected components"),
        ("diametro", "Diameter (range)"),
        ("grado_medio", "Average degree (range)"),
    ]
    filas_res: list[dict[str, str]] = []
    for suf, etiqueta in sufijos:
        rh = _rango_columna(det, f"hvg_{suf}")
        rr = _rango_columna(det, f"rec_{suf}")
        rn = _rango_columna(det, f"nvg_{suf}")
        es_entero = suf in ("m", "componentes")
        filas_res.append(
            {
                "property": etiqueta,
                "HVG": _formatear_rango_resumen(rh, entero=es_entero),
                "Recurrence": _formatear_rango_resumen(rr, entero=es_entero),
                "NVG": _formatear_rango_resumen(rn, entero=es_entero),
            }
        )
    res_df = pd.DataFrame(filas_res)
    res_path = out_dir / "resumen_estructural_grafos_ceemdan.csv"
    res_df.to_csv(res_path, index=False)
    logger.info("Resumen: %s", res_path)

    # Texto para pegar en LaTeX (4 decimales en densidades y grados cuando aplica)
    md_lines = [
        "# Resumen estructural (CEEMDAN)",
        "",
        res_df.to_string(index=False),
        "",
        "Valores numéricos completos en `metricas_estructurales_por_componente_ceemdan.csv`.",
    ]
    md_path = out_dir / "resumen_estructural_grafos_ceemdan.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Markdown: %s", md_path)


if __name__ == "__main__":
    main()
