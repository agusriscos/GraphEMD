"""
Script para obtener las metricas de los grafos obtenidos en la aplicacion de las transformaciones de grafos a los componentes IMFs de la ETF XLV.

1. Carga los datos necesarios para la obtencion de las metricas de los grafos obtenidos en la aplicacion de las transformaciones de grafos a los componentes IMFs de la ETF XLV.
2. Obten las metricas de densidad, numero de enlaces, componentes conexas, diametro y grado medio de los grafos obtenidos en la aplicacion de las transformaciones de grafos a los componentes IMFs de la ETF XLV.
3. Calcula los coeficientes de clustering (media, mediana y desviacion estandar) de los grafos obtenidos en la aplicacion de las transformaciones de grafos a los componentes IMFs de la ETF XLV.
4. Calcula las centralidades (degree, betweenness, closeness y eigenvector) de los grafos obtenidos en la aplicacion de las transformaciones de grafos a los componentes IMFs de la ETF XLV.
5. Documenta en este mismo script la logica implementada y los resultados obtenidos.
6. No cambies por el momento ningun codigo que ya exista

Lógica implementada
---------------------
- **Carga**: grafos guardados por ``05_graph_transformations_xlv.py`` en
  ``data/GraphEMD/xlv_analysis/grafos/{hvg,nvg,recurrencia}/{imf_k,residuo}/*.pt``
  (PyTorch Geometric → ``networkx.Graph``).
- **Estructurales** (por grafo): densidad, número de enlaces, componentes conexas,
  diámetro (doble BFS en la componente conexa mayor si el grafo es conexo; NaN si no),
  grado medio ``2m/n``.
- **Clustering**: media (``average_clustering``), mediana y desviación típica de los
  coeficientes locales; omitido en NVG con ``m > 4×10⁵`` (grafos demasiado densos).
- **Centralidades**: media de grado normalizado, intermediación, cercanía y vector propio;
  mismos umbrales que ``ejecutar_centralidad_grafos_ceemdan_20abr26.py`` (muestreo /
  omisión en NVG muy denso).

Resultados obtenidos (ejecución 2026-05-17, 27 grafos = 9×3)
-------------------------------------------------------------

**HVG:** todos conexos (1 componente); diámetro crece en modos lentos (IMF_8 ≈ 1306,
Residuo ≈ 2036); clustering medio ≈ 0.33–0.66.

**NVG IMF_1–IMF_6:** métricas completas; densidad hasta ≈ 0.035 (IMF_6). **NVG IMF_7+**:
clustering omitido (``m > 4×10⁵``); **IMF_8** y **Residuo**: centralidad omitida
(``m > 1.5×10⁶`` y ``> 2×10⁶`` respectivamente); estructurales sí calculadas
(Residuo: densidad ≈ 0.65, grado medio ≈ 2316).

**Recurrencia:** muchas componentes conexas (salvo IMF_6 conexo, diámetro ≈ 332);
intermediación/cercanía aproximadas si ``m > 1.2×10⁴``.

**Salidas:** ``xlv_metricas_grafos_imf.csv``, ``xlv_resumen_metricas_grafos.json``.
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.sparse.linalg import ArpackNoConvergence

warnings.filterwarnings("ignore", category=UserWarning)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DIR_DATOS = _REPO_ROOT / "data" / "GraphEMD" / "xlv_analysis"
_DIR_GRAFOS = _DIR_DATOS / "grafos"
_RUTA_METRICAS_CSV = _DIR_DATOS / "xlv_metricas_grafos_imf.csv"
_RUTA_RESUMEN_JSON = _DIR_DATOS / "xlv_resumen_metricas_grafos.json"

TIPOS_GRAFO: tuple[str, ...] = ("hvg", "nvg", "recurrencia")

# Umbrales alineados con scripts MSCI (exploracion/); clustering NVG acotado
# por coste en grafos densos (IMF_7+ en XLV superan ~4×10⁵ aristas).
MAX_ARISTAS_NVG_CLUSTERING = 400_000
MAX_ARISTAS_NVG_CENTRALIDAD = 1_500_000
UMBRAL_ARISTAS_BETWEENNESS_APROX = 200_000
K_MUESTRA_BETWEENNESS = 600
SEMILLA_BETWEENNESS = 42
UMBRAL_ARISTAS_CERCANIA_EXACTA = 50_000
K_MUESTRA_CERCANIA = 500
SEMILLA_CERCANIA = 42
UMBRAL_ARISTAS_RECURRENCIA_APROX = 12_000
N_MAX_EIGENVECTOR_NUMPY = 4096

logger = logging.getLogger(__name__)


def id_carpeta_a_componente(id_carpeta: str) -> str:
    """
    Convierte el nombre de carpeta (``imf_1``, ``residuo``) al id de componente.

    Parameters
    ----------
    id_carpeta : str
        Nombre del subdirectorio bajo ``grafos/<tipo>/``.

    Returns
    -------
    str
        ``IMF_1``, …, ``IMF_8`` o ``Residuo``.
    """
    if id_carpeta == "residuo":
        return "Residuo"
    if id_carpeta.startswith("imf_"):
        sufijo = id_carpeta.split("_", 1)[1]
        return f"IMF_{sufijo}"
    return id_carpeta


def listar_componentes_grafos(dir_grafos: Path = _DIR_GRAFOS) -> list[str]:
    """
    Lista componentes disponibles a partir de las carpetas HVG.

    Parameters
    ----------
    dir_grafos : Path
        Directorio base ``grafos/``.

    Returns
    -------
    list[str]
        Nombres de componente ordenados (``IMF_1``, …, ``Residuo``).
    """
    carpeta_hvg = dir_grafos / "hvg"
    if not carpeta_hvg.is_dir():
        raise FileNotFoundError(
            f"No existe {carpeta_hvg}. Ejecute antes 05_graph_transformations_xlv.py."
        )
    ids = sorted(p.name for p in carpeta_hvg.iterdir() if p.is_dir())
    return [id_carpeta_a_componente(n) for n in ids]


def ruta_archivo_grafo(
    dir_grafos: Path,
    tipo_grafo: str,
    componente: str,
) -> Path:
    """
    Devuelve la ruta al archivo ``.pt`` del grafo.

    Parameters
    ----------
    dir_grafos : Path
        Directorio base de grafos.
    tipo_grafo : str
        ``hvg``, ``nvg`` o ``recurrencia``.
    componente : str
        ``IMF_k`` o ``Residuo``.

    Returns
    -------
    Path
        Ruta al objeto Data serializado.
    """
    if componente == "Residuo":
        id_carpeta = "residuo"
    else:
        id_carpeta = componente.lower()
    return dir_grafos / tipo_grafo / id_carpeta / f"grafo_{tipo_grafo}_{id_carpeta}.pt"


def cargar_grafo_desde_pt(ruta_pt: Path) -> nx.Graph:
    """
    Carga un grafo PyG desde ``.pt`` y lo convierte a NetworkX no dirigido.

    Parameters
    ----------
    ruta_pt : Path
        Archivo ``grafo_<tipo>_<id>.pt``.

    Returns
    -------
    nx.Graph
        Grafo simple con nodos ``0 … n-1``.

    Raises
    ------
    FileNotFoundError
        Si no existe el archivo.
    """
    if not ruta_pt.is_file():
        raise FileNotFoundError(f"No se encuentra el grafo: {ruta_pt}")
    grafo_data = torch.load(str(ruta_pt), map_location="cpu", weights_only=False)
    grafo = nx.Graph()
    grafo.add_nodes_from(range(int(grafo_data.num_nodes)))
    edge_index = grafo_data.edge_index.cpu().numpy()
    aristas = [
        (int(edge_index[0, i]), int(edge_index[1, i]))
        for i in range(edge_index.shape[1])
    ]
    grafo.add_edges_from(aristas)
    return grafo


def diametro_aproximado(grafo: nx.Graph) -> float:
    """
    Estima el diámetro con dos BFS sobre la componente conexa principal.

    Parameters
    ----------
    grafo : nx.Graph
        Grafo no dirigido.

    Returns
    -------
    float
        Diámetro estimado, 0 si no hay aristas, NaN si hay varias componentes.
    """
    num_nodos = grafo.number_of_nodes()
    if num_nodos == 0:
        return float("nan")
    if grafo.number_of_edges() == 0:
        return 0.0
    if nx.number_connected_components(grafo) != 1:
        return float("nan")
    nodo_inicial = next(iter(grafo.nodes))
    distancias_1 = nx.single_source_shortest_path_length(grafo, nodo_inicial)
    nodo_lejano = max(distancias_1, key=distancias_1.get)
    distancias_2 = nx.single_source_shortest_path_length(grafo, nodo_lejano)
    return float(max(distancias_2.values()))


def metricas_estructurales(grafo: nx.Graph) -> dict[str, Any]:
    """
    Calcula densidad, enlaces, componentes, diámetro y grado medio.

    Parameters
    ----------
    grafo : nx.Graph
        Grafo a analizar.

    Returns
    -------
    dict
        Métricas estructurales del grafo.
    """
    num_nodos = grafo.number_of_nodes()
    num_enlaces = grafo.number_of_edges()
    densidad = float(nx.density(grafo))
    num_componentes = int(nx.number_connected_components(grafo))
    diametro = diametro_aproximado(grafo)
    grado_medio = (2.0 * num_enlaces / num_nodos) if num_nodos else 0.0
    return {
        "num_nodos": num_nodos,
        "num_enlaces": num_enlaces,
        "densidad": densidad,
        "num_componentes_conexas": num_componentes,
        "diametro": diametro,
        "grado_medio": grado_medio,
    }


def metricas_clustering(
    grafo: nx.Graph,
    tipo_grafo: str,
) -> dict[str, Any]:
    """
    Calcula estadísticas del coeficiente de clustering local.

    Parameters
    ----------
    grafo : nx.Graph
        Grafo no dirigido.
    tipo_grafo : str
        Tipo de grafo (para aplicar umbral NVG).

    Returns
    -------
    dict
        Media, mediana, desviación típica u omitido si el grafo es demasiado denso.
    """
    num_enlaces = grafo.number_of_edges()
    if tipo_grafo == "nvg" and num_enlaces > MAX_ARISTAS_NVG_CLUSTERING:
        return {
            "clustering_media": float("nan"),
            "clustering_mediana": float("nan"),
            "clustering_desviacion_tipica": float("nan"),
            "clustering_omitido": True,
        }
    if grafo.number_of_nodes() == 0:
        return {
            "clustering_media": float("nan"),
            "clustering_mediana": float("nan"),
            "clustering_desviacion_tipica": float("nan"),
            "clustering_omitido": False,
        }
    coeficientes = nx.clustering(grafo)
    valores = np.fromiter(coeficientes.values(), dtype=np.float64, count=len(coeficientes))
    return {
        "clustering_media": float(nx.average_clustering(grafo)),
        "clustering_mediana": float(np.median(valores)),
        "clustering_desviacion_tipica": float(np.std(valores, ddof=0)),
        "clustering_omitido": False,
    }


def _media_dict_escalar(mapa: dict[Any, float]) -> float:
    """
    Devuelve la media de un diccionario nodo → escalar.

    Parameters
    ----------
    mapa : dict
        Valores de centralidad por nodo.

    Returns
    -------
    float
        Media o NaN si el mapa está vacío.
    """
    if not mapa:
        return float("nan")
    valores = np.fromiter(mapa.values(), dtype=np.float64, count=len(mapa))
    return float(np.mean(valores))


def _betweenness_media(grafo: nx.Graph, *, aproximar: bool) -> tuple[float, bool]:
    """
    Calcula la media de centralidad de intermediación normalizada.

    Parameters
    ----------
    grafo : nx.Graph
        Grafo no dirigido.
    aproximar : bool
        Si es True, usa muestreo de nodos.

    Returns
    -------
    tuple[float, bool]
        Media y flag de aproximación.
    """
    num_nodos = grafo.number_of_nodes()
    if num_nodos == 0:
        return float("nan"), False
    if aproximar:
        k = min(K_MUESTRA_BETWEENNESS, num_nodos)
        mapa = nx.betweenness_centrality(
            grafo, k=k, normalized=True, seed=SEMILLA_BETWEENNESS
        )
        return _media_dict_escalar(mapa), True
    mapa = nx.betweenness_centrality(grafo, normalized=True)
    return _media_dict_escalar(mapa), False


def _closeness_media(grafo: nx.Graph, *, usar_muestreo: bool) -> tuple[float, bool]:
    """
    Calcula la media de centralidad de cercanía.

    Parameters
    ----------
    grafo : nx.Graph
        Grafo no dirigido.
    usar_muestreo : bool
        Si es True, promedia sobre una muestra de nodos.

    Returns
    -------
    tuple[float, bool]
        Media y flag de muestreo.
    """
    if grafo.number_of_nodes() == 0:
        return float("nan"), False
    if not usar_muestreo:
        return _media_dict_escalar(nx.closeness_centrality(grafo)), False
    rng = np.random.default_rng(SEMILLA_CERCANIA)
    nodos = list(grafo.nodes())
    k = min(K_MUESTRA_CERCANIA, len(nodos))
    muestra = rng.choice(nodos, size=k, replace=False)
    valores: list[float] = []
    for u in muestra:
        cercania = nx.closeness_centrality(grafo, u=int(u))
        valores.append(
            float(cercania) if isinstance(cercania, (int, float)) else float(cercania[u])
        )
    return float(np.mean(valores)), True


def _eigenvector_media(grafo: nx.Graph) -> float:
    """
    Calcula la media de centralidad de vector propio en la componente principal.

    Parameters
    ----------
    grafo : nx.Graph
        Grafo no dirigido.

    Returns
    -------
    float
        Media de eigenvector centrality o NaN si no converge.
    """
    if grafo.number_of_nodes() == 0:
        return float("nan")
    subgrafo: nx.Graph = grafo
    if not nx.is_connected(grafo):
        nodos = max(nx.connected_components(grafo), key=len)
        subgrafo = grafo.subgraph(nodos).copy()
    if subgrafo.number_of_nodes() <= N_MAX_EIGENVECTOR_NUMPY:
        try:
            mapa = nx.eigenvector_centrality_numpy(subgrafo)
            return _media_dict_escalar(mapa)
        except (nx.NetworkXError, np.linalg.LinAlgError, ArpackNoConvergence):
            pass
    try:
        mapa = nx.eigenvector_centrality(subgrafo, max_iter=5000, tol=1e-06)
        return _media_dict_escalar(mapa)
    except (nx.NetworkXError, ArpackNoConvergence):
        return float("nan")


def metricas_centralidad(
    grafo: nx.Graph,
    tipo_grafo: str,
) -> dict[str, Any]:
    """
    Calcula medias de centralidad de grado, intermediación, cercanía y vector propio.

    Parameters
    ----------
    grafo : nx.Graph
        Grafo no dirigido.
    tipo_grafo : str
        ``hvg``, ``nvg`` o ``recurrencia`` (define umbrales de aproximación).

    Returns
    -------
    dict
        Medias de centralidad y flags de aproximación u omisión.
    """
    num_enlaces = grafo.number_of_edges()
    if tipo_grafo == "nvg" and num_enlaces > MAX_ARISTAS_NVG_CENTRALIDAD:
        return {
            "degree_centrality_media": float("nan"),
            "betweenness_centrality_media": float("nan"),
            "closeness_centrality_media": float("nan"),
            "eigenvector_centrality_media": float("nan"),
            "centralidad_omitida": True,
            "betweenness_aproximada": False,
            "closeness_muestreada": False,
        }

    grado_media = _media_dict_escalar(nx.degree_centrality(grafo))

    if tipo_grafo == "recurrencia":
        aproximar_bt = num_enlaces > UMBRAL_ARISTAS_RECURRENCIA_APROX
        muestrear_cl = aproximar_bt
    elif tipo_grafo == "nvg":
        aproximar_bt = num_enlaces > UMBRAL_ARISTAS_BETWEENNESS_APROX
        muestrear_cl = num_enlaces > UMBRAL_ARISTAS_CERCANIA_EXACTA
    else:
        aproximar_bt = False
        muestrear_cl = False

    betweenness_media, bt_aprox = _betweenness_media(grafo, aproximar=aproximar_bt)
    closeness_media, cl_muest = _closeness_media(grafo, usar_muestreo=muestrear_cl)
    eigenvector_media = _eigenvector_media(grafo)

    return {
        "degree_centrality_media": grado_media,
        "betweenness_centrality_media": betweenness_media,
        "closeness_centrality_media": closeness_media,
        "eigenvector_centrality_media": eigenvector_media,
        "centralidad_omitida": False,
        "betweenness_aproximada": bt_aprox,
        "closeness_muestreada": cl_muest,
    }


def calcular_metricas_grafo(
    grafo: nx.Graph,
    componente: str,
    tipo_grafo: str,
) -> dict[str, Any]:
    """
    Agrega métricas estructurales, de clustering y de centralidad de un grafo.

    Parameters
    ----------
    grafo : nx.Graph
        Grafo cargado desde disco.
    componente : str
        Nombre del componente IMF o residuo.
    tipo_grafo : str
        ``hvg``, ``nvg`` o ``recurrencia``.

    Returns
    -------
    dict
        Fila lista para un DataFrame.
    """
    fila: dict[str, Any] = {
        "componente": componente,
        "tipo_grafo": tipo_grafo,
    }
    fila.update(metricas_estructurales(grafo))
    fila.update(metricas_clustering(grafo, tipo_grafo))
    fila.update(metricas_centralidad(grafo, tipo_grafo))
    return fila


def procesar_todos_los_grafos(
    dir_grafos: Path = _DIR_GRAFOS,
) -> pd.DataFrame:
    """
    Calcula métricas para cada componente y tipo de grafo disponible.

    Parameters
    ----------
    dir_grafos : Path
        Directorio base con subcarpetas ``hvg``, ``nvg``, ``recurrencia``.

    Returns
    -------
    pd.DataFrame
        Tabla con una fila por (componente, tipo_grafo).
    """
    componentes = listar_componentes_grafos(dir_grafos)
    filas: list[dict[str, Any]] = []

    for componente in componentes:
        for tipo in TIPOS_GRAFO:
            ruta_pt = ruta_archivo_grafo(dir_grafos, tipo, componente)
            logger.info("Procesando %s — %s ...", componente, tipo.upper())
            grafo = cargar_grafo_desde_pt(ruta_pt)
            filas.append(calcular_metricas_grafo(grafo, componente, tipo))

    return pd.DataFrame(filas)


def guardar_salidas(
    df_metricas: pd.DataFrame,
    ruta_csv: Path = _RUTA_METRICAS_CSV,
    ruta_json: Path = _RUTA_RESUMEN_JSON,
) -> None:
    """
    Persiste CSV detallado y JSON resumen.

    Parameters
    ----------
    df_metricas : pd.DataFrame
        Métricas por componente y tipo.
    ruta_csv : Path
        Ruta del CSV.
    ruta_json : Path
        Ruta del JSON de resumen.
    """
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    df_metricas.to_csv(ruta_csv, index=False)
    logger.info("Métricas guardadas: %s", ruta_csv)

    omitidos_clustering = df_metricas.loc[
        df_metricas["clustering_omitido"] == True,  # noqa: E712
        ["componente", "tipo_grafo"],
    ].to_dict(orient="records")
    omitidos_centralidad = df_metricas.loc[
        df_metricas["centralidad_omitida"] == True,  # noqa: E712
        ["componente", "tipo_grafo"],
    ].to_dict(orient="records")

    payload = {
        "num_filas": int(len(df_metricas)),
        "componentes": sorted(df_metricas["componente"].unique().tolist()),
        "tipos_grafo": sorted(df_metricas["tipo_grafo"].unique().tolist()),
        "clustering_omitido": omitidos_clustering,
        "centralidad_omitida": omitidos_centralidad,
        "metricas": df_metricas.to_dict(orient="records"),
    }
    with open(ruta_json, "w", encoding="utf-8") as archivo:
        json.dump(payload, archivo, indent=2, ensure_ascii=False)
    logger.info("Resumen JSON: %s", ruta_json)


def main(dir_grafos: Optional[Path] = None) -> dict[str, Any]:
    """
    Punto de entrada: calcula y guarda métricas de todos los grafos XLV.

    Parameters
    ----------
    dir_grafos : Path, optional
        Directorio base de grafos. Por defecto ``data/.../grafos``.

    Returns
    -------
    dict
        Rutas de salida y DataFrame de métricas.
    """
    carpeta = dir_grafos or _DIR_GRAFOS
    df_metricas = procesar_todos_los_grafos(carpeta)
    guardar_salidas(df_metricas)
    return {
        "ruta_metricas_csv": str(_RUTA_METRICAS_CSV),
        "ruta_resumen_json": str(_RUTA_RESUMEN_JSON),
        "metricas": df_metricas,
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Métricas estructurales, clustering y centralidad de grafos IMF XLV."
    )
    parser.add_argument(
        "--dir-grafos",
        type=Path,
        default=None,
        help="Directorio base con hvg/nvg/recurrencia (por defecto data/.../grafos).",
    )
    args = parser.parse_args()
    try:
        salida = main(dir_grafos=args.dir_grafos)
        logger.info("Completado: %s", salida["ruta_metricas_csv"])
    except Exception:
        logger.exception("Error al calcular métricas de grafos XLV")
        sys.exit(1)
