"""
Script to apply graph transformations to CEEMDAN IMF components of XLP (Consumer Staples Select Sector SPDR Fund).

1. Load data required to apply graph transformations to IMF components of XLP. On one hand, IMF components from the CEEMDAN CEEMDAN decomposition of XLP and on the other hand graph data from graph transformations applied to IMF components of XLP.
2. Apply graph transformations to XLP CEEMDAN IMF components.
3. Document implemented logic and results in this script.
4. Do not change any existing code for now

Implemented logic
---------------------
- **Load**: ``xlp_imfs_ceemdan.parquet`` (8 IMFs + ``Residuo``, 3587 observations; output from
  ``03_ceemdan_xlp.py``).
- **Visibility** (``ts2vg``, same convention as MSCI in ``graph_imf_transform_utils``):
  - **HVG** (Horizontal Visibility Graph): edges between points with horizontal visibility
    (``HorizontalVG``, directed ``left_to_right``).
  - **NVG** (Natural Visibility Graph): natural visibility (``NaturalVG``).
- **Recurrence** (delay embedding + distance percentile threshold, ``random_state=42``):
  ``tau`` selection (mutual information), dimension ``d`` (false nearest neighbours),
  ``epsilon`` threshold at the 10th percentile of distances in embedded space; undirected graph
  with nodes = time points and edges where Euclidean distance ``< epsilon``.
- **Implementation**: ``build_all_imf_graphs`` saves per component parquet + ``.pt``
  (PyTorch Geometric) in ``data/GraphEMD/xlp_analysis/grafos/{hvg,nvg,recurrencia}/``.
- **Auxiliary table**: ``xlp_parametros_recurrencia_ceemdan.csv`` (``tau``, ``d``, ``epsilon`` per IMF).

Results obtained (run 2026-05-16, n=3587)
-------------------------------------------------

**9 components** (IMF_1–IMF_8 + Residuo): HVG, NVG, and recurrence generated without error.

**Visibility:** each series has **3587 nodes** (one node per time point). HVG ≈ 5.9–7.2×10³
edges; NVG grows with IMF scale (IMF_1 ≈ 10⁴, Residuo ≈ 4.15×10⁶ edges).

**Recurrence** (``tau``, ``d``, ``epsilon`` in ``xlp_parametros_recurrencia_ceemdan.csv``):
IMF_1–IMF_5 with ``d=4``; IMF_6 ``d=3``; IMF_7–IMF_8 and Residual ``d=2``; ``tau`` increases
in slow modes (IMF_7: 41, Residual: 21). Embedding nodes ≈ 3529–3581; edges
≈ 3.3×10⁴–3.5×10⁴ per component.

**Outputs:** ``grafos/{hvg,nvg,recurrencia}/``, ``xlp_resumen_grafos_imf.csv``,
``xlp_parametros_recurrencia_ceemdan.csv``, ``xlp_grafos_imf_manifest.json``.
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC_PYTHON = _REPO_ROOT / "src" / "python"
_EXPLORATION = _REPO_ROOT / "scripts" / "exploration"
_DIR_DATOS = _REPO_ROOT / "data" / "GraphEMD" / "xlp_analysis"
_RUTA_IMFS_XLP = _DIR_DATOS / "xlp_imfs_ceemdan.parquet"
_DIR_GRAFOS = _DIR_DATOS / "grafos"
_RUTA_RESUMEN_CSV = _DIR_DATOS / "xlp_resumen_grafos_imf.csv"
_RUTA_PARAM_RECURRENCIA_CSV = _DIR_DATOS / "xlp_parametros_recurrencia_ceemdan.csv"
_RUTA_RESUMEN_JSON = _DIR_DATOS / "xlp_grafos_imf_manifest.json"

# Same values as MSCI (``run_graph_subsection_outputs_ceemdan_20abr26.py``)
TAU_MAX: int = 50
DIM_MAX: int = 10
UMBRAL_PERCENTIL_RECURRENCIA: float = 10.0
RANDOM_STATE_RECURRENCIA: int = 42

if str(_SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(_SRC_PYTHON))
if str(_EXPLORATION) not in sys.path:
    sys.path.insert(0, str(_EXPLORATION))

from GraphEMD.data.graph_imf_transform_utils import build_all_imf_graphs  # noqa: E402
from run_graph_subsection_outputs_ceemdan_20abr26 import (  # noqa: E402
    compute_recurrence_params_table,
)

logger = logging.getLogger(__name__)


def cargar_imfs_ceemdan_xlp(
    ruta_imfs: Path = _RUTA_IMFS_XLP,
) -> pd.DataFrame:
    """
    Load the CEEMDAN IMF parquet for XLP.

    Parameters
    ----------
    ruta_imfs : Path
        Path to ``xlp_imfs_ceemdan.parquet``.

    Returns
    -------
    pd.DataFrame
        Columns ``IMF_1`` … ``IMF_n`` and ``Residuo``.

    Raises
    ------
    FileNotFoundError
        If the IMF file does not exist.
    ValueError
        If there are no IMF or Residuo columns.
    """
    if not ruta_imfs.is_file():
        raise FileNotFoundError(
            f"Not found: {ruta_imfs}. Run 03_ceemdan_xlp.py first."
        )
    df_imfs = pd.read_parquet(ruta_imfs, engine="pyarrow")
    columnas = [c for c in df_imfs.columns if c.startswith("IMF_") or c == "Residuo"]
    if not columnas:
        raise ValueError(
            f"No IMF/Residuo columns in {ruta_imfs}. Columns: {list(df_imfs.columns)}"
        )
    logger.info(
        "XLP IMFs loaded: %d rows, components %s",
        len(df_imfs),
        columnas,
    )
    return df_imfs


def resumir_resultados_grafos(resultados: dict[str, Any]) -> pd.DataFrame:
    """
    Convert the ``build_all_imf_graphs`` dictionary to a flat table.

    Parameters
    ----------
    resultados : dict
        Output of :func:`build_all_imf_graphs`.

    Returns
    -------
    pd.DataFrame
        One row per component with nodes/edges per graph type.
    """
    filas: list[dict[str, Any]] = []
    for id_imf, tipos in resultados.items():
        fila: dict[str, Any] = {"componente": id_imf}
        for tipo_grafo in ("hvg", "nvg", "recurrencia"):
            info = tipos.get(tipo_grafo, {})
            if isinstance(info, dict) and info.get("exito"):
                fila[f"nodos_{tipo_grafo}"] = info.get("num_nodes")
                fila[f"enlaces_{tipo_grafo}"] = info.get("num_edges")
                if tipo_grafo == "recurrencia":
                    fila["tau"] = info.get("tau")
                    fila["dim_embedding"] = info.get("dim_embedding")
                    fila["epsilon"] = info.get("umbral_recurrencia")
            else:
                fila[f"error_{tipo_grafo}"] = (
                    info.get("error", "failure") if isinstance(info, dict) else "failure"
                )
        filas.append(fila)
    return pd.DataFrame(filas)


def aplicar_transformaciones_grafos(
    ruta_imfs: Path = _RUTA_IMFS_XLP,
    dir_grafos: Path = _DIR_GRAFOS,
    tau_max: int = TAU_MAX,
    dim_max: int = DIM_MAX,
    umbral_percentil: float = UMBRAL_PERCENTIL_RECURRENCIA,
    random_state: int = RANDOM_STATE_RECURRENCIA,
) -> dict[str, Any]:
    """
    Generate HVG, NVG, and recurrence graphs for each IMF/residual of XLP.

    Parameters
    ----------
    ruta_imfs : Path
        Parquet with CEEMDAN IMFs.
    dir_grafos : Path
        Base output directory (``hvg``, ``nvg``, ``recurrencia`` subfolders).
    tau_max : int
        Maximum delay for recurrence.
    dim_max : int
        Maximum embedding dimension (FNN).
    umbral_percentil : float
        Percentile for the distance threshold in recurrence.
    random_state : int
        Seed for the recurrence threshold.

    Returns
    -------
    dict
        Per-component results (same structure as ``build_all_imf_graphs``).
    """
    dir_grafos.mkdir(parents=True, exist_ok=True)
    logger.info("Generating graphs in %s ...", dir_grafos)
    return build_all_imf_graphs(
        df_imfs=str(ruta_imfs),
        carpeta_salida_base=str(dir_grafos),
        tau_max=tau_max,
        dim_max=dim_max,
        umbral_percentil=umbral_percentil,
        random_state=random_state,
    )


def guardar_tabla_recurrencia(
    df_imfs: pd.DataFrame,
    ruta_csv: Path = _RUTA_PARAM_RECURRENCIA_CSV,
    umbral_percentil: float = UMBRAL_PERCENTIL_RECURRENCIA,
    random_state: int = RANDOM_STATE_RECURRENCIA,
) -> pd.DataFrame:
    """
    Compute and save tau, d, and epsilon per component (without rebuilding graphs).

    Parameters
    ----------
    df_imfs : pd.DataFrame
        CEEMDAN IMFs for XLP.
    ruta_csv : Path
        Output CSV path.
    umbral_percentil : float
        Distance threshold percentile.
    random_state : int
        Seed for threshold computation.

    Returns
    -------
    pd.DataFrame
        Recurrence parameter table.
    """
    df_params = compute_recurrence_params_table(
        df_imfs,
        umbral_percentil=umbral_percentil,
        random_state=random_state,
    )
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    df_params.to_csv(ruta_csv, index=False)
    logger.info("Recurrence parameters: %s", ruta_csv)
    return df_params


def guardar_manifest(
    resultados: dict[str, Any],
    df_resumen: pd.DataFrame,
    df_params: pd.DataFrame,
    ruta_json: Path = _RUTA_RESUMEN_JSON,
) -> None:
    """
    Write a JSON with paths and metrics of generated graphs.

    Parameters
    ----------
    resultados : dict
        Output of :func:`aplicar_transformaciones_grafos`.
    df_resumen : pd.DataFrame
        Summary table per component.
    df_params : pd.DataFrame
        Recurrence embedding parameters.
    ruta_json : Path
        Output JSON file.
    """
    payload = {
        "ruta_imfs": str(_RUTA_IMFS_XLP),
        "dir_grafos": str(_DIR_GRAFOS),
        "parametros_recurrencia": df_params.to_dict(orient="records"),
        "resumen_por_componente": df_resumen.to_dict(orient="records"),
        "detalle_grafos": resultados,
    }
    ruta_json.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_json, "w", encoding="utf-8") as archivo:
        json.dump(payload, archivo, indent=2, ensure_ascii=False)
    logger.info("Manifiesto: %s", ruta_json)


def main(
    ruta_imfs: Optional[Path] = None,
    dir_grafos: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Run the IMF → graph transformation pipeline for XLP.

    Parameters
    ----------
    ruta_imfs : Path, optional
        IMF parquet. Default ``xlp_imfs_ceemdan.parquet``.
    dir_grafos : Path, optional
        Base graph folder. Default ``data/.../grafos``.

    Returns
    -------
    dict
        Output paths and summary tables.
    """
    ruta = ruta_imfs or _RUTA_IMFS_XLP
    carpeta = dir_grafos or _DIR_GRAFOS

    df_imfs = cargar_imfs_ceemdan_xlp(ruta)
    df_params = guardar_tabla_recurrencia(df_imfs)

    resultados = aplicar_transformaciones_grafos(
        ruta_imfs=ruta,
        dir_grafos=carpeta,
    )
    df_resumen = resumir_resultados_grafos(resultados)
    _RUTA_RESUMEN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_resumen.to_csv(_RUTA_RESUMEN_CSV, index=False)
    logger.info("Graph summary: %s", _RUTA_RESUMEN_CSV)

    guardar_manifest(resultados, df_resumen, df_params)

    return {
        "ruta_imfs": str(ruta),
        "dir_grafos": str(carpeta),
        "ruta_resumen_csv": str(_RUTA_RESUMEN_CSV),
        "ruta_param_recurrencia_csv": str(_RUTA_PARAM_RECURRENCIA_CSV),
        "ruta_manifest_json": str(_RUTA_RESUMEN_JSON),
        "resumen": df_resumen,
        "parametros_recurrencia": df_params,
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="IMF→graph transformations (HVG, NVG, recurrence) for XLP CEEMDAN."
    )
    parser.add_argument(
        "--parquet-imfs",
        type=Path,
        default=None,
        help="Parquet with IMF_1,...,Residuo (default xlp_imfs_ceemdan.parquet).",
    )
    parser.add_argument(
        "--dir-grafos",
        type=Path,
        default=None,
        help="Base output directory for graphs.",
    )
    args = parser.parse_args()
    try:
        salida = main(ruta_imfs=args.parquet_imfs, dir_grafos=args.dir_grafos)
        logger.info("Done. Summary: %s", salida["ruta_resumen_csv"])
    except Exception:
        logger.exception("Error in graph transformations for XLP")
        sys.exit(1)
