"""
Reproduce las salidas usadas en ``docs/20abr26/main.tex`` (subsección *Graph transformation*):

- Tabla de parámetros de embedding y umbral $\\varepsilon$ para grafos de recurrencia
  por IMF/residuo (CEEMDAN).
- Figuras ``docs/20abr26/images/english/hvg_imf.png`` e
  ``imf_decomposition.png`` (CEEMDAN + EEMD en la misma figura; ver
  ``exportar_figuras_documento_20abr26`` en ``info_msci_world_data.py``).

Orígenes en el repositorio
--------------------------
- **HVG por IMF / NVG / tamaños**: ``info_msci_world_data.py`` (CEEMDAN + ``calcular_tamaño_grafos_por_imf``).
- **Exploración interactiva HVG CEEMDAN**:
  ``analysis/20abr26/04_transformacion_imf_grafo_ceemdan/042_transformacion_imf_grafo_hvg_ceemdan.ipynb``.
- **Funciones de recurrencia** ($\\tau$, FNN, umbral): ``GraphEMD.data.graph_imf_transform_utils``.

Ejecución típica (desde la raíz del repo)::

    PYTHONPATH=src/python python scripts/GraphEMD/exploracion/ejecutar_salidas_subseccion_grafos_ceemdan_20abr26.py

Las salidas CSV/Markdown se escriben en ``docs/20abr26/out/``.
"""

from __future__ import annotations

import contextlib
import io
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Raíz del repositorio GraphEMD (scripts/GraphEMD/exploracion -> 4 niveles arriba)
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

# Importar generación de figuras desde el script hermano
_EXPLORACION = Path(__file__).resolve().parent
if str(_EXPLORACION) not in sys.path:
    sys.path.insert(0, str(_EXPLORACION))

from info_msci_world_data import (  # noqa: E402
    EEMD_AVAILABLE,
    exportar_figuras_documento_20abr26,
    obtener_imfs_ceemdan,
    obtener_imfs_eemd,
)

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)


def _ruta_imfs_ceemdan() -> Path:
    """
    Devuelve la ruta preferida al parquet de IMFs CEEMDAN.

    Returns
    -------
    Path
        Ruta a ``data/20abr26/msci_world_imfs_ceemdan.parquet``.
    """
    return _REPO_ROOT / "data" / "20abr26" / "msci_world_imfs_ceemdan.parquet"


def _ruta_imfs_eemd() -> Path:
    """
    Devuelve la ruta al parquet de IMFs EEMD (MSCI World).

    Returns
    -------
    Path
        Ruta a ``data/20abr26/msci_world_imfs_eemd.parquet``.
    """
    return _REPO_ROOT / "data" / "20abr26" / "msci_world_imfs_eemd.parquet"


def _ruta_precios() -> Path:
    """
    Devuelve la ruta a la serie de precios de cierre.

    Returns
    -------
    Path
        Ruta a ``data/20abr26/msci_world.parquet``.
    """
    return _REPO_ROOT / "data" / "20abr26" / "msci_world.parquet"


def calcular_tabla_parametros_recurrencia(
    df_imfs: pd.DataFrame,
    umbral_percentil: float = 10.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Calcula $\\tau$, dimensión de embedding $d$ y umbral $\\varepsilon$ por componente.

    Misma lógica que ``obtener_grafo_recurrencia_imf`` (sin construir el objeto Data completo).

    Parameters
    ----------
    df_imfs : pd.DataFrame
        Columnas ``IMF_1`` ... ``IMF_n`` y ``Residuo``.
    umbral_percentil : float, optional
        Percentil para el umbral de distancias. Por defecto 10.
    random_state : int, optional
        Semilla para el cálculo del umbral. Por defecto 42.

    Returns
    -------
    pd.DataFrame
        Columnas: componente, tau, d, epsilon.
    """
    buf = io.StringIO()
    filas: list[dict[str, object]] = []
    columnas = [c for c in df_imfs.columns if c.startswith("IMF_") or c == "Residuo"]
    for nombre in columnas:
        x = np.asarray(df_imfs[nombre].values, dtype=np.float64).copy()
        tau = int(seleccionar_tau(x, tau_max=50))
        d = int(calcular_false_nearest_neighbors(x, tau=tau, dim_max=10))
        emb = construir_espacio_embedding(x, d, tau)
        with contextlib.redirect_stdout(buf):
            _, eps = calcular_matriz_recurrencia(
                emb,
                umbral_percentil=umbral_percentil,
                random_state=random_state,
            )
        filas.append({"componente": nombre, "tau": tau, "d": d, "epsilon": float(eps)})
    return pd.DataFrame(filas)


def generar_markdown_tabla_recurrencia(df_params: pd.DataFrame) -> str:
    """
    Genera un fragmento LaTeX/tabular Markdown con los parámetros calculados.

    Parameters
    ----------
    df_params : pd.DataFrame
        Salida de :func:`calcular_tabla_parametros_recurrencia`.

    Returns
    -------
    str
        Texto Markdown con tabla y bloque ``tabular`` copiable.
    """
    lineas = [
        "# Parámetros de recurrencia (CEEMDAN)",
        "",
        "| Componente | tau | d | epsilon |",
        "|------------|-----|---|---------|",
    ]
    for row in df_params.itertuples(index=False):
        lineas.append(
            f"| {getattr(row, 'componente')} | {getattr(row, 'tau')} | "
            f"{getattr(row, 'd')} | {getattr(row, 'epsilon'):.4f} |"
        )
    lineas.extend(["", "## Fragmento LaTeX (tabular)", "```"])
    lineas.append(r"\begin{tabular}{lccc}")
    lineas.append(r"\toprule")
    lineas.append(
        r"\textbf{Component} & \textbf{$\tau$} & \textbf{$d$} & \textbf{$\varepsilon$} \\"
    )
    lineas.append(r"\midrule")
    for row in df_params.itertuples(index=False):
        comp = getattr(row, "componente")
        if comp == "Residuo":
            comp_tex = "Residue"
        else:
            comp_tex = comp.replace("_", r"\_")
        lineas.append(
            f"{comp_tex} & {getattr(row, 'tau')} & {getattr(row, 'd')} & "
            f"{getattr(row, 'epsilon'):.4f} \\\\"
        )
    lineas.append(r"\bottomrule")
    lineas.append(r"\end{tabular}")
    lineas.append("```")
    return "\n".join(lineas)


def main(parquet_imfs: Optional[Path] = None, regenerar_figuras: bool = True) -> None:
    """
    Escribe CSV/Markdown de parámetros de recurrencia y opcionalmente regenera figuras.

    Parameters
    ----------
    parquet_imfs : Path, optional
        Parquet con IMFs. Si es None, usa ``data/20abr26/msci_world_imfs_ceemdan.parquet``
        o ejecuta CEEMDAN sobre la serie de precios si falta.
    regenerar_figuras : bool, optional
        Si True, escribe ``hvg_imf.png`` e ``imf_decomposition.png`` en
        ``docs/20abr26/images/english/``.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out_dir = _REPO_ROOT / "docs" / "20abr26" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    ruta_imfs = parquet_imfs or _ruta_imfs_ceemdan()
    ruta_precios = _ruta_precios()

    if not ruta_imfs.is_file():
        logger.info("No existe %s; calculando CEEMDAN...", ruta_imfs)
        if not ruta_precios.is_file():
            raise FileNotFoundError(
                f"Se necesita {ruta_precios} o un parquet de IMFs existente."
            )
        serie = np.asarray(
            pd.read_parquet(ruta_precios, engine="pyarrow")["Close"].values,
            dtype=np.float64,
        ).copy()
        df_imfs = obtener_imfs_ceemdan(serie)
        ruta_imfs.parent.mkdir(parents=True, exist_ok=True)
        df_imfs.to_parquet(ruta_imfs, index=False)
        logger.info("Guardado %s", ruta_imfs)
    else:
        df_imfs = pd.read_parquet(ruta_imfs, engine="pyarrow")

    logger.info("Calculando parámetros de recurrencia (%d filas)...", len(df_imfs))
    df_params = calcular_tabla_parametros_recurrencia(df_imfs)
    csv_path = out_dir / "parametros_recurrencia_ceemdan.csv"
    df_params.to_csv(csv_path, index=False)
    logger.info("CSV: %s", csv_path)

    md_path = out_dir / "parametros_recurrencia_ceemdan.md"
    md_path.write_text(
        generar_markdown_tabla_recurrencia(df_params) + "\n", encoding="utf-8"
    )
    logger.info("Markdown: %s", md_path)

    readme = out_dir / "README_fuentes_subseccion_grafos.txt"
    readme.write_text(
        "\n".join(
            [
                "Fuentes para docs/20abr26 main.tex (Graph transformation)",
                "",
                "1) Figuras hvg_imf.png e imf_decomposition.png:",
                "   - exportar_figuras_documento_20abr26() en info_msci_world_data.py",
                "   - ejecutar_salidas_subseccion_grafos_ceemdan_20abr26.py --figuras",
                "",
                "2) Tabla tau, d, epsilon (recurrencia):",
                "   - GraphEMD/data/graph_imf_transform_utils.py (seleccionar_tau, FNN, umbral)",
                "   - salida: docs/20abr26/out/parametros_recurrencia_ceemdan.csv",
                "",
                "3) Notebook exploratorio HVG:",
                "   - analysis/20abr26/04_transformacion_imf_grafo_ceemdan/042_transformacion_imf_grafo_hvg_ceemdan.ipynb",
                "",
            ]
        ),
        encoding="utf-8",
    )
    logger.info("README: %s", readme)

    if regenerar_figuras:
        if not ruta_precios.is_file():
            logger.warning("No hay %s; omitiendo figuras.", ruta_precios)
        else:
            serie = np.asarray(
                pd.read_parquet(ruta_precios, engine="pyarrow")["Close"].values,
                dtype=np.float64,
            ).copy()
            img_dir = _REPO_ROOT / "docs" / "20abr26" / "images" / "english"
            df_eemd = None
            if EEMD_AVAILABLE:
                logger.info("Ejecutando EEMD (parámetros docs/16dic25)...")
                df_eemd = obtener_imfs_eemd(serie)
                ruta_eemd = _ruta_imfs_eemd()
                ruta_eemd.parent.mkdir(parents=True, exist_ok=True)
                df_eemd.to_parquet(ruta_eemd, index=False)
                logger.info("Guardado %s", ruta_eemd)
            else:
                logger.warning("PyEMD sin EEMD; figura solo CEEMDAN.")
            exportar_figuras_documento_20abr26(
                df_imfs, serie, img_dir, df_imfs_eemd=df_eemd
            )
            logger.info("Figuras en %s", img_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Salidas CEEMDAN para la subsección Graph transformation (docs/20abr26)."
    )
    parser.add_argument(
        "--parquet-imfs",
        type=Path,
        default=None,
        help="Parquet con IMF_1,...,Residuo (por defecto data/20abr26/msci_world_imfs_ceemdan.parquet).",
    )
    parser.add_argument(
        "--sin-figuras",
        action="store_true",
        help="No regenerar PNG en docs/20abr26/images/english/",
    )
    args = parser.parse_args()
    main(parquet_imfs=args.parquet_imfs, regenerar_figuras=not args.sin_figuras)
