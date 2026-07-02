"""
Reproduce outputs used in ``docs/20abr26/main.tex`` (*Graph transformation* subsection):

- Table of embedding parameters and threshold $\\varepsilon$ for recurrence graphs
  per IMF/residue (CEEMDAN).
- Figures ``docs/20abr26/images/english/hvg_imf.png`` and
  ``imf_decomposition.png`` (CEEMDAN + EEMD in the same figure; see
  ``exportar_figuras_documento_20abr26`` in ``info_msci_world_data.py``).

Repository sources
------------------
- **HVG per IMF / NVG / sizes**: ``info_msci_world_data.py`` (CEEMDAN + ``compute_graph_sizes_per_imf``).
- **Interactive CEEMDAN HVG exploration**:
  ``analysis/20abr26/04_ceemdan_imf_to_graph_transform/042_ceemdan_imf_to_graph_hvg_transform.ipynb``.
- **Recurrence functions** ($\\tau$, FNN, threshold): ``GraphEMD.data.graph_imf_transform_utils``.

Typical execution (from repo root)::

    PYTHONPATH=src/python python scripts/exploration/run_graph_subsection_outputs_ceemdan_20abr26.py

CSV/Markdown outputs are written to ``docs/20abr26/out/``.
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

# GraphEMD repository root (scripts/exploration -> 4 levels up)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC_PYTHON = _REPO_ROOT / "src" / "python"
if str(_SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(_SRC_PYTHON))

from GraphEMD.data.graph_imf_transform_utils import (
    calcular_false_nearest_neighbors,
    calcular_matriz_recurrencia,
    construir_espacio_embedding,
    seleccionar_tau,
)

# Import figure generation from sibling script
_EXPLORATION = Path(__file__).resolve().parent
if str(_EXPLORATION) not in sys.path:
    sys.path.insert(0, str(_EXPLORATION))

from info_msci_world_data import (  # noqa: E402
    EEMD_AVAILABLE,
    exportar_figuras_documento_20abr26,
    extract_ceemdan_imfs,
    extract_eemd_imfs,
)

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)


def _ruta_imfs_ceemdan() -> Path:
    """
    Return the preferred path to the CEEMDAN IMFs parquet.

    Returns
    -------
    Path
        Path to ``data/20abr26/msci_world_imfs_ceemdan.parquet``.
    """
    return _REPO_ROOT / "data" / "20abr26" / "msci_world_imfs_ceemdan.parquet"


def _ruta_imfs_eemd() -> Path:
    """
    Return the path to the EEMD IMFs parquet (MSCI World).

    Returns
    -------
    Path
        Path to ``data/20abr26/msci_world_imfs_eemd.parquet``.
    """
    return _REPO_ROOT / "data" / "20abr26" / "msci_world_imfs_eemd.parquet"


def _ruta_precios() -> Path:
    """
    Return the path to the closing-price series.

    Returns
    -------
    Path
        Path to ``data/20abr26/msci_world.parquet``.
    """
    return _REPO_ROOT / "data" / "20abr26" / "msci_world.parquet"


def compute_recurrence_params_table(
    df_imfs: pd.DataFrame,
    umbral_percentil: float = 10.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute $\\tau$, embedding dimension $d$, and threshold $\\varepsilon$ per component.

    Same logic as ``build_recurrence_imf_graph`` (without building the full Data object).

    Parameters
    ----------
    df_imfs : pd.DataFrame
        Columns ``IMF_1`` ... ``IMF_n`` and ``Residuo``.
    umbral_percentil : float, optional
        Percentile for the distance threshold. Default is 10.
    random_state : int, optional
        Seed for threshold computation. Default is 42.

    Returns
    -------
    pd.DataFrame
        Columns: componente, tau, d, epsilon.
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


def generate_recurrence_params_markdown(df_params: pd.DataFrame) -> str:
    """
    Generate a LaTeX/tabular Markdown fragment with the computed parameters.

    Parameters
    ----------
    df_params : pd.DataFrame
        Output of :func:`compute_recurrence_params_table`.

    Returns
    -------
    str
        Markdown text with table and copyable ``tabular`` block.
    """
    lineas = [
        "# Recurrence parameters (CEEMDAN)",
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
    Write recurrence-parameter CSV/Markdown and optionally regenerate figures.

    Parameters
    ----------
    parquet_imfs : Path, optional
        Parquet with IMFs. If None, uses ``data/20abr26/msci_world_imfs_ceemdan.parquet``
        or runs CEEMDAN on the price series if missing.
    regenerar_figuras : bool, optional
        If True, writes ``hvg_imf.png`` and ``imf_decomposition.png`` to
        ``docs/20abr26/images/english/``.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out_dir = _REPO_ROOT / "docs" / "20abr26" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    ruta_imfs = parquet_imfs or _ruta_imfs_ceemdan()
    ruta_precios = _ruta_precios()

    if not ruta_imfs.is_file():
        logger.info("Not found: %s; computing CEEMDAN...", ruta_imfs)
        if not ruta_precios.is_file():
            raise FileNotFoundError(
                f"Need {ruta_precios} or an existing IMF parquet."
            )
        serie = np.asarray(
            pd.read_parquet(ruta_precios, engine="pyarrow")["Close"].values,
            dtype=np.float64,
        ).copy()
        df_imfs = extract_ceemdan_imfs(serie)
        ruta_imfs.parent.mkdir(parents=True, exist_ok=True)
        df_imfs.to_parquet(ruta_imfs, index=False)
        logger.info("Saved %s", ruta_imfs)
    else:
        df_imfs = pd.read_parquet(ruta_imfs, engine="pyarrow")

    logger.info("Computing recurrence parameters (%d rows)...", len(df_imfs))
    df_params = compute_recurrence_params_table(df_imfs)
    csv_path = out_dir / "parametros_recurrencia_ceemdan.csv"
    df_params.to_csv(csv_path, index=False)
    logger.info("CSV: %s", csv_path)

    md_path = out_dir / "parametros_recurrencia_ceemdan.md"
    md_path.write_text(
        generate_recurrence_params_markdown(df_params) + "\n", encoding="utf-8"
    )
    logger.info("Markdown: %s", md_path)

    readme = out_dir / "README_fuentes_subseccion_grafos.txt"
    readme.write_text(
        "\n".join(
            [
                "Sources for docs/20abr26 main.tex (Graph transformation)",
                "",
                "1) Figuras hvg_imf.png e imf_decomposition.png:",
                "   - exportar_figuras_documento_20abr26() en info_msci_world_data.py",
                "   - run_graph_subsection_outputs_ceemdan_20abr26.py --figuras",
                "",
                "2) Tabla tau, d, epsilon (recurrencia):",
                "   - GraphEMD/data/graph_imf_transform_utils.py (seleccionar_tau, FNN, umbral)",
                "   - salida: docs/20abr26/out/parametros_recurrencia_ceemdan.csv",
                "",
                "3) Notebook exploratorio HVG:",
                "   - analysis/20abr26/04_ceemdan_imf_to_graph_transform/042_ceemdan_imf_to_graph_hvg_transform.ipynb",
                "",
            ]
        ),
        encoding="utf-8",
    )
    logger.info("README: %s", readme)

    if regenerar_figuras:
        if not ruta_precios.is_file():
            logger.warning("Missing %s; skipping figures.", ruta_precios)
        else:
            serie = np.asarray(
                pd.read_parquet(ruta_precios, engine="pyarrow")["Close"].values,
                dtype=np.float64,
            ).copy()
            img_dir = _REPO_ROOT / "docs" / "20abr26" / "images" / "english"
            df_eemd = None
            if EEMD_AVAILABLE:
                logger.info("Running EEMD (docs/16dic25 parameters)...")
                df_eemd = extract_eemd_imfs(serie)
                ruta_eemd = _ruta_imfs_eemd()
                ruta_eemd.parent.mkdir(parents=True, exist_ok=True)
                df_eemd.to_parquet(ruta_eemd, index=False)
                logger.info("Saved %s", ruta_eemd)
            else:
                logger.warning("PyEMD without EEMD; CEEMDAN-only figure.")
            exportar_figuras_documento_20abr26(
                df_imfs, serie, img_dir, df_imfs_eemd=df_eemd
            )
            logger.info("Figures in %s", img_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CEEMDAN outputs for the Graph transformation subsection (docs/20abr26)."
    )
    parser.add_argument(
        "--parquet-imfs",
        type=Path,
        default=None,
        help="Parquet with IMF_1,...,Residuo (default: data/20abr26/msci_world_imfs_ceemdan.parquet).",
    )
    parser.add_argument(
        "--sin-figuras",
        action="store_true",
        help="Do not regenerate PNGs in docs/20abr26/images/english/.",
    )
    args = parser.parse_args()
    main(parquet_imfs=args.parquet_imfs, regenerar_figuras=not args.sin_figuras)
