#!/usr/bin/env python3
"""
Generates a matrix figure with ICA (FastICA) output components for the
empirical panel after CEEMDAN decomposition.

Layout: ``N`` rows (maximum ``k`` ICA across any series) by ``M`` columns
(one per series). Only sources ``Z_j`` are shown (no residual).

Example::

    python scripts/generate_ica_panel_figure.py

    python scripts/generate_ica_panel_figure.py \\
        --salida-png /home/agusriscos/proyectos/PAPER/figures/ica_components_panel.png
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_VMD = _REPO_ROOT / "scripts" / "run_vmd_all_assets.py"
_PAPER_FIGURES = _REPO_ROOT.parent / "PAPER" / "figures"
_MSCI_ICA = (
    _REPO_ROOT / "docs" / "20abr26" / "out" / "imfs_dim_red" / "k4" / "fastica"
)

logger = logging.getLogger(__name__)

ETIQUETAS_COLUMNAS: Tuple[str, ...] = (
    "MSCI World",
    "XLE",
    "XLP",
    "XLV",
    "XAU/USD",
)


@dataclass(frozen=True)
class ConfigIcaPanel:
    """
    Input paths for one asset in the ICA panel figure.

    Attributes
    ----------
    ruta_parquet_ica : Path
        Parquet with ``Z_*`` and ``Residuo`` columns.
    ruta_parquet_imfs : Path
        Original CEEMDAN parquet to count oscillatory IMFs.
    ruta_precios : Path
        Price parquet for the time index.
    columna_precio : str
        Closing price column name.
    """

    ruta_parquet_ica: Path
    ruta_parquet_imfs: Path
    ruta_precios: Path
    columna_precio: str = "Close"


def _cargar_modulo_vmd_activos() -> object:
    """
    Load ``run_vmd_all_assets`` to reuse ``ACTIVOS``.

    Returns
    -------
    module
        Module with the ``ACTIVOS`` tuple.
    """
    spec = importlib.util.spec_from_file_location("vmd_activos_ica_panel", _SCRIPT_VMD)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {_SCRIPT_VMD}")
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _contar_k_ica(ruta_parquet_ica: Path) -> int:
    """
    Count ICA sources ``Z_j`` in the reduced parquet.

    Parameters
    ----------
    ruta_parquet_ica : Path
        FastICA output parquet.

    Returns
    -------
    int
        Number of ``Z_*`` columns.
    """
    df = pd.read_parquet(ruta_parquet_ica, engine="pyarrow")
    return len([c for c in df.columns if c.startswith("Z_")])


def _cargar_indice_temporal(cfg: ConfigIcaPanel, n_filas: int) -> pd.Index:
    """
    Obtain the time index aligned to the ICA parquet length.

    Parameters
    ----------
    cfg : ConfigIcaPanel
        Asset paths.
    n_filas : int
        Number of rows in the ICA parquet.

    Returns
    -------
    pd.Index
        Date index or integer positions.
    """
    precios = pd.read_parquet(cfg.ruta_precios, engine="pyarrow")
    if cfg.columna_precio in precios.columns:
        serie = precios[cfg.columna_precio]
    else:
        serie = precios.iloc[:, 0]
    if len(serie) >= n_filas:
        return serie.index[:n_filas]
    return pd.RangeIndex(n_filas)


def _configuracion_activos(mod: object) -> Tuple[ConfigIcaPanel, ...]:
    """
    Build ICA configuration per panel asset.

    Parameters
    ----------
    mod : module
        Module with ``ACTIVOS``.

    Returns
    -------
    tuple[ConfigIcaPanel, ...]
        Five configurations in panel order.
    """
    activos = mod.ACTIVOS
    configs: list[ConfigIcaPanel] = []
    for cfg in activos:
        if cfg.id_activo == "msci_world":
            ruta_ica = _MSCI_ICA / "imfs_reducidas.parquet"
            ruta_imfs = cfg.dir_datos / "msci_world_imfs_ceemdan.parquet"
        else:
            ruta_ica = cfg.dir_datos / "ica" / "fastica" / "imfs_reducidas.parquet"
            ruta_imfs = cfg.dir_datos / f"{cfg.prefijo}_imfs_ceemdan.parquet"
        configs.append(
            ConfigIcaPanel(
                ruta_parquet_ica=ruta_ica,
                ruta_parquet_imfs=ruta_imfs,
                ruta_precios=cfg.ruta_precios,
                columna_precio=cfg.columna_precio,
            )
        )
    return tuple(configs)


def _etiqueta_fila(indice_fila: int) -> str:
    """
    Y-axis label for one matrix row.

    Parameters
    ----------
    indice_fila : int
        Row index (0-based).

    Returns
    -------
    str
        Readable label.
    """
    return f"$Z_{{{indice_fila + 1}}}$"


def generar_figura(
    ruta_salida_png: Path,
    ruta_salida_pdf: Optional[Path] = None,
    dpi: int = 200,
) -> Path:
    """
    Build the ``N`` rows × ``M`` columns ICA component figure.

    Parameters
    ----------
    ruta_salida_png : Path
        Output PNG path.
    ruta_salida_pdf : Path, optional
        Optional vector PDF.
    dpi : int
        PNG resolution.

    Returns
    -------
    Path
        Written PNG path.
    """
    mod = _cargar_modulo_vmd_activos()
    configs = _configuracion_activos(mod)
    n_columnas = len(configs)
    n_filas = max(_contar_k_ica(c.ruta_parquet_ica) for c in configs)
    max_longitud = max(
        len(pd.read_parquet(c.ruta_parquet_ica, engine="pyarrow")) for c in configs
    )

    fig, axes = plt.subplots(
        n_filas,
        n_columnas,
        figsize=(4.2 * n_columnas, 2.35 * n_filas),
        sharex=True,
        gridspec_kw={"hspace": 0.12, "wspace": 0.05},
    )
    if n_filas == 1:
        axes = np.array([axes])
    if n_columnas == 1:
        axes = axes.reshape(n_filas, 1)

    k_por_columna: list[int] = []

    for col, (cfg, titulo) in enumerate(zip(configs, ETIQUETAS_COLUMNAS)):
        df = pd.read_parquet(cfg.ruta_parquet_ica, engine="pyarrow")
        z_cols = sorted(
            [c for c in df.columns if c.startswith("Z_")],
            key=lambda nombre: int(nombre.split("_")[1]),
        )
        k = len(z_cols)
        k_por_columna.append(k)
        tiempo = np.arange(len(df))

        axes[0, col].set_title(titulo, fontsize=11, pad=5)

        for fila in range(n_filas):
            ax = axes[fila, col]
            if fila < k:
                ax.plot(tiempo, df[z_cols[fila]].values, linewidth=0.65, color="C0")
                ax.set_xlim(0, max_longitud - 1)
                ax.grid(True, alpha=0.22, linewidth=0.35)
                ax.tick_params(labelsize=7.5)
                if fila < n_filas - 1:
                    plt.setp(ax.get_xticklabels(), visible=False)
            else:
                ax.set_visible(False)

        axes[0, col].text(
            0.02,
            0.96,
            f"$k={k}$",
            transform=axes[0, col].transAxes,
            fontsize=8.5,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.7},
        )

    for fila in range(n_filas):
        eje_etiqueta = None
        for col, k in enumerate(k_por_columna):
            if fila < k:
                eje_etiqueta = axes[fila, col]
                break
        if eje_etiqueta is not None:
            eje_etiqueta.set_ylabel(_etiqueta_fila(fila), fontsize=10)

    for col, k in enumerate(k_por_columna):
        if n_filas - 1 < k:
            axes[n_filas - 1, col].set_xlabel("Time index", fontsize=9)
            break

    fig.subplots_adjust(left=0.06, right=0.995, top=0.94, bottom=0.08, hspace=0.12, wspace=0.05)
    ruta_salida_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        ruta_salida_png,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    if ruta_salida_pdf is not None:
        fig.savefig(
            ruta_salida_pdf,
            bbox_inches="tight",
            pad_inches=0.02,
            facecolor="white",
        )
    plt.close(fig)
    logger.info("ICA matrix figure (%d×%d): %s", n_filas, n_columnas, ruta_salida_png)
    return ruta_salida_png


def _parsear_argumentos() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Output and resolution options.
    """
    parser = argparse.ArgumentParser(
        description="FastICA CEEMDAN matrix figure (N rows × M columns)."
    )
    parser.add_argument(
        "--salida-png",
        type=Path,
        default=_PAPER_FIGURES / "ica_components_panel.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--salida-pdf",
        type=Path,
        default=None,
        help="Optional vector PDF path.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution.")
    return parser.parse_args()


def main() -> None:
    """
    Entry point: generate the empirical panel ICA matrix figure.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parsear_argumentos()
    ruta = generar_figura(
        ruta_salida_png=args.salida_png,
        ruta_salida_pdf=args.salida_pdf,
        dpi=args.dpi,
    )
    print(f"Escrito: {ruta}")


if __name__ == "__main__":
    main()
