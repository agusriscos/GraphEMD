#!/usr/bin/env python3
"""
Generates a figure with five CEEMDAN/EEMD/VMD panels from the empirical panel in
a 3 rows × 2 columns layout; the fifth asset is centered on the bottom row.

Each cell replicates the panel from ``generate_comparative_residual_figure`` in
``run_vmd_all_assets.py`` (signal + IMF sum on top; residual gap below).
No global title or legend (color convention in the LaTeX figure caption).

Example::

    python scripts/generate_decomposition_panel_figure.py

    python scripts/generate_decomposition_panel_figure.py \\
        --salida-png /home/agusriscos/proyectos/PAPER/figures/empirical_decomposition_panel.png
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path
from typing import Any, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_VMD = _REPO_ROOT / "scripts" / "run_vmd_all_assets.py"
_PAPER_FIGURES = _REPO_ROOT.parent / "PAPER" / "figures"
_DOCS_IMAGES = _REPO_ROOT / "docs" / "20abr26" / "images" / "english"

logger = logging.getLogger(__name__)

ETIQUETAS_PANEL: Tuple[Tuple[str, str], ...] = (
    ("(a) MSCI World", "MSCI World"),
    ("(b) XLE", "XLE (energy ETF)"),
    ("(c) XLP", "XLP (consumer staples ETF)"),
    ("(d) XLV", "XLV (health care ETF)"),
    ("(e) XAU/USD", "Spot gold"),
)


def _cargar_modulo_vmd_activos() -> Any:
    """
    Load ``run_vmd_all_assets`` to reuse paths and IMF utilities.

    Returns
    -------
    module
        Module with ``ACTIVOS``, ``cargar_serie_precios``, and residual-gap helpers.
    """
    spec = importlib.util.spec_from_file_location("vmd_activos_panel", _SCRIPT_VMD)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {_SCRIPT_VMD}")
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _leer_vmd_dc(ruta_parametros: Path) -> int:
    """
    Read the ``DC`` value used in VMD calibration for the asset.

    Parameters
    ----------
    ruta_parametros : pathlib.Path
        JSON ``*_vmd_parametros.json``.

    Returns
    -------
    int
        ``DC`` value (defaults to 1 if the file is missing).
    """
    if not ruta_parametros.is_file():
        logger.warning("Missing %s; assuming DC=1.", ruta_parametros)
        return 1
    with ruta_parametros.open(encoding="utf-8") as fh:
        datos = json.load(fh)
    return int(datos["calibracion"]["mejor"]["DC"])


def _dibujar_panel_activo(
    mod: Any,
    cfg: Any,
    ax_superior: plt.Axes,
    ax_inferior: plt.Axes,
) -> None:
    """
    Draw the CEEMDAN/EEMD/VMD top/bottom pair for one asset.

    Parameters
    ----------
    mod : module
        ``run_vmd_all_assets`` module.
    cfg : ConfigActivo
        Asset configuration.
    ax_superior : matplotlib.axes.Axes
        Axis for original signal and IMF sums.
    ax_inferior : matplotlib.axes.Axes
        Axis for trend/residual implicit gaps.
    """
    rutas = mod._rutas_activo(cfg)
    _, serie = mod.cargar_serie_precios(cfg)
    n = len(serie)

    if not rutas["imfs_ceemdan"].is_file():
        raise FileNotFoundError(f"Missing CEEMDAN: {rutas['imfs_ceemdan']}")
    if not rutas["imfs"].is_file():
        raise FileNotFoundError(f"Missing VMD: {rutas['imfs']}")

    df_ceemdan = mod._alinear_df_imfs_longitud(
        pd.read_parquet(rutas["imfs_ceemdan"], engine="pyarrow"),
        n,
    )
    df_vmd = mod._alinear_df_imfs_longitud(
        pd.read_parquet(rutas["imfs"], engine="pyarrow"),
        n,
    )
    vmd_dc = _leer_vmd_dc(rutas["parametros"])

    brecha_ceemdan = mod._brecha_residuo_implicito(serie, df_ceemdan)
    suma_ceemdan = mod._suma_imfs_sin_residuo(df_ceemdan)

    ax_superior.plot(serie, label="Original", linewidth=0.8, color="C0")
    ax_superior.plot(
        suma_ceemdan,
        label="Σ IMFs (CEEMDAN, without residual)",
        linewidth=0.8,
        alpha=0.85,
        color="C1",
    )
    ax_inferior.plot(
        brecha_ceemdan,
        label="|Original − Σ IMFs| (CEEMDAN)",
        linewidth=0.7,
        color="C1",
    )

    if rutas["imfs_eemd"].is_file():
        df_eemd = mod._alinear_df_imfs_longitud(
            pd.read_parquet(rutas["imfs_eemd"], engine="pyarrow"),
            n,
        )
        suma_eemd = mod._suma_imfs_sin_residuo(df_eemd)
        brecha_eemd = mod._brecha_residuo_implicito(serie, df_eemd)
        ax_superior.plot(
            suma_eemd,
            label="Σ IMFs (EEMD, without residual)",
            linewidth=0.8,
            alpha=0.85,
            color="C2",
        )
        ax_inferior.plot(
            brecha_eemd,
            label="|Original − Σ IMFs| (EEMD)",
            linewidth=0.7,
            color="C2",
            alpha=0.9,
        )

    suma_vmd = mod._suma_imfs_oscilatorias_vmd(df_vmd, vmd_dc=vmd_dc)
    etiqueta_suma_vmd = (
        "Σ IMF$_{2..K}$ (VMD, DC=1)"
        if vmd_dc == 1
        else "Σ IMFs (VMD, without residual)"
    )
    ax_superior.plot(
        suma_vmd,
        label=etiqueta_suma_vmd,
        linewidth=0.8,
        alpha=0.85,
        color="C3",
    )

    if vmd_dc == 1:
        brecha_vmd = mod._brecha_tendencia_vmd_dc1(serie, df_vmd)
        ax_inferior.plot(
            brecha_vmd,
            label="|Original − Σ IMF$_{2..K}$| (VMD, DC=1)",
            linewidth=0.7,
            color="C3",
            alpha=0.9,
        )
    else:
        brecha_vmd = mod._brecha_residuo_implicito(serie, df_vmd)
        ax_inferior.plot(
            brecha_vmd,
            label="|Original − Σ IMFs| (VMD)",
            linewidth=0.7,
            color="C3",
            alpha=0.9,
        )

    ax_superior.set_ylabel("Level", fontsize=9)
    ax_inferior.set_ylabel("|Original − Σ IMFs|", fontsize=9)
    ax_superior.grid(True, alpha=0.35, linestyle="--", linewidth=0.5)
    ax_inferior.grid(True, alpha=0.35, linestyle="--", linewidth=0.5)
    ax_superior.tick_params(labelsize=9)
    ax_inferior.tick_params(labelsize=9)
    plt.setp(ax_superior.get_xticklabels(), visible=False)


def generar_figura_msci_ilustrativa(
    ruta_salida_png: Path,
    ruta_salida_pdf: Optional[Path] = None,
    dpi: int = 200,
) -> Path:
    """
    Illustrative CEEMDAN/EEMD/VMD comparison figure for MSCI World.

    Top panel: original signal and oscillatory IMF sums by method.
    Bottom panel: slow-trend gap (implicit residual) by method.

    Parameters
    ----------
    ruta_salida_png : pathlib.Path
        PNG output path.
    ruta_salida_pdf : pathlib.Path, optional
        If provided, also saves a vector PDF.
    dpi : int
        PNG resolution.

    Returns
    -------
    pathlib.Path
        Written PNG path.
    """
    mod = _cargar_modulo_vmd_activos()
    cfg = mod.ACTIVOS[0]

    fig, (ax_sup, ax_inf) = plt.subplots(
        2, 1, sharex=True, figsize=(9.8, 5.2), gridspec_kw={"hspace": 0.06}
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "MSCI World: CEEMDAN vs. EEMD vs. VMD\n"
        r"Top: original and $\sum$ oscillatory IMFs $\cdot$ "
        r"Bottom: slow-trend gap $|x-\sum\mathrm{IMF}|$",
        fontsize=10.0,
        y=0.98,
    )
    _dibujar_panel_activo(mod, cfg, ax_sup, ax_inf)
    ax_inf.set_xlabel("Time index", fontsize=9)
    leyenda = [
        Line2D([0], [0], color="C0", linewidth=1.2, label="Original"),
        Line2D([0], [0], color="C1", linewidth=1.2, label="CEEMDAN"),
        Line2D([0], [0], color="C2", linewidth=1.2, label="EEMD"),
        Line2D([0], [0], color="C3", linewidth=1.2, label="VMD"),
    ]
    ax_sup.legend(
        handles=leyenda,
        loc="upper left",
        fontsize=8,
        framealpha=0.92,
        ncol=2,
        columnspacing=0.8,
        handlelength=1.6,
    )

    fig.subplots_adjust(left=0.07, right=0.995, top=0.86, bottom=0.12, hspace=0.06)
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
    logger.info("Illustrative MSCI figure: %s", ruta_salida_png)
    return ruta_salida_png


def generar_figura(
    ruta_salida_png: Path,
    ruta_salida_pdf: Optional[Path] = None,
    dpi: int = 200,
) -> Path:
    """
    Build the 3×2 figure with five assets (two internal rows per cell).

    The grid uses four logical columns: rows 1–2 occupy column pairs;
    row 3 centers the fifth asset in the middle columns.

    Parameters
    ----------
    ruta_salida_png : pathlib.Path
        PNG output path.
    ruta_salida_pdf : pathlib.Path, optional
        If provided, also saves a vector PDF.
    dpi : int
        PNG resolution.

    Returns
    -------
    pathlib.Path
        Written PNG path.
    """
    mod = _cargar_modulo_vmd_activos()
    activos = mod.ACTIVOS

    if len(activos) != len(ETIQUETAS_PANEL):
        raise ValueError(
            f"Se esperaban {len(ETIQUETAS_PANEL)} activos; hay {len(activos)}."
        )

    fig = plt.figure(figsize=(12.0, 11.0))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        3,
        4,
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.55,
        wspace=0.28,
    )
    slots = [
        gs[0, 0:2],
        gs[0, 2:4],
        gs[1, 0:2],
        gs[1, 2:4],
        gs[2, 1:3],
    ]

    for idx, (slot, cfg, (titulo, subtitulo)) in enumerate(
        zip(slots, activos, ETIQUETAS_PANEL)
    ):
        subfig = fig.add_subfigure(slot)
        ax_sup, ax_inf = subfig.subplots(2, 1, sharex=True, gridspec_kw={"hspace": 0.06})
        subfig.suptitle(f"{titulo}\n{subtitulo}", fontsize=9.0, y=1.02)
        _dibujar_panel_activo(mod, cfg, ax_sup, ax_inf)
        if idx < 4:
            plt.setp(ax_inf.get_xticklabels(), visible=False)
        else:
            ax_inf.set_xlabel("Time index", fontsize=9)

    fig.tight_layout()

    ruta_salida_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    if ruta_salida_pdf is not None:
        fig.savefig(ruta_salida_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Asset panel figure: %s", ruta_salida_png)
    return ruta_salida_png


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Output and resolution options.
    """
    parser = argparse.ArgumentParser(
        description="CEEMDAN/EEMD/VMD figure for the five panel assets."
    )
    parser.add_argument(
        "--salida-png",
        type=Path,
        default=_PAPER_FIGURES / "empirical_decomposition_panel.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--salida-pdf",
        type=Path,
        default=None,
        help="Optional vector PDF path.",
    )
    parser.add_argument(
        "--copia-docs",
        action="store_true",
        help="Also write under docs/20abr26/images/english/.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution.")
    parser.add_argument(
        "--msci-ilustrativa",
        action="store_true",
        help="Generate only the MSCI illustrative figure (reconstruction + CEEMDAN residual).",
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry point: generate the empirical panel figure.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    if args.msci_ilustrativa:
        ruta = generar_figura_msci_ilustrativa(
            ruta_salida_png=args.salida_png,
            ruta_salida_pdf=args.salida_pdf,
            dpi=args.dpi,
        )
    else:
        ruta = generar_figura(
            ruta_salida_png=args.salida_png,
            ruta_salida_pdf=args.salida_pdf,
            dpi=args.dpi,
        )
    print(f"Escrito: {ruta}")
    if args.copia_docs:
        ruta_docs = _DOCS_IMAGES / "empirical_decomposition_panel.png"
        generar_figura(ruta_salida_png=ruta_docs, dpi=args.dpi)
        print(f"Escrito: {ruta_docs}")


if __name__ == "__main__":
    main()
