#!/usr/bin/env python3
"""
Generate a PDF/PNG figure with the five synthetic signals used in the paper
(5 rows by 1 column; no global title; the caption goes in LaTeX).

Parameters match ``construir_escenarios`` in ``run_emdsynth_decompositions.py``.

Example::

    python3 scripts/emdsynth/generate_synthetic_signals_figure.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DOCS_IMAGES = _REPO_ROOT / "docs" / "20abr26" / "images" / "english"


def _cargar_modulo_ejecutar():
    """
    Load the decomposition pipeline module (same scenario definitions).

    Returns
    -------
    module
        Module with ``construir_escenarios``.
    """
    ruta = Path(__file__).resolve().parent / "run_emdsynth_decompositions.py"
    spec = importlib.util.spec_from_file_location("emdsynth_ejecutar", ruta)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {ruta}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generar_figura(
    ruta_salida_png: Path,
    ruta_salida_pdf: Path | None = None,
    dpi: int = 200,
) -> None:
    """
    Build the figure with five stacked subplots (one row per signal).

    Parameters
    ----------
    ruta_salida_png : pathlib.Path
        Path to the PNG.
    ruta_salida_pdf : pathlib.Path, optional
        If provided, also saves vector PDF.
    dpi : int
        PNG resolution.
    """
    mod = _cargar_modulo_ejecutar()
    escenarios = mod.construir_escenarios(
        duracion=5.0,
        frecuencia_muestreo=500.0,
        semilla_ruido=42,
    )

    titulos = [
        (
            "(a) Two close tones",
            r"$f_1=10\,\mathrm{Hz}$, $f_2=11.3\,\mathrm{Hz}$",
        ),
        (
            "(b) Linear chirp",
            r"$f_0=1.5\,\mathrm{Hz}$, sweep $k=1.2\,\mathrm{Hz/s}$",
        ),
        (
            "(c) Burst on low-frequency carrier",
            r"$f_\mathrm{low}=0.8\,\mathrm{Hz}$, burst $12\,\mathrm{Hz}$ on $[1.2, 3.2]\,\mathrm{s}$",
        ),
        (
            "(d) Superposition + noise",
            r"close tones $+$ $0.35\times$ chirp $+$ scaled Gaussian noise",
        ),
        (
            "(e) Multi-component benchmark",
            r"mode mixing $+$ chirps $+$ close tones $+$ sinusoids $+$ $\mathcal{N}(0,0.2)$ $+$ slow uptrend",
        ),
    ]

    fig, axes = plt.subplots(5, 1, figsize=(10.0, 12.0), sharex=True)
    fig.patch.set_facecolor("white")

    for ax, (escenario, (titulo, subtit)) in zip(axes, zip(escenarios, titulos)):
        _nombre, t_vec, x_vec, _meta = escenario
        ax.plot(t_vec, x_vec, color="#1f77b4", linewidth=0.85)
        ax.set_title(f"{titulo}\n{subtit}", fontsize=9.5)
        ax.set_xlim(t_vec[0], t_vec[-1])
        ax.grid(True, alpha=0.35, linestyle="--", linewidth=0.5)
        ax.tick_params(labelsize=9)
        ax.set_ylabel("Amplitude (a.u.)", fontsize=9)

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    fig.tight_layout()

    ruta_salida_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    if ruta_salida_pdf is not None:
        fig.savefig(ruta_salida_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    """
    Entry point: write PNG (and PDF) under ``docs/20abr26/images/english/``.
    """
    png = _DOCS_IMAGES / "synthetic_signals_emdsynth.png"
    pdf = _DOCS_IMAGES / "synthetic_signals_emdsynth.pdf"
    generar_figura(png, ruta_salida_pdf=pdf)
    print(f"Escrito: {png}")
    print(f"Escrito: {pdf}")


if __name__ == "__main__":
    main()
