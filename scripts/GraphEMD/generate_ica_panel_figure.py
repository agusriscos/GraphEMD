#!/usr/bin/env python3
"""
Genera una figura matricial con las componentes ICA (FastICA) de salida para el
panel empírico tras la descomposición CEEMDAN.

Disposición: ``N`` filas (máximo ``k`` ICA en cualquier serie) por ``M`` columnas
(una por serie). Solo se representan las fuentes ``Z_j`` (sin residuo).

Ejemplo::

    python scripts/GraphEMD/generar_figura_ica_panel_activos.py

    python scripts/GraphEMD/generar_figura_ica_panel_activos.py \\
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

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT_VMD = _REPO_ROOT / "scripts" / "GraphEMD" / "ejecutar_vmd_todos_activos.py"
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
    Rutas de entrada para un activo en la figura panel ICA.

    Attributes
    ----------
    ruta_parquet_ica : Path
        Parquet con columnas ``Z_*`` y ``Residuo``.
    ruta_parquet_imfs : Path
        Parquet CEEMDAN original para contar IMFs oscilatorias.
    ruta_precios : Path
        Parquet de precios para el índice temporal.
    columna_precio : str
        Nombre de la columna de cierre.
    """

    ruta_parquet_ica: Path
    ruta_parquet_imfs: Path
    ruta_precios: Path
    columna_precio: str = "Close"


def _cargar_modulo_vmd_activos() -> object:
    """
    Carga ``ejecutar_vmd_todos_activos`` para reutilizar ``ACTIVOS``.

    Returns
    -------
    module
        Módulo con la tupla ``ACTIVOS``.
    """
    spec = importlib.util.spec_from_file_location("vmd_activos_ica_panel", _SCRIPT_VMD)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar {_SCRIPT_VMD}")
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _contar_k_ica(ruta_parquet_ica: Path) -> int:
    """
    Cuenta fuentes ICA ``Z_j`` en el parquet reducido.

    Parameters
    ----------
    ruta_parquet_ica : Path
        Parquet con salida FastICA.

    Returns
    -------
    int
        Número de columnas ``Z_*``.
    """
    df = pd.read_parquet(ruta_parquet_ica, engine="pyarrow")
    return len([c for c in df.columns if c.startswith("Z_")])


def _cargar_indice_temporal(cfg: ConfigIcaPanel, n_filas: int) -> pd.Index:
    """
    Obtiene el índice temporal alineado a la longitud del parquet ICA.

    Parameters
    ----------
    cfg : ConfigIcaPanel
        Rutas del activo.
    n_filas : int
        Número de filas del parquet ICA.

    Returns
    -------
    pd.Index
        Índice de fechas o posiciones enteras.
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
    Construye la configuración ICA por activo del panel.

    Parameters
    ----------
    mod : module
        Módulo con ``ACTIVOS``.

    Returns
    -------
    tuple[ConfigIcaPanel, ...]
        Cinco configuraciones en orden del panel.
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
    Etiqueta del eje Y para una fila de la matriz.

    Parameters
    ----------
    indice_fila : int
        Índice de fila (0-based).

    Returns
    -------
    str
        Etiqueta legible.
    """
    return f"$Z_{{{indice_fila + 1}}}$"


def generar_figura(
    ruta_salida_png: Path,
    ruta_salida_pdf: Optional[Path] = None,
    dpi: int = 200,
) -> Path:
    """
    Construye la figura ``N`` filas × ``M`` columnas con componentes ICA.

    Parameters
    ----------
    ruta_salida_png : Path
        Ruta del PNG de salida.
    ruta_salida_pdf : Path, optional
        PDF vectorial opcional.
    dpi : int
        Resolución del PNG.

    Returns
    -------
    Path
        Ruta del PNG escrito.
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
    logger.info("Figura matriz ICA (%d×%d): %s", n_filas, n_columnas, ruta_salida_png)
    return ruta_salida_png


def _parsear_argumentos() -> argparse.Namespace:
    """
    Parsea argumentos de línea de comandos.

    Returns
    -------
    argparse.Namespace
        Opciones de salida y resolución.
    """
    parser = argparse.ArgumentParser(
        description="Figura matricial FastICA CEEMDAN (N filas × M columnas)."
    )
    parser.add_argument(
        "--salida-png",
        type=Path,
        default=_PAPER_FIGURES / "ica_components_panel.png",
        help="Ruta del PNG de salida.",
    )
    parser.add_argument(
        "--salida-pdf",
        type=Path,
        default=None,
        help="Ruta opcional del PDF vectorial.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Resolución del PNG.")
    return parser.parse_args()


def main() -> None:
    """
    Punto de entrada: genera la figura matricial ICA del panel empírico.
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
