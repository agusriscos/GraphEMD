"""
Módulo GraphEMD.data: utilidades pandas y dataloader PyTorch.

VisualGraphDataloader se importa directamente; el resto de nombres se cargan
bajo demanda para evitar cargar pandas_datareader cuando solo se usa el dataloader.
"""
from .torch_utils import VisualGraphDataloader

__all__ = [
    "str_to_datetime",
    "download_dcoilwtico",
    "scale_data",
    "revert_scale_data",
    "train_test_split",
    "series_to_window_list",
    "VisualGraphDataloader",
]

_PANDAS_UTILS_NAMES = frozenset(
    {
        "str_to_datetime",
        "download_dcoilwtico",
        "scale_data",
        "revert_scale_data",
        "train_test_split",
        "series_to_window_list",
    }
)


def __getattr__(name: str):
    """Carga perezosa de pandas_utils para no importar pandas_datareader si no hace falta."""
    if name in _PANDAS_UTILS_NAMES:
        from .pandas_utils import (
            download_dcoilwtico,
            revert_scale_data,
            scale_data,
            series_to_window_list,
            str_to_datetime,
            train_test_split,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
