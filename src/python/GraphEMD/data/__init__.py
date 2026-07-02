"""
GraphEMD.data module: pandas utilities and PyTorch dataloader.

VisualGraphDataloader is imported directly; other names are loaded on demand
to avoid importing pandas_datareader when only the dataloader is needed.
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
    """Lazy-load pandas_utils to avoid importing pandas_datareader when unnecessary."""
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
