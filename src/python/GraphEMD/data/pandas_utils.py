import os
from datetime import datetime
from typing import Tuple, Iterable, Optional

from random import seed, randint
from pandas import DataFrame, Series

try:
    from pandas_datareader.data import DataReader
except ModuleNotFoundError:
    pass

from plotly.express import line


def str_to_datetime(date_str: str, year_first: bool = True) -> datetime:
    date_str = date_str.replace("/", "-")
    year_idx = 0 if year_first else 2
    day_idx = 2 if year_first else 0

    date_year, date_month, date_day = (
        int(date_str.split("-")[year_idx]), int(date_str.split("-")[1]), int(date_str.split("-")[day_idx])
    )
    return datetime(year=date_year, month=date_month, day=date_day)


def download_dcoilwtico(
        start: str, end: str, year_first: bool,
        data_dir: str = r"C:\Users\LDAARM4\proyectos\GraphEMD\data", plot_result: bool = False
):
    start_dt = str_to_datetime(start, year_first=year_first)
    end_dt = str_to_datetime(end, year_first=year_first)

    os.makedirs(data_dir, exist_ok=True)
    file_path = "{}/dcoilwtico.csv".format(data_dir)

    df = DataReader("DCOILWTICO", "fred", start_dt, end_dt)

    if plot_result:
        fig = line(df)
        plot_path = "{}/dcoilwtico_raw_data.html".format(data_dir)
        fig.write_html(plot_path)

    df.to_csv(file_path)


def scale_data(df: DataFrame, min_val: float, max_val: float) -> DataFrame:
    return (df.clip(lower=min_val, upper=max_val) - min_val) / (max_val - min_val)


def revert_scale_data(df: DataFrame, min_val: float, max_val: float) -> DataFrame:
    return df * (max_val - min_val) + min_val


def train_test_split(df: DataFrame, threshold_date: str) -> Tuple[DataFrame, DataFrame]:
    threshold = threshold_date.replace("/", "-")
    return df.loc[df.index.min():threshold], df.loc[threshold: df.index.max()]


def series_to_window_list(
        data: Series, max_window_size: int, min_window_size, mode: int = 1,
        window_num: Optional[int] = None, random_state: Optional[int] = None
) -> Iterable[Series]:
    """
    Útil de generación de ventanas de datos.

    :param data: Datos
    :param max_window_size: Tamaño máximo de ventana
    :param min_window_size: Tamaño mínimo de ventana
    :param mode: Hay dos modos de generación. 1 - Sin solape temporal. 2 - Con solape temporal
    :param window_num: Obligatorio especificar al usar el modo 2. Número de ventanas.
    :param random_state:  Estado aleatorio. None igual a sin semilla
    :return: Lista de ventanas de los datos de entrada.
    """
    if mode != 1 and mode != 2:
        raise ValueError("Modo de generación de ventanas temporales no válido")

    if mode == 2 and window_num is None:
        raise ValueError("window_num es obligatorio cuando mode == 2")

    if random_state:
        seed(random_state)

    sort_data = data.dropna().sort_index(ascending=False)
    windows = []
    start_index = 0

    if mode == 1:
        while start_index < len(data):
            window_size = randint(min_window_size, max_window_size)
            window = sort_data[start_index:start_index + window_size]
            start_index += window_size
            windows.append(window)
    else:
        assert window_num is not None, "window_num debe ser un entero cuando mode == 2"
        num_windows: int = window_num
        for _ in range(num_windows):
            max_start_index = max(0, len(sort_data) - 1)
            start_index = randint(0, max_start_index)
            window_size = randint(
                min(min_window_size, len(sort_data) - start_index), min(max_window_size, len(sort_data) - start_index)
            )
            window = sort_data[start_index:start_index + window_size]
            if window.shape[0] > 1:
                windows.append(window)

    return windows

