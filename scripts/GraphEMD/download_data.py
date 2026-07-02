from GraphEMD.conf import DATA_PATH
from GraphEMD.data import download_dcoilwtico

if __name__ == '__main__':
    
    # download_dcoilwtico(
    #     start="1983/01/10", end="2022/06/15", year_first=True,
    #     data_dir=DATA_PATH, plot_result=True
    # )

    download_dcoilwtico(
        start="1983/01/10", end="2026/02/22", year_first=True,
        data_dir=DATA_PATH, plot_result=True
    )

