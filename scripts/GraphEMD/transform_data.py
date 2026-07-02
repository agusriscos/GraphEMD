import os
import shutil

from pandas import read_csv
import sklearn.model_selection as ms
from plotly.express import line
from GraphEMD.conf import DATA_PATH, FILE_PATH, TransformConfig
from GraphEMD.data import scale_data, train_test_split, series_to_window_list

if __name__ == '__main__':
    raw_data = read_csv(FILE_PATH, index_col="DATE")
    transform_config = TransformConfig.to_dict()

    data = scale_data(df=raw_data, min_val=transform_config["MIN_VAL"], max_val=transform_config["MAX_VAL"])
    train, test = train_test_split(data, threshold_date="2016/12/23")

    # FREEZE TEST DATA
    line(test).write_html("{}/dcoilwtico_test_data.html".format(DATA_PATH))
    test.to_csv("{}/dcoilwtico_test_data.csv".format(DATA_PATH))

    # SAVE TRAIN DATA WINDOWS
    for dir_ in ("train_data", "val_data"):
        if os.path.exists("{0}/{1}".format(DATA_PATH, dir_)):
            shutil.rmtree("{0}/{1}".format(DATA_PATH, dir_))
        os.makedirs("{0}/{1}".format(DATA_PATH, dir_))

    windows = series_to_window_list(
        train.iloc[:, 0], transform_config["MAX_WINDOW_SIZE"], transform_config["MIN_WINDOW_SIZE"],
        transform_config["WINDOW_MODE"], transform_config["WINDOW_NUM"]
    )

    # SPLIT TRAIN AND VALIDATION
    train_windows, val_windows = ms.train_test_split(windows, test_size=0.2, shuffle=True, random_state=42)
    [
        w.to_frame().to_parquet(r"{0}/train_data/train_{1}.parquet".format(DATA_PATH, str(i).zfill(5)), index=False)
        for i, w in enumerate(train_windows)
    ]

    [
        w.to_frame().to_parquet(r"{0}/val_data/val_{1}.parquet".format(DATA_PATH, str(i).zfill(5)), index=False)
        for i, w in enumerate(val_windows)
    ]
