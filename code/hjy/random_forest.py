# Created by Jiayun Hong, 2019-05-14

import os.path
import utils
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def random_forest(X_train, y_train, X_test, y_test):
    rf = RandomForestRegressor()


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(__file__, '../../..'))
    input_path = os.path.join(base_dir, 'output/hjy/data.pkl')
    output_dir = os.path.join(base_dir, 'output/hjy')
    utils.ensure_dir(output_dir)

    utils.cv_data(input_path, output_dir)
