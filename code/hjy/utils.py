# Created by Jiayun Hong, 2019-05-17                                                                                  

import os
import pickle
import pandas as pd


# load pickle
def load_pickle(input_path):
    with open(input_path, 'rb') as f:
        res = pickle.load(f)
    return res


# dump pickle
def dump_pickle(pkl, output_path):
    with open(output_path, "wb") as f:
        pickle.dump(pkl, f)


# ensure directory
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


# cross validation datasets
def cv_data(input_path, output_dir):
    data_lst = []

    data = load_pickle(input_path)

    # Drop np.nan in fut_ret
    # data.dropna(subset=['fut_ret'],inplace=True)
    # data.replace(np.nan,0,inplace=True)

    cols = data.columns.tolist()
    # print(cols)

    cols_x = cols.copy()
    cols_x.remove('Date')
    if 'Time' in cols_x:
        cols_x.remove('Time')
    if 'sec_id' in cols_x:
        cols_x.remove("sec_id")
    cols_x.remove('fut_ret')

    k = 5
    kfold = [(32 * x, 32 * x + 31) for x in range(k)]

    data_cv = data[data.Date < 160]

    for i in range(k):
        train_idx = kfold[i]
        X_test = data_cv[(data_cv.Date <= train_idx[1]) & (data_cv.Date >= train_idx[0])][cols_x]
        y_test = data_cv[(data_cv.Date <= train_idx[1]) & (data_cv.Date >= train_idx[0])]['fut_ret']
        X_train = data_cv[(data_cv.Date >= train_idx[1] + 5) | (data_cv.Date <= train_idx[0] - 5)][cols_x]
        y_train = data_cv[(data_cv.Date >= train_idx[1] + 5) | (data_cv.Date <= train_idx[0] - 5)]['fut_ret']
        data_lst.append((X_train, y_train, X_test, y_test))

    dump_pickle(data_lst, os.path.join(output_dir, "cv_data.pkl"))
