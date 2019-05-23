# Created by Jiayun Hong, 2019-05-17                                                                                  

import os.path
import utils
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import *


# delete rows with fut_ret = 0 and vol = 0
def clean_nan(input_path, output_dir):
    data = pd.read_csv(input_path, index_col=None)
    data.drop(columns=["Time", "sec_id"], inplace=True)
    cols = data.columns.tolist()
    cols.remove("Date")
    data.loc[:, cols] = data.loc[:, cols].replace(0., np.nan)
    data.dropna(subset=["fut_ret", "vol"], inplace=True)
    utils.dump_pickle(data, os.path.join(output_dir, "data.pkl"))


# discretize continuous
def discretize(data, bins):
    split = np.array_split(np.sort(data.dropna()), bins)
    cutoffs = [x[-1] for x in split]
    cutoffs = cutoffs[:-1]
    discrete = np.digitize(data, cutoffs, right=True)
    discrete = discrete.astype(float)
    discrete[np.isnan(data)] = np.nan
    return discrete, cutoffs


# discretize all features
def disc_feat(input_path, output_dir, disc_cols):
    data = utils.load_pickle(input_path)
    for col in disc_cols:
        if col in ["X3", "X4", "X6", "X7"]:
            bins = 5
        else:
            bins = 50
        data.loc[:, col + "_disc"], cutoffs = discretize(data.loc[:, col], bins)
    utils.dump_pickle(data, os.path.join(output_dir, "data_dsc.pkl"))


# draw pattern
def pattern_plot(input_path, output_dir, disc_cols):
    data_dsc = utils.load_pickle(input_path)
    cols_len = len(disc_cols)
    for i in range(cols_len - 1):
        for j in range(i + 1, cols_len):
            col1 = disc_cols[i]
            col2 = disc_cols[j]
            res = pd.crosstab(data_dsc.loc[:, col1], data_dsc.loc[:, col2], normalize='all')
            if len(res.columns) > 20:
                res.drop([0, len(res.columns) - 1], axis=1, inplace=True)
            if len(res.index) > 20:
                res.drop([0, len(res.index) - 1], axis=0, inplace=True)
            f, ax = plt.subplots(1, 1, figsize=(14, 10))
            sns.heatmap(res, ax=ax, cmap="YlGnBu")
            ax.set_title(col1 + "_" + col2 + " pattern")
            f.savefig(os.path.join(output_dir, col1 + "_" + col2 + "_pattern.pdf"))
            plt.close(f)


# denote fut ret direction, >0 -> +1, <0 -> -1
def denote_direction(input_path, output_dir):
    data_dsc = utils.load_pickle(input_path)
    data_dsc.loc[:, "fut_ret_direction"] = np.nan
    is_pos = data_dsc.loc[:, "fut_ret"] > 0
    data_dsc.loc[is_pos, "fut_ret_direction"] = 1
    data_dsc.loc[~is_pos, "fut_ret_direction"] = -1
    utils.dump_pickle(data_dsc, os.path.join(output_dir, "data_dsc.pkl"))


# calculate mutual information
def mut_info(input_path, output_dir):
    data_dsc = utils.load_pickle(input_path)
    cols = [col for col in data_dsc.columns if not col.startswith("fut_ret")]
    cols.remove("Date")
    res = pd.Series(index=cols, data=np.nan)
    for col in cols:
        data_part = data_dsc.loc[:, [col, "fut_ret_direction"]].copy()
        data_part.dropna(subset=[col], inplace=True)
        mut_info = mutual_info_classif(data_part.loc[:, [col]], data_part.loc[:, "fut_ret_direction"])
        res.loc[col] = mut_info[0]
    utils.dump_pickle(res, os.path.join(output_dir, "mutual_info.pkl"))


# distribution of feature with pos and neg ret
def dist_direction(input_path, output_dir):
    # data_dsc = utils.load_pickle(input_path)
    data_dsc = pd.read_csv(input_path, index_col=None)
    data_dsc.replace(0.0, np.nan, inplace=True)
    cols = [col for col in data_dsc.columns
            if not col.startswith("fut_ret") and not col.endswith("_disc")]
    cols.remove("Date")

    if "sec_id" in cols:
        cols.remove("sec_id")
    if "fut_ret_direction" in data_dsc.columns:
        is_pos = data_dsc.loc[:, "fut_ret_direction"] == 1
    else:
        is_pos = data_dsc.loc[:, "fut_ret"] > 0
    for col in cols:
        f, ax = plt.subplots(1, 1)
        sns.distplot(data_dsc.loc[is_pos, col].dropna(), label="pos ret", ax=ax)
        sns.distplot(data_dsc.loc[~is_pos, col].dropna(), label="neg ret", ax=ax)
        ax.set_title("distribution of " + col)
        ax.legend()
        f.savefig(os.path.join(output_dir, col + "_dist.pdf"))
        plt.close(f)


# train test split
def split_train_test(input_path, output_dir):
    # data_dsc = utils.load_pickle(input_path
    data_dsc = pd.read_csv(input_path, index_col=None)
    is_train = data_dsc.loc[:, "Date"] <= 150
    is_test = data_dsc.loc[:, "Date"] >= 155
    train = data_dsc.loc[is_train, :]
    test = data_dsc.loc[is_test, :]
    utils.dump_pickle(train, os.path.join(output_dir, "train_scale.pkl"))
    utils.dump_pickle(test, os.path.join(output_dir, "test_scale.pkl"))


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(__file__, '../../..'))
    input_path = os.path.join(base_dir, 'output/hjy/data_dsc.pkl')
    output_dir = os.path.join(base_dir, 'output/hjy/pattern2')
    utils.ensure_dir(output_dir)

    # clean_nan(input_path, output_dir)
    disc_cols = ["fut_ret_disc", "vol_disc", "X1_disc", "X2_disc", "X3_disc", "X4_disc", "X5_disc", "X6_disc",
                 "X7_disc"]
    # disc_cols = ["fut_ret", "vol", "X1", "X2", "X3", "X4", "X5", "X6",
    #              "X7"]
    # disc_feat(input_path, output_dir, disc_cols)

    pattern_plot(input_path, output_dir, disc_cols)
    # denote_direction(input_path, output_dir)
    # mut_info(input_path, output_dir)
    # dist_direction(input_path, output_dir)
    # split_train_test(input_path, output_dir)
