# Created by Jiayun Hong, 2019-05-17                                                                                  

import os.path
import utils
import numpy as np
import pandas as pd


# delete rows with fut_ret = 0 and vol = 0
def clean_nan(input_path, output_dir):
    data = pd.read_csv(input_path, index_col=None)
    data.drop(columns=["Time", "sec_id"], inplace=True)
    cols = data.columns.tolist()
    cols.remove("Date")
    data.loc[:, cols] = data.loc[:, cols].replace(0., np.nan)
    data.dropna(subset=["fut_ret", "vol"], inplace=True)
    utils.dump_pickle(data, os.path.join(output_dir, "data.pkl"))


# discretize features
def discretize(data, bins):
    split = np.array_split(np.sort(data), bins)
    cutoffs = [x[-1] for x in split]
    cutoffs = cutoffs[:-1]
    discrete = np.digitize(data, cutoffs, right=True)
    return discrete, cutoffs





if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(__file__, '../../..'))
    input_path = os.path.join(base_dir, 'data/dat_final.csv')
    output_dir = os.path.join(base_dir, 'output/hjy')
    utils.ensure_dir(output_dir)

    clean_nan(input_path, output_dir)
