# Created by Jiayun Hong, 2019-05-21                                                                                  

import utils
import os.path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import r2_score


# predict scale
def linear_regression(train, test, cols_x, col_y, output_dir):
    reg = LinearRegression()
    X_train = train.loc[:, cols_x].copy()
    y_train = train.loc[:, col_y].copy()
    X_test = test.loc[:, cols_x].copy()
    y_test = test.loc[:, col_y].copy()

    X_train.replace(np.nan, 0.0, inplace=True)
    X_test.replace(np.nan, 0.0, inplace=True)

    X_train = X_train.abs()
    y_train = y_train.abs()
    X_test = X_test.abs()
    y_test = y_test.abs()

    # res = np.correlate(X_train.loc[:, cols_x[0]], X_train.loc[:, cols_x[1]])
    # print(res)

    reg.fit(X_train, y_train)
    print(reg.coef_)
    yhat_test = reg.predict(X_test)
    utils.dump_pickle(yhat_test, os.path.join(output_dir, "scale_yhat_test.pkl"))
    print("insample R2: ", reg.score(X_train, y_train))
    print("outofsample R2: ", reg.score(X_test, y_test))


# predict direction
def logistic_regression(train, test, cols_x, col_y):
    X_train = train.loc[:, cols_x].copy()
    y_train = train.loc[:, col_y].copy()
    X_test = test.loc[:, cols_x].copy()
    y_test = test.loc[:, col_y].copy()

    X_train.replace(np.nan, 0.0, inplace=True)
    X_test.replace(np.nan, 0.0, inplace=True)

    clf = LogisticRegression(penalty='l2')
    clf.fit(X_train, y_train)
    yhat_test = clf.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, yhat_test).ravel()
    all = tp + fp + fn + tn
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp+tn)/all
    f_score = 2 * recall * precision / (recall + precision)

    print(str(tp/all) + "  " + str(fp/all))
    print(str(fn/all) + "  " + str(tn/all))
    print("recall: ", recall)
    print("precision: ", precision)
    print("accuracy: ", accuracy)
    print("f_score: ", f_score)


# y moment
def y_moment(train):
    data = train.loc[:, ["fut_ret", "fut_ret_disc"]].copy()
    grouped = data.grouby("fut_ret_disc")
    res = grouped.mean()
    return res.values


# only use top and bottom 30%
def process_train(train, threshold):
    data = train.loc[:, ["Date", "fut_ret"]].copy()
    grouped = data.groupby("Date")
    top = grouped.quantile(1 - threshold)
    bottom = grouped.quantile(threshold)
    train_new = train.set_index("Date")
    train_new.loc[:, "fut_ret_top_thresh"]=top
    train_new.loc[:, "fut_ret_bot_thresh"]=bottom
    train_new.reset_index(inplace=True)
    is_top = train_new.loc[:, "fut_ret"] > train_new.loc[:, "fut_ret_top_thresh"]
    is_bot = train_new.loc[:, "fut_ret"] < train_new.loc[:, "fut_ret_bot_thresh"]
    is_train = is_top | is_bot
    train_new.drop(columns=["fut_ret_top_thresh", "fut_ret_bot_thresh"], inplace=True)
    return train_new.loc[is_train, :].copy()


# predict directly
def naive_bayes(train, test, cols_x, col_y, y_moment):
    X_train = train.loc[:, cols_x].copy()
    y_train = train.loc[:, col_y].copy()
    X_test = test.loc[:, cols_x].copy()
    y_test = test.loc[:, col_y].copy()
    y_true_train = train.loc[:, "fut_ret"].copy()
    y_true_test = test.loc[:, "fut_ret"].copy()

    clf = GaussianNB()
    clf.fit(X_train, y_train)
    yhat_train = clf.predict_proba(X_train)
    yhat_test = clf.predict_proba(X_test)
    yhat_true_train = np.dot(yhat_train, y_moment).flatten()
    yhat_true_test = np.dot(yhat_test, y_moment).flatten()
    print("insample R2: ", r2_score(y_true_train, yhat_true_train))
    print("outofsample R2: ", r2_score(y_true_test, yhat_true_test))
    return y_true_train, y_true_test, yhat_true_train, yhat_true_test


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(__file__, '../../..'))
    # input_path = os.path.join(base_dir, 'output/hjy/data_dsc.pkl')
    output_dir = os.path.join(base_dir, 'output/hjy')
    utils.ensure_dir(output_dir)

    train = utils.load_pickle(os.path.join(output_dir, "train.pkl"))
    test = utils.load_pickle(os.path.join(output_dir, "test.pkl"))
    y_moment = utils.load_pickle(os.path.join(output_dir, "y_moment.pkl"))
    cols_x = [col for col in train.columns
              if not col.startswith("fut_ret") and col.endswith("_disc")]
    col_y = "fut_ret_disc"
    train_y, test_y, train_yhat, test_yhat = naive_bayes(train, test, cols_x, col_y, y_moment)
    # cols_x.remove("Date")
    # cols_x.remove("X5")
    # col_y = "fut_ret_direction"
    # logistic_regression(train, test, cols_x, col_y)
    # cols_x = ["vol", "X1"]
    # col_y = ["fut_ret"]
    # linear_regression(train, test, cols_x, col_y, output_dir)
