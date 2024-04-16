import numpy as np  # linear algebra, use `np`as the alias
import pandas as pd  # data processing, use `pd` as the alias
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt  # native visualization
import seaborn as sns  # visualization but more friendly interface


def get_acc_rf(x_train, x_test, y_train, y_test, n_estimators, max_depth):
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    clf.fit(x_train, y_train)
    acc = clf.score(x_test, y_test).round(3)
    return acc


def grid_search_rf(x_train, x_test, y_train, y_test, ls_n_estimators, ls_max_depth):
    # grid search
    ls_results = []
    for n_estimators in ls_n_estimators:
        for max_depth in ls_max_depth:
            acc = get_acc_rf(x_train, x_test, y_train, y_test, n_estimators, max_depth)
            ls_results.append([n_estimators, max_depth, acc])
    # return the results
    return np.array(ls_results)


def val_method1(x_train, x_test, y_train, y_test, ls_n_estimators, ls_max_depth):
    # grid search the entire dataset
    result = grid_search_rf(
        x_train, x_test, y_train, y_test, ls_n_estimators, ls_max_depth
    )
    # find the best hyperparameters
    idx_best = np.argmax(result[:, 2])
    n_estimators_best, max_depth_best = result[idx_best, :2]
    # final evaluation
    final_acc = get_acc_rf(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        n_estimators=int(n_estimators_best),
        max_depth=int(max_depth_best),
    )
    # report
    return final_acc


def val_method2(x_train, x_test, y_train, y_test, ls_n_estimators, ls_max_depth):
    # split the entire dataset into training and testing sets
    x_train_s, x_val, y_train_s, y_val = train_test_split(
        x_train, y_train, test_size=0.2
    )
    # grid search the entire dataset
    result = grid_search_rf(
        x_train_s, x_val, y_train_s, y_val, ls_n_estimators, ls_max_depth
    )
    # find the best hyperparameters
    idx_best = np.argmax(result[:, 2])
    n_estimators_best, max_depth_best = result[idx_best, :2]
    # final evaluation
    final_acc = get_acc_rf(
        x_train=x_train,
        x_test=x_test,  # the FIRST time we use the testing set!
        y_train=y_train,
        y_test=y_test,  # the FIRST time we use the testing set!
        n_estimators=int(n_estimators_best),
        max_depth=int(max_depth_best),
    )
    # report
    return final_acc
