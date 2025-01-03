import os
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd

# local imports
from utils import (
    idx_top_ft,
    sample_data,
    get_SVR_score,
    suggest_kernel,
    count_most_freq_str,
)

# CONSTANTS
N_SAMPLE = 100
N_FT = 1000
N_FT_SELECT = 50
N_ITER = 1000
K = 5
SEED = 24061
FILE_OUT = "hparm2.csv"

np.random.seed(SEED)
pd.DataFrame(columns=["approach", "r", "iter"]).to_csv(
    FILE_OUT, index=False, mode="w", header=True
)
for i in range(N_ITER):
    scores = {}
    X, Y = sample_data(N_SAMPLE, N_FT)

    # scenario 1: RVT, RV/T
    score_k = []
    kfold = KFold(n_splits=K)
    # ft select (RVT)
    idx_select = idx_top_ft(X, Y, N_FT_SELECT)
    X_select = X[:, idx_select]
    for idx_train, idx_test in kfold.split(X_select):
        # split
        X_train, Y_train = X_select[idx_train], Y[idx_train]
        X_test, Y_test = X_select[idx_test], Y[idx_test]
        # hp tuning (RV/T)
        kernel = suggest_kernel(X_train, Y_train, X_test, Y_test)
        score_k.append(get_SVR_score(X_train, Y_train, X_test, Y_test, kernel))
    # average scores
    scores["s1"] = np.mean(score_k)

    # scenario 2: RV, RV/T
    score_k = []
    kfold = KFold(n_splits=K)
    for idx_train, idx_test in kfold.split(X):
        # split
        X_train, Y_train = X[idx_train], Y[idx_train]
        X_test, Y_test = X[idx_test], Y[idx_test]
        # ft select (RV)
        idx_select = idx_top_ft(X_train, Y_train, N_FT_SELECT)
        X_train, X_test = X_train[:, idx_select], X_test[:, idx_select]
        # hp tuning (RV/T)
        kernel = suggest_kernel(X_train, Y_train, X_test, Y_test)
        score_k.append(get_SVR_score(X_train, Y_train, X_test, Y_test, kernel))
    scores["s2"] = np.mean(score_k)

    # scenario 3: RV, R/V
    score_k = []
    kfold_out = KFold(n_splits=K)
    # ft select (RVT)
    idx_select = idx_top_ft(X, Y, N_FT_SELECT)
    X_select = X[:, idx_select]
    for idx_train_out, idx_test in kfold_out.split(X_select):
        # outer split
        X_train_o, Y_train_o = X_select[idx_train_out], Y[idx_train_out]
        X_test, Y_test = X_select[idx_test], Y[idx_test]

        kernels = []
        kfold_in = KFold(n_splits=K)
        for idx_train, idx_val in kfold_in.split(X_train_o):
            # inner split
            X_train, Y_train = X_train_o[idx_train], Y_train_o[idx_train]
            X_val, Y_val = X_train_o[idx_val], Y_train_o[idx_val]
            # hp tuning (R/V)
            kernel = suggest_kernel(X_train, Y_train, X_val, Y_val)
            kernels.append(kernel)

        kernel_select = count_most_freq_str(kernels)
        score_k.append(
            get_SVR_score(X_train_o, Y_train_o, X_test, Y_test, kernel_select)
        )
    scores["s3"] = np.mean(score_k)

    # scenario 4: R, R/V
    score_k = []
    kfold_out = KFold(n_splits=K)
    for idx_train_out, idx_test in kfold_out.split(X):
        # outer split
        X_train_o, Y_train_o = X[idx_train_out], Y[idx_train_out]
        X_test, Y_test = X[idx_test], Y[idx_test]

        kernels = []
        kfold_in = KFold(n_splits=K)
        for idx_train, idx_val in kfold_in.split(X_train_o):
            # inner split
            X_train, Y_train = X_train_o[idx_train], Y_train_o[idx_train]
            X_val, Y_val = X_train_o[idx_val], Y_train_o[idx_val]
            # ft select (R)
            idx_select = idx_top_ft(X_train, Y_train, N_FT_SELECT)
            X_train, X_val = X_train[:, idx_select], X_val[:, idx_select]
            # hp tuning (R/V)
            kernel = suggest_kernel(X_train, Y_train, X_val, Y_val)
            kernels.append(kernel)
        kernel_select = count_most_freq_str(kernels)
        idx_select_o = idx_top_ft(X_train_o, Y_train_o, N_FT_SELECT)
        X_train_o, X_test = X_train_o[:, idx_select_o], X_test[:, idx_select_o]
        score_k.append(
            get_SVR_score(X_train_o, Y_train_o, X_test, Y_test, kernel_select)
        )
    scores["s4"] = np.mean(score_k)

    # output
    df_out = pd.DataFrame(scores, index=[0]).melt()
    df_out["iter"] = i
    df_out.to_csv(FILE_OUT, index=False, mode="a", header=False)


# # visualization
data = pd.read_csv("hparm.csv")

# set s1, s3 to the column: val=1
data.loc[:, ["use val"]] = 0
data.loc[data["approach"].isin(["s3", "s4"]), ["use val"]] = 1

data.loc[:, ["select train"]] = 0
data.loc[data["approach"].isin(["s2", "s4"]), ["select train"]] = 1

from matplotlib import pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
sns.set_palette("Set2")
g = sns.boxplot(
    x="select train",
    y="r",
    hue="use val",
    gap=0.05,
    data=data,
)
g.axhline(y=0, color="black", linestyle="--", linewidth=0.7)
# remove legend
g.get_legend().remove()
g.set_yticks(ticks=np.arange(-0.5, 1, 0.1), minor=True)
plt.savefig("hparm.png", dpi=300)


# summary
data.groupby("approach").median()
# s1 0.797384	499.5	0.0	0.0
# s2 0.112664	499.5	0.0	1.0
# s3 0.761103	499.5	1.0	0.0
# s4 -0.007723	499.5   1.0 1.0, p-value=0.562
st = data.query("approach == 's4'")["r"].values
# check p-value greater than 0
from scipy.stats import ttest_1samp

ttest_1samp(st, 0)

np.mean(st > 0)
