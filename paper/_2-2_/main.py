from matplotlib import pyplot as plt
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR, LinearSVR
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

# local imports
from utils import get_PLSR_score, rmse, sample_data, select_top_ft

# CONSTANTS
N_SAMPLE = 100
N_FT = 1000
N_FT_SELECT = 50
K = 5


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
        # ft select (RV)
        idx_select = idx_top_ft(X_train_o, Y_train_o, N_FT_SELECT)
        X_train, X_val = X_train[:, idx_select], X_val[:, idx_select]
        # hp tuning (R/V)
        kernel = suggest_kernel(X_train, Y_train, X_val, Y_val)
        kernels.append(kernel)

    kernel_select = count_most_freq_str(kernels)
    idx_select_o = idx_top_ft(X_train_o, Y_train_o, N_FT_SELECT)
    X_train_o, X_test = X_train_o[:, idx_select_o], X_test[:, idx_select_o]
    score_k.append(get_SVR_score(X_train_o, Y_train_o, X_test, Y_test, kernel_select))
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
    score_k.append(get_SVR_score(X_train_o, Y_train_o, X_test, Y_test, kernel_select))
scores["s4"] = np.mean(score_k)


scores
