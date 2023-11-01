import enum
from math import exp
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd


N_ITER = 100
SEED = 24061
P = 10
N = [50, 100, 500]

# unseen data
NEW_N = 100
NEW_ITER = 10


def LOOCV(X, Y):
    y_hat = []
    rmse = []
    for i in range(len(X)):
        # split data
        X_train = np.delete(X, i, axis=0)
        Y_train = np.delete(Y, i, axis=0)
        X_test = X[i].reshape(1, -1)  # cast to 2d array (P, ) -> (1, P)
        # fit OLS
        model = LinearRegression()
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        y_hat.append(y_pred)
        # calculate squared error
        rmse.append(np.sqrt((Y[i] - y_pred) ** 2))
    # calculate scores
    score = get_scores(Y, y_hat)
    var_rmse = np.var(np.array(rmse))
    return score, var_rmse


def k_fold_cv(X, Y, K):
    scores = []
    rmse = []
    kfold = KFold(n_splits=K)
    for train_indices, test_indices in kfold.split(X):
        # split data
        X_train, Y_train = X[train_indices], Y[train_indices]
        X_test, Y_test = X[test_indices], Y[test_indices]
        # fit OLS
        score = get_OLS_scores(X_train, Y_train, X_test, Y_test)
        rmse.append(score[2])
        scores.append(score)
    # taking average of scores
    scores = np.array(scores)  # (K, 3)
    var_rmse = np.var(np.array(rmse))
    final_score = np.mean(scores, axis=0)  # (3, )
    return final_score, var_rmse


def truth_estimation(X, Y, niter=NEW_ITER):
    model = LinearRegression()
    model.fit(X, Y)
    cor = []
    r2 = []
    rmse = []
    for _ in range(niter):
        new_X = np.random.randn(NEW_N, P)
        new_Y = np.random.uniform(0, 1, NEW_N)
        new_Y_pred = model.predict(new_X)
        new_cor, new_r2, new_rmse = get_scores(new_Y, new_Y_pred)
        cor.append(new_cor)
        r2.append(new_r2)
        rmse.append(new_rmse)
    return cor, r2, rmse


def get_OLS_scores(X_train, Y_train, X_test, Y_test):
    model = LinearRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    score = get_scores(Y_test, Y_pred)
    return score


def get_scores(x, y):
    x = np.array(x).reshape(-1)  # cast to 1d array (N, 1) -> (N, )
    y = np.array(y).reshape(-1)  # cast to 1d array (N, 1) -> (N, )
    cor = pearsonr(x, y)[0]
    r2 = r2_score(x, y)
    rmse = np.sqrt(np.mean((x - y) ** 2))
    return cor, r2, rmse


np.random.seed(SEED)
scores = dict()
for n in N:
    score = dict(
        {
            "In-Sample": [],
            "2-Fold CV": [],
            "5-Fold CV": [],
            "10-Fold CV": [],
            "LOOCV": [],
            "Truth": [],
        }
    )
    for i in range(N_ITER):
        X = np.random.randn(n, P)
        Y = np.random.uniform(0, 1, n)
        score["In-Sample"].append(get_OLS_scores(X, Y, X, Y))
        score["2-Fold CV"].append(k_fold_cv(X, Y, 2))
        score["5-Fold CV"].append(k_fold_cv(X, Y, 5))
        score["10-Fold CV"].append(k_fold_cv(X, Y, 10))
        score["LOOCV"].append(LOOCV(X, Y))
        score["Truth"].append(truth_estimation(X, Y))
    scores[f"N = {n}"] = score


data = pd.DataFrame(columns=["N", "Method", "Metric", "Bias", "Variance"])
row = {}
for n in N:
    d = scores[f"N = {n}"]
    for key in ["In-Sample", "2-Fold CV", "5-Fold CV", "10-Fold CV", "LOOCV"]:
        for i in range(len(d[key])):
            if key == "In-Sample":
                # since In-Sample only makes one prediction
                (est_cor, est_r2, est_rmse) = d[key][i]
                var_rmse = 0
            else:
                (est_cor, est_r2, est_rmse), var_rmse = d[key][i]
            (exp_cor, exp_r2, exp_rmse) = d["Truth"][i]
            exp_cor, exp_r2, exp_rmse = (
                np.mean(exp_cor),
                np.mean(exp_r2),
                np.mean(exp_rmse),
            )
            bias_cor, bias_r2, bias_rmse = (
                est_cor - exp_cor,
                est_r2 - exp_r2,
                est_rmse - exp_rmse,
            )
            row["N"] = [n] * 3
            row["Method"] = [key] * 3
            row["Metric"] = ["Correlation", "R2", "RMSE"]
            row["Bias"] = [bias_cor, bias_r2, bias_rmse]
            row["Variance"] = [0, 0, var_rmse]
            data = pd.concat([data, pd.DataFrame(row)], ignore_index=True)

data.to_csv("cv.csv", index=False)


# figure 1
original_palette = sns.color_palette("Set3", 12)
shifted_palette = original_palette[1:] + [original_palette[0]]
sns.set_theme(style="whitegrid")
sns.set_palette(shifted_palette)
data_fg1 = data.query("Metric == 'RMSE' and Method != 'In-Sample'")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for i, m in enumerate(["Bias", "Variance"]):
    axes[i].axhline(
        y=0,
        color="black",
        linestyle="--",
        linewidth=1,
    )
    sns.boxplot(
        x="N",
        y=m,
        hue="Method",
        data=data_fg1,
        ax=axes[i],
    )
    axes[i].set_xlabel("Sample Size (N)")
    axes[i].set_ylabel(m)
axes[0].set_title("Bias")
axes[0].get_legend().remove()
axes[0].set_ylim(-0.12, 0.2)
axes[1].set_title("Variance")
axes[1].set_ylim(-0.005, 0.045)
fig.suptitle("Bias and Variance of RMSE by Method")

# figure 2
sns.set_palette("Set3")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, m in enumerate(["Correlation", "R2", "RMSE"]):
    axes[i].axhline(
        y=0,
        color="black",
        linestyle="--",
        linewidth=1,
    )
    sns.boxplot(
        x="N",
        y="Bias",
        hue="Method",
        data=data.query("Metric == '%s'" % m),
        ax=axes[i],
    )
    axes[i].set_xlabel("Sample Size (N)")
    axes[i].set_ylabel(m)

fig.suptitle("Bias of Metrics by Validation Method")
axes[0].set_title("Correlation (r)")
axes[0].get_legend().remove()
axes[1].set_title("Coefficient of Determination (R2)")
axes[1].set_yscale("symlog", base=2)
axes[1].set_ylim(-4.5, 1.3)
axes[1].set_yticks([-4, -2, -1, 0, 1])
axes[2].set_title("RMSE")
axes[2].set_ylim(-0.2, 0.2)
axes[2].get_legend().remove()
