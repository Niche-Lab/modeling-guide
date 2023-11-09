from os import times_result
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import time as t

N_ITER = 1000
SEED = 24061
P = 10
N = [50, 100, 500]

# unseen data
NEW_N = 100  # sample size of unseen data
NEW_ITER = 100  # number of times to sample unseen data


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
        new_X, new_Y = sample_data(NEW_N, P)
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


def sample_data(n, p):
    X = np.random.normal(0, 1, (n, p))
    Y = np.random.normal(0, 1, n)
    return X, Y


np.random.seed(SEED)
dict_scores = dict()
for n in N:
    score = dict(
        {
            "In-Sample": [None] * N_ITER,
            "2-Fold CV": [None] * N_ITER,
            "5-Fold CV": [None] * N_ITER,
            "10-Fold CV": [None] * N_ITER,
            "LOOCV": [None] * N_ITER,
            "Truth": [None] * N_ITER,
        }
    )
    for i in range(N_ITER):
        print(f"Sample Size: {n}, Iteration: {i}")
        X, Y = sample_data(n, P)
        # in-sample
        time_cur = t.time()
        scores = get_OLS_scores(X, Y, X, Y)
        time_elapsed = t.time() - time_cur
        scores += (time_elapsed,)
        score["In-Sample"][i] = scores
        # 2-fold, 5-fold, 10-fold
        for k in [2, 5, 10]:
            time_cur = t.time()
            scores = k_fold_cv(X, Y, k)
            time_elapsed = t.time() - time_cur
            scores += (time_elapsed,)
            score[f"{k}-Fold CV"][i] = scores
        # LOOCV
        time_cur = t.time()
        scores = LOOCV(X, Y)
        time_elapsed = t.time() - time_cur
        scores += (time_elapsed,)
        score["LOOCV"][i] = scores
        # truth
        scores = truth_estimation(X, Y)
        score["Truth"][i] = scores
    dict_scores[f"N = {n}"] = score

data = pd.DataFrame(columns=["N", "Method", "Metric", "Bias", "Variance", "Time"])
row = {}
for n in N:
    d = dict_scores[f"N = {n}"]
    for key in ["In-Sample", "2-Fold CV", "5-Fold CV", "10-Fold CV", "LOOCV"]:
        for i in range(len(d[key])):
            if key == "In-Sample":
                # since In-Sample only makes one prediction
                (est_cor, est_r2, est_rmse, time) = d[key][i]
                var_rmse = 0
            else:
                (est_cor, est_r2, est_rmse), var_rmse, time = d[key][i]
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
            row["Time"] = [time] * 3
            data = pd.concat([data, pd.DataFrame(row)], ignore_index=True)

data.to_csv("cv10.csv", index=False)

# 1000 * 3 * 5 *

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
        linewidth=0.7,
    )
    sns.boxplot(
        x="N",
        y=m,
        hue="Method",
        data=data_fg1,
        ax=axes[i],
        linewidth=0.5,
        fliersize=4,
    )
    axes[i].set_xlabel("Sample Size (N)")
    axes[i].set_ylabel(m)
axes[0].set_title("Bias")
axes[0].get_legend().remove()
axes[1].set_title("Variance")
fig.suptitle("Bias and Variance of RMSE by Method")
fig.savefig("bias_var_rmse.png", dpi=300)


# figure 2
sns.set_palette("Set3")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, m in enumerate(["Correlation", "R2", "RMSE"]):
    axes[i].axhline(
        y=0,
        color="black",
        linestyle="--",
        linewidth=0.7,
    )
    sns.boxplot(
        x="N",
        y="Bias",
        hue="Method",
        data=data.query("Metric == '%s'" % m),
        ax=axes[i],
        linewidth=0.5,
        fliersize=4,
    )
    axes[i].set_xlabel("Sample Size (N)")
    axes[i].set_ylabel(m)

# titles
fig.suptitle("Bias of Metrics by Validation Method with 10 Predictors")
axes[0].set_title("Correlation (r)")
axes[1].set_title("Coefficient of Determination (R2)")
axes[2].set_title("RMSE")

# y axis
axes[0].set_ylim(-0.55, 0.7)
axes[1].set_ylim(-2.5, 1.3)
axes[1].set_yscale("symlog", base=2)
axes[1].set_yticks([-2, -1, 0, 1])
axes[2].set_ylim(-0.55, 0.7)

# legend
axes[0].get_legend().remove()
axes[1].legend(loc="lower right", ncol=1)
axes[2].get_legend().remove()

# save
fig.savefig("bias_metrics.png", dpi=300)

# report
data.groupby(["Method", "Metric", "N"]).aggregate(["median"]).reset_index().to_csv(
    "cv_summary.csv", index=False
)


# figure 3 time
data_time = data.query("Metric == 'RMSE'").loc[:, ["N", "Method", "Time"]]

fig, axe = plt.subplots(figsize=(8, 8))
sns.barplot(
    data=data_time,
    x="Method",
    y="Time",
    hue="N",
    errorbar="sd",
    err_kws={"linewidth": 1},
    ax=axe,
)
axe.set_yscale("log", base=10)


data_time.groupby(["Method", "N"]).agg("median").reset_index()
# 	Method	N	Time
# 0	10-Fold CV	50	0.006544
# 1	10-Fold CV	100	0.006502
# 2	10-Fold CV	500	0.006801
# 3	2-Fold CV	50	0.001401
# 4	2-Fold CV	100	0.001396
# 5	2-Fold CV	500	0.001449
# 6	5-Fold CV	50	0.003316
# 7	5-Fold CV	100	0.003307
# 8	5-Fold CV	500	0.003437
# 9	In-Sample	50	0.000711
# 10	In-Sample	100	0.000720
# 11	In-Sample	500	0.000783
# 12	LOOCV	50	0.009949
# 13	LOOCV	100	0.019340
# 14	LOOCV	500	0.112297
0.001401
0.003316
0.006544
data_time.groupby(["Method"]).agg("median").reset_index()

# Method	N	Time
# 0	10-Fold CV	100.0	0.006591
# 1	2-Fold CV	100.0	0.001392
# 2	5-Fold CV	100.0	0.003334
# 3	In-Sample	100.0	0.000713
# 4	LOOCV	100.0	0.019251
