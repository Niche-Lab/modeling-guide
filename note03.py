import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import statsmodels.formula.api as smf


def sim_bw(sdu=100, sde=50, n_months=24, n_sires=3, n_off=10):
    # constants
    n_cows = n_sires * n_off
    n = n_months * n_cows
    # dataframe
    data = _init_data(n_months, n_sires, n_off)
    # incidence matrix
    X = _make_X(data)
    Z = _make_Z(data, n_months, n_sires)
    # fixed effects
    b = _define_b()
    # random effects
    u = _sample_u(sdu, n_months, n_sires)
    e = _sample_e(sde, n)
    # simulate
    data["bw"] = X @ b + Z @ u + e
    # correct pseudo sires for the computational covenience
    # {1, 2, 3} -> {3, 4, 5}
    data["sire"] += 2
    # return
    return data


def vis_bw(data, figsize=(12, 8)):
    sns.set_theme(style="whitegrid", palette="Set2")
    plt.figure(figsize=figsize)
    sns.lineplot(x="month", y="bw", hue="sire", palette=["C0", "C1", "C2"], data=data)
    plt.xlim(1, 24)
    plt.xlabel("Month")
    plt.ylabel("Body weight (lbs)")
    plt.title("Simulated body weight")


def merge_train_test(data_train, data_test):
    col_final = ["cow", "sire", "month", "observed_bw", "predicted_bw", "split"]

    # test
    data_all = data_test.loc[:, ["cow", "sire", "month", "bw", "bw_pre"]]
    data_all.loc[:, "split"] = "test"
    data_all.columns = col_final

    # train
    data_train.loc[:, "predicted_bw"] = data_train["bw"]
    data_train.loc[:, "split"] = "train"
    data_train_tmp = data_train.loc[
        :, ["cow", "sire", "month", "bw", "predicted_bw", "split"]
    ]
    data_train_tmp.columns = col_final

    # merge and return
    data_all = pd.concat([data_all, data_train_tmp], axis=0)
    data_all = data_all.melt(
        id_vars=["cow", "sire", "month", "split"],
        value_vars=["observed_bw", "predicted_bw"],
    )
    data_all.loc[data_all["split"] == "train", "variable"] = "trained_bw"
    return data_all


def fit_and_predict(data_train, data_test):
    # learn coefficients from the training ste
    model_ols = smf.ols("bw ~ month + x1 + x2 + x3", data_train).fit()
    # predict on the testing set
    data_test.loc[:, "bw_pre"] = model_ols.predict(data_test)
    # evaluation
    obs = data_test["bw"]
    pre = data_test["bw_pre"]
    # evaluation
    r = pearsonr(obs, pre)[0].round(3)
    mse = mean_squared_error(obs, pre).round(3)
    # print results
    print(f"r = {r}, mse = {mse}")
    # merge
    data_all = merge_train_test(data_train, data_test)
    return data_all, r, mse


def vis_linechart(data_test):
    data_plot = data_test.melt(id_vars=["sire", "month"], value_vars=["bw", "bw_pre"])
    plot = sns.relplot(
        x="month",
        y="value",
        hue="variable",
        style="variable",
        col="sire",
        col_wrap=2,
        kind="line",
        height=3,
        aspect=1.5,
        data=data_plot,
    )
    (
        plot.map(plt.axvline, x=16, color="k", linestyle="--")
        .set_axis_labels("Month", "Body weight (lb)")
        .tight_layout(w_pad=0)
    )


def vis_obspre(data_test, r, mse):
    sns.scatterplot(x="bw", y="bw_pre", hue="sire", data=data_test)
    plt.title(f"Model accuracy: r = {r}, mse = {mse}")
    plt.xlabel("Observed body weight (lbs)")
    plt.ylabel("Predicted body weight (lbs)")


# private ---------------------------------------------------------------------


def _init_data(n_months=24, n_sires=3, n_off=10):
    n = n_months * n_sires * n_off
    data = np.zeros((n, 3))  # cow, sires, months
    count = 0
    for i in range(1, n_sires + 1):
        for j in range(1, n_off + 1):
            for k in range(1, n_months + 1):
                data[count, :] = [i, j, k]
                count += 1
    data = pd.DataFrame(data, columns=["sire", "cow", "month"], dtype=int)
    # add three arbitrary predictors
    data["x1"] = np.random.normal(0, 1, n)
    data["x2"] = np.random.normal(0, 1, n)
    data["x3"] = np.random.normal(0, 1, n)
    # return
    return data


def _define_b():
    # reference
    # https://extension.psu.edu/growth-charts-for-dairy-heifers
    # range from 100-150 lbs to 1300-1500 lbs
    # fixed effects
    b = (
        np.array(
            [
                [100]  # intercept
                + [100] * 10  # first 10 months gains 50 lbs per month
                + [70] * 5  # next 5 months gains 30 lbs per month
                + [10] * 5  # next 5 months gains 20 lbs per month
                + [5] * 4  # next 4 months gains 10 lbs per month
            ]
        )
        .flatten()
        .cumsum()
    )
    # arbitrary predictors effects (x1, x2, x3)
    b = np.concatenate((b, np.array([50, 100, 200])))
    return b


def _sample_u(sdu, n_months=24, n_sires=3):
    # random effects
    n_test = int(n_sires / 3)
    n_train = n_sires - n_test
    ped = pd.DataFrame(
        {
            "id": [1, 2] + list(range(3, 3 + n_sires)),
            "sire": [0, 0] + [1] * n_train + [0] * n_test,
            "dam": [0, 0] + [2] * n_train + [0] * n_test,
        }
    )
    A = _get_A(ped)[-n_sires:, -n_sires:]
    vu = sdu**2
    ucov = vu * A
    u = np.random.multivariate_normal(
        [0] * n_train + [100] * n_test, ucov, n_months + 1
    )
    u = u.T.reshape(-1)  # s1, s1m1, s1m2,...,s1m24, s2, s2m1,...
    return u


def _get_A(df_ped):
    # Mrode, R. A. (2014). Linear models for the prediction of animal breeding values.
    print("--- Simulated Pedigree ---")
    print(df_ped)
    print("--------------------------")
    n = len(df_ped)
    A = np.identity(n)
    for i in range(n):
        s = df_ped.loc[i, "sire"] - 1  # matrix index
        d = df_ped.loc[i, "dam"] - 1  # matrix index
        for j in range(i):
            if s != -1 and d != -1:
                ajs = A[s, j]
                ajd = A[d, j]
                A[i, j] = 0.5 * (ajs + ajd)
                A[j, i] = A[i, j]
                A[i, i] = 1 + 0.5 * A[s, d]
            elif s == -1 and d != -1:
                ajd = A[d, j]
                A[i, j] = 0.5 * ajd
                A[j, i] = A[i, j]
                A[i, i] = 1
            elif s != -1 and d == -1:
                ajs = A[s, j]
                A[i, j] = 0.5 * ajs
                A[j, i] = A[i, j]
                A[i, i] = 1
            else:
                A[i, j] = 0
                A[j, i] = A[i, j]
                A[i, i] = 1
    return A


def _sample_e(sde, n):
    # random effects
    return np.random.normal(0, sde, n)


def _make_X(data):
    n = data.shape[0]
    # month effects (fixed)
    X_month = pd.get_dummies(data["month"], prefix="m").to_numpy() * 1
    # arbitrary predictors
    X_arb = data[["x1", "x2", "x3"]].to_numpy() * 1
    # concatenate
    X = np.concatenate((X_month, X_arb), axis=1)
    # add intercept
    X = np.concatenate((np.ones((n, 1)), X), axis=1)
    # return
    return X


def _make_Z(data, n_months=24, n_sires=3):
    # I. sire effects
    Zs = pd.get_dummies(data["sire"], prefix="s").to_numpy() * 1

    # II. interaction sire and month
    n = data.shape[0]
    # create interaction terms
    inter_terms = data["month"].astype(str) + "_" + data["sire"].astype(str)
    pd_dummy = pd.get_dummies(inter_terms, prefix="m/s")

    # III. create Z
    Z = np.zeros((n, n_months * n_sires + n_sires))
    i = 0
    for s in range(n_sires):
        Z[:, i] = Zs[:, s]
        i += 1
        for m in range(n_months):
            col = "m/s_%d_%d" % (m + 1, s + 1)
            Z[:, i] = pd_dummy.loc[:, col]
            i += 1

    # IV. return
    return Z


# # data = sim_bw()
# # vis_bw(data)


# # constants
# n_cows = n_sires * n_off
# n = n_months * n_cows
# # dataframe
# data = _init_data(n_months, n_sires, n_off)
# # incidence matrix
# X = _make_X(data)
# Z = _make_Z(data, n_months, n_sires)
# # fixed effects
# b = _define_b()
# # random effects
# u = _sample_u(sdu, n_months, n_sires)

# e = _sample_e(sde, n)
# # simulate
# data["bw"] = X @ b + Z @ u + e
# # correct pseudo sires for the computational covenience
# # {1, 2, 3} -> {3, 4, 5}
# data["sire"] += 2

# sns.set_theme(style="whitegrid", palette="Set2")
# plt.figure(figsize=(12, 8))
# sns.heatmap(X, cmap="Blues", cbar=False)
# sns.heatmap(Z, cmap="Blues", cbar=False)
# sns.heatmap(u.reshape((1, -1)), cmap="Blues", cbar=False)


# A = np.array([[1, 2, 3], [2, 1, 3], [3, 3, 1]])

# I = np.eye(2)

# np.kron(A, I)
