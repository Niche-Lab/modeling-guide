import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def vis_bw(data, figsize=(12, 8)):
    plt.figure(figsize=figsize)
    sns.lineplot(x="month", y="bw", hue="sire", data=data)


def sim_bw(sdu=100, sde=100, n_months=24, n_sires=3, n_off=10, seed=0):
    np.random.seed(seed)
    # constants
    n_cows = n_sires * n_off
    n = n_months * n_cows
    # dataframe
    data = init_data(n_months, n_sires, n_off)
    # incidence matrix
    X = make_X(data)
    Z = make_Z(data, n_months, n_sires)
    # fixed effects
    b = define_b()
    # random effects
    u = sample_u(sdu, n_months, n_sires)
    e = sample_e(sde, n)
    # simulate
    data["bw"] = X @ b + Z @ u + e
    # return
    return data


def init_data(n_months=24, n_sires=3, n_off=10):
    n = n_months * n_sires * n_off
    data = np.zeros((n, 3))  # cow, sires, months
    count = 0
    for i in range(1, n_sires + 1):
        for j in range(1, n_off + 1):
            for k in range(1, n_months + 1):
                data[count, :] = [i, j, k]
                count += 1
    data = pd.DataFrame(data, columns=["sire", "cow", "month"], dtype=int)
    # return
    return data


def define_b():
    # fixed effects
    b = (
        np.array(
            [
                [100]
                + [80] * 10  # intercept  # first 10 months gains 50 lbs per month
                + [50] * 5  # next 5 months gains 30 lbs per month
                + [30] * 5  # next 5 months gains 20 lbs per month
                + [20] * 4  # next 4 months gains 10 lbs per month
            ]
        )
        .flatten()
        .cumsum()
    )
    return b


def sample_A(n):
    a = np.identity(n)
    for i in range(1, n):
        for j in range(i):
            a[i, j] = a[j, i] = 1 / np.random.choice([2, 4, 8])
    return a


def sample_e(sde, n):
    return np.random.normal(0, sde, n)


def sample_u(sdu, n_months=24, n_sires=3):
    A = sample_A(n_sires)
    vu = np.random.normal(0, sdu) ** 2
    ucov = vu * A
    u = np.random.multivariate_normal(np.zeros(n_sires), ucov, n_months + 1)
    u = u.T.reshape(-1)  # s1, s1m1, s1m2,...,s1m24, s2, s2m1,...
    return u


def make_X(data):
    n = data.shape[0]
    # month effects (fixed)
    X = pd.get_dummies(data["month"], prefix="m").to_numpy() * 1
    # add intercept
    X = np.concatenate((np.ones((n, 1)), X), axis=1)
    # return
    return X


def make_Z(data, n_months=24, n_sires=3):
    # I. sire effects
    Zs = pd.get_dummies(data["sire"], prefix="s").to_numpy() * 1

    # II. interaction sire and month
    n = data.shape[0]
    # create interaction terms
    data.loc[:, "m/s"] = data["month"].astype(str) + "_" + data["sire"].astype(str)
    pd_dummy = pd.get_dummies(data["m/s"], prefix="m/s")
    # create pd_temp with all columns
    col_all = [
        "m/s_%d_%d" % (i, j)
        for i in range(1, n_months + 1)
        for j in range(1, n_sires + 1)
    ]
    pd_temp = pd.DataFrame(np.zeros((n, len(col_all))), columns=col_all)
    # assign columns from data to pd_temp
    for col in pd_dummy.columns:
        if col in col_all:
            pd_temp.loc[:, col] = pd_dummy.loc[:, col]
    # convert to numpy
    Zinter = pd_temp.to_numpy() * 1

    # III. concatenate Zs and Zinter
    Z = np.concatenate((Zs, Zinter), axis=1)

    # IV. return
    return Z
