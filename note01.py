# python libs imports
import numpy as np  # linear algebra, use `np`as the alias
import pandas as pd  # dataframe, use `pd` as the alias
from scipy.stats import pearsonr  # correlation coefficient
from sklearn.linear_model import LinearRegression  # OLS
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt  # native visualization
import seaborn as sns  # visualization but more friendly interface


def make_data(n, p, rank=-1, p_assoc=0, dist="normal", precision=4):
    """
    Generate a random dataset with X having n observations and p variables and a target variable Y.
    The first p_assoc variables are associated with the target variable.

    Parameters
    ----------
    n : int
        Number of observations.
    p : int
        Number of variables.
    rank : int
        Rank of X. If -1, X is a full-rank matrix.
    p_assoc : int
        Number of variables associated with the target variable Y.
    dist : str
        Distribution of Y. "normal", "exponential", or "uniform".

    Returns
    -------
    X : array-like of shape (n, p)
        The feature matrix.
    y : array-like of shape (n,)
        The target values.
    """
    # generate n random variables
    X = np.random.normal(loc=0, scale=1, size=(n, p))

    # if the matrix is not full-rank
    if rank != -1:
        # compute the SVD of X
        U, S, V = np.linalg.svd(X, full_matrices=False)
        # set the smallest singular values to zero
        S[rank:] = 0
        # reconstruct the matrix X
        X = U.dot(np.diag(S)).dot(V)

    # if there is no association between X and Y
    if p_assoc == 0:
        if dist == "normal":
            Y = np.random.normal(loc=0.0, scale=1, size=n)
        elif dist == "exponential":
            Y = np.random.exponential(scale=1, size=n)
        elif dist == "uniform":
            Y = np.random.uniform(low=0, high=1, size=n)
        else:
            raise ValueError(
                "dist must be one of 'normal', 'exponential', or 'uniform'."
            )
    # if there are p_assoc variables associated with Y
    else:
        # Y = Xb + error
        beta = np.random.normal(loc=0, scale=1, size=p_assoc)
        idx_p = np.random.choice(p, p_assoc, replace=False)
        error = np.random.normal(loc=0, scale=1, size=n)
        Y = X[:, idx_p].dot(beta) + error

    return X.round(precision), Y.round(precision)


def get_dataframe(X, y):
    """
    Generate a dataframe from X and y.
    """
    n, p = X.shape
    data = pd.DataFrame(["cow %d" % i for i in range(1, n + 1)], columns=["id"])
    for i in range(p):
        data.loc[:, "x_" + str(i + 1)] = X[:, i]
    data.loc[:, "y"] = y
    return data


def get_p_value(dist, threshold):
    dist = np.array(dist)
    bol_pass = dist <= threshold
    p_value = np.mean(bol_pass)
    return p_value.round(4)


def ols_r_dist(n, p, niter, p_assoc=0):
    # run the simulation
    dist_r = []
    for i in range(niter):
        X, y = make_data(n, p, p_assoc=p_assoc)  # sample a dataset
        y_hat = get_yhat_by_ols(X, y)  # obtain the fitted values
        r = pearsonr(y, y_hat)[0]  # the first element is the correlation coefficient
        dist_r.append(r)  # concatenate the corre
    return np.array(dist_r)


def ols_r_dist_val(n, p, niter, p_assoc=0):
    dist_r = []
    for i in range(niter):
        X, y = make_data(n, p, p_assoc=p_assoc)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        y_hat = get_yhat_by_ols_val(x_train, x_test, y_train)
        r = pearsonr(y_test, y_hat)[0]
        dist_r.append(r)
    return np.array(dist_r)


def get_yhat_by_ols(X, y):
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    y_hat = linear_model.predict(X)
    return y_hat


def get_yhat_by_ols_val(x_train, x_test, y_train):
    linear_model = LinearRegression()
    linear_model.fit(x_train, y_train)
    y_hat = linear_model.predict(x_test)
    return y_hat
