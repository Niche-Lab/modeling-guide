import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


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


def get_r(x, y):
    """
    Compute the Pearson's correlation coefficient r.

    Parameters
    ----------
    x : array-like of shape (n,)
        The true values.
    y : array-like of shape (n,)
        The predicted values.

    Returns
    -------
    rr : float
        The Pearson's correlation coefficient r.
    """
    return np.corrcoef(x, y)[0, 1]


def get_p_value(dist, threshold):
    dist = np.array(dist)
    bol_pass = dist > threshold
    prop_pass = np.mean(bol_pass)
    p_value = 1 - prop_pass
    return p_value.round(4)


def get_yhat_by_ols(X, y):
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    y_hat = linear_model.predict(X)
    return y_hat
