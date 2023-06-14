import numpy as np
import pandas as pd


def make_data(n, p, rank=-1, p_assoc=0, dist="normal"):
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
    # Generate n random variables
    X = np.random.rand(n, p).round(4)

    # if the matrix is not full-rank
    if rank != -1:
        # compute the SVD of X
        U, S, V = np.linalg.svd(X, full_matrices=False)
        # set the smallest singular values to zero
        S[rank:] = 0
        # reconstruct the matrix X
        X = U.dot(np.diag(S)).dot(V)

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
    else:
        # Y = Xb + error
        beta = np.random.uniform(low=0, high=1, size=p_assoc)
        idx_p = np.random.choice(p, p_assoc, replace=False)
        error = np.random.normal(loc=0, scale=0.5, size=n)
        Y = X[:, idx_p].dot(beta) + error

    return X, Y


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


def get_r2(x, y):
    """
    Compute the squared correlation coefficient R^2.

    Parameters
    ----------
    x : array-like of shape (n,)
        The true values.
    y : array-like of shape (n,)
        The predicted values.

    Returns
    -------
    r2 : float
        The squared correlation coefficient R^2.
    """
    return np.corrcoef(x, y)[0, 1] ** 2
