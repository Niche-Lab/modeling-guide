from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


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


def truth_estimation(X, Y, n, p, niter=100):
    model = LinearRegression()
    model.fit(X, Y)
    cor = []
    r2 = []
    rmse = []
    for _ in range(niter):
        new_X, new_Y = sample_data(n, p)
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
