from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR


def idx_top_ft(X, Y, top_n=50):
    n, p = X.shape
    cor = [None] * p
    for i in range(p):
        cor[i] = pearsonr(X[:, i], Y)[0] ** 2
    cor = np.array(cor)
    top_idx = np.argsort(cor)[-top_n:]
    return top_idx


def sample_data(n, p):
    X = np.random.normal(0, 1, (n, p))
    Y = np.random.normal(0, 1, n)
    return X, Y


def cor_score(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0] ** 2


def get_PLSR_score(x_train, y_train, x_test, y_test, n_comp):
    model = PLSRegression(n_components=n_comp).fit(x_train, y_train)
    return model.score(x_test, y_test)


def get_SVR_score(x_train, y_train, x_test, y_test, kernel):
    model = SVR(kernel=kernel).fit(x_train, y_train)
    return model.score(x_test, y_test)


def suggest_kernel(x_train, y_train, x_test, y_test):
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    scores = []
    for kernel in kernels:
        score = get_SVR_score(x_train, y_train, x_test, y_test, kernel)
        scores.append(score)
    return kernels[np.argmax(scores)]


def count_most_freq_str(lst):
    return max(set(lst), key=lst.count)
