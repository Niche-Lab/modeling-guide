from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR


def sample_data(n, p):
    X = np.random.normal(0, 1, (n, p))
    Y = np.random.normal(0, 1, n)
    return X, Y
