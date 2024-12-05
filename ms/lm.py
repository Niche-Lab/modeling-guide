# import linear model from sklearn
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
import pandas as pd


# ccc correlation
def CCC(y_true, y_pred):
    """Concordance correlation coefficient."""
    # Raw data
    dct = {"y_true": y_true, "y_pred": y_pred}
    df = pd.DataFrame(dct)
    # Remove NaNs
    df = df.dropna()
    # Pearson product-moment correlation coefficients
    y_true = df["y_true"]
    y_pred = df["y_pred"]
    cor = np.corrcoef(y_true, y_pred)[0][1]
    # Means
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    # Population variances
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    # Population standard deviations
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    return numerator / denominator


# generate X and y
P = 2
N = int(50)

n_train = 20
X = np.random.rand(N, P)
b = np.random.rand(P)
y = np.dot(X, b) + np.random.randn(N) * 0.1
# fit
model = lm.LinearRegression()
x_train, x_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]
model.fit(x_train, y_train)

# predict
pre = model.predict(x_train)
obs = y_train
errors = pre - obs
np.mean(errors)
np.cov(pre, errors)
plt.scatter(obs, pre)


CCC(obs, pre * 10)
np.corrcoef(obs, pre * 10)[0, 1]
r2_score(obs, pre * 10)

# correlation
cor = np.corrcoef(obs, pre)[0, 1]
# R2 coefficient
r2 = r2_score(obs, pre)
cor**2, r2

np.var(pre) / np.var(obs)


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([2, 3, 4, 5, 8])
rmse(y_true, y_pred)
mae(y_true, y_pred)
