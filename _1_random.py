import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# correlation


# Number of random variables
n = 5

# Number of observations
m = 5

# Generate n random variables
X = np.random.rand(m, n)
X[:, 0] = X[:, 1] * 20 + 3
X[:, 2] = X[:, 1] * 10
# Create a target variable Y as a mixture of normal distributions
Y = np.random.normal(loc=0.0, scale=4.0, size=m)
# follow other distributions
# Y = np.random.exponential(scale=4.0, size=m)
# plt.hist(Y, bins=30, edgecolor="black")

# Fit a linear regression model
model = LinearRegression()
model.fit(X, Y)
# Print the correlation coefficients
Y_pred = model.predict(X)
plt.scatter(Y, Y_pred, alpha=0.5)

np.corrcoef(Y, Y_pred)


def correlation(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_std = np.std(x)
    y_std = np.std(y)
    return np.mean((x - x_mean) * (y - y_mean)) / (x_std * y_std)


correlation(Y, Y_pred)


# Plot true vs predicted Y
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(Y, bins=30, alpha=0.5)
plt.title("True Y")

plt.subplot(1, 2, 2)
plt.hist(Y_pred, bins=30, alpha=0.5)
plt.title("Predicted Y")

plt.show()
