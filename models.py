from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# pLsr
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from data.loader import SpectralData
from data.splitter import Splitter

loader = SpectralData()
splitter = Splitter(X=loader.X(), y=loader.y(), K=5)
cov = loader.cov()
# splits = splitter.sample_splits(cov["season"])
splits = splitter.sample_splits("MC")

split = splits[1]
X_train, X_test = split["X_train"], split["X_test"]
y_train, y_test = split["y_train"], split["y_test"]

model = RandomForestRegressor()
# model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# pearson correlation
cor = np.corrcoef(y_test, y_pred)[0, 1]
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
rmspe = np.sqrt(np.mean(((y_test - y_pred) / y_test) ** 2))
print("correlation:", cor)
print("RMSE:", rmse)
print("RMSPE:", rmspe)
sns.scatterplot(x=y_pred, y=y_test)
plt.title(f"correlation: {cor:.2f}, RMSE: {rmse:.2f}, RMSPE: {rmspe*100:.2f}%")
plt.xlabel("Predicted NDF")
plt.ylabel("Actual NDF (y)")