

# native
import pandas as pd 
# visualization
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
# local
from data.loader import SimulatedSpectralData
from data.splitter import Splitter
from evaluate import Evaluator
import numpy as np


evaluator = Evaluator("regression")

spectral_data = SimulatedSpectralData()
X, y = spectral_data.sample(500, smallset=True)
n, p = X.shape
splitter = Splitter(X, y)

MODEL = RandomForestRegressor

splits = splitter.sample("In-Sample")
X_train, X_test = splits["X_train"], splits["X_test"]
y_train, y_test = splits["y_train"], splits["y_test"]
# step 2: fit and predict
model = MODEL().fit(X_train, y_train)
y_pred = model.predict(X_test)
# step 3: log the results
evaluator.log(y_test, y_pred)

splits = splitter.sample("KF", K=5)
X_train, X_test = splits[0]["X_train"], splits[0]["X_test"]
y_train, y_test = splits[0]["y_train"], splits[0]["y_test"]
# step 2: fit and predict
model = MODEL().fit(X_train, y_train)
y_pred = model.predict(X_test)
# step 3: log the results
evaluator.log(y_test, y_pred)


evaluator.to_dataframe()




splits = splitter.sample("KF", K=5)
i = 0
X_train, y_train = splits[i]['X_train'], splits[i]['y_train']
X_test, y_test = splits[i]['X_test'], splits[i]['y_test']

from sklearn.svm import SVR
model = SVR()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
evaluator.log(y_test, y_pred)
sns.scatterplot(x=y_test, y=y_pred)
evaluator.to_dataframe()





n_effect = n // 3
effects = ["summer"] * n_effect + ["fall"] * n_effect + ["winter"] * n_effect

idx_select = np.arange(p)[::p // 10] # select 10 features
           

# splits = splitter.sample(effects)


# PLS
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
# model = PLSRegression(n_components=5)
# model = RandomForestRegressor()
model = SVR()

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)


# eval
evaluator.log(y_test, y_pred)
sns.scatterplot(x=y_test, y=y_pred)
evaluator.to_dataframe()
