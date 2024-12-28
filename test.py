# native
import pandas as pd 
# visualization
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
# local
from data.loader import SimulatedSpectralData, SpectralData
from data.splitter import Splitter
from evaluate import Evaluator
import numpy as np

from study1 import eval_kfold, eval_loocv, eval_trueG, eval_insample

N_SAMPLE = 500
SEED = 24061
MODEL = RandomForestRegressor

dataloader = SimulatedSpectralData()
X, y = dataloader.sample(n=N_SAMPLE) 
season = dataloader.cov()

dataloader = SpectralData()
X, y = dataloader.load() 
season = dataloader.cov()['season']

splitter = Splitter(X, y)
splits_KF = splitter.sample("KF", K=5)
k = 0
X_train, y_train = splits_KF[k]["X_train"], splits_KF[k]["y_train"]
X_test, y_test = splits_KF[k]["X_test"], splits_KF[k]["y_test"]
idx_test = splits_KF[k]["idx_test"]
c_test = season.iloc[idx_test]

model = RandomForestRegressor().fit(X_train, y_train)
y_pred = model.predict(X_test)
sns.scatterplot(x=y_pred, y=y_test, hue=c_test)

model = SVR(kernel="poly").fit(X_train, y_train)
y_pred = model.predict(X_test)
sns.scatterplot(x=y_pred, y=y_test, hue=c_test)

evaluator = Evaluator("regression")
evaluator.log(y_pred, y_test)
evaluator.summary()