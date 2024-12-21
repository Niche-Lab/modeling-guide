from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
# svm
from sklearn.svm import SVC, SVR
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from data.loader import SpectralData
from data.splitter import Splitter
