import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd


def LOOCV(X, Y):
    y_hat = []
    for i in range(len(X)):
        # split data
        X_train = np.delete(X, i, axis=0)
        Y_train = np.delete(Y, i, axis=0)
        X_test = X[i].reshape(1, -1)  # cast to 2d array (P, ) -> (1, P)
        # fit OLS
        model = LinearRegression()
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        y_hat.append(y_pred[0][0])
    # calculate r
    score = get_scores(Y[:, 0], y_hat)
    return score


def k_fold_cv(X, Y, K):
    scores = []
    kfold = KFold(n_splits=K)
    for train_indices, test_indices in kfold.split(X):
        # split data
        X_train, Y_train = X[train_indices], Y[train_indices]
        X_test, Y_test = X[test_indices], Y[test_indices]
        # fit OLS
        score = get_OLS_scores(X_train, Y_train, X_test, Y_test)
        scores.append(score)
    # taking average of scores
    final_score = np.mean(scores)
    return final_score


def get_OLS_scores(X_train, Y_train, X_test, Y_test):
    model = LinearRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    score = get_scores(Y_test[:, 0], Y_pred[:, 0])
    return score


def get_scores(x, y):
    score = pearsonr(x, y)[0]
    return score


N_ITER = 1000
SEED = 24061
P = 10
N = [50, 100, 500]

scores = dict()
for n in N:
    score = dict(
        {
            "fit": [],
            "K = 2": [],
            "K = 5": [],
            "K = 10": [],
            "LOOCV": [],
        }
    )
    np.random.seed(SEED)
    for i in range(N_ITER):
        X = np.random.randn(n, P)
        Y = np.random.randn(n, 1)
        score["fit"].append(get_OLS_scores(X, Y, X, Y))
        score["K = 2"].append(k_fold_cv(X, Y, 2))
        score["K = 5"].append(k_fold_cv(X, Y, 5))
        score["K = 10"].append(k_fold_cv(X, Y, 10))
        score["LOOCV"].append(LOOCV(X, Y))
    scores[f"N = {n}"] = score


long_form_data = []
for n, score in scores.items():
    for method, score_list in score.items():
        for s in score_list:
            long_form_data.append({"N": n, "Method": method, "Score": s})
df = pd.DataFrame(long_form_data)

# figure size
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
sns.boxplot(x="N", y="Score", hue="Method", data=df, palette="Set3")
# Add titles and labels
plt.title("Distribution of Scores by Method")
plt.xlabel("Sample Size (N)")
plt.ylabel("Correlation Coefficient (r)")
# Show the plot
plt.show()
