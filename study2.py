"""

"""

# native imports
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
# local imports
from data.loader import SimulatedData, SpectralData
from data.splitter import Splitter
from evaluate import Evaluator

# constants
SEED = 24061
N_ITER = 1000  # number of iterations
N_SAMPLE = 100  # sample size
N_FT = 1000  # number of features
N_FT_SELECT = 10  # number of features to select
K = 5  # number of folds
# HP_SPACE = dict({ # hyperparameter space for random forest
#     "n_estimators": [2, 32, 128], # number of trees
#     "max_depth": [1, 2, 4], 
#     # "criterion": ["squared_error", "absolute_error"], # loss function
# })
HP_SPACE = dict({ # hyperparameter space for random forest
    "C": [1e-2, 1e-1, 1e-0], # number of trees
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
})
MODEL = SVR
PATH_OUT = Path(__file__).resolve().parent / "out" / "study2.csv"

def main():
    np.random.seed(SEED)
    for i in tqdm(range(N_ITER), desc="Iteration"):
        # Simultated Data
        X_sim, y_sim = SimulatedData(n=N_SAMPLE, p=N_FT).sample()
        run(X_sim,  y_sim, i=i, dataset="simulated")    
        # Spectral Data
        X_spec, y_spec = SpectralData().load()
        run(X_spec, y_spec, i=i, dataset="spectral")

def run(X, y, i, dataset):
    # sample the data splits
    splits = Splitter(X, y).sample(method="KF", K=K)
    # compare different strategies
    dict_out = {
        "FS0_HT0": FS0_HT0(splits, X, y),
        "FS0_HT1": FS0_HT1(splits, X, y),
        "FS1_HT0": FS1_HT0(splits, X, y),
        "FS1_HT1": FS1_HT1(splits, X, y),
    }
    # save the results
    save_results(dict_out, i, dataset)

def FS0_HT0(splits, X, y):
    evaluator = Evaluator("regression")
    for k in range(len(splits)):
        idx_train = splits[k]["idx_train"]
        idx_test = splits[k]["idx_test"]

        # step 1: use the full dataset to select the top features
        idx_sel, scores = select_features(X, y)
        Xs = X[:, idx_sel]

        # step 2: split the data based on the selected features
        Xs_train, Xs_test = Xs[idx_train], Xs[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]

        # step 3: use the full dataset to tune the hyperparameters
        param_tuned = hyperparam_tuning(Xs_train, y_train, Xs_test, y_test)

        # step 4: train with the tuned hyperparameters and the selected features
        model = MODEL(**param_tuned).fit(Xs_train, y_train)
        y_pred = model.predict(Xs_test)

        # step 5: log the results
        evaluator.log(y_test, y_pred)

    return evaluator.summary()


def FS0_HT1(splits, X, y):
    evaluator = Evaluator("regression")
    for k in range(len(splits)):
        idx_train = splits[k]["idx_train"]
        idx_test = splits[k]["idx_test"]

        # step 1: use the full dataset to select the top features
        idx_sel, scores = select_features(X, y)
        Xs = X[:, idx_sel]

        # step 2: split the data based on the selected features
        Xs_train, Xs_test = Xs[idx_train], Xs[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]

        # step 3: use only the training set to tune the hyperparameters
        param_tuned = hyperparam_tuning(Xs_train, y_train)

        # step 4: train with the tuned hyperparameters and the selected features
        model = MODEL(**param_tuned).fit(Xs_train, y_train)
        y_pred = model.predict(Xs_test)

        # step 5: log the results
        evaluator.log(y_test, y_pred)

    return evaluator.summary()



def FS1_HT0(splits, X, y):
    evaluator = Evaluator("regression")
    for k in range(len(splits)):
        idx_train = splits[k]["idx_train"]
        idx_test = splits[k]["idx_test"]

        # step 1: split the data first
        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]

        # step 2: use only the training set to select the top features
        idx_sel, scores = select_features(X_train, y_train)
        Xs_train, Xs_test = X_train[:, idx_sel], X_test[:, idx_sel]

        # step 3: use the full dataset to tune the hyperparameters
        param_tuned = hyperparam_tuning(Xs_train, y_train, Xs_test, y_test)

        # step 4: train with the tuned hyperparameters and the selected features
        model = MODEL(**param_tuned).fit(Xs_train, y_train)
        y_pred = model.predict(Xs_test)

        # step 5: log the results
        evaluator.log(y_test, y_pred)

    return evaluator.summary()


def FS1_HT1(splits, X, y):
    evaluator = Evaluator("regression")
    for k in range(len(splits)):
        idx_train = splits[k]["idx_train"]
        idx_test = splits[k]["idx_test"]

        # step 1: split the data first
        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]

        # step 2: use only the training set to select the top features
        idx_sel, scores = select_features(X_train, y_train)
        Xs_train, Xs_test = X_train[:, idx_sel], X_test[:, idx_sel]

        # step 3: use only the training set to tune the hyperparameters
        param_tuned = hyperparam_tuning(Xs_train, y_train)

        # step 4: train with the tuned hyperparameters and the selected features
        model = MODEL(**param_tuned).fit(Xs_train, y_train)
        y_pred = model.predict(Xs_test)

        # step 5: log the results
        evaluator.log(y_test, y_pred)

    return evaluator.summary()

def hyperparam_tuning(X, y, X_test=None, y_test=None, K=5, metric="r"):
    """
    Grid search for hyperparameters of a random forest regressor.
    
    args
    ---
    X: np.ndarray, shape (n, p), feature matrix of the training set
    y: np.ndarray, shape (n, ), target vector of the training set
    X_test: np.ndarray, shape (n_test, p), feature matrix of the test set
    y_test: np.ndarray, shape (n_test, ), target vector of the test set
    
    return
    ---
    params_suggested: dict, suggested hyperparameters
    
    """

    # step 1: initialize the evaluator and hyperparameter space
    hps = list(HP_SPACE.keys()) # all hyperparameters
    ls_hp0 = [] # list of hyperparameter 0
    ls_hp1 = [] # list of hyperparameter 1
    evaluator = Evaluator("regression")
    if X_test is None or y_test is None:
        # step 2a (HT=1): split the data if the test set is not provided
        splitter = Splitter(X, y)
        splits = splitter.sample("KF", K=K)
        for i in range(K):
            X_train, X_test = splits[i]["X_train"], splits[i]["X_test"]
            y_train, y_test = splits[i]["y_train"], splits[i]["y_test"]

            # step 3: grid search based on the performance on the test set
            for hp0, hp1 in zip(HP_SPACE[hps[0]], HP_SPACE[hps[1]]):
                params = {hps[0]: hp0, hps[1]: hp1}
                # train and predict
                model = MODEL(**params).fit(X_train, y_train)
                y_pred = model.predict(X_test)
                # log the results
                evaluator.log(y_test, y_pred)
                ls_hp0.append(hp0)
                ls_hp1.append(hp1)
    else:
        # step 2b (HT=0): direct grid search if the test set is provided
        for hp0, hp1 in zip(HP_SPACE[hps[0]], HP_SPACE[hps[1]]):
            params = {hps[0]: hp0, hps[1]: hp1}
            # train and predict
            model = MODEL(**params).fit(X, y)
            y_pred = model.predict(X_test)
            # log the results
            evaluator.log(y_test, y_pred)
            ls_hp0.append(hp0)
            ls_hp1.append(hp1)

    # step 4: select the best hyperparameters
    scores = evaluator.to_dataframe()[metric]
    idx_high = np.argmax(scores)
    params_suggested = {
        hps[0]: ls_hp0[idx_high],
        hps[1]: ls_hp1[idx_high]
    }
    return params_suggested


def select_features(X, y, n_select=N_FT_SELECT):
    """
    Use the available dataset (X, y) to select the top n_select features X
    with the highest association with y.
    
    args
    ---
    X: np.ndarray, shape (n, p)
    y: np.ndarray, shape (n, )
    n_select: int, number of features to select
    method: str, method to select features
        - "OLS": select features based on OLS coefficients
        - "corr": select features based on correlation with y
    
    return
    ---
    idx_select: np.ndarray, shape (n_select, )
    """

    # step 1: calculate the association between features and y
    scores = np.abs(np.corrcoef(X.T, y.T)[:-1, -1])

    # step 2: select the top n_select features with the highest scores
    idx_select = np.argsort(scores)[-n_select:]

    return idx_select, scores


def save_results(dict_out, i, dataset):
    for k, v in dict_out.items():
        v["method"] = k
        v["i"] = i
        v["dataset"] = dataset
        v.loc[:, ["metric", "mean", "method", "i", "dataset"]].\
            to_csv(PATH_OUT, mode="a", index=False, 
                   header=not PATH_OUT.exists())


if __name__ == "__main__":
    main()