# native imports
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
# local imports
from data.loader import SimulatedData
from data.splitter import Splitter
from evaluate import Evaluator, mae_var, rmspe_var, rmse_var
from sklearn.linear_model import LinearRegression

# constants
SEED = 24061
N_ITER = 1000  # number of iterations
N_SAMPLE = [50, 100, 500]  # sample size
N_FT = 10  # number of features
N_UNSEEN_ITER = 100  # number of times to sample unseen data
N_UNSEEN_SAMPLE = 100  # sample size of unseen data
PATH_OUT = Path(__file__).resolve().parent / "out" / "study1.csv"

def main():
    np.random.seed(SEED)
    # iterate over the sample sizes
    for n in tqdm(N_SAMPLE, desc="Sample Size"):
        # generate the data
        for i in tqdm(range(N_ITER), desc="Iteration"):
            X, y = SimulatedData(n=n, p=N_FT).sample()
            splitter = Splitter(X, y)
            # evaluate each estimator
            results = {}            
            results["In-Sample"] = eval_insample(splitter)
            results["2-Fold CV"] = eval_kfold(splitter, K=2)
            results["5-Fold CV"] = eval_kfold(splitter, K=5)
            results["10-Fold CV"] = eval_kfold(splitter, K=10)
            results["LOOCV"] = eval_loocv(splitter)
            # simulate the true generalization
            results["TrueG"] = eval_trueG(splitter)
            # organize the results and save
            df_out = concat_results(results, n, i)
            df_out.to_csv(PATH_OUT, mode="a", index=False,
                          header=not PATH_OUT.exists())


def eval_insample(splitter):
    evaluator = Evaluator("regression")
    # step 1: split the data
    splits = splitter.sample("In-Sample")
    X_train, X_test = splits["X_train"], splits["X_test"]
    y_train, y_test = splits["y_train"], splits["y_test"]
    # step 2: fit and predict
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # step 3: log the results
    evaluator.log(y_test, y_pred)
    return evaluator.summary(estimator="In-Sample")

def eval_kfold(splitter, K):
    evaluator = Evaluator("regression")
    # step 1: split the data
    splits = splitter.sample("KF", K=K)
    for i in range(K):
        X_train, X_test = splits[i]["X_train"], splits[i]["X_test"]
        y_train, y_test = splits[i]["y_train"], splits[i]["y_test"]
        # step 2: fit and predict
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # step 3: log the results
        evaluator.log(y_test, y_pred)
    return evaluator.summary(estimator=f"{K}-Fold CV")


def eval_loocv(splitter):
    evaluator = Evaluator("regression")
    # step 1: split the data
    splits = splitter.sample("LOOCV")
    # step 2: create containers for obs_y and pre_y
    obs_y = []
    pre_y = []
    for i in range(splitter.n):
        X_train, X_test = splits[i]["X_train"], splits[i]["X_test"]
        y_train, y_test = splits[i]["y_train"], splits[i]["y_test"]
        # step 3: fit and predict
        model = LinearRegression().fit(X_train, y_train)
        obs_y += y_test.tolist()
        pre_y += model.predict(X_test).tolist()
    # step 4: log the results
    obs_y, pre_y = np.array(obs_y), np.array(pre_y)
    evaluator.log(obs_y, pre_y)
    out = evaluator.summary(estimator="LOOCV")
    # step 5: add variance of the metrics
    out.index = out["metric"]
    out.loc["RMSE", "var"] = rmse_var(obs_y, pre_y)
    out.loc["MAE", "var"] = mae_var(obs_y, pre_y)
    out.loc["RMSPE", "var"] = rmspe_var(obs_y, pre_y)
    return out

def eval_trueG(splitter, n=N_UNSEEN_SAMPLE, p=N_FT, niter=N_UNSEEN_ITER):
    evaluator = Evaluator("regression")
    # step 1: fit the model with the available data
    X, y = splitter.X, splitter.y
    model = LinearRegression()
    model.fit(X, y)
    # step 2: simulate $niter sets of new data to estimate the true generalization
    for _ in range(niter):
        new_X, new_y = SimulatedData(n, p).sample()
        new_y_pred = model.predict(new_X)
        # step 3: log the results
        evaluator.log(new_y, new_y_pred)
    return evaluator.summary(estimator="TrueG")


def concat_results(results, n, i):
    df_tmp = pd.DataFrame(columns=["metric", "mean", "var", "estimator"])
    for key in results.keys():
        if key != "TrueG":
            # concat estimator
            if len(df_tmp) == 0:
                df_tmp = results[key]
            else:
                df_tmp = pd.concat([df_tmp, results[key]])
        else:
            # add true generalization
            df_G = results[key].loc[:, ["metric", "mean"]]
            df_G.columns = ["metric", "true"]
            df_tmp = pd.merge(df_tmp, df_G, on="metric")
    # calculate bias and variance
    df_tmp["bias"] = df_tmp["mean"] - df_tmp["true"]
    df_tmp["variance"] = df_tmp["var"]
    df_out = df_tmp.loc[:, ["metric", "estimator", "bias", "variance"]]
    df_out["n"] = n
    df_out["i"] = i
    return df_out

if __name__ == "__main__":
    main()