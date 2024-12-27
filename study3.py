# native imports
from tqdm import tqdm
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
# local imports
from data.loader import SpectralData, SimulatedSpectralData
from data.splitter import Splitter
from evaluate import Evaluator

SEED = 24061
N_ITER = 500
N_SAMPLE = 600
K = 5
MODEL = RandomForestRegressor
PATH_OUT = Path(__file__).resolve().parent / "out" / "study3.csv"

def main():
    for i in tqdm(range(N_ITER), desc="Iteration"):
        # simulated data
        loader = SimulatedSpectralData()
        X, y = loader.sample(N_SAMPLE, seed=SEED + i)
        ls_season = loader.cov()
        run(X, y, ls_season, i=i, dataset="simulated") 
        # real data
        loader = SpectralData()
        X, y = loader.load()
        ls_season = loader.cov()["season"]
        run(X, y, ls_season, i=i, dataset="real")

def run(X, y, ls_season, i, dataset):
    # sample the data splits
    splitter = Splitter(X, y)
    splits_KF = splitter.sample("KF", K=K)
    splits_block = splitter.sample(ls_season)
    # compare different cross-validation splits
    dict_out = {
        "KF": cross_validation(splits_KF),
        "block": cross_validation(splits_block)
    }
    # save the results
    save_results(dict_out, i, dataset)
    
def cross_validation(splits):
    evaluator = Evaluator("regression")
    for key, split in splits.items():
        # step 1: split the data
        X_train, y_train = split["X_train"], split["y_train"]
        X_test, y_test = split["X_test"], split["y_test"]

        # step 2: fit and predict
        model = MODEL().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # step3: evaluate the results
        evaluator.log(y_test, y_pred)

    return evaluator.summary()

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