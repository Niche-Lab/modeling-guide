import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

# local imports
from evaluate import Evaluator

MU_Y = 0
STD_Y = 2 # intend to show the difference between RMSE and RSR
BIAS = [0, .5, 1, 2, 4, 8]
STD =  [0, .5, 1, 2, 4, 8]
N_SAMPLE = 10000
N_ITER = 500

PATH_DEMO = Path(__file__).resolve().parent / "out" / "study4_demo.csv"
PATH_OUT = Path(__file__).resolve().parent / "out" / "study4.csv"
LS_TRIALS = [f'bias_{b}-std_{s}' for b in BIAS for s in STD]

def main():
    make_demo()
    dict_eval = run_sim()
    save_results(dict_eval)

def make_demo(n_demo=100):
    dict_demo = dict()
    y = np.random.normal(MU_Y, STD_Y, n_demo) 
    for b in BIAS:
        for s in STD:
            name = f'bias_{b}-std_{s}'
            dict_demo[name] = y + np.random.normal(b, s, n_demo) 
    df_demo = pd.DataFrame(dict_demo)
    df_demo['y'] = y
    df_demo.to_csv(PATH_DEMO, index=False)


def init_evaluators():
    dict_eval = dict()
    for trial in LS_TRIALS:
        dict_eval[trial] = Evaluator("regression")
    return dict_eval

def run_sim():
    dict_eval = init_evaluators()
    for i in tqdm(range(N_ITER), desc="Iteration"): 
        y = np.random.normal(MU_Y, STD_Y, N_SAMPLE)
        for b in BIAS:
            for s in STD:
                name = f'bias_{b}-std_{s}'
                yhat = y + np.random.normal(b, s, N_SAMPLE)
                dict_eval[name].log(y, yhat)
    return dict_eval


def save_results(dict_eval):
    # init the dataframe with the first trial
    t = LS_TRIALS[0]
    df = dict_eval[t].summary()
    df["trial"] = t
    # append the rest of the trials
    for t in LS_TRIALS[1:]:
        df_new = dict_eval[t].summary()
        df_new["trial"] = t
        df = pd.concat([df, df_new], ignore_index=True)
    # pivot the dataframe from long to wide
    df = df.pivot(index="trial", columns="metric", values="mean")
    df.reset_index(inplace=True)
    # add the bias and std columns
    df["bias"] = df["trial"].apply(lambda x: x.split("-")[0].split("_")[1])
    df["std"] = df["trial"].apply(lambda x: x.split("-")[1].split("_")[1])
    df.to_csv(PATH_OUT, index=False)

    
if __name__ == "__main__":
    main()
