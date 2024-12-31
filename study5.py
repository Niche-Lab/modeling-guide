import numpy as np
from tqdm import tqdm
from pathlib import Path

# local imports
from evaluate import Evaluator

N_SAMPLE = 1000
N_ITER = 500
BALANCED = [0.1, 0.3, 0.5, 0.7, 0.9] # portion of positive cases
CONF = [0.1, 0.3, 0.5, 0.7, 0.9] # confidence level
SEED = 24061
PATH_OUT = Path(__file__).resolve().parent / "out" / "study5.csv"
BETA_POS_A = 5
BETA_POS_B = 2
BETA_NEG_A = 2
BETA_NEG_B = 5

def main():
    dict_eval = init_evaluators()
    for i in tqdm(range(N_ITER), desc="Iteration"): 
        for b in BALANCED:
            y_obs, y_pre = generate_data(b)
            for c in CONF:
                name = f"pos_{int(b * 10)}-conf_{int(c * 10)}"
                dict_eval[name].log(y_obs, y_pre, conf=c)
    save_results(dict_eval)

def init_evaluators():
    dict_eval = dict()
    for b in BALANCED:
        for c in CONF:
            name = f"pos_{int(b * 10)}-conf_{int(c * 10)}"
            dict_eval[name] = Evaluator("classification")
    return dict_eval

def generate_data(pos):
    y_obs = np.random.choice([0, 1], size=N_SAMPLE, p=[1 - pos, pos])
    y_pre = np.zeros(N_SAMPLE)
    for i, y in enumerate(y_obs): 
        # beta distribution
        if y == 1:
            # when the observation is positive
            y_pre[i] = np.random.beta(BETA_POS_A, BETA_POS_B)
        else:
            # when the observation is negative
            y_pre[i] = np.random.beta(BETA_NEG_A, BETA_NEG_B)
    return y_obs, y_pre

def save_results(dict_eval):
    for key in dict_eval.keys():
        df = dict_eval[key].summary()
        df["trial"] = key
        df = df.pivot(index="trial", columns="metric", values="mean").reset_index()
        df["pos"] = df["trial"].apply(lambda x: x.split("-")[0].split("_")[1])
        df["conf"] = df["trial"].apply(lambda x: x.split("-")[1].split("_")[1])
        if PATH_OUT.exists():
            df.to_csv(PATH_OUT, index=False, header=False, mode='a')  
        else:
            df.to_csv(PATH_OUT, index=False)
          

if __name__ == "__main__":
    main()


