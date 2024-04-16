# native imports
from matplotlib.pyplot import plot
import numpy as np
import pandas as pd
import time as t

# local imports
from utils import (
    sample_data,
    get_OLS_scores,
    k_fold_cv,
    LOOCV,
    truth_estimation,
)
from visualization import (
    plot_bias_var_rmse,
    plot_bias_metrics,
    plot_time,
)

# CONSTANTS
SEED = 24061  # seed for reproducibility
N_ITER = 1000  # number of iterations
N_SAMPLE = [50, 100, 500]  # sample size
N_FT = 10  # number of features

# unseen data
N_UNSEEN_ITER = 100  # number of times to sample unseen data
N_UNSEEN_SAMPLE = 100  # sample size of unseen data


def main():
    # STEP 1: run the experiment and store the results in a struct ============
    dict_scores = dict()
    for n in N_SAMPLE:
        dict_record = dict(
            {
                "In-Sample": [None] * N_ITER,
                "2-Fold CV": [None] * N_ITER,
                "5-Fold CV": [None] * N_ITER,
                "10-Fold CV": [None] * N_ITER,
                "LOOCV": [None] * N_ITER,
                "Truth": [None] * N_ITER,
            }
        )
        for i in range(N_ITER):
            print(f"Sample Size: {n}, Iteration: {i}")
            X, Y = sample_data(n, N_FT)
            # in-sample -------------------------------------------
            time_cur = t.time()
            scores = get_OLS_scores(X, Y, X, Y)
            time_elapsed = t.time() - time_cur
            scores += (time_elapsed,)
            dict_record["In-Sample"][i] = scores
            # 2-fold, 5-fold, 10-fold ----------------------------
            for k in [2, 5, 10]:
                time_cur = t.time()
                scores = k_fold_cv(X, Y, k)
                time_elapsed = t.time() - time_cur
                scores += (time_elapsed,)
                dict_record[f"{k}-Fold CV"][i] = scores
            # LOOCV -----------------------------------------------
            time_cur = t.time()
            scores = LOOCV(X, Y)
            time_elapsed = t.time() - time_cur
            scores += (time_elapsed,)
            dict_record["LOOCV"][i] = scores
            # truth -----------------------------------------------
            scores = truth_estimation(
                X,
                Y,
                n=N_UNSEEN_SAMPLE,
                p=N_FT,
                niter=N_UNSEEN_ITER,
            )
            dict_record["Truth"][i] = scores
        # save scores
        dict_scores[f"N = {n}"] = dict_record

    # STEP 2: transform the struct into a dataframe ===========================
    data = pd.DataFrame(columns=["N", "Method", "Metric", "Bias", "Variance", "Time"])
    row = {}
    for n in N_SAMPLE:
        d = dict_scores[f"N = {n}"]
        for key in ["In-Sample", "2-Fold CV", "5-Fold CV", "10-Fold CV", "LOOCV"]:
            for i in range(len(d[key])):
                # extract scores
                if key == "In-Sample":
                    # since In-Sample only makes one prediction
                    (est_cor, est_r2, est_rmse, time) = d[key][i]
                    var_rmse = 0
                else:
                    (est_cor, est_r2, est_rmse), var_rmse, time = d[key][i]
                (exp_cor, exp_r2, exp_rmse) = d["Truth"][i]

                # calculate bias
                exp_cor, exp_r2, exp_rmse = (
                    np.mean(exp_cor),
                    np.mean(exp_r2),
                    np.mean(exp_rmse),
                )
                bias_cor, bias_r2, bias_rmse = (
                    est_cor - exp_cor,
                    est_r2 - exp_r2,
                    est_rmse - exp_rmse,
                )

                # feed into dataframe
                row["N"] = [n] * 3
                row["Method"] = [key] * 3
                row["Metric"] = ["Correlation", "R2", "RMSE"]
                row["Bias"] = [bias_cor, bias_r2, bias_rmse]
                row["Variance"] = [0, 0, var_rmse]
                row["Time"] = [time] * 3
                data = pd.concat([data, pd.DataFrame(row)], ignore_index=True)
    # save
    data.to_csv("results.csv", index=False)
    data.groupby(["Method", "Metric", "N"]).aggregate(["median"]).reset_index().to_csv(
        "summary.csv", index=False
    )
    data_time = data.query("Metric == 'RMSE'").loc[:, ["N", "Method", "Time"]]
    data_time.to_csv("time.csv", index=False)

    # STEP 3: Visualize the results ===========================================
    plot_bias_var_rmse(data)
    plot_bias_metrics(data)
    plot_time(data_time)


if __name__ == "__main__":
    np.random.seed(SEED)
    main()


# 	Method	N	Time
# 0	10-Fold CV	50	0.006544
# 1	10-Fold CV	100	0.006502
# 2	10-Fold CV	500	0.006801
# 3	2-Fold CV	50	0.001401
# 4	2-Fold CV	100	0.001396
# 5	2-Fold CV	500	0.001449
# 6	5-Fold CV	50	0.003316
# 7	5-Fold CV	100	0.003307
# 8	5-Fold CV	500	0.003437
# 9	In-Sample	50	0.000711
# 10	In-Sample	100	0.000720
# 11	In-Sample	500	0.000783
# 12	LOOCV	50	0.009949
# 13	LOOCV	100	0.019340
# 14	LOOCV	500	0.112297
# 0.001401
# 0.003316
# 0.006544

# Method	N	Time
# 0	10-Fold CV	100.0	0.006591
# 1	2-Fold CV	100.0	0.001392
# 2	5-Fold CV	100.0	0.003334
# 3	In-Sample	100.0	0.000713
# 4	LOOCV	100.0	0.019251
