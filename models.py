from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# pLsr
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from data.loader import SpectralData
from data.splitter import Splitter


def get_metrics(y_test, y_pred):
    cor = np.corrcoef(y_test, y_pred)[0, 1]
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    rmspe = np.sqrt(np.mean(((y_test - y_pred) / y_test) ** 2))
    rsr = rmse / np.std(y_test)
    return dict({
        "cor": cor,
        "rmse": rmse,
        "rmspe": rmspe,
        "rsr": rsr
    })
    
loader = SpectralData()
splitter = Splitter(X=loader.X(), y=loader.y(), K=5)
cov = loader.cov()

ls_r = dict({"MC": [], "season": []})
ls_rmse = dict({"MC": [], "season": []})
ls_rmspe = dict({"MC": [], "season": []})
ls_rsr = dict({"MC": [], "season": []})

for i in range(50):
    splits = splitter.sample_splits("MC")
    for key in splits:
        split = splits[key]
        X_train, X_test = split["X_train"], split["X_test"]
        y_train, y_test = split["y_train"], split["y_test"]

        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = get_metrics(y_test, y_pred)
        ls_r["MC"].append(metrics["cor"])
        ls_rmse["MC"].append(metrics["rmse"])
        ls_rmspe["MC"].append(metrics["rmspe"])
        ls_rsr["MC"].append(metrics["rsr"])
    splits = splitter.sample_splits(cov["season"])
    for key in splits:
        split = splits[key]
        X_train, X_test = split["X_train"], split["X_test"]
        y_train, y_test = split["y_train"], split["y_test"]

        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = get_metrics(y_test, y_pred)
        ls_r["season"].append(metrics["cor"])
        ls_rmse["season"].append(metrics["rmse"])
        ls_rmspe["season"].append(metrics["rmspe"])
        ls_rsr["season"].append(metrics["rsr"])



import pandas as pd
df_eval = pd.DataFrame(
    data={
        "split": ["MC"] * 250 + ["season"] * 150,
        "cor": ls_r["MC"] + ls_r["season"],
        "rmse": ls_rmse["MC"] + ls_rmse["season"],
        "rmspe": ls_rmspe["MC"] + ls_rmspe["season"],
        "rsr": ls_rsr["MC"] + ls_rsr["season"]
    }
).melt(id_vars="split", var_name="metric")
# facet by metrics
sns.set(style="whitegrid")
g = sns.FacetGrid(df_eval, col="metric", 
                  col_wrap=2, sharey=False)
g.map(sns.boxplot, "split", "value", "split", palette="muted")
plt.show()

  
  # # 

    
    # sns.scatterplot(x=y_pred, y=y_test, hue=cov["season"][idx_test])
    # plt.title(f"correlation: {cor:.2f}, RMSE: {rmse:.2f},\
    #     RMSPE: {rmspe*100:.2f}%, RSR: {rsr*100:.2f}%")
    # plt.xlabel("Predicted NDF")
    # plt.ylabel("Actual NDF (y)")