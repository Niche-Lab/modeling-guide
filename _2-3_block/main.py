import os
from tkinter import N
import numpy as np
import pandas as pd

os.chdir("../_2-3_block")
# CONSTANTS
N_SAMPLE = 100
N_FT = 10
N_RUN = 1000
BLOCK_EFF = [0.5, 1, 1.5, 2, 2.5, 3]
N_ITER = N_RUN * len(BLOCK_EFF)
SEED = 24061
K = 5
N_BLOCK = int(N_SAMPLE / K)

np.random.seed(SEED)
scores = {
    "iter": [None] * N_ITER,
    "effect": [None] * N_ITER,
    "block": [None] * N_ITER,
    "random": [None] * N_ITER,
}
for i, block in enumerate(BLOCK_EFF):
    print("block_eff: %f" % block)
    for j in range(N_RUN):
        iter = i * N_RUN + j
        X, Y = sample_data(N_SAMPLE, N_FT)
        data = pd.DataFrame({"Y": Y, "block": np.repeat(range(K), N_BLOCK)})
        data["Y"] = data["Y"] + data["block"] * block
        Y = data["Y"].values
        block_eff = data["block"].values + np.random.normal(0, 1, N_SAMPLE)
        X[:, 0] = block_eff

        scores["iter"][iter] = j
        scores["effect"][iter] = block
        # block CV
        score_tmp = [None] * K
        for k in range(K):
            idx_train = data["block"] != k
            idx_test = data["block"] == k
            x_train, x_test = X[idx_train], X[idx_test]
            y_train, y_test = Y[idx_train], Y[idx_test]
            model = LinearRegression().fit(x_train, y_train)
            y_pred = model.predict(x_test)
            score = cor_score(y_test, y_pred)
            # score = model.score(x_test, y_test)
            score_tmp[k] = score
        scores["block"][iter] = np.mean(score_tmp)

        # random CV
        score_tmp = [None] * K
        kfold = KFold(n_splits=K, shuffle=True, random_state=SEED + iter)
        for k, (idx_train, idx_test) in enumerate(kfold.split(X)):
            x_train, x_test = X[idx_train], X[idx_test]
            y_train, y_test = Y[idx_train], Y[idx_test]
            model = LinearRegression().fit(x_train, y_train)
            y_pred = model.predict(x_test)
            score = cor_score(y_test, y_pred)
            # score = model.score(x_test, y_test)
            score_tmp[k] = score
        scores["random"][iter] = np.mean(score_tmp)

df_scores = pd.DataFrame(scores)
df_scores = df_scores.melt(
    id_vars=["iter", "effect"], var_name="CV", value_name="correlation r"
)
df_scores.loc[:, ["effect", "CV", "correlation r"]].groupby(["effect", "CV"]).aggregate(
    "median"
)

# correlation r
# effect	CV
# 0.5	block	0.001939
# random	0.382582
# 1.0	block	-0.009064
# random	0.623485
# 1.5	block	0.002237
# random	0.705711
# 2.0	block	-0.001324
# random	0.744552
# 2.5	block	0.006413
# random	0.761071
# 3.0	block	0.002597
# random	0.771366

# p-value


# visualize
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
sns.set_theme(style="whitegrid")
sns.set_palette("Set2")
plt.axhline(y=0, color="black", linestyle="--", linewidth=0.7)
sns.boxplot(
    x="effect",
    y="correlation r",
    hue="CV",
    data=df_scores,
)
# rm legend
ax.get_legend().remove()
plt.savefig("cv.png", dpi=300)


# visualization
data_block = data.copy()
data_block["color"] = data_block["block"].apply(lambda x: "fold " + str(x + 1))

data_rdm = data.copy()
data_rdm["color"] = ["fold %d" % (i + 1) for i in range(K)] * N_BLOCK

JITTER = 0.2
sns.set_style("whitegrid")
sns.set_palette("Set3")
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# jitter scatter plot
sns.stripplot(
    x="block",
    y="Y",
    hue="color",
    data=data_block,
    ax=axes[0],
    jitter=JITTER,
    # dot size
    size=10,
    linewidth=1,
    alpha=0.7,
)
axes[0].set_title("Block Cross Validation")
axes[0].set_xlabel("Block Level")
axes[0].set_ylabel("Response Variable (Y)")
# random scatter plot
sns.stripplot(
    x="block",
    y="Y",
    hue="color",
    data=data_rdm,
    ax=axes[1],
    jitter=JITTER,
    # dot size
    size=10,
    linewidth=1,
    alpha=0.7,
)
axes[1].set_title("Random Cross Validation")
axes[1].set_xlabel("Block Level")
axes[1].set_ylabel("Response Variable (Y)")
plt.tight_layout()
plt.savefig("folds.png", dpi=300)

data


import scipy.stats as stats

# Compute the overall mean
y_bar = data["Y"].mean()

# Compute the sum of squares between (SSB)
ssb = sum(data.groupby("block")["Y"].mean().apply(lambda x: 20 * (x - y_bar) ** 2))
# Compute sum of squares within (SSW)
ssw = sum(data.groupby("block")["Y"].apply(lambda x: sum((x - x.mean()) ** 2)))
# Compute total sum of squares (SST)
sst = sum((data["Y"] - y_bar) ** 2)

# Compute degrees of freedom between (DFB)
dfb = data["block"].nunique() - 1

# Compute degrees of freedom within (DFW)
dfw = data.shape[0] - data["block"].nunique()

# Compute mean square between (MSB)
msb = ssb / dfb

# Compute mean square within (MSW)
msw = ssw / dfw

# Compute F-Value
f_value = msb / msw

# Compute p-value
p_value = 1 - stats.f.cdf(f_value, dfb, dfw)

# Create ANOVA table
anova_table = pd.DataFrame(
    {
        "source": ["between", "within", "total"],
        "SS": [ssb, ssw, sst],
        "df": [dfb, dfw, dfb + dfw],
        "MS": [msb, msw, ""],
        "F": [f_value, "", ""],
        "p-value": [p_value, "", ""],
    }
)

anova_table
