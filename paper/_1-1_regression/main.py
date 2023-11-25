import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score


def pearson_r(x, y):
    x_mu = np.mean(x)
    y_mu = np.mean(y)
    x_std = np.sqrt(np.mean((x - x_mu) ** 2))
    y_std = np.sqrt(np.mean((y - y_mu) ** 2))
    covxy = np.mean((x - x_mu) * (y - y_mu))
    r = covxy / (x_std * y_std)
    return r


def rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2))


def determination_r2(obs, pred):
    obs_mu = np.mean(obs)
    ssr = np.sum((obs - pred) ** 2)
    sst = np.sum((obs - obs_mu) ** 2)
    return 1 - ssr / sst


def print_status(x, y):
    print(f"Correlation r = {pearson_r(x, y):.2f}")
    print(f"Numpy Correlation r2 = {np.corrcoef(x, y)[0, 1]**2:.2f}")
    print(f"Determination r2 = {determination_r2(x, y):.2f}")
    print(f"Sklearn r2 = {r2_score(x, y):.2f}")
    print(f"RMSE = {rmse(x, y):.2f}")


RCORR = 0.3
NOISE = 1 - RCORR
BLOCK_EFF = 3
SEED = 24061

np.random.seed(SEED)
obs = np.concatenate(
    [
        np.random.normal(BLOCK_EFF, 1, 50),
        np.random.normal(-BLOCK_EFF, 1, 50),
    ]
)
pred_1 = obs * RCORR + np.random.normal(0, NOISE, 100)
pred_2 = pred_1 * 5
pred_3 = pred_1**5
pred_3[np.abs(pred_3) < 16.7] = 0
pred_4 = []
for i, o in enumerate(obs):
    if o > 0:
        pred_4 += [pred_1[i] + np.random.normal(BLOCK_EFF, NOISE)]
    else:
        pred_4 += [pred_1[i] + np.random.normal(-BLOCK_EFF, NOISE)]
pred_4 = np.array(pred_4)

SIZE = 100
sns.set_style("whitegrid")
sns.set_palette("Set2")
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
sns.scatterplot(
    x=obs,
    y=pred_1,
    ax=axes[0, 0],
    color="grey",
    s=SIZE,
)
sns.scatterplot(
    x=obs,
    y=pred_2,
    ax=axes[0, 1],
    color="grey",
    s=SIZE,
)
sns.scatterplot(
    x=obs,
    y=pred_3,
    ax=axes[1, 0],
    color="grey",
    s=SIZE,
)
sns.scatterplot(
    x=obs,
    y=pred_4,
    ax=axes[1, 1],
    hue=obs > 0,
    s=SIZE,
)
# remove legend
axes[1, 1].get_legend().remove()


fig.savefig("regression.png", dpi=300)


ax = sns.scatterplot(x=obs, y=pred_1)
print_status(obs, pred_1)
# Correlation r = 0.83
# Determination r2 = 0.47
# RMSE = 2.41

ax = sns.scatterplot(x=obs, y=pred_2)
print_status(obs, pred_2)
# Correlation r = 0.83
# Determination r2 = -0.21
# RMSE = 3.63

ax = sns.scatterplot(x=obs, y=pred_3)
print_status(obs, pred_3)
# Correlation r = 0.36
# Determination r2 = -59.19
# RMSE = 25.56

ax = sns.scatterplot(x=obs, y=pred_4)
print_status(obs, pred_4)
# Correlation r = 0.94
# Determination r2 = 0.80
# RMSE = 1.49

print_status(obs[obs > 0], pred_4[obs > 0])
print_status(obs[obs < 0], pred_4[obs < 0])
# Correlation r = 0.33
# Determination r2 = -0.71
# RMSE = 1.46

# Correlation r = 0.25
# Determination r2 = -1.10
# RMSE = 1.52
