import numpy as np
import matplotlib as plt
import seaborn as sns


def pearson_r2(x, y):
    x_mu = np.mean(x)
    y_mu = np.mean(y)
    x_std = np.sqrt(np.mean((x - x_mu) ** 2))
    y_std = np.sqrt(np.mean((y - y_mu) ** 2))
    covxy = np.mean((x - x_mu) * (y - y_mu))
    r = covxy / (x_std * y_std)
    return r**2


from sklearn.metrics import r2_score


def rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2))


def determination_r2(obs, pred):
    obs_mu = np.mean(obs)
    ssr = np.sum((obs - pred) ** 2)
    sst = np.sum((obs - obs_mu) ** 2)
    return 1 - ssr / sst


def print_status(x, y):
    print(f"Correlation r2 = {pearson_r2(x, y):.2f}")
    # print(f"Numpy Correlation r2 = {np.corrcoef(x, y)[0, 1]**2:.2f}")
    print(f"Determination r2 = {determination_r2(x, y):.2f}")
    # print(f"Sklearn r2 = {r2_score(x, y):.2f}")
    print(f"RMSE = {rmse(x, y):.2f}")


RCORR = 0.5
NOISE = 1 - RCORR
BLOCK_EFF = 3
SEED = 24060

np.random.seed(SEED)
obs = np.concatenate([np.random.normal(2, 1, 50), np.random.normal(-2, 1, 50)])

pred_1 = obs * RCORR + np.random.normal(0, NOISE, 100)
ax = sns.scatterplot(x=obs, y=pred_1)
print_status(obs, pred_1)

pred_2 = pred_1 * 3
ax = sns.scatterplot(x=obs, y=pred_2)
print_status(obs, pred_2)

pred_3 = pred_1**3
ax = sns.scatterplot(x=obs, y=pred_3)
print_status(obs, pred_3)

pred_4 = []
for i, o in enumerate(obs):
    if o > 0:
        pred_4 += [pred_1[i] + np.random.normal(BLOCK_EFF, NOISE)]
    else:
        pred_4 += [pred_1[i] + np.random.normal(-BLOCK_EFF, NOISE)]
pred_4 = np.array(pred_4)
ax = sns.scatterplot(x=obs, y=pred_4)
print_status(obs, pred_4)

print_status(obs[obs > 0], pred_4[obs > 0])
print_status(obs[obs < 0], pred_4[obs < 0])
