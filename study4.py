import pandas as pd
import numpy as np
import sys, os
from matplotlib import pyplot as plt
import seaborn as sns

# local imports
from evaluate import Evaluator

BIAS = [0, 1, 2]
STD = [0, 1, 2]
N_SAMPLE = 100

y = np.random.normal(0, 1, N_SAMPLE)
y_dict = dict()
for b in BIAS:
    for s in STD:
        name = f'bias_{b}_std_{s}'
        y_dict[name] = y + np.random.normal(b, s, N_SAMPLE) 
df = pd.DataFrame(y_dict)
df['y'] = y
df

evaluator = Evaluator("regression")
for col in df.columns:
    print(col)
    if col != 'y':
        evaluator.log(y, df[col])
evaluator.to_dataframe()


# 3 by 3 subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
for i, b in enumerate(BIAS):
    for j, s in enumerate(STD):
        name = f'bias_{b}_std_{s}'
        sns.scatterplot(x=name, y=y, data=df, ax=axes[i, j])
        # set range
        axes[i, j].set_xlim(-10, 10)
        axes[i, j].set_ylim(-10, 10)
        