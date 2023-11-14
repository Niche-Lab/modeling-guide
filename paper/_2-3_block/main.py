# CONSTANTS
from random import shuffle
from IPython import core


N_SAMPLE = 100
N_FT = 10
N_ITER = 1000
SEED = 24061
BLOCK_EFF = 3
K = 5
N_BLOCK = int(N_SAMPLE / K)


X, Y = sample_data(N_SAMPLE, N_FT)
data = pd.DataFrame({"Y": Y, "block": np.repeat(range(K), N_BLOCK)})
data["Y"] = data["Y"] + data["block"] * BLOCK_EFF
Y = data["Y"].values

# block CV
ik = 0
idx_train = data["block"] != ik
idx_test = data["block"] == ik
x_train, x_test = X[idx_train], X[idx_test]
y_train, y_test = Y[idx_train], Y[idx_test]
model = LinearRegression().fit(x_train, y_train)
y_pred = model.predict(x_test)
cor_score(y_test, y_pred)

# random CV
kfold = KFold(n_splits=K, shuffle=True)
idx_train, idx_test = next(kfold.split(X))
x_train, x_test = X[idx_train], X[idx_test]
y_train, y_test = Y[idx_train], Y[idx_test]
model = LinearRegression().fit(x_train, y_train)
y_pred = model.predict(x_test)
cor_score(y_test, y_pred)

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
