import matplotlib.pyplot as plt
import seaborn as sns


# figure 1
def plot_bias_var_rmse(data):
    original_palette = sns.color_palette("Set3", 12)
    shifted_palette = original_palette[1:] + [original_palette[0]]
    sns.set_theme(style="whitegrid")
    sns.set_palette(shifted_palette)
    data_fg1 = data.query("Metric == 'RMSE' and Method != 'In-Sample'")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for i, m in enumerate(["Bias", "Variance"]):
        axes[i].axhline(
            y=0,
            color="black",
            linestyle="--",
            linewidth=0.7,
        )
        sns.boxplot(
            x="N",
            y=m,
            hue="Method",
            data=data_fg1,
            ax=axes[i],
            linewidth=0.5,
            fliersize=4,
        )
        axes[i].set_xlabel("Sample Size (N)")
        axes[i].set_ylabel(m)
    axes[0].set_title("Bias")
    axes[0].get_legend().remove()
    axes[1].set_title("Variance")
    fig.suptitle("Bias and Variance of RMSE by Method")
    fig.savefig("bias_var_rmse.png", dpi=300)


# figure 2
def plot_bias_metrics(data):
    sns.set_palette("Set3")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, m in enumerate(["Correlation", "R2", "RMSE"]):
        axes[i].axhline(
            y=0,
            color="black",
            linestyle="--",
            linewidth=0.7,
        )
        sns.boxplot(
            x="N",
            y="Bias",
            hue="Method",
            data=data.query("Metric == '%s'" % m),
            ax=axes[i],
            linewidth=0.5,
            fliersize=4,
        )
        axes[i].set_xlabel("Sample Size (N)")
        axes[i].set_ylabel(m)

    # titles
    fig.suptitle("Bias of Metrics by Validation Method with 10 Predictors")
    axes[0].set_title("Correlation (r)")
    axes[1].set_title("Coefficient of Determination (R2)")
    axes[2].set_title("RMSE")

    # y axis
    axes[0].set_ylim(-0.55, 0.7)
    axes[1].set_ylim(-2.5, 1.3)
    axes[1].set_yscale("symlog", base=2)
    axes[1].set_yticks([-2, -1, 0, 1])
    axes[2].set_ylim(-0.55, 0.7)

    # legend
    axes[0].get_legend().remove()
    axes[1].legend(loc="lower right", ncol=1)
    axes[2].get_legend().remove()

    # save
    fig.savefig("bias_metrics.png", dpi=300)


# figure 3
def plot_time(data_time):
    fig, axe = plt.subplots(figsize=(8, 8))
    sns.barplot(
        data=data_time,
        x="Method",
        y="Time",
        hue="N",
        errorbar="sd",
        err_kws={"linewidth": 1},
        ax=axe,
    )
    axe.set_yscale("log", base=10)
    fig.savefig("time.png", dpi=300)
