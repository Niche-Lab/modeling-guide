import matplotlib.pyplot as plt
import seaborn as sns


def show_first_n_cows(data, xaxis, n=3):
    """
    Show the first 3 cows by the specified variable.

    """
    data_sub = data.iloc[:n]
    data_long = data_sub.melt(id_vars="id", value_name="value", var_name="variable")
    sns.set_theme(style="darkgrid", palette="Set2")
    plt.figure(figsize=(10, 5))
    sns.pointplot(
        data=data_long,
        x="variable" if xaxis == "x" else "id",
        y="value",
        hue="id" if xaxis == "x" else "variable",
    )
