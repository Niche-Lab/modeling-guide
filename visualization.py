import matplotlib.pyplot as plt
import seaborn as sns


def show_first_3_cows(data, xaxis):
    """
    Show the first 3 cows by the specified variable.

    """
    sns.set_theme(style="darkgrid", palette="Set2")
    data_long = data.melt(id_vars="id", value_name="value", var_name="variable")
    plt.figure(figsize=(10, 5))
    sns.pointplot(
        data=data_long.query("variable != 'y' and id in ['cow 1', 'cow 2', 'cow 3']"),
        x="variable" if xaxis == "x" else "id",
        y="value",
        hue="id" if xaxis == "x" else "variable",
    )
