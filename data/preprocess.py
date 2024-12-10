"""
This script:
1. Filters the data to only include the field data
2. Adds categorical columns for season, pasture coverage, and NDF (quantile 50%)
3. Performs PCA on the spectral bands
4. Renames the columns 
"""
# native
from pathlib import Path
import pandas as pd 

# statistics/ML
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.decomposition import PCA

# constants
PATH_DATA = Path(__file__).parent / "raw.csv"
PATH_OUT = Path(__file__).parent / "spectral.csv"
LS_WV = [
    # https://cdn.sparkfun.com/assets/8/5/f/0/3/AS7265x_Design_Considerations.pdf
    410, 435, 460, 485, 510,
    535, 560, 585, 610, 645,
    680, 705, 730, 760, 810,
    860, 900, 940
]

def main():
    data = pd.read_csv(PATH_DATA)
    # add categorical columns
    data = format_date(data)
    data = add_season(data) # groupped by 2 months
    data = add_pasture(data) # groupped by pasture coverage
    data = add_ndfgrp(data) # groupped by NDF
    # filter non-field data
    data = data.query("Type == 'Field'").dropna().reset_index()
    # PCA
    data = concat_pcs(data) # 97.12%, 1.43%
    # finalization
    data = rename_data(data)
    inspect_anova(data)
    data.to_csv(PATH_OUT, index=False)

def format_date(data):
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%b')
    data['Date'] = data['Date'].apply(lambda x: x.replace(year=2023))
    return data

def add_season(data):
    # MayJune
    data["Season"] = "MayJun"
    # JulyAugust
    data.loc[data['Date'].dt.month.isin([7, 8]), "Season"] = "JulAug"
    # SeptemberOctober
    data.loc[data['Date'].dt.month.isin([9, 10]), "Season"] = "SepOct"
    # return
    return data

def add_pasture(data):
    """
    add pasture coverage
    poor: < 60% coverage
    moderate: 60-85% coverage
    good: > 85% coverage
    """
    data["pasture"] = "NONE"
    # P1, P2, K5, K6 is "Poor"
    data.loc[data['Field'].isin(["P1", "P2", "K5", "K6"]), "pasture"] = "Poor"
    # P3, P4, K1, K2 is "Moderate"
    data.loc[data['Field'].isin(["P3", "P4", "K1", "K2"]), "pasture"] = "Moderate"
    # P5, P6, K3, K4 is "Good"
    data.loc[data['Field'].isin(["P5", "P6", "K3", "K4"]), "pasture"] = "Good"
    # is K farm
    data["is_kfarm"] = data["Field"].apply(lambda x: "K" in str(x))
    # return
    return data

def add_ndfgrp(data):
    """
    categorize NDF into 4 groups or binary (50%)
    """
    q25 = data["NDF"].quantile(0.25)
    q50 = data["NDF"].quantile(0.50)
    q75 = data["NDF"].quantile(0.75)
    data["NDFgrp"] = "q1"
    data.loc[data["NDF"] > q25, "NDFgrp"] = "q2"
    data.loc[data["NDF"] > q50, "NDFgrp"] = "q3"
    data.loc[data["NDF"] > q75, "NDFgrp"] = "q4"
    data["NDFbin"] = data["NDF"].apply(lambda x: x > q50)
    return data

def get_pca(data):
    # V1 to V18 spectral bands
    X = data.loc[:, ["V" + str(i) for i in range(1, 19)]]
    # standardize
    X = (X - X.mean()) / X.std()
    pca = PCA(n_components=2)
    variances = pca.fit(X).explained_variance_ratio_
    pcs = pca.transform(X)
    df_pcs = pd.DataFrame(pcs, columns=["PC1", "PC2"])
    return df_pcs, variances

def rename_data(data):
    cols = ["NDF", "NDFbin", "is_kfarm", "Season", "pasture", "PC1", "PC2"] +\
         ["V" + str(i) for i in range(1, 20)]
    cols_rename = ["id", "ndf", "ndfq50", "is_kfarm", "season", "pasture", "PC1", "PC2"]
    data = data.loc[:, cols].reset_index()
    data.columns = cols_rename + LS_WV + ["lidar"]
    return data

def concat_pcs(data):
    df_pcs, vars_pcs = get_pca(data)
    data = pd.concat([data, df_pcs], axis=1)
    print(vars_pcs) 
    return data

def inspect_anova(data):
    formula = 'ndf ~ 1 + is_kfarm + pasture'
    model = ols(formula, data).fit()
    aov_table = anova_lm(model, typ=2)
    print(aov_table)

if __name__ == "__main__":
    main()