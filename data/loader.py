from pathlib import Path
import pandas as pd
import numpy as np

PATH_DATA = Path(__file__).parent / "spectral.csv"
LS_WV = [
    # https://cdn.sparkfun.com/assets/8/5/f/0/3/AS7265x_Design_Considerations.pdf
    410, 435, 460, 485, 510,
    535, 560, 585, 610, 645,
    680, 705, 730, 760, 810,
    860, 900, 940
]
LS_WV_STR = [str(w) for w in LS_WV]
LS_COV = ["is_kfarm", "pasture", "season"]
LS_COL = ["id", "ndf", "ndfq50", "is_kfarm", "season", "pasture", "PC1", "PC2"]

class SimulatedData:
    """
    - n: number of samples
    - p: number of features
    """
    def __init__(self, n, p):
        self.n = n
        self.p = p
        self.X, self.y = self.sample()
    
    def sample(self):
        X = np.random.normal(0, 1, (self.n, self.p))
        y = np.random.normal(0, 1, self.n)
        return X, y
    
    def X(self):
        return self.X

    def y(self):
        return self.y
    

class SpectralData:
    """
    Spectral data loader to return
    - X: spectral data (lidar optional)
    - y: NDF or NDFq50
    - PC: principal components
    - cov: covariates (is_kfarm, pasture, season)
    """
    def __init__(self):
        self.data = pd.read_csv(PATH_DATA)
    
    def X(self, in_array=True, lidar=False):
        ls_wv = [str(w) for w in LS_WV_STR]
        if lidar:
            ls_wv = ls_wv + ["lidar"]
        pd_X = self.data.loc[:, ls_wv]
        return pd_X.to_numpy() if in_array else pd_X

    def y(self, cat=False):
        if cat:
            pd_y = self.data.loc[:, "ndfq50"]
        else:
            pd_y = self.data.loc[:, "ndf"]
        array_y = pd_y.to_numpy()
        return array_y
    
    def PC(self, in_array=True):
        pd_PC = self.data.loc[:, ["PC1", "PC2"]]
        return pd_PC.to_numpy() if in_array else pd_PC
    
    def cov(self):
        pd_cov = self.data.loc[:, LS_COV]
        return pd_cov
  
    def tidy(self):
        data_long = pd.melt(self.data,
            id_vars=LS_COL, 
            value_vars=LS_WV_STR,
            var_name="wv", value_name="value")
        return data_long
