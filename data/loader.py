from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# local imports
from data.spectral import make_T, make_P, apply_effect, make_y

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

class SimulatedSpectralData:
    
    def __init__(self):
        self.n = None
        self.p = 300
        # useful matrix
        self.Tu, self.Pu, self.Xu = None, None, None
        # detirmental matrix
        self.Td, self.Pd, self.Xd = None, None, None
        # feature matrix
        self.X = None
        self.y = None
    
    def sample(self, n, smallset=False, seed=None):
        if seed is None:
            seed = np.random.randint(1e6)
        self.n = n
        # useful matrix
        Tu = make_T(n, sds=[2e-2, 1e-1], seed=seed)
        Tu = apply_effect(Tu, effects=[1, 1.10, 1.07], seed=seed + 1)
        Pu = make_P(self.p, mus=[-30, 200], sds=[100, 60], amps=[.35, .24])
        Xu = Tu @ Pu
        # detrimental matrix
        Td = make_T(n, sds=[1e-1, 2e-2], seed=seed)
        Td = apply_effect(Td, effects=[1.07, 1, 1], seed=seed + 1)
        Pd = make_P(self.p, mus=[90, 345], sds=[40, 60], amps=[.06, .35])
        Xd = Td @ Pd
        # feature matrix
        X = Xu + Xd + np.random.normal(0, 1e-2, (n, self.p))
        # response variable derived from useful matrix (Xu)
        y = make_y(Xu, seed=seed + 2)
        # assignment and return
        self.Tu, self.Pu, self.Xu = Tu, Pu, Xu
        self.Td, self.Pd, self.Xd = Td, Pd, Xd
        self.X, self.y = X, y
        if smallset:
            idx_select = np.arange(self.p)[::self.p // 10] # select 10 features
            self.X = self.X[:, idx_select]
        return self.X, self.y
    
    def cov(self):
        n_season = self.n // 3
        season = ["summer"] * n_season\
                + ["fall"] * n_season\
                + ["winter"] * n_season
        return pd.Series(season, name="season")
            
class SimulatedData:
    """
    - n: number of samples
    - p: number of features
    """
    def __init__(self, n, p):
        self.n = n
        self.p = p
        self.X, self.y = self.sample()
    
    def sample(self, seed=None):
        if seed:
            np.random.seed(seed)
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
    
    def load(self, scale=True):
        return self.X(scale), self.y()
    
    def X(self, in_array=True, lidar=False, scale=True):
        ls_wv = [str(w) for w in LS_WV_STR]
        if lidar:
            ls_wv = ls_wv + ["lidar"]
        pd_X = self.data.loc[:, ls_wv]
        if scale:
            scaler = StandardScaler()
            pd_X = pd.DataFrame(scaler.fit_transform(pd_X), columns=ls_wv)
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
