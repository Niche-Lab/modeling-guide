import pandas as pd

import numpy as np

def get_u(model):



def make_Z(data):
    Z_sire = pd.get_dummies(data["sire"]).to_numpy() * 1
    n_sire = Z_sire.shape[1]
    Z_month = np.tile(data["month"], n_sire).reshape(n_sire, -1).T
    
    Z = pd.concat([Z_sire, Z_month], axis=1)
    Z = Z.to_numpy() * 1
    return Z

Z_sire * data["month"]
