import numpy as np

class Splitter:
    """
    Splitter class for cross-validation
    returns a dictionary of splits:
    {
        0: {
            'X_train': np.array,
            'X_test': np.array,
            'y_train': np.array,
            'y_test': np.array
        },
        1: {
            'X_train': np.array,
            'X_test': np.array,
            'y_train': np.array,
            'y_test': np.array
        },
        ...
        K: {
            'X_train': np.array,
            'X_test': np.array,
            'y_train': np.array,
            'y_test': np.array
        }
    }
    """
    def __init__(self, X, y, K):
        """
        X: np.array, features matrix n x p
        y: np.array, target vector n x 1
        K: int, number of splits
        """
        self.X = X
        self.y = y
        self.K = K
        self.n = X.shape[0]
        self.n_test = int((1 / K) * self.n)

    def sample_splits(self, method):
        """
        method: str or list, 
            'MC': Monte Carlo (random sampling with replacement)
            'KF': K-Fold (random sampling without replacement)
            'LOOCV': Leave-One-Out Cross-Validation
            list: a variable to split on. E.g., ["s1", "s1", "s2", "s2", "s3", "s3"...]
        
        return: dict, dictionary of splits
            dict_idx = {
                0: [idx_test],
                1: [idx_test],
                ...
                K: [idx_test]
            }
        """
        if isinstance(method, str):
            if method == 'MC':
                dict_idx = self.sample_MC_idx()
            elif method == 'KF':
                dict_idx = self.sample_KF_idx()
            elif method == "LOOCV":
                dict_idx = self.sample_LOOCV_idx()
        else:
            dict_idx = self.sample_custom_idx(method)
    
        return self.assign_splits(dict_idx)

    def sample_LOOCV_idx(self):
        dict_idx = {}
        for i in range(self.n):
            dict_idx[i] = np.array([i])
    
        return dict_idx

    def sample_MC_idx(self):
        dict_idx = {}
        for k in range(self.K):
            dict_idx[k] = {}
            idx_rdm = np.random.permutation(self.n)
            dict_idx[k] = idx_rdm[:self.n_test]

        return dict_idx

    def sample_KF_idx(self):
        idx_rdm = np.random.permutation(self.n)
        dict_idx = np.array_split(idx_rdm, self.K)
        
        return dict_idx

    def sample_custom_idx(self, values):
        # use a custom list (values) to split the data
        dict_idx = {}
        unique_levels = np.unique(values)
        for i, v in enumerate(unique_levels):
            dict_idx[i] = np.where(values == v)[0]

        return dict_idx

    def assign_splits(self, dict_idx):
        dict_splits = {}
        set_idx = set(range(self.n))

        for k in range(len(dict_idx)):
            dict_splits[k] = {}

            idx_test = dict_idx[k]
            idx_train = list(set_idx - set(idx_test))

            dict_splits[k]['X_train'] = self.X[idx_train]
            dict_splits[k]['X_test'] = self.X[idx_test]
            dict_splits[k]['y_train'] = self.y[idx_train]
            dict_splits[k]['y_test'] = self.y[idx_test]

        return dict_splits

