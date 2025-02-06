import numpy as np
import pandas as pd

class Evaluator:
    def __init__(self, task="regression"):
        self.logs = dict()
        if task == "regression":
            self.metrics = dict({       
                # linearity metrics
                "R2": r2,
                "r": pearsonr,
                "r2": pearsonr2,
                "CCC": ccc,
                # error-based metrics
                "MAE": mae,
                "RMSE": rmse,
                "RMSPE": rmspe,
                "RSR": rsr
            })
        elif task == "classification":
            self.metrics = dict({
                "accuracy": accuracy,
                # machine learning metrics
                "precision": precision,
                "recall": recall,
                # medical diagnostics metrics
                "sensitivity": sensitivity,
                "specificity": specificity,
                # self-explanatory metrics
                "TPR": TPR,
                "FNR": FNR,
                "FPR": FPR,
                "TNR": TNR,
                # composite metrics
                "f1": f1,
                "f2": f2,
                "f05": f05,
                "mcc": mcc
            })
        for m in self.metrics:
            self.logs[m] = []
    
    def log(self, y, y_hat, conf=None):
        y = np.array(y)
        if conf is not None:
            y_hat = (np.array(y_hat) > conf).astype(int)
        else:   
            y_hat = np.array(y_hat)
        for k, func in self.metrics.items():
            metric = func(y, y_hat)
            self.logs[k].append(metric)
    
    def to_dataframe(self):
        """
        return
        ------
        iteration | {metrics}
        """ 
        df = pd.DataFrame(self.logs).reset_index()
        df.columns = ["iteration"] + df.columns[1:].tolist()
        return df
    
    def to_tidy(self):
        """
        return
        ------
        iteration | metric | value | (estimator)
        """
        df = self.to_dataframe()
        df = df.melt(id_vars="iteration", var_name="metric")
        return df

    def summary(self, estimator=None, func=["mean", "var"]):
        """
        return
        ------
        metric | {func} | (estimator)
        """
        df = self.to_tidy().iloc[:, 1:].\
                groupby("metric").agg(func)
        df = df.reset_index()
        df.columns = ["metric"] + func
        if estimator:
            df["estimator"] = estimator
        return df
        
def is_single_input(x):
    """
    Check if input is a single value or an array with only one element
    """
    is_single = isinstance(x, (int, float))
    is_array = isinstance(x, np.ndarray) and len(x) == 1
    return is_single or is_array

# Regression metrics ---------------------------------------------------------
def r2(y, yhat):
    """
    R-squared
    """
    if is_single_input(y):
        return np.nan
    ss_residual = np.sum((y - yhat) ** 2)
    ss_total = np.sum((y - np.mean(y)) ** 2)
    R2 = 1 - (ss_residual / ss_total)
    return R2

def pearsonr(y, yhat):
    """
    correlation coefficient
    """
    if is_single_input(y):
        return np.nan
    cov = np.cov(y, yhat)[0, 1] # get the covariance (off-diagonal element)
    std_y = np.std(y)
    std_yhat = np.std(yhat)
    return cov / (std_y * std_yhat)
    
def pearsonr2(y, yhat):
    """
    squared correlation coefficient
    """
    if is_single_input(y):
        return np.nan
    return pearsonr(y, yhat) ** 2

def ccc(y, yhat):
    """
    Lins' concordance correlation coefficient
    """
    if is_single_input(y):
        return np.nan
    r = pearsonr(y, yhat)
    mean_y = np.mean(y)
    mean_yhat = np.mean(yhat)
    var_y = np.var(y)
    var_yhat = np.var(yhat)
    sd_y = np.std(y)
    sd_yhat = np.std(yhat)
    num = 2 * r * sd_y * sd_yhat
    den = var_y + var_yhat + (mean_y - mean_yhat) ** 2
    return num / den

def mae(y, yhat):
    """
    mean absolute error
    """
    return np.mean(np.abs(y - yhat))

def rmse(y, yhat):
    """
    root mean squared error
    """
    return np.sqrt(np.mean((y - yhat) ** 2))

def rmspe(y, yhat):
    """
    root mean squared percentage error
    """
    return np.sqrt(np.mean(((y - yhat) / y) ** 2))

def rsr(y, yhat):
    """
    RMSE standard deviation ratio
    """
    if is_single_input(y):
        return np.nan
    return rmse(y, yhat) / np.std(y)

def mae_var(y, yhat):
    """
    MAE variance
    """
    return np.var(np.abs(y - yhat))

def rmse_var(y, yhat):
    """
    RMSE variance
    """
    return np.var(np.sqrt((y - yhat) ** 2))

def rmspe_var(y, yhat):
    """
    RMSPE variance
    """
    return np.var(np.sqrt(((y - yhat) / y) ** 2))


# Classification metrics -----------------------------------------------------
def accuracy(y, yhat):
    """
    accuracy
    """
    return np.mean(y == yhat)

# machine learning metrics

def precision(y, yhat):
    """
    precision
    """
    tp = np.sum((y == 1) & (yhat == 1))
    fp = np.sum((y == 0) & (yhat == 1))
    return tp / (tp + fp)

def recall(y, yhat):
    """
    recall = sensitivity = true positive rate
    """
    tp = np.sum((y == 1) & (yhat == 1))
    fn = np.sum((y == 1) & (yhat == 0))
    return tp / (tp + fn)

# medical diagnostics metrics
def sensitivity(y, yhat):
    """
    sensitivity = recall = true positive rate
    """
    return recall(y, yhat)

def specificity(y, yhat):
    """
    specificity = true negative rate
    """
    tn = np.sum((y == 0) & (yhat == 0))
    fp = np.sum((y == 0) & (yhat == 1))
    return tn / (tn + fp)

# self-explanatory metrics
def TPR(y, yhat):
    """
    true positive rate
    """
    return sensitivity(y, yhat)

def FNR(y, yhat):
    """
    false negative rate
    """
    return 1 - sensitivity(y, yhat)

def FPR(y, yhat):
    """
    false positive rate
    """
    return 1 - specificity(y, yhat)


def TNR(y, yhat):
    """
    true negative rate
    """
    return specificity(y, yhat)     

# composite metrics
def f1(y, yhat):
    """
    f1 score
    """
    p = precision(y, yhat)
    r = recall(y, yhat)
    return 2 * (p * r) / (p + r)

def f2(y, yhat):
    return fbeta(y, yhat, beta=2)

def f05(y, yhat):
    return fbeta(y, yhat, beta=0.5)

def fbeta(y, yhat, beta=2):
    """
    f-beta score
    """
    p = precision(y, yhat)
    r = recall(y, yhat)
    return (1 + beta ** 2) * (p * r) / ((beta ** 2 * p) + r)

def mcc(y, yhat):
    """
    Matthews correlation coefficient
    """
    tp = np.sum((y == 1) & (yhat == 1))
    tn = np.sum((y == 0) & (yhat == 0))
    fp = np.sum((y == 0) & (yhat == 1))
    fn = np.sum((y == 1) & (yhat == 0))
    num = (tp * tn) - (fp * fn)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return num / den