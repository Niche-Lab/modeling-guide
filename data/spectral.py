import numpy as np
from sklearn.preprocessing import StandardScaler

def make_T(n, sds, seed=None):
    """
    generate a score matrix T (n x m), where:
    - n is the number of samples (rows).
    - m is the number of latent components (columns).
    
    parameters
    ---
    - n: number of samples
    - sds: list of 'response' to each latent component by sample
    """
    if seed:
        np.random.seed(seed)
    m = len(sds) # number of latent components
    T = np.ones((n, m))

    # add random noise to each component for all samples
    for i in range(m):
        T[:, i] += np.random.normal(0, sds[i], n)

    return T

def make_P(p, mus, sds, amps):
    """
    generate a loading matrix P (m x p), where m is the number of latent components
    - p: number of projected bands
    - mus: list of means to define the peaks
    - sds: list of standard deviations to define the peaks
    - amps: list of amplitudes to define the peaks    
    """
    if len(mus) != len(sds) or len(mus) != len(amps):
        raise ValueError("mus, sds, and amps must have the same length")

    m = len(mus) # number of components
    P = np.zeros((m, p))
    for i in range(p): 
        P[:, i] = [amp * gaussian_pdf(i, mu, sd)\
            for mu, sd, amp in zip(mus, sds, amps)]
    return P / P.std()
    
def apply_effect(T, effects, seed=None):
    """
    apply effects to the score matrix T 
    """
    if seed:
        np.random.seed(seed)
    n, m = T.shape # number of samples, number of components

    # determine how many effects we have and how many samples per effect
    n_effects = len(effects)
    n_eff_per = n // n_effects

    # randomly choose which components will be affected by each effect
    affected_comps = np.random.choice(m, n_effects)

    # apply each effect to the corresponding subset of samples and chosen component
    for i in range(n_effects):        
        i_start = i * n_eff_per
        i_end = (i + 1) * n_eff_per
        T[i_start:i_end, affected_comps[i]] *= effects[i]
        # print(f"Effect {i}: {effects[i]} on component {affected_comps[i]}")
    return T

    
def make_y(X, p_effect=10, noise=0.5, seed=None):
    if seed:
        np.random.seed(seed)
    # standardize feature matrix
    Xstd = StandardScaler().fit_transform(X)
    n, p = X.shape

    # define linear and non-linear effects
    n_linear = p_effect // 3
    n_nonlinear = (p_effect // 3) * 2
    idx_effects = np.random.choice(p, p_effect, replace=False)
    # print("Linear effects: ", idx_effects[:n_linear])
    # print("Non-linear effects: ", idx_effects[n_linear:])

    # linear effects
    X_linear = Xstd[:, idx_effects[:n_linear]]
    beta_linear = np.random.normal(0, 1, (n_linear, 1))
    y_linear = (X_linear @ beta_linear).reshape(-1)
    y_linear /= y_linear.std()
    
    # non-linear effects (sine and cosine)
    X_nonlinear0 = Xstd[:, idx_effects[n_linear:n_nonlinear]]
    X_nonlinear1 = Xstd[:, idx_effects[n_nonlinear:]]
    y_nonlinear0 = np.sin(X_nonlinear0).mean(axis=1)
    y_nonlinear1 = np.cos(X_nonlinear1**2).mean(axis=1)
    y_nonlinear0 /= y_nonlinear0.std()
    y_nonlinear1 /= y_nonlinear1.std()
    
    # combine effects to derive y
    y = y_linear + y_nonlinear0 + y_nonlinear1
    
    # apply noise
    y = np.log(y - y.min() + 1)
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).reshape(-1)
    y += np.random.normal(0, noise, n)
    
    # return
    return y
  
# pdf gaussian
def gaussian_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
