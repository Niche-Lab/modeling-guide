import numpy as np

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
    T = np.random.normal(1, sds, (n, m))
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
    return P / P.std() # avoid small values in sine and cosine 
    
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

    
def make_y(X, noise=1, seed=None):
    if seed:
        np.random.seed(seed)
    # standardize feature matrix
    #Xstd = StandardScaler().fit_transform(X)
    n, p = X.shape

    # define non-linear effects
    poly = 3
    idx_effects = [50, 100, 180, 230] 
    n_effects = len(idx_effects)

    deg = [i * np.pi / 4 for i in range(n_effects)]
    x_nonlinear = X[: , idx_effects]
    y = np.zeros(n)
    for i in range(n_effects):
        y += np.sin(x_nonlinear[:, i] ** poly + deg[i])
   
    # apply noise
    #y = np.log(y - y.min() + 1)
    ystd = np.std(y)
    y += np.random.normal(0, noise * ystd, n)
    
    # return
    return y
  
# pdf gaussian
def gaussian_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
