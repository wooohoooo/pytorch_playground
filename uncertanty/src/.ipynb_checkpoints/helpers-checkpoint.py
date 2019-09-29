import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import numpy as np
import seaborn as sns

N = 100
min_x, max_x = -3, 1
    
    
def uncertainity_estimate(X, model, iters, l2=0.005, range_fn=trange):
    outputs = np.hstack([model(X[:, np.newaxis]).data.numpy() for i in range_fn(iters)])
    y_mean = outputs.mean(axis=1)
    y_variance = outputs.var(axis=1)
    tau = l2 * (1-model.dropout_p) / (2*N*model.decay)
    y_variance += (1/tau)
    y_std = np.sqrt(y_variance) + (1/tau)
    return y_mean, y_std

def ensemble_uncertainity_estimate(X, ensemble, iters, l2=0.005, range_fn=trange):
    outputs = np.hstack([model(X[:, np.newaxis]).data.numpy() for model in ensemble.models])
    y_mean = outputs.mean(axis=1)
    y_variance = outputs.var(axis=1)
    tau = l2 * (1-ensemble.dropout_p) / (2*N*ensemble.decay)
    y_variance += (1/tau)
    y_std = np.sqrt(y_variance)# + (1/tau)
    return y_mean, y_std



def plot_model(model,X_true,y_true,X_obs,y_obs, iters=200, l2=0.005, n_std=3, ax=None, uncertainty_function=uncertainity_estimate):
    if ax is None:
        plt.close("all")
        plt.clf()
        fig, ax = plt.subplots(1,1)
    y_mean, y_std = uncertainty_function(X_true, model, iters, l2=l2)
    
    ax.plot(X_obs, y_obs, ls="none", marker="o", color="0.1", alpha=0.5, label="observed")
    ax.plot(X_true, y_true, ls="-", color="r", label="true")
    ax.plot(X_true, y_mean, ls="-", color="b", label="mean")
    for i in range(n_std):
        ax.fill_between(
            X_true,
            y_mean - y_std * ((i+1)/2),
            y_mean + y_std * ((i+1)/2),
            color="b",
            alpha=0.1
        )
    ax.legend()
    sns.despine(offset=10)
    return ax