import numpy as np



def append_x_min(array,xmin,xmax):
    return np.insert(np.insert(array,0,xmin),-1,xmax)

def get_data(N, min_x, max_x):
    w, b = np.random.randn(2)
    def true_model(X):
        lin_factor = w * X + b
        y = np.sin(lin_factor*10)*(lin_factor**2)
        return y
    X_true = np.arange(min_x, max_x, 0.01)
    X_true = append_x_min(X_true,min_x,max_x)
    
    y_true = true_model(X_true)
    span = (max_x - min_x)
    scale = 0.2
    X_obs = min_x + span*scale + np.random.rand(N)*(span - 2*scale*span)
    #X_obs = np.append(np.array([min_x]),[list(X_obs), max_x])
    X_obs = append_x_min(X_obs,min_x,max_x)
    #return X_obs
    y_obs = true_model(X_obs) + np.random.randn(N+2)*0.4
    y_obs[-1] = y_obs[-1]+3

    return (X_obs, y_obs, X_true, y_true), (w, b, true_model)