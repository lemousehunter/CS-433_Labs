import numpy as np


def compute_subgradient_mae(y, tx, w):
    """Compute a subgradient of the MAE at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the subgradient of the MAE at w.
    """
    
    N = y.shape[0]
    e = y - tx.dot(w)
    differentials = np.zeros(e.shape)
        
    for i in range(len(e)):
        if e[i] < 0:
            differentials[i] = -1.0
        elif e[i] == 0:
            differentials[i] = np.random.uniform(-1.0, 1.0)  # getting any subgradient for when e == 0, since all lines for y = c, with c being any value between -1.0 and 1.0 is considered a subgradient of the absolute function
        else:  # e[i] > 0
            differentials[i] = 1.0
    #print("subgradient shape:", differentials.shape)

    return -(1/N)*differentials.dot(tx)
