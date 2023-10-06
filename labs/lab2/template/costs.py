# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def compute_mse_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    N = y.shape[0]
    e = y - tx.dot(w)
    loss = 1 / (2 * N) * e.T.dot(e)
    return loss


def compute_mae_loss(y, tx, w):
    N = y.shape[0]
    e = y - tx.dot(w)
    return np.absolute((1 / N) * np.sum(e))