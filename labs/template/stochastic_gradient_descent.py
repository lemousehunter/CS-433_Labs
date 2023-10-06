# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Stochastic Gradient Descent
"""
from helpers import batch_iter
from costs import compute_mae_loss, compute_mse_loss


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from a data sample batch of size B, where B < N, and their corresponding labels.

    Args:
        y: numpy array of shape=(B, )
        tx: numpy array of shape=(B,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    B = y.shape[0]
    gradients = np.zeros((B, 2))
    for idx in range(B):
        gradients[idx] = compute_gradient(np.array([y[idx]]), np.array([tx[idx, :]]), w)
    return np.average(gradients, axis=0)


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    # Get batched data
    iter = 0
    for batched_y, batched_tx in batch_iter(y, tx, batch_size, max_iters):
        # Get gradient
        gradient = compute_stoch_gradient(batched_y, batched_tx, w)
        
        # Stochastic Gradient Descent
        w = w - gamma * gradient
        loss = compute_mse_loss(y, tx, w)
        
        # Track losses and weights histories
        losses.append(loss)
        ws.append(w)

        print(
            "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )        
        )
        iter += 1
    return losses, ws
