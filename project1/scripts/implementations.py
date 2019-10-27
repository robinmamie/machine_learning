import numpy as np
from common_functions import *


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent.
    Uses MSE.

    Parameters
    ----------
    y:  ndarray
        the labels
    tx: ndarray
        matrix x tilde, i.e. the parameters with a bias term
    initial_w: ndarray
        initial weight vector
    max_iters: int
        maximum number of iterations
    gamma: float
        learning rate

    Returns
    -------
    (ndarray, float)
        Last weight vector and the corresponding loss value
    """
    loss = 0
    w = initial_w
    n = y.shape[0]
    n2 = n*2

    for n_iter in range(max_iters):
        e = compute_e(y, tx, w)
        gradient = compute_gradient_MSE(tx, n, e)
        loss = compute_loss_MSE(n2, e)
        # Update weights
        w -= gamma * gradient

    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent.
    Uses MSE.

    Parameters
    ----------
    y:  ndarray
        the labels
    tx: ndarray
        matrix x tilde, i.e. the parameters with a bias term
    initial_w: ndarray
        initial weight vector
    max_iters: int
        maximum number of iterations
    gamma: float
        learning rate

    Returns
    -------
    (ndarray, float)
        Last weight vector and the corresponding loss value
    """

    loss = 0
    w = initial_w[:, np.newaxis]
    n = y.shape[0]
    n2 = n*2

    # Data shuffle
    data_size = len(y)
    shuffled_indices = np.random.permutation(np.arange(data_size))
    shuffled_y = y[shuffled_indices]
    shuffled_tx = tx[shuffled_indices]
    shuffled_y = shuffled_y[:,np.newaxis]

    for n_iter, by, btx in zip(range(max_iters), shuffled_y, shuffled_tx):
        by = by[np.newaxis]
        btx = btx[np.newaxis, :]
        e = compute_e(by, btx, w)
        gradient = compute_gradient_MSE(btx, n, e)
        loss = compute_loss_MSE(n2, e)

        # Update weights
        w -= gamma * gradient
    return w, compute_loss_MSE(n2, compute_e(y, tx, w[:,0]))

def least_squares(y, tx):
    """
    Linear regression using normal equations.
    Use MSE loss function

    Parameters
    ----------
    y:  ndarray
        the labels
    tx: ndarray
        matrix x tilde, i.e. the parameters with a bias term

    Returns
    -------
    (ndarray, float)
        Last weight vector and the corresponding loss value
    """
    w = la.solve(tx.T @ tx, tx.T @ y)
    return w, compute_loss_MSE(y.shape[0]*2, compute_e(y, tx, w))

def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations.

    Parameters
    ----------
    y : ndarray
        the labels
    tx: ndarray
        matrix x tilde, i.e. the parameters with a bias term
    lambda_: float
        the ridge parameter

    Returns
    -------
    (ndarray, float)
        Last weight vector and the corresponding loss value
    """
    X = tx.T @ tx
    n = y.shape[0]
    w = la.solve(X + lambda_ * (2 * n) * np.eye(X.shape[0]), tx.T @ y)
    return w, compute_loss_rmse(2 * n, compute_e(y, tx, w))

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent or SGD.

    Parameters
    ----------
    y:  ndarray
        the labels
    tx: ndarray
        matrix x tilde, i.e. the parameters with a bias term
    initial_w: ndarray
        initial weight vector
    max_iters: int
        maximum number of iterations
    gamma: float
        learning rate

    Returns
    -------
    (ndarray, float)
        Last weight vector and the corresponding loss value
    """
    threshold = 1e-8
    w = initial_w
    y = y[:,np.newaxis]
    losses = []

    for iter in range(max_iters):
        loss, gradient = logistic_regression_step(y, tx, w)
        w -= gamma * gradient
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, losses[-1]

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent or SGD.

    Parameters
    ----------
    y:  ndarray
        the labels
    tx: ndarray
        matrix x tilde, i.e. the parameters with a bias term
    lambda_: float
        regularization term
    initial_w: ndarray
        initial weight vector
    max_iters: int
        maximum number of iterations
    gamma: float
        learning rate

    Returns
    -------
    (ndarray, float)
        Last weight vector and the corresponding loss value
    """
    threshold = 1e-8
    losses = []
    y = (y+1)/2
    w = initial_w
    y = y[:,np.newaxis]

    for iter in range(max_iters):
        loss, gradient = logistic_regression_step(y, tx, w)
        loss     += lambda_ * np.squeeze(w.T @ w)
        gradient += 2 * lambda_ * w
        w -= gamma * gradient

        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, losses[-1]

