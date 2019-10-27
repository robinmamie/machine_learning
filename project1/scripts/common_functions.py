import numpy as np

def compute_e(y, tx, w):
    """
    Computes the error vector.

    Parameters
    ----------
    y:  ndarray
        the labels
    tx: ndarray
        matrix x tilde, i.e. the parameters with a bias term
    w: ndarray
        weight vector
    """
    return y - tx @ w

def compute_loss_MSE(n2, e):
    """
    Computes the mean square error.

    Parameters
    ----------
    n2: int
        the number of data points times 2 (already multiplied externally for
        performance reasons)
    w: ndarray
        the error vector (computed externally for performance reasons)
    """
    return (e.T @ e) / n2

def compute_gradient_MSE(tx, n, e):
    return - tx.T @ e / n

def compute_loss_MAE(n, e):
    return 1/n * np.sum(np.abs(e))

def compute_gradient_MAE(tx, n, e):
    return -1/n*tx.T @ np.sign(e)

def compute_loss_rmse(n2, e):
    return np.sqrt(2 * compute_loss_MSE(n2, e))

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

def compute_loss_logistic(y, tx, w):
    """compute the cost by negative log likelihood."""
    sigmoid_param = tx @ w
    # Thresholding to prevent any overflow
    sigmoid_param[sigmoid_param > 20] = 20
    sigmoid_param[sigmoid_param < -20] = -20
    sigm = sigmoid(sigmoid_param)
    loss = (y.T @ np.log(sigm)) + ((1 - y).T @ np.log(1 - sigm))
    return np.squeeze(-loss)

def compute_gradient_logistic(y, tx, w):
    """compute the gradient of loss."""
    sigmoid_param = tx @ w
    # Thresholding to prevent any overflow
    sigmoid_param[sigmoid_param > 20] = 20
    sigmoid_param[sigmoid_param < -20] = -20

    return tx.T @ (sigmoid(sigmoid_param) - y)

def logistic_regression_step(y, tx, w):
    """return the loss, gradient"""
    return compute_loss_logistic(y, tx, w), compute_gradient_logistic(y, tx, w)
