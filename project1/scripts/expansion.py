import numpy as np

"""
Two functions to expand a feature matrix

Example of usage of both functions :
polynomial_expansion(np.array([[2,3],[2,3]]), 3)


func = [lambda x: x * x, lambda x: x * x * x]
expansion(np.array([[2,3],[2,3]]), func)
"""

def polynomial_expansion(x, degree = 2):
    """
    Do a polynomial expansion of matrix x up to degree
    The minimum degree is 2.
    
    Parameters
    ----------
    x: ndarray
        feature matrix x
    degree: int
        highest degree of the expansion matrix
    
    Returns
    -------
    ndarray
        x expanded up to degree
    """
    def poly(x, degree):
        """
        polynomial basis functions for input data x, for j=2 up to j=degree.
        """
        acc = x*x
        phi = acc.copy()
        for i in range(2, degree):
            acc *= x
            phi = np.c_[phi, acc]
        return phi

    for i in range(0,x.shape[1]):
        x = np.c_[x, np.apply_along_axis(poly , 0, x[:,i], degree)]
    return x


def expansion(x, functions):
    """
    Expand the matrix x with the list of functions
    
    Parameters
    ----------
    x: ndarray
        feature matrix x
    functions: ndarray of lambdas
        array of functions used to expand the feature matrix
    
    Returns
    -------
    ndarray
        feature matrix x expanded with the list of functions
    """
    for i in range(0, x.shape[1]):
        for f in functions:
            x = np.c_[x, np.apply_along_axis(f , 0, x[:,i])]
    return x

