# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def cross_validation_visualization(lambds, mse_tr, mse_te, err_tr, err_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.errorbar(lambds, mse_tr, err_tr, marker='')
    plt.errorbar(lambds, mse_te, err_te, marker='')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")
    plt.show()
