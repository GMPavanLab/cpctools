import numpy as np
from scipy.spatial.distance import jensenshannon


def KernelSoap(x, y, n):
    """
    Soap Kernel
    """
    return ( np.dot(x, y) / (np.dot(x, x) * np.dot(y, y)) ** 0.5 ) ** n


def DistanceSoap(x, y, n=1):
    """
    Distance based on Soap Kernel.
    """
    try:
        return (2.0 - 2.0 * KernelSoap(x, y, n)) ** 0.5
    except FloatingPointError:
        return 0

        
def KL(p, q):
    """
    Kullback-Leibler divergence
    """
    return np.sum(p * np.log(p / q))


def JS(p, q):
    """
    Jensen–Shannon divergence
    """
    return jensenshannon(np.exp(p), np.exp(q))
