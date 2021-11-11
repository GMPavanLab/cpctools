import numpy as np
import numpy.linalg as la
import scipy

def simpleKernelSoap(x: np.ndarray, y: np.ndarray) -> float:
    """
    Soap Kernel
    """

    return 1 - scipy.spatial.distance.cosine(x, y)
    # return np.dot(x, y) / (la.norm(x) * la.norm(y))


def simpleSOAPdistance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Distance based on Soap Kernel.
    """
    try:
        return np.sqrt(2.0 - 2.0 * simpleKernelSoap(x, y))
    except FloatingPointError:
        return 0.0


def KernelSoap(x: np.ndarray, y: np.ndarray, n: int) -> float:
    """
    Soap Kernel
    """

    # return (1 - scipy.spatial.distance.cosine(x, y)) ** n
    return (np.dot(x, y) / (la.norm(x) * la.norm(y))) ** n


def SOAPdistance(x: np.ndarray, y: np.ndarray, n: int = 1) -> float:
    """
    Distance based on Soap Kernel.
    """
    try:
        return np.sqrt(2.0 - 2.0 * KernelSoap(x, y, n))
    except FloatingPointError:
        return 0
        