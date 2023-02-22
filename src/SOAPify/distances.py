"""This submodules contaisn a collection of SOAP distances"""
import numpy as np
import numpy.linalg as la
import scipy


def simpleKernelSoap(x: np.ndarray, y: np.ndarray) -> float:
    """a simpler SOAP Kernel than :func:`KernelSoap`, power is always 1

    Args:
        x (np.ndarray): a SOAP fingerprint
        y (np.ndarray): a SOAP fingerprint

    Returns:
        float: kernel value
    """

    return 1 - scipy.spatial.distance.cosine(x, y)
    # return np.dot(x, y) / (la.norm(x) * la.norm(y))


def simpleSOAPdistance(x: np.ndarray, y: np.ndarray) -> float:
    """a simpler SOAP distance than :func:`SOAPdistance`, power is always 1

    Args:
        x (np.ndarray): a SOAP fingerprint
        y (np.ndarray): a SOAP fingerprint

    Returns:
        float: the distance between the two fingerprints, between :math:`0` and :math:`2`
    """
    try:
        return np.sqrt(2.0 - 2.0 * simpleKernelSoap(x, y))
    except FloatingPointError:
        return 0.0


def kernelSoap(x: np.ndarray, y: np.ndarray, n: int) -> float:
    """The SOAP Kernel with a variable power

    Args:
        x (np.ndarray): a SOAP fingerprint
        y (np.ndarray): a SOAP fingerprint
        n (int): the power to elevate the result of the kernel

    Returns:
        float: kernel value
    """

    # return (1 - scipy.spatial.distance.cosine(x, y)) ** n
    return (np.dot(x, y) / (la.norm(x) * la.norm(y))) ** n


def SOAPdistance(x: np.ndarray, y: np.ndarray, n: int = 1) -> float:
    """the SOAP distance between two SOAP fingerprints

    Args:
        x (np.ndarray): a SOAP fingerprint
        y (np.ndarray): a SOAP fingerprint
        n (int): the power to elevate the result of the kernel

    Returns:
        float: the distance between the two fingerprints, between :math:`0` and :math:`2`
    """
    try:
        return np.sqrt(2.0 - 2.0 * kernelSoap(x, y, n))
    except FloatingPointError:
        return 0.0


def SOAPdistanceNormalized(x: np.ndarray, y: np.ndarray) -> float:
    """the SOAP distance between two normalized SOAP fingerprints
    The pre-normalized vectors should net some performace over the classic kernel

    Args:
        x (np.ndarray): a normalized SOAP fingerprint
        y (np.ndarray): a normalized SOAP fingerprint

    Returns:
        float: the distance between the two fingerprints, between :math:`0` and :math:`2`
    """

    return np.sqrt(np.abs(2.0 - 2.0 * x.dot(y)))
