from .SOAPClassify import SOAPclassification
import numpy as np


def transitionMatrixFromSOAPClassification(
    data: SOAPclassification, stride: int = 1
) -> "np.ndarray[float]":
    """Generates the unnormalized matrix of the transitions from a :func:`classifyWithSOAP`

        The matrix is organized in the following way:
        for each atom in each frame we increment by one the cell whose row is the
        state at the frame `n-stride` and the column is the state at the frame `n`
        If the classification includes an error with a `-1` values the user should add an 'error' class in the legend

    Args:
        data (SOAPclassification): the results of the soapClassification from :func:`classifyWithSOAP`
        stride (int): the stride in frames between each state confrontation. Defaults to 1.
        of the groups that contain the references in hdf5FileReference
    Returns:
        np.ndarray[float]: the unnormalized matrix of the transitions
    """
    nframes = len(data.references)
    nat = len(data.references[0])

    nclasses = len(data.legend)
    transMat = np.zeros((nclasses, nclasses), np.dtype(float))

    for frameID in range(stride, nframes, 1):
        for atomID in range(0, nat):
            classFrom = data.references[frameID - stride][atomID]
            classTo = data.references[frameID][atomID]
            transMat[classFrom, classTo] += 1
    return transMat


def normalizeMatrix(transMat: "np.ndarray[float]") -> "np.ndarray[float]":
    """normalizes a matrix that is an ouput of :func:`transitionMatrixFromSOAPClassification`

    The matrix is normalized with the criterion that the sum of each **row** is `1`

    Args:
        np.ndarray[float]: the unnormalized matrix of the transitions

    Returns:
        np.ndarray[float]: the normalized matrix of the transitions
    """
    for row in range(transMat.shape[0]):
        sum = np.sum(transMat[row, :])
        if sum != 0:
            transMat[row, :] /= sum
    return transMat


def transitionMatrixFromSOAPClassificationNormalized(
    data: SOAPclassification, stride: int = 1, withErrors=False
) -> "np.ndarray[float]":
    """Generates the normalized matrix of the transitions from a :func:`classifyWithSOAP` and normalize it

        The matrix is organized in the following way:
        for each atom in each frame we increment by one the cell whose row is the
        state at the frame `n-stride` and the column is the state at the frame `n`

        The matrix is normalized with the criterion that the sum of each **row** is `1`

    Args:
        data (SOAPclassification): the results of the soapClassification from :func:`classifyWithSOAP`
        stride (int): the stride in frames between each state confrontation. Defaults to 1.
        of the groups that contain the references in hdf5FileReference
    Returns:
        np.ndarray[float]: the normalized matrix of the transitions
    """
    transMat = transitionMatrixFromSOAPClassification(data, stride, withErrors)
    return normalizeMatrix(transMat)
