"""Simple submodule for generating trasition matrices from SOAPclassification
"""
import numpy

from ..classify import SOAPclassification
from .tracker import *


# TODO add stride/window selection
def transitionMatrixFromSOAPClassification(
    data: SOAPclassification, stride: int = 1
) -> "numpy.ndarray[float]":
    """Generates the unnormalized matrix of the transitions

       see :func:`calculateTransitionMatrix` for a detailed description of an
       unnormalized transition matrix

    Args:
        data (SOAPclassification):
            the results of the soapClassification from :func:`classify`
        stride (int):
            the stride in frames between each state confrontation.
            Defaults to 1.
    Returns:
        numpy.ndarray[float]: the unnormalized matrix of the transitions
    """
    nframes = len(data.references)
    nat = len(data.references[0])

    nclasses = len(data.legend)
    transMat = numpy.zeros((nclasses, nclasses), dtype=numpy.float64)

    for frameID in range(stride, nframes, 1):
        for atomID in range(0, nat):
            classFrom = data.references[frameID - stride][atomID]
            classTo = data.references[frameID][atomID]
            transMat[classFrom, classTo] += 1
    return transMat


def normalizeMatrixByRow(transMat: "numpy.ndarray[float]") -> "numpy.ndarray[float]":
    """Normalizes a transition matrix by row

        The matrix is normalized with the criterion that the sum of each
        **row** is `1`

    Args:
        numpy.ndarray[float]: the unnormalized matrix of the transitions

    Returns:
        numpy.ndarray[float]: the normalized matrix of the transitions
    """
    toRet = transMat.copy()
    for row in range(transMat.shape[0]):
        mysum = numpy.sum(toRet[row, :])
        if mysum != 0:
            toRet[row, :] /= mysum
    return toRet


# TODO add stride/window selection
def transitionMatrixFromSOAPClassificationNormalized(
    data: SOAPclassification, stride: int = 1
) -> "numpy.ndarray[float]":
    """Generates a normalized matrix of the transitions

        The matrix is organized in the following way:
        for each atom in each frame we increment by one the cell whose row is
        the state at the frame `n-stride` and the column is the state at the
        frame `n`

        The matrix is normalized with the criterion that the sum of each
        **row** is `1`

    Args:
        data (SOAPclassification):
            the results of the soapClassification from :func:`classify`
        stride (int):
            the stride in frames between each state confrontation.
            Defaults to 1.
    Returns:
        numpy.ndarray[float]: the normalized matrix of the transitions
    """
    transMat = transitionMatrixFromSOAPClassification(data, stride)
    return normalizeMatrixByRow(transMat)


# TODO add stride here?
def calculateResidenceTimesFromClassification(
    classification: SOAPclassification,
) -> "list[numpy.ndarray]":
    """Calculates the resindence times for each state in the classification.

        Given a SOAPclassification, calculate the resindence time for each
        element of the classification. The residence time is how much an atom
        stays in a determined state:
        this function calculates the redidence time for each atom and for each
        state, and returns an ordered list of residence times for each state
        (hence losing the atom identity)

    Args:
        classification (SOAPclassification):
        the classified trajectory

    Returns:
        list[numpy.ndarray]:
        an ordered list of the residence times for each state
    """

    nofFrames = classification.references.shape[0]
    nofAtoms = classification.references.shape[1]
    residenceTimes = [[] for i in range(len(classification.legend))]
    for atomID in range(nofAtoms):
        atomTraj = classification.references[:, atomID]
        time = 0
        state = atomTraj[0]
        for frame in range(1, nofFrames):
            if atomTraj[frame] != state:
                residenceTimes[state].append(time)
                state = atomTraj[frame]
                time = 0
            time += 1
        # the last state does not have an out transition,
        # appendig negative time to make it clear
        residenceTimes[state].append(-time)

    return [
        numpy.sort(numpy.array(residenceTimes[i])) for i in range(len(residenceTimes))
    ]


def calculateResidenceTimes(
    data: SOAPclassification, statesTracker: list = None
) -> "list[numpy.ndarray]":
    """Calculates the resindence time for each state in the classification.

        The function decides automatically if calling
        :func:`calculateResidenceTimesFromClassification`
        or :func:`tracker.getResidenceTimesFromStateTracker`

    Args:
        data (SOAPclassification):
            the classified trajectory, is statesTracker is passed will be used
            for getting the legend of the states
        statesTracker (list, optional):
            a list of list of state trackers, organized by atoms, or a list of
            state trackers. Defaults to None.

    Returns:
        list[numpy.ndarray]:
        an ordered list of the residence times for each state
    """

    if not statesTracker:
        return calculateResidenceTimesFromClassification(data)
    return getResidenceTimesFromStateTracker(statesTracker, data.legend)


# TODO add stride/window selection
def calculateTransitionMatrix(
    data: SOAPclassification, stride: int = 1, statesTracker: list = None
) -> numpy.ndarray:
    """Generates the unnormalized matrix of the transitions
        from a :func:`classify` of from a statesTracker

        The matrix is organized in the following way:
        for each atom in each frame we increment by one the cell whose row is
        the state at theframe `n-stride` and the column is the state
        at the frame `n`
        If the classification includes an error with a `-1` values the user
        should add an 'error' class in the legend

    Args:
        data (SOAPclassification):
            the results of the soapClassification from :func:`classify`
        stride (int):
            the stride in frames between each state confrontation.
            Defaults to 1.
        statesTracker (list, optional):
            a list of list of state trackers, organized by atoms, or a list of
            state trackers. Defaults to None.

    Returns:
        numpy.ndarray[float]: the unnormalized matrix of the transitions
    """

    if not statesTracker:
        return transitionMatrixFromSOAPClassification(data, stride)
    return transitionMatrixFromStateTracker(statesTracker, data.legend)
