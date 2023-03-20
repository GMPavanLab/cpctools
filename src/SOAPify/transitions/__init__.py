"""Simple submodule for generating trasition matrices from SOAPclassification
Author: Daniele Rapetti"""
import numpy

from ..classify import SOAPclassification
from .tracker import *


def transitionMatrixFromSOAPClassification(
    data: SOAPclassification, stride: int = 1, window: "int|None" = None
) -> "numpy.ndarray[float]":
    """Generates the unnormalized matrix of the transitions

        see :func:`calculateTransitionMatrix` for a detailed description of an
        unnormalized transition matrix.

        If the user specifies windows equal to None the windows is set equal to
        the stride

    Args:
        data (SOAPclassification):
            the results of the soapClassification from :func:`classify`
            stride (int): the stride in frames between each state confrontation.
            Defaults to 1.
        window (int):
            the dimension of the windows between each state confrontations.
            Defaults to None.
    Returns:
        numpy.ndarray[float]: the unnormalized matrix of the transitions
    """
    if window is None:
        window = stride
    if window < stride:
        raise ValueError("the window must be bigger than the stride")
    if window > data.references.shape[0]:
        raise ValueError("stride and window must be smaller than simulation lenght")
    nframes = len(data.references)
    nat = len(data.references[0])

    nclasses = len(data.legend)
    transMat = numpy.zeros((nclasses, nclasses), dtype=numpy.float64)

    for frameID in range(window, nframes, stride):
        for atomID in range(0, nat):
            classFrom = data.references[frameID - window][atomID]
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
        rowSum = numpy.sum(toRet[row, :])
        if rowSum != 0:
            toRet[row, :] /= rowSum
    return toRet


def transitionMatrixFromSOAPClassificationNormalized(
    data: SOAPclassification, stride: int = 1, window: "int|None" = None
) -> "numpy.ndarray[float]":
    """Generates the normalized transition matrix from a :func:`classify`

        The matrix is organized in the following way:
        for each atom in each frame we increment by one the cell whose row is
        the state at the frame `n-stride` and the column is the state at the
        frame `n`

        The matrix is normalized with the criterion that the sum of each **row**
          is `1`

    Args:
        data (SOAPclassification):
            the results of the soapClassification from :func:`classify`
        stride (int):
            the stride in frames between each state confrontation. Defaults to 1.
        window (int):
            the dimension of the windows between each state confrontations.
            Defaults to None.
    Returns:
        numpy.ndarray[float]: the normalized matrix of the transitions
    """
    transMat = transitionMatrixFromSOAPClassification(data, stride, window)
    return normalizeMatrixByRow(transMat)


def calculateResidenceTimesFromClassification(
    data: SOAPclassification, window: int = 1, stride: "int|None" = None
) -> "list[numpy.ndarray]":
    """Calculates the resindence time for each element of the classification.

        The residence time is how much an atom stays in a determined state: this
        function calculates the residence time for each atom and for each state,
        and returns an ordered list of residence times for each state
        hence losing the atom identity)

        When a `stride` is specified, the algorithm accumulates the resiedence
        times of up to `window` simulations on hte same trajectory samplet at
        the `window` rate

        The first and the last residence time for each atom are saved as negative
        numbers to signal the user that that time has to be considered more carefully

    Args:
        data (SOAPclassification):
            the classified trajectory
        window (int):
            the dimension of the windows between each state confrontations.
            Defaults to 1.
        stride (int):
            the stride in frames between each state confrontation.
            Defaults to None.


    Returns:
        list[numpy.ndarray]:
        an ordered list of the residence times for each state
    """
    if stride is None:
        stride = window
    if stride > window:
        raise ValueError("the window must be bigger than the stride")

    if window > data.references.shape[0] or stride > data.references.shape[0]:
        raise ValueError("stride and window must be smaller than simulation lenght")

    nofFrames = data.references.shape[0]
    nofAtoms = data.references.shape[1]
    residenceTimes = [[] for i in range(len(data.legend))]

    for atomID in range(nofAtoms):
        atomTraj = data.references[:, atomID]
        for initialFrame in range(0, window, stride):
            time = 0
            state = atomTraj[initialFrame]
            initialStep = True
            for frame in range(window + initialFrame, nofFrames, window):
                time += window
                if atomTraj[frame] != state:
                    residenceTimes[state].append(time)
                    if initialStep:
                        residenceTimes[state][-1] = -residenceTimes[state][-1]
                        initialStep = False
                    state = atomTraj[frame]
                    time = 0

            # the last state does not have an out transition:
            # appendig negative time to make it clear
            # TODO: correct the window going past the last frame, may be secondary
            # NB: the -window is here because the last frame skips a +=
            residenceTimes[state].append(-time - window)
    for i, rts in enumerate(residenceTimes):
        residenceTimes[i] = numpy.sort(numpy.array(rts))

    return residenceTimes


def calculateResidenceTimes(
    data: SOAPclassification, statesTracker: list = None, **algokwargs
) -> "list[numpy.ndarray]":
    """Given a classification (and the state tracker) generates a ordered list
    of residence times per state

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
        algokwargs:
            arguments passed to the called functions
    Returns:
        list[numpy.ndarray]:
        an ordered list of the residence times for each state
    """

    if not statesTracker:
        return calculateResidenceTimesFromClassification(data, **algokwargs)
    else:
        return getResidenceTimesFromStateTracker(statesTracker, data.legend)


def calculateTransitionMatrix(
    data: SOAPclassification, statesTracker: list = None, **algokwargs
) -> numpy.ndarray:
    """Generates the unnormalized matrix of the transitions
        from a :func:`classify`
        of from a statesTracker

        The matrix is organized in the following way:
        for each atom in each frame we increment by one the cell whose row is
        the state at theframe `n-stride` and the column is the state
        at the frame `n`
        If the classification includes an error with a `-1` values the user
        should
        add an 'error' class in the legend

    Args:
        data (SOAPclassification):
            the results of the soapClassification from :func:`classify`
        stride (int):
            the stride in frames between each state confrontation.
            Defaults to 1.
        statesTracker (list, optional):
            a list of list of state trackers, organized by atoms, or a list of
            state trackers. Defaults to None.
        algokwargs:
            arguments passed to the called functions

    Returns:
        numpy.ndarray[float]: the unnormalized matrix of the transitions
    """

    if not statesTracker:
        return transitionMatrixFromSOAPClassification(data, **algokwargs)
    else:
        return transitionMatrixFromStateTracker(statesTracker, data.legend)
