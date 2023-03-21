"""Module that contains various analysis routines"""
import numpy
from numpy import ndarray
from .distances import simpleSOAPdistance


def tempoSOAP(
    SOAPTrajectory: ndarray,
    window: int = 1,
    stride: int = None,
    backward: bool = False,
    distanceFunction: callable = simpleSOAPdistance,
) -> "tuple[ndarray, ndarray]":
    """performs the 'tempoSOAP' analysis on the given SOAP trajectory

        Original author: Cristina Caruso
        Mantainer: Daniele Rapetti
    Args:
        SOAPTrajectory (int):
            _description_
        window (int):
            the dimension of the windows between each state confrontations.
            Defaults to 1.
        stride (int):
            the stride in frames between each state confrontation.
            Defaults to None.
        deltaT (int): number of frames to skip
        distanceFunction (callable, optional):
            the defini. Defaults to :func:`SOAPify.distances.simpleSOAPdistance`.

    Returns:
        tuple[numpy.ndarray,numpy.ndarray]: _description_
    """
    if stride is None:
        stride = window
    if stride > window:
        raise ValueError("the window must be bigger than the stride")
    if window >= SOAPTrajectory.shape[0] or stride >= SOAPTrajectory.shape[0]:
        raise ValueError("stride and window must be smaller than simulation lenght")
    timedSOAP = numpy.zeros((SOAPTrajectory.shape[0] - window, SOAPTrajectory.shape[1]))

    for frame in range(window, SOAPTrajectory.shape[0]):
        for molecule in range(0, SOAPTrajectory.shape[1]):
            x = SOAPTrajectory[frame, molecule, :]
            y = SOAPTrajectory[frame - window, molecule, :]
            # fill the matrix (each molecule for each frame)
            timedSOAP[frame - window, molecule] = distanceFunction(x, y)

    expectedDeltaTimedSOAP = []
    for molecule in range(0, timedSOAP.shape[1]):
        derivative = numpy.diff(timedSOAP[:, molecule])
        expectedDeltaTimedSOAP.append(derivative)

    return timedSOAP, expectedDeltaTimedSOAP
