"""Module that contains various analysis routines"""
import numpy
from numpy import ndarray
from .distances import simpleSOAPdistance
from MDAnalysis import Universe, AtomGroup
from MDAnalysis.lib.NeighborSearch import AtomNeighborSearch


def tempoSOAP(
    SOAPTrajectory: ndarray,
    window: int = 1,
    stride: int = None,
    backward: bool = False,
    distanceFunction: callable = simpleSOAPdistance,
) -> "tuple[ndarray, ndarray]":
    """performs the 'tempoSOAP' analysis on the given SOAP trajectory

        * Original author: Cristina Caruso
        * Mantainer: Daniele Rapetti
    Args:
        SOAPTrajectory (int):
            _description_
        window (int):
            the dimension of the windows between each state confrontations.
            Defaults to 1.
        stride (int):
            the stride in frames between each state confrontation. -NON IN USE-
            Defaults to None.
        deltaT (int): number of frames to skip
        distanceFunction (callable, optional):
            the function that define the distance. Defaults to :func:`SOAPify.distances.simpleSOAPdistance`.

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
    # vectorizedDistanceFunction = numpy.vectorize(
    #     distanceFunction, signature="(n),(n)->(1)"
    # )
    # print(SOAPTrajectory.shape)

    expectedDeltaTimedSOAP = numpy.diff(timedSOAP, axis=-1)

    return timedSOAP, expectedDeltaTimedSOAP


def tempoSOAPsimple(
    SOAPTrajectory: ndarray,
    window: int = 1,
    stride: int = None,
    backward: bool = False,
) -> "tuple[ndarray, ndarray]":
    """performs the 'tempoSOAP' analysis on the given SOAP trajectory

        this is optimized to use :func:`SOAPify.distances.simpleSOAPdistance`
        without calling it

        .. warning:: this function works **only** with normalized numpy.float64 soap vectors!


        * Original author: Cristina Caruso
        * Mantainer: Daniele Rapetti
    Args:
        SOAPTrajectory (int):
            _description_
        window (int):
            the dimension of the windows between each state confrontations.
            Defaults to 1.
        stride (int):
            the stride in frames between each state confrontation. -NON IN USE-
            Defaults to None.
        deltaT (int): number of frames to skip

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
    prev = SOAPTrajectory[0]
    for frame in range(window, SOAPTrajectory.shape[0]):
        actual = SOAPTrajectory[frame]
        # this is equivalent to distance of two normalized SOAP vector
        timedSOAP[frame - window] = numpy.linalg.norm(actual - prev, axis=1)
        prev = actual

    expectedDeltaTimedSOAP = numpy.diff(timedSOAP, axis=-1)

    return timedSOAP, expectedDeltaTimedSOAP


def listNeighboursAlongTrajectory(
    inputUniverse: Universe, cutOff: float, trajSlice: slice = slice(None)
) -> "list[list[AtomGroup]]":
    """produce a per frame list of the neighbours, atom per atom

    Args:
        inputUniverse (Universe):
            the universe, or the atomgroup containing the trajectory
        cutOff (float):
            the maximum neighbour distance
        trajSlice (slice, optional):
            the slice of the trajectory to consider. Defaults to slice(None).

    Returns:
        list[list[AtomGroup]]:
            list of AtomGroup wint the neighbours of each atom for each frame
    """
    nnListPerFrame = []
    for ts in inputUniverse.universe.trajectory[trajSlice]:
        nnListPerAtom = []
        nnSearch = AtomNeighborSearch(inputUniverse.atoms, box=inputUniverse.dimensions)
        for atom in inputUniverse.atoms:
            nnListPerAtom.append(nnSearch.search(atom, cutOff))
        nnListPerFrame.append(nnListPerAtom)
    return nnListPerFrame


def neighbourChangeInTime(
    nnListPerFrame: "list[list[AtomGroup]]",
) -> "tuple[list,list,list,list]":
    """return, listed per each atoms the parameters used in the LENS analysis

    Args:
        nnListPerFrame (list[list[AtomGroup]]): _description_

    Returns:
        tuple[list,list,list,list]: _description_
    """
    # this is
    ncontTot = []
    # this is
    nnTot = []
    # this is
    numTot = []
    # this is
    denTot = []
    nAt = numpy.shape(nnListPerFrame)[1]
    nFrames = numpy.shape(nnListPerFrame)[0]
    print(nnListPerFrame)
    for atomID in range(nAt):
        ncont = numpy.zeros((nFrames,))
        for frame in range(1, nFrames):
            continue
        ncontTot.append(ncont)
        nnTot.append([])
        numTot.append([])
        denTot.append([])

    return ncontTot, nnTot, numTot, denTot
