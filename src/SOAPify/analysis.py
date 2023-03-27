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
        nnListPerFrame.append([at.ix for at in nnListPerAtom])
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
    nAt = numpy.shape(nnListPerFrame)[1]
    nFrames = numpy.shape(nnListPerFrame)[0]
    # this is the number of common NN between frames
    ncontTot = numpy.zeros((nAt, nFrames))
    # this is the number of NN at that frame
    nnTot = numpy.zeros((nAt, nFrames))
    # this is the numerator of LENS
    numTot = numpy.zeros((nAt, nFrames))
    # this is the denominator of lens
    denTot = numpy.zeros((nAt, nFrames))
    # each nnlist contains also the atom that generates them,
    # so 0 nn is a 1 element list

    for atomID in range(nAt):
        # nnTot[atomID, 0] = nnListPerFrame[0][atomID].shape[0] - 1
        for frame in range(1, nFrames):
            nnTot[atomID, frame] = nnListPerFrame[frame][atomID].shape[0] - 1
            denTot[atomID, frame] = (
                nnListPerFrame[frame][atomID].shape[0]
                + nnListPerFrame[frame - 1][atomID].shape[0]
                - 2
            )
            numTot[atomID, frame] = numpy.setxor1d(
                nnListPerFrame[frame][atomID], nnListPerFrame[frame - 1][atomID]
            ).shape[0]

            intesectionNN = numpy.intersect1d(
                nnListPerFrame[frame][atomID], nnListPerFrame[frame - 1][atomID]
            )
            # all neighBours have changed or no NN for both frames:
            if intesectionNN.shape[0] == 1:
                # non NN for both frames
                if (
                    nnListPerFrame[frame][atomID].shape[0] == 1
                    and nnListPerFrame[frame - 1][atomID].shape[0] == 1
                ):
                    # we are using zeros as initializer
                    # ncontTot[atomID, frame] = 0
                    # numTot[atomID, frame] = 0
                    continue
                # ncontTot[atomID, frame] = 1
                # numTot[atomID, frame] = 1
    denIsNot0 = denTot != 0
    ncontTot[denIsNot0] = numTot[denIsNot0] / denTot[denIsNot0]
    return ncontTot, nnTot, numTot, denTot
