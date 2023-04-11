"""Module that contains various analysis routines"""

import numpy
from numpy import ndarray
from .distances import simpleSOAPdistance
from MDAnalysis import Universe, AtomGroup
from MDAnalysis.lib.NeighborSearch import AtomNeighborSearch


def timeSOAP(
    SOAPTrajectory: ndarray,
    window: int = 1,
    stride: int = None,
    backward: bool = False,
    distanceFunction: callable = simpleSOAPdistance,
) -> "tuple[ndarray, ndarray]":
    """performs the 'timeSOAP' analysis on the given SOAP trajectory

        * Original author: Cristina Caruso
        * Mantainer: Daniele Rapetti
    Args:
        SOAPTrajectory (int):
            a trajectory of SOAP fingerprints, should have shape (nFrames,nAtoms,SOAPlenght)
        window (int):
            the dimension of the windows between each state confrontations.
            Defaults to 1.
        stride (int):
            the stride in frames between each state confrontation. **NOT IN USE**.
            Defaults to None.
        deltaT (int): number of frames to skip
        distanceFunction (callable, optional):
            the function that define the distance. Defaults to :func:`SOAPify.distances.simpleSOAPdistance`.

    Returns:
        tuple[numpy.ndarray,numpy.ndarray]:
            - **timedSOAP** the timeSOAP values, shape(frames-1,natoms)
            - **deltaTimedSOAP** the derivatives of timeSOAP, shape(natoms, frames-2)
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

    deltaTimedSOAP = numpy.diff(timedSOAP.T, axis=-1)

    return timedSOAP, deltaTimedSOAP


def timeSOAPsimple(
    SOAPTrajectory: ndarray,
    window: int = 1,
    stride: int = None,
    backward: bool = False,
) -> "tuple[ndarray, ndarray]":
    r"""performs the 'timeSOAP' analysis on the given **normalized** SOAP trajectory

        this is optimized to use :func:`SOAPify.distances.simpleSOAPdistance`,
        without calling it.

        .. warning:: this function works **only** with normalized numpy.float64
          soap vectors!

            The SOAP distance is calculated with

            .. math::
                d(\vec{a},\vec{b})=\sqrt{2-2\frac{\vec{a}\cdot\vec{b}}{\left\|\vec{a}\right\|\left\|\vec{b}\right\|}}

            That is equivalent to

            .. math::
                d(\vec{a},\vec{b})=\sqrt{2-2\hat{a}\cdot\hat{b}} = \sqrt{\hat{a}\cdot\hat{a}+\hat{b}\cdot\hat{b}-2\hat{a}\cdot\hat{b}} =

                \sqrt{(\hat{a}-\hat{b})\cdot(\hat{a}-\hat{b})}

            That is the euclidean distance between the versors

        * Original author: Cristina Caruso
        * Mantainer: Daniele Rapetti
    Args:
        SOAPTrajectory (int):
            a **normalize ** trajectory of SOAP fingerprints, should have shape
            (nFrames,nAtoms,SOAPlenght)
        window (int):
            the dimension of the windows between each state confrontations.
            Defaults to 1.
        stride (int):
            the stride in frames between each state confrontation. **NOT IN USE**.
            Defaults to None.
        deltaT (int): number of frames to skip

    Returns:
        tuple[numpy.ndarray,numpy.ndarray]:
            - **timedSOAP** the timeSOAP values, shape(frames-1,natoms)
            - **deltaTimedSOAP** the derivatives of timeSOAP, shape(natoms, frames-2)
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

    deltaTimedSOAP = numpy.diff(timedSOAP.T, axis=-1)

    return timedSOAP, deltaTimedSOAP


def listNeighboursAlongTrajectory(
    inputUniverse: Universe, cutOff: float, trajSlice: slice = slice(None)
) -> "list[list[AtomGroup]]":
    """produce a per frame list of the neighbours, atom per atom

        * Original author: Martina Crippa
        * Mantainer: Daniele Rapetti
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
) -> "tuple[ndarray,ndarray,ndarray,ndarray]":
    """return, listed per each atoms the parameters used in the LENS analysis

        * Original author: Martina Crippa
        * Mantainer: Daniele Rapetti
    Args:
        nnListPerFrame (list[list[AtomGroup]]):
             a frame by frame list of the neighbours of each atom: output of
             :func:`listNeighboursAlongTrajectory
    Returns:
        tuple[numpy.ndarray,numpy.ndarray,numpy.ndarray,numpy.ndarray]:
            - **lensArray** The calculated LENS parameter
            - **numberOfNeighs** the count of neighbours per frame
            - **lensNumerators** the numerators used for calculating LENS parameter
            - **lensDenominators** the denominators used for calculating LENS parameter
    """
    nAt = numpy.asarray(nnListPerFrame, dtype=object).shape[1]
    nFrames = numpy.asarray(nnListPerFrame, dtype=object).shape[0]
    # this is the number of common NN between frames
    lensArray = numpy.zeros((nAt, nFrames))
    # this is the number of NN at that frame
    numberOfNeighs = numpy.zeros((nAt, nFrames))
    # this is the numerator of LENS
    lensNumerators = numpy.zeros((nAt, nFrames))
    # this is the denominator of lens
    lensDenominators = numpy.zeros((nAt, nFrames))
    # each nnlist contains also the atom that generates them,
    # so 0 nn is a 1 element list
    for atomID in range(nAt):
        numberOfNeighs[atomID, 0] = nnListPerFrame[0][atomID].shape[0] - 1
        # let's calculate the numerators and the denominators
        for frame in range(1, nFrames):
            numberOfNeighs[atomID, frame] = nnListPerFrame[frame][atomID].shape[0] - 1
            lensDenominators[atomID, frame] = (
                nnListPerFrame[frame][atomID].shape[0]
                + nnListPerFrame[frame - 1][atomID].shape[0]
                - 2
            )
            lensNumerators[atomID, frame] = numpy.setxor1d(
                nnListPerFrame[frame][atomID], nnListPerFrame[frame - 1][atomID]
            ).shape[0]

    denIsNot0 = lensDenominators != 0
    # lens
    lensArray[denIsNot0] = lensNumerators[denIsNot0] / lensDenominators[denIsNot0]
    return lensArray, numberOfNeighs, lensNumerators, lensDenominators
