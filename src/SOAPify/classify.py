"""Submodule to classify trajectories

    Contains the definition of the container for the :class:`SOAPclassification`
    and for the references container :class:`SOAPReferences`.
    Along with the definition of function to apply a classification to a given 
    dataset.
"""
from typing import Callable
from dataclasses import dataclass
import numpy as np
import h5py

from .distances import SOAPdistanceNormalized
from .utils import fillSOAPVectorFromdscribe, normalizeArray


@dataclass
class SOAPclassification:
    """
    Stores the information about the SOAP classification of a trajectory.
    """

    #: stores the (per frame) per atom information about the distance from the
    #: closest reference fingerprint
    distances: "np.ndarray[float]"
    #: stores the (per frame) per atom index of the closest reference
    references: "np.ndarray[int]"
    legend: "list[str]"  #:: stores the references legend


@dataclass
class SOAPReferences:
    """
    Stores the spectra selected for a environments dictionary.
    """

    names: "list[str]"  #: the names of the references
    spectra: "np.ndarray[np.float64]"  #: the SOAP spectra of the references
    lmax: int  #: the parameter lmax used in the calculation
    nmax: int  #: the parameter nmax used in the calculation

    def __len__(self) -> int:
        """returns the lenght of the dictionary

        Returns:
            int: the number of stored spectra
        """
        return len(self.names)


def createReferencesFromTrajectory(
    h5SOAPDataSet: h5py.Dataset,
    addresses: dict,
    lmax: int,
    nmax: int,
    doNormalize=True,
) -> SOAPReferences:
    """Generate a SOAPReferences object.

    by storing the data found from h5SOAPDataSet.
    The atoms are selected trough the addresses dictionary.

    Args:
        h5SOAPDataSet (h5py.Dataset): the dataset with the SOAP fingerprints
        addresses (dict):
            the dictionary with the names and the addresses of the fingerprints.
            The keys will be used as the names of the references and the values
            assigned to the keys must be tuples or similar with the number of
            the chosen frame and the atom number (for example
            ``dict(exaple=(framenum, atomID))``)
        doNormalize (bool, optional):
            If True normalizes the SOAP vector before storing them.
            Defaults to True.
        settingsUsedInDscribe (dscribeSettings|None, optional):
            If none the SOAP vector are not preprocessed, if not none the SOAP
            vectors are decompressed, as dscribe omits the symmetric part of the
            spectra. Defaults to None.

    Returns:
        SOAPReferences: the container with the selected references
    """
    nofData = len(addresses)
    names = list(addresses.keys())
    SOAPDim = h5SOAPDataSet.shape[2]
    SOAPexpectedDim = lmax * nmax * nmax
    SOAPSpectra = np.empty((nofData, SOAPDim), dtype=h5SOAPDataSet.dtype)
    for i, key in enumerate(addresses):
        SOAPSpectra[i] = h5SOAPDataSet[addresses[key][0], addresses[key][1]]
    if SOAPexpectedDim != SOAPDim:
        SOAPSpectra = fillSOAPVectorFromdscribe(SOAPSpectra, lmax, nmax)
    if doNormalize:
        SOAPSpectra = normalizeArray(SOAPSpectra)
    return SOAPReferences(names, SOAPSpectra, lmax, nmax)


def getDistanceBetween(
    data: np.ndarray, spectra: np.ndarray, distanceCalculator: Callable
) -> np.ndarray:
    """Generate an array with the distances between the the data and the given `spectra`

        TODO: enforce the np.ndarray

    Args:
        data (np.ndarray): the array of the data
        spectra (np.ndarray): the references
        distanceCalculator (Callable):
            the function to calculate the distances

    Returns:
        np.ndarray:
            the array of the distances (the shape is `(data.shape[0], spectra.shape[0])`)
    """
    toret = np.zeros((data.shape[0], spectra.shape[0]), dtype=data.dtype)
    for j in range(spectra.shape[0]):
        for i in range(data.shape[0]):
            toret[i, j] = distanceCalculator(data[i], spectra[j])
    return toret


def getDistancesFromRef(
    SOAPTrajData: h5py.Dataset,
    references: SOAPReferences,
    distanceCalculator: Callable,
    doNormalize: bool = False,
) -> np.ndarray:
    """generates the distances between a SOAP-hdf5 trajectory and the given references

    Args:
        SOAPTrajData (h5py.Dataset): the dataset containing the SOAP trajectory
        references (SOAPReferences): the contatiner of the references
        distanceCalculator (Callable): the function to calculate the distances
        doNormalize (bool, optional):
            informs the function if the given data needs to be normalized before
            caclulating the distance. Defaults to False.

    Returns:
        np.ndarray: the "trajectory" of distance from the given references
    """

    chunkDims = min(100, SOAPTrajData.chunks[0])
    # assuming shape is (nframes, natoms, nsoap)
    currentFrame = 0
    doconversion = SOAPTrajData.shape[-1] != references.spectra.shape[-1]
    distanceFromReference = np.zeros(
        (SOAPTrajData.shape[0], SOAPTrajData.shape[1], len(references))
    )
    while SOAPTrajData.shape[0] > currentFrame:
        upperFrame = min(SOAPTrajData.shape[0], currentFrame + chunkDims)
        frames = SOAPTrajData[currentFrame:upperFrame]
        if doconversion:
            frames = fillSOAPVectorFromdscribe(frames, references.lmax, references.nmax)
        if doNormalize:
            frames = normalizeArray(frames)
        for i, frame in enumerate(frames):
            distanceFromReference[currentFrame + i] = getDistanceBetween(
                frame, references.spectra, distanceCalculator
            )
        currentFrame += chunkDims

    return distanceFromReference


def getDistancesFromRefNormalized(
    SOAPTrajData: h5py.Dataset, references: SOAPReferences
):
    """shortcut for :func:`SOAPify.classify.getDistancesFromRef` forcing normalization

        see :func:`SOAPify.SOAPClassify.getDistancesFromRef`,
        the distance calculator is :func:`SOAPdistanceNormalized` and
        doNormalize is set to True

    Args:
        SOAPTrajData (h5py.Dataset):
            the dataset containing the SOAP trajectory
        references (SOAPReferences):
            the contatiner of the references
    Returns:
        np.ndarray: the trajectory of distance from the given references
    """
    return getDistancesFromRef(
        SOAPTrajData, references, SOAPdistanceNormalized, doNormalize=True
    )


def mergeReferences(*x: SOAPReferences) -> SOAPReferences:
    """Merges a list of :class:`SOAPReferences` into a single object

    Raises:
        ValueError: if the lmax and the nmax of the references are not the same

    Returns:
        SOAPReferences:
            a new `SOAPReferences` that contains the concatenated list of references
    """
    names = []
    for i in x:
        names += i.names
        if x[0].nmax != i.nmax or x[0].lmax != i.lmax:
            raise ValueError("nmax or lmax are not the same in the two references")
    return SOAPReferences(
        names,
        np.concatenate([i.spectra for i in x]),
        nmax=x[0].nmax,
        lmax=x[0].lmax,
    )


def saveReferences(
    h5position: "h5py.Group|h5py.File", targetDatasetName: str, refs: SOAPReferences
):
    """Export the given references in the indicated group/hdf5 file

    Args:
        h5position (h5py.Group|h5py.File):
            The file object of the group where to save the references
        targetDatasetName (str): the name to give to the list of references
        refs (SOAPReferences): the `SOAPReferences` object to be exported
    """
    whereToSave = h5position.require_dataset(
        targetDatasetName,
        shape=refs.spectra.shape,
        dtype=refs.spectra.dtype,
        compression="gzip",
        compression_opts=9,
    )
    whereToSave[:] = refs.spectra
    whereToSave.attrs.create("nmax", refs.nmax)
    whereToSave.attrs.create("lmax", refs.lmax)
    whereToSave.attrs.create("names", refs.names)


def getReferencesFromDataset(dataset: h5py.Dataset) -> SOAPReferences:
    """Given a `h5py.Dataset` returns a :class:`SOAPReferences` with the initializated data

        TODO: check if the dataset contains the needed references

    Args:
        dataset (h5py.Dataset): the dataset with the references

    Returns:
        SOAPReferences: the prepared references container
    """
    fingerprints = dataset[:]
    names = dataset.attrs["names"].tolist()
    lmax = dataset.attrs["lmax"]
    nmax = dataset.attrs["nmax"]
    return SOAPReferences(names=names, spectra=fingerprints, lmax=lmax, nmax=nmax)


def applyClassification(
    SOAPTrajData: h5py.Dataset,
    references: SOAPReferences,
    distanceCalculator: Callable,
    doNormalize: bool = False,
) -> SOAPclassification:
    """Applies the references to a dataset.

    generates the distances from the given references and then classify all of
    the atoms by the closest element in the dictionary

    Args:
        SOAPTrajData (h5py.Dataset): the dataset containing the SOAP trajectory
        references (SOAPReferences):  the contatiner of the references
        distanceCalculator (Callable): the function to calculate the distances
        doNormalize (bool, optional):
            informs the function if the given data needs to be normalized
            before caclulating the distance. Defaults to False.
    Returns:
        SOAPclassification: The result of the classification
    """
    info = getDistancesFromRef(
        SOAPTrajData, references, distanceCalculator, doNormalize
    )
    minimumDistID = np.argmin(info, axis=-1)
    minimumDist = np.amin(info, axis=-1)
    return SOAPclassification(minimumDist, minimumDistID, references.names)
