from ase.io.formats import F
import h5py
import numpy as np
from .SOAPbase import SOAPdistance
from dataclasses import dataclass


@dataclass
class SOAPclassification:
    """Utility class to store the information about the SOAP classification of a system."""

    distances: "np.ndarray[float]"  #: stores the (per frame) per atom information about the distance from the closes reference fingerprint
    references: "np.ndarray[int]"  #: stores the (per frame) per atom index of the closest reference
    legend: "list[str]"  #:: stores the references legend


def classifyWithSOAP(
    SOAPTrajData: h5py.Dataset, hdf5FileReference: h5py.File, referenceAddresses: list
) -> SOAPclassification:
    """classifies all atoms in a system, given the precalculated set of SOAP fingerprints, and the references.

    Args:
        SOAPTrajData (h5py.Dataset): The hdf5 dataset that contaisn the SOAP fingerprints
        hdf5FileReference (h5py.File): the hdf5 file that contains the references
        referenceAddresses (list): a list of the addresses of the references and\/or of the groups that contain the references in hdf5FileReference

    Returns:
        SOAPclassification: the information of the whole trajectory divided frame by frame and atom by atom, along with the legend
    """
    spectra, legend = loadRefs(hdf5FileReference, referenceAddresses)
    nframes = SOAPTrajData.shape[0]
    nat = SOAPTrajData.shape[1]
    distances = np.zeros((nframes, nat), np.dtype(float))
    references = np.zeros((nframes, nat), np.dtype(int))
    n = 0
    # nchunks=len(T1.iter_chunks())
    # TODO verify if this loop can be parallelized
    for chunk in SOAPTrajData.iter_chunks():
        print(f"working on chunk {n}, {chunk}")
        n += 1
        for chunkID, frame in enumerate(SOAPTrajData[chunk]):
            frameID = chunk[0].start + chunkID
            for atomID, atom in enumerate(frame):
                distances[frameID, atomID] = 1000.0
                references[frameID, atomID] = -1
                for j in range(len(spectra)):
                    try:
                        tdist = SOAPdistance(atom, spectra[j])
                        if tdist < distances[frameID, atomID]:
                            distances[frameID, atomID] = tdist
                            references[frameID, atomID] = j
                    except:
                        print(f"at {j}:")
                        print(f"spectra: {spectra[j]}")
                        print(f"atom: {atomID}")
    return SOAPclassification(distances, references, legend)


def loadRefs(
    hdf5FileReference: h5py.File, referenceAddresses: list
) -> "tuple[np.ndarray[float],list[str]]":
    """loads the references given in referenceAddresses from an hdf5 file

    Args:
        hdf5FileReference (h5py.File): the hdf5 file that contains the references
        referenceAddresses (list): a list of the addresses of the references and/or
        of the groups that contain the references in hdf5FileReference
    Returns:
        tuple[np.ndarray[float],list[str]]: returns a tuple with the fingerprint and the relative names of the references
    """
    spectra = []  # np.zeros((0, 0), dtype=np.dtype(float))
    legend = []
    for address in referenceAddresses:
        data = hdf5FileReference[address]
        if isinstance(data, h5py.Group):
            for refName in data.keys():
                dataset = hdf5FileReference[f"{address}/{refName}"]
                legend.append(refName)
                spectra.append(np.mean(dataset[:], axis=0))

        elif isinstance(data, h5py.Dataset):
            legend.append(data.name.rsplit("/")[-1])
            spectra.append(np.mean(data[:], axis=0))
        else:
            print(
                f"loadRefs cannot create a reference from given input: repr={repr(data)}"
            )
            exit(255)
    spectra = np.array(spectra)
    return spectra, legend


def transitionMatrixFromSOAPClassification(
    data: SOAPclassification, stride: int = 1
) -> "np.ndarray[float]":
    """Generates the unnormalized matrix of the transitions from a :func:`classifyWithSOAP`

        The matrix is organized in the following way:
        for each atom in each frame we increment by one the cell whose row is the
        state at the frame `n-stride` and the column is the state at the frame `n`

    Args:
        data (SOAPclassification): the results of the soapClassification from :func:`classifyWithSOAP`
        stride (int): the stride in frames between each state confrontation. Defaults to 1.
        of the groups that contain the references in hdf5FileReference
    Returns:
        np.ndarray[float]: the unnormalized matrix of the transitions
    """
    nframes = len(data.references)
    nat = len(data.references[0])
    # +1 in case of errors
    nclasses = len(data.legend) + 1
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
    data: SOAPclassification, stride: int = 1
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
    transMat = transitionMatrixFromSOAPClassification(data, stride)
    return normalizeMatrix(transMat)


if __name__ == "__main__":
    import h5py
    import numpy as np

    trajLoader = h5py.File("WaterSOAP__.hdf5", "r")
    SOAPtraj = trajLoader["SOAP/1ns"]
    refFile = h5py.File("ReferenceWater.hdf5", "r")
    data = classifyWithSOAP(SOAPtraj, refFile, ["Waters/R10/tip4p2005", "Ices/R10"])
    print(data)
