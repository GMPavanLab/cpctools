from re import A
import numpy
from ase.data import atomic_numbers
from itertools import combinations_with_replacement


def normalizeArray(a: numpy.ndarray) -> numpy.ndarray:
    """Normalizes the futher axis of the given array

    (eg. in an array of shape (100,50,3) normalizes all the  5000 3D vectors)

    Args:
        a (numpy.ndarray): the arra to be normalized

    Returns:
        numpy.ndarray: the normalized array
    """
    norm = numpy.linalg.norm(a, axis=-1, keepdims=True)
    norm[norm == 0] = 1
    return a / norm


def getSlicesFromAttrs(attrs: dict) -> "tuple(list,dict)":
    """Given the attributes of from a SOAP dataset returns the slices of the SOAP vector that contains the pair information

    Args:
        attrs (dict): the attributes of the SOAP dataset

    Returns:
        tuple(list,dict): the slices of the SOAP vector to be extracted and the atomic types
    """
    species = attrs["species"]
    slices = {}
    for s1 in species:
        for s2 in species:
            if (f"species_location_{s1}-{s2}") in attrs:
                slices[s1 + s2] = slice(
                    attrs[f"species_location_{s1}-{s2}"][0],
                    attrs[f"species_location_{s1}-{s2}"][1],
                )

    return species, slices


def fillsingleSOAPVectorFromdscribe(
    soapFromdscribe: numpy.ndarray,
    l_max: int,
    n_max: int,
    atomTypes: list = [None],
    atomicSlices: dict = None,
) -> numpy.ndarray:
    """Given the resul of a SOAP calculation from dscribe returns the SOAP power
        spectrum with also the symmetric part explicitly stored, see the note in https://singroup.github.io/dscribe/1.2.x/tutorials/descriptors/soap.html

        No controls are implemented on the shape of the soapFromdscribe vector.

    Args:
        soapFromdscribe (numpy.ndarray): the result of the SOAP calculation from the dscribe utility
        l_max (int, optional): the l_max specified in the calculation. Defaults to 8.
        n_max (int, optional): the n_max specified in the calculation. Defaults to 8.

    Returns:
        numpy.ndarray: The full soap spectrum, with the symmetric part sored explicitly
    """
    nOfFeatures = (l_max + 1) * n_max * n_max
    if atomTypes == [None]:
        nofCombinations = 1
        completeData = numpy.zeros(
            ((l_max + 1), n_max, n_max), dtype=soapFromdscribe.dtype
        )
        limitedID = 0
        for l in range(l_max + 1):
            for n in range(n_max):
                for np in range(n, n_max):
                    completeData[l, n, np] = soapFromdscribe[limitedID]
                    completeData[l, np, n] = soapFromdscribe[limitedID]
                    limitedID += 1
    else:
        nofCombinations = len(list(combinations_with_replacement(atomTypes, 2)))
        completeData = numpy.zeros(
            nOfFeatures * nofCombinations, dtype=soapFromdscribe.dtype
        )
        combinationID = 0
        for i, s1 in enumerate(atomTypes):
            for j in range(i, len(atomTypes)):
                s2 = atomTypes[j]
                myslice = atomicSlices[s1 + s2]
                completeID = combinationID * nOfFeatures
                completeSlice = slice(
                    completeID,
                    completeID + nOfFeatures,
                )
                if s1 == s2:
                    limitedID = 0
                    completeDataSlice = numpy.zeros(
                        ((l_max + 1), n_max, n_max), dtype=soapFromdscribe.dtype
                    )
                    temp = soapFromdscribe[myslice]
                    for l in range(l_max + 1):
                        for n in range(n_max):
                            for np in range(n, n_max):
                                completeDataSlice[l, n, np] = temp[limitedID]
                                completeDataSlice[l, np, n] = temp[limitedID]
                                limitedID += 1
                    completeData[completeSlice] = completeDataSlice.reshape(-1)
                else:
                    completeData[completeSlice] = soapFromdscribe[myslice]
                combinationID += 1
    # TODO finish this with the desired positioning of the atom couples
    return completeData.reshape(-1)


def fillSOAPVectorFromdscribeMonoAtomic(
    soapFromdscribe: numpy.ndarray,
    l_max: int,
    n_max: int,
) -> numpy.ndarray:
    """Given the resul of a SOAP calculation from dscribe returns the SOAP power
        spectrum with also the symmetric part explicitly stored, see the note in https://singroup.github.io/dscribe/1.2.x/tutorials/descriptors/soap.html

        No controls are implemented on the shape of the soapFromdscribe vector.

    Args:
        soapFromdscribe (numpy.ndarray): the result of the SOAP calculation from the dscribe utility
        l_max (int): the l_max specified in the calculation.
        n_max (int): the n_max specified in the calculation.

    Returns:
        numpy.ndarray: The full soap spectrum, with the symmetric part sored explicitly
    """
    limitedSOAPdim = int(((l_max + 1) * (n_max + 1) * n_max) / 2)
    if soapFromdscribe.shape[-1] != limitedSOAPdim:
        raise Exception(
            "fillSOAPVectorFromdscribe: the given soap vector do not have the expected dimensions"
        )
    if len(soapFromdscribe.shape) == 1:
        return fillsingleSOAPVectorFromdscribe(soapFromdscribe, l_max, n_max)
    fullSOAPdim = (l_max + 1) * n_max * n_max
    retShape = list(soapFromdscribe.shape)
    retShape[-1] = fullSOAPdim
    retdata = numpy.empty((retShape), dtype=soapFromdscribe.dtype)
    if len(retShape) == 2:
        for i in range(soapFromdscribe.shape[0]):
            retdata[i] = fillsingleSOAPVectorFromdscribe(
                soapFromdscribe[i], l_max, n_max
            )
    elif len(retShape) == 3:
        for i in range(soapFromdscribe.shape[0]):
            for j in range(soapFromdscribe.shape[1]):
                retdata[i, j] = fillsingleSOAPVectorFromdscribe(
                    soapFromdscribe[i, j], l_max, n_max
                )
    else:
        raise Exception("fillSOAPVectorFromdscribe: cannot convert array with shape >3")

    return retdata


def fillSOAPVectorFromdscribe(
    soapFromdscribe: numpy.ndarray,
    l_max: int,
    n_max: int,
    atomTypes: list = [None],
    atomicSlices: dict = None,
) -> numpy.ndarray:
    """Given the resul of a SOAP calculation from dscribe returns the SOAP power
        spectrum with also the symmetric part explicitly stored, see the note in https://singroup.github.io/dscribe/1.2.x/tutorials/descriptors/soap.html

        No controls are implemented on the shape of the soapFromdscribe vector.

    Args:
        soapFromdscribe (numpy.ndarray): the result of the SOAP calculation from the dscribe utility
        l_max (int): the l_max specified in the calculation.
        n_max (int): the n_max specified in the calculation.

    Returns:
        numpy.ndarray: The full soap spectrum, with the symmetric part sored explicitly
    """
    upperDiag = int((l_max + 1) * (n_max) * (n_max + 1) / 2)
    fullmat = n_max * n_max * (l_max + 1)
    limitedSOAPdim = upperDiag * len(atomTypes) + fullmat * int(
        (len(atomTypes) - 1) * len(atomTypes) / 2
    )
    # enforcing the order of the atomTypes
    if len(atomTypes) > 1:
        atomTypes = list(atomTypes)
        atomTypes.sort(key=lambda x: atomic_numbers[x])

    if soapFromdscribe.shape[-1] != limitedSOAPdim:
        raise Exception(
            "fillSOAPVectorFromdscribe: the given soap vector do not have the expected dimensions"
        )
    if len(soapFromdscribe.shape) == 1:
        return fillsingleSOAPVectorFromdscribe(
            soapFromdscribe, l_max, n_max, atomTypes, atomicSlices
        )
    # TODO: len(list(combinations_with_replacement(atomTypes, 2))) should be more automated, it is calculated at each passage
    fullSOAPdim = (
        (l_max + 1)
        * n_max
        * n_max
        * len(list(combinations_with_replacement(atomTypes, 2)))
    )
    retShape = list(soapFromdscribe.shape)
    retShape[-1] = fullSOAPdim
    retdata = numpy.empty((retShape), dtype=soapFromdscribe.dtype)
    if len(retShape) == 2:
        for i in range(soapFromdscribe.shape[0]):
            retdata[i] = fillsingleSOAPVectorFromdscribe(
                soapFromdscribe[i], l_max, n_max, atomTypes, atomicSlices
            )
    elif len(retShape) == 3:
        for i in range(soapFromdscribe.shape[0]):
            for j in range(soapFromdscribe.shape[1]):
                retdata[i, j] = fillsingleSOAPVectorFromdscribe(
                    soapFromdscribe[i, j], l_max, n_max, atomTypes, atomicSlices
                )
    else:
        raise Exception("fillSOAPVectorFromdscribe: cannot convert array with shape >3")

    return retdata
