"""Utilities submodule, cointains unclassified support functions
Author: Daniele Rapetti"""
from itertools import combinations_with_replacement
import numpy
from ase.data import atomic_numbers
import h5py


def _SOAPpstr(l, Z, n, Zp, np) -> str:
    if atomic_numbers[Z] < atomic_numbers[Zp]:
        Z, Zp = Zp, Z
        n, np = np, n
    return f"{l}_{Z}{n}_{Zp}{np}"


def getdscribeSOAPMapping(
    lmax: int, nmax: int, species: "list[str]", crossover: bool = True
) -> numpy.ndarray:
    """returns how dscribe saves the SOAP results

       return a list of string with the identities of the data returned from
       dscribe, see the note in
       https://singroup.github.io/dscribe/1.2.x/tutorials/descriptors/soap.html

    Args:
        lmax (int): the lmax specified in the calculation.
        nmax (int): the nmax specified in the calculation.
        species (list[str]): the list of atomic species.
        crossover (bool):
            if True, the SOAP descriptors are generated for the mixed species.
            Defaults to True.

    Returns:
        numpy.ndarray:
        an array of strings with the mapping of the output of the analysis
    """
    species = orderByZ(species)
    pdscribe = []
    for Z in species:
        for Zp in species:
            if not crossover and Z != Zp:
                continue
            for l in range(lmax + 1):
                for n in range(nmax):
                    for np in range(nmax):
                        if (np, atomic_numbers[Zp]) >= (n, atomic_numbers[Z]):
                            pdscribe.append(_SOAPpstr(l, Z, n, Zp, np))
    return numpy.array(pdscribe)


def _getRSindex(nmax: int, species: "list[str]") -> numpy.ndarray:
    """Support function for quippy"""
    rsIndex = numpy.zeros((2, nmax * len(species)), dtype=numpy.int32)
    i = 0
    for iSpecies in range(len(species)):
        for na in range(nmax):
            rsIndex[:, i] = na, iSpecies
            i += 1
    return rsIndex


def getquippySOAPMapping(
    lmax: int, nmax: int, species: "list[str]", diagonalRadial: bool = False
) -> numpy.ndarray:
    """returns how quippi saves the SOAP results


        return a list of string with the identities of the data returned from quippy,
        see https://github.com/libAtoms/GAP/blob/main/descriptors.f95#L7588

    Args:
        lmax (int): the lmax specified in the calculation.
        nmax (int): the nmax specified in the calculation.
        species (list[str]): the list of atomic species.
        diagonalRadial (bool):
            if True, Only return the n1=n2 elements of the power spectrum.
            **NOT IMPLEMENTED**. Defaults to False.

    Returns:
        numpy.ndarray:
        an array of strings with the mapping of the output of the analysis
    """
    species = orderByZ(species)
    rsIndex = _getRSindex(nmax, species)
    pquippy = []
    for ia in range(len(species) * nmax):
        np = rsIndex[0, ia]
        Zp = species[rsIndex[1, ia]]
        for jb in range(ia + 1):  # ia is  in the range
            n = rsIndex[0, jb]
            Z = species[rsIndex[1, jb]]
            # if diagonalRadial and np != n:
            #    continue

            for l in range(lmax + 1):
                pquippy.append(_SOAPpstr(l, Z, n, Zp, np))
    return numpy.array(pquippy)


def orderByZ(species: "list[str]") -> "list[str]":
    """Orders the list of species by their atomic number

    Args:
        species (list[str]): the list of atomic species to be ordered

    Returns:
        list[str]: the ordered list of atomic species
    """
    return sorted(species, key=lambda x: atomic_numbers[x])


def getAddressesQuippyLikeDscribe(
    lmax: int, nmax: int, species: "list[str]"
) -> numpy.ndarray:
    """create a support bdarray to reorder the quippy output in a dscribe fashion

        Given the lmax and nmax of a SOAP calculation and the species of the atoms
        returns an array of idexes for reordering the quippy results as the dscribe results

    Args:
        lmax (int): the lmax specified in the calculation.
        nmax (int): the nmax specified in the calculation.
        species (list[str]): the list of atomic species.

        Returns:
            numpy.ndarray: an array of indexes
    """
    species = orderByZ(species)
    nsp = len(species)
    addresses = numpy.zeros(
        (lmax + 1) * ((nmax * nsp) * (nmax * nsp + 1)) // 2, dtype=int
    )
    quippyOrder = getquippySOAPMapping(lmax, nmax, species)
    dscribeOrder = getdscribeSOAPMapping(lmax, nmax, species)
    for i, _ in enumerate(addresses):
        addresses[i] = numpy.where(quippyOrder == dscribeOrder[i])[0][0]
    return addresses


def normalizeArray(x: numpy.ndarray) -> numpy.ndarray:
    """Normalizes the futher axis of the given array

    (eg. in an array of shape (100,50,3) normalizes all the  5000 3D vectors)

    Args:
        x (numpy.ndarray): the array to be normalized

    Returns:
        numpy.ndarray: the normalized array
    """
    norm = numpy.linalg.norm(x, axis=-1, keepdims=True)
    norm[norm == 0] = 1
    return x / norm


def getSlicesFromAttrs(attrs: dict) -> "tuple(list,dict)":
    """returns the positional slices for the calculated fingerprints

        Given the attributes of from a SOAP dataset returns the slices of the
        SOAP vector that contains the pair information

    Args:
        attrs (dict): the attributes of the SOAP dataset

    Returns:
        tuple(list,dict):
        the slices of the SOAP vector to be extracted and the atomic types
    """
    species = attrs["species"]
    slices = {}
    for symbol1 in species:
        for symbol2 in species:
            if f"species_location_{symbol1}-{symbol2}" in attrs:
                slices[symbol1 + symbol2] = slice(
                    attrs[f"species_location_{symbol1}-{symbol2}"][0],
                    attrs[f"species_location_{symbol1}-{symbol2}"][1],
                )

    return species, slices


def _getIndexesForFillSOAPVectorFromdscribeSameSpecies(
    lMax: int,
    nMax: int,
) -> numpy.ndarray:
    """returns the indexes of the SOAP vector to be reordered

    given lMax and nMax returns the indexes of the SOAP vector to be
    reordered to fill a  complete vector.

    useful to calculate the correct distances between the SOAP vectors

    Args:
        lMax (int): the lmax specified in the calculation.
        nMax (int): the nmax specified in the calculation.

    Returns:
        numpy.ndarray: the array of the indexes in the correct order
    """

    completeData = numpy.zeros(((lMax + 1), nMax, nMax), dtype=int)
    limitedID = 0
    for l in range(lMax + 1):
        for n in range(nMax):
            for nP in range(n, nMax):
                completeData[l, n, nP] = limitedID
                completeData[l, nP, n] = limitedID
                limitedID += 1
    return completeData.reshape(-1)


def _getIndexesForFillSOAPVectorFromdscribe(
    lMax: int,
    nMax: int,
    atomTypes: list = None,
    atomicSlices: dict = None,
) -> numpy.ndarray:
    """Given the data of a SOAP calculation from dscribe returns the SOAP power
        spectrum with also the symmetric part explicitly stored, see the note in
        https://singroup.github.io/dscribe/1.2.x/tutorials/descriptors/soap.html

        No controls are implemented on the shape of the soapFromdscribe vector.

    Args:
        soapFromdscribe (numpy.ndarray):
            the result of the SOAP calculation from the dscribe utility
        lmax (int):
            the l_max specified in the calculation. Defaults to 8.
        nMax (int):
            the n_max specified in the calculation. Defaults to 8.
        atomTypes (list[str]):
            the list of atomic species. Defaults to [None].
        atomicSlices (dict):
            the slices of the SOAP vector relative to che atomic species combinations.
            Defaults to None.

    Returns:
        numpy.ndarray:
            The full soap spectrum, with the symmetric part sored explicitly
    """
    if atomTypes is None:
        atomTypes = [None]
    if atomTypes == [None]:
        return _getIndexesForFillSOAPVectorFromdscribeSameSpecies(lMax, nMax)
    nOfFeatures = (lMax + 1) * nMax * nMax
    nofCombinations = len(list(combinations_with_replacement(atomTypes, 2)))
    completeData = numpy.zeros(nOfFeatures * nofCombinations, dtype=int)
    combinationID = 0
    for i, symbol1 in enumerate(atomTypes):
        for j in range(i, len(atomTypes)):
            symbol2 = atomTypes[j]
            completeID = combinationID * nOfFeatures
            completeSlice = slice(
                completeID,
                completeID + nOfFeatures,
            )
            if symbol1 == symbol2:
                completeData[completeSlice] = (
                    _getIndexesForFillSOAPVectorFromdscribeSameSpecies(lMax, nMax)
                    + atomicSlices[symbol1 + symbol2].start
                )
            else:
                completeData[completeSlice] = (
                    numpy.arange(nOfFeatures, dtype=int)
                    + atomicSlices[symbol1 + symbol2].start
                )
            combinationID += 1
    return completeData


def fillSOAPVectorFromdscribe(
    soapFromdscribe: numpy.ndarray,
    lMax: int,
    nMax: int,
    atomTypes: list = None,
    atomicSlices: dict = None,
) -> numpy.ndarray:
    """Given the result of a SOAP calculation from dscribe returns the SOAP power spectrum
        with also the symmetric part explicitly stored, see the note in
        https://singroup.github.io/dscribe/1.2.x/tutorials/descriptors/soap.html

        No controls are implemented on the shape of the soapFromdscribe vector.

    Args:
        soapFromdscribe (numpy.ndarray):
            the result of the SOAP calculation from the dscribe utility
        lMax (int):
            the l_max specified in the calculation.
        nMax (int):
            the n_max specified in the calculation.

    Returns:
        numpy.ndarray:
            The full soap spectrum, with the symmetric part sored explicitly
    """
    if atomTypes is None:
        atomTypes = [None]
    upperDiag = int((lMax + 1) * nMax * (nMax + 1) / 2)
    fullmat = nMax * nMax * (lMax + 1)
    limitedSOAPdim = upperDiag * len(atomTypes) + fullmat * int(
        (len(atomTypes) - 1) * len(atomTypes) / 2
    )
    # enforcing the order of the atomTypes
    if len(atomTypes) > 1:
        atomTypes = list(atomTypes)
        atomTypes.sort(key=lambda x: atomic_numbers[x])

    if soapFromdscribe.shape[-1] != limitedSOAPdim:
        raise ValueError(
            "fillSOAPVectorFromdscribe: the given soap vector do not have the expected dimensions"
        )
    indexes = _getIndexesForFillSOAPVectorFromdscribe(
        lMax, nMax, atomTypes, atomicSlices
    )
    if len(soapFromdscribe.shape) == 1:
        return soapFromdscribe[indexes]
    if len(soapFromdscribe.shape) == 2:
        return soapFromdscribe[:, indexes]
    if len(soapFromdscribe.shape) == 3:
        return soapFromdscribe[:, :, indexes]

    raise ValueError(
        "fillSOAPVectorFromdscribe: cannot convert array with len(shape) >=3"
    )


def getSOAPSettings(fitsetData: h5py.Dataset) -> dict:
    """Gets the settings of the SOAP calculation

        you can feed directly this output to :func:`fillSOAPVectorFromdscribe`

        #TODO: make tests for this
    Args:
        fitsetData (h5py.Dataset): A soap dataset with attributes

    Returns:
        dict: a dictionary with the following components:
            - **nMax**
            - **lMax**
            - **atomTypes**
            - **atomicSlices**

    """
    lmax = fitsetData.attrs["l_max"]
    nmax = fitsetData.attrs["n_max"]
    symbols, atomicSlices = getSlicesFromAttrs(fitsetData.attrs)

    return {
        "nMax": nmax,
        "lMax": lmax,
        "atomTypes": symbols,
        "atomicSlices": atomicSlices,
    }
