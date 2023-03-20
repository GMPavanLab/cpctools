"""This submodule contains the settings and the function to call the SOAP engines"""
from typing import Iterable, Literal
import abc
import warnings
from ase.data import atomic_numbers, chemical_symbols
import ase
import numpy

from .utils import orderByZ, getAddressesQuippyLikeDscribe

try:
    from dscribe.descriptors import SOAP as dscribeSOAP

    HAVE_DSCRIBE = True
except ImportError:  # pragma: no cover
    HAVE_DSCRIBE = False
try:
    from quippy.descriptors import Descriptor

    HAVE_QUIPPY = True
except ImportError:  # pragma: no cover
    HAVE_QUIPPY = False

KNOWNSOAPENGINES = Literal[
    "dscribe", "quippy"
]  #:Literal type for the Known SOAP engine


def centerMaskCreator(
    SOAPatomMask: "list[str]",
    symbols: "list[str]",
) -> "list[int]":
    """Creates a mask for calculating SOAP only on a subset of the given symbols

        given the list of types of atoms to select and the list od types of atoms
        in the topology returns the mask of the selected species

        the mask is 1 if the atom i is in the `SOAPatomMask` list, else is 0

    Args:
        SOAPatomMask (list[str]): the mask to apply
        symbols (list[str]): the list of the type of atoms in the trajectory

    Returns:
        list[int]: the mask of selected atoms
    """
    return [i for i in range(len(symbols)) if symbols[i] in SOAPatomMask]


class SOAPengineContainer(abc.ABC):
    """A container for the SOAP engine

    Attributes:
        SOAPengine (SOAP): the soap engine already set up

    """

    def __init__(self, SOAPengine, centerMask, SOAPengineKind):
        self.SOAPenginekind_ = SOAPengineKind
        self.SOAPengine = SOAPengine
        self.centersMask_ = centerMask

    @property
    def engine(self):
        """retunrs the sotred engine

        Returns:
            varies: the stored engine
        """
        return self.SOAPengine

    @property
    def SOAPenginekind(self):
        """retunrs the name of the library used for the engine

        Returns:
            str: the name of the engine stored
        """
        return self.SOAPenginekind_

    @property
    def centersMask(self):
        """Retunr the centersMask"""
        return self.centersMask_

    @property
    @abc.abstractmethod
    def features(self) -> int:  # pragma: no cover
        """returns the features of the engine"""

    @property
    @abc.abstractmethod
    def nmax(self) -> int:  # pragma: no cover
        """returns the nmax"""

    @property
    @abc.abstractmethod
    def lmax(self) -> int:  # pragma: no cover
        """returns the lmax"""

    @property
    @abc.abstractmethod
    def rcut(self) -> int:  # pragma: no cover
        """returns the cutoff radius of the engine"""

    @property
    @abc.abstractmethod
    def species(self) -> list:  # pragma: no cover
        """returns the species set up fro the engine"""

    @property
    @abc.abstractmethod
    def crossover(self) -> bool:  # pragma: no cover
        """returns True if the engine wil calculate the terms for the non same species"""

    @abc.abstractmethod
    def getLocation(self, specie1, specie2):  # pragma: no cover
        """returns the slice where the two asked species are stored in the ouput array"""

    @abc.abstractmethod
    def __call__(self, atoms, **kwargs):  # pragma: no cover
        """calculate the fingerprints of the given atoms"""


class dscribeSOAPengineContainer(SOAPengineContainer):
    """A container for the SOAP engine from dscribe

    Attributes:
        SOAPengine (SOAP): the soap engine already set up

    """

    def __init__(self, SOAPengine, centerMask):
        super().__init__(SOAPengine, centerMask, "dscribe")

    @property
    def features(self):
        return self.SOAPengine.get_number_of_features()

    @property
    def nmax(self):
        # this will automatically produce a miss in the coverage, becasue I cannot
        # have two different versions of dscribe installed at the same time
        if hasattr(self.SOAPengine, "_nmax"):
            return self.SOAPengine._nmax
        if hasattr(self.SOAPengine, "_n_max"):
            return self.SOAPengine._n_max
        return None

    @property
    def lmax(self):
        # this will automatically produce a miss in the coverage, becasue I cannot
        # have two different versions of dscribe installed at the same time
        if hasattr(self.SOAPengine, "_lmax"):
            return self.SOAPengine._lmax
        if hasattr(self.SOAPengine, "_l_max"):
            return self.SOAPengine._l_max
        return None

    @property
    def rcut(self):
        # this will automatically produce a miss in the coverage, becasue I cannot
        # have two different versions of dscribe installed at the same time
        if hasattr(self.SOAPengine, "_rcut"):
            return self.SOAPengine._rcut
        if hasattr(self.SOAPengine, "_r_cut"):
            return self.SOAPengine._r_cut
        return None

    @property
    def species(self):
        return self.SOAPengine.species

    @property
    def crossover(self) -> bool:
        return self.SOAPengine.crossover

    def getLocation(self, specie1, specie2):
        """returns the slice where the two asked species are stored in the ouput array"""
        return self.SOAPengine.get_location((specie1, specie2))

    def __call__(self, atoms, **kwargs):
        toret = self.SOAPengine.create(atoms, **kwargs)
        if toret.ndim == 2:
            return numpy.expand_dims(toret, axis=0)
        return toret


class quippySOAPengineContainer(SOAPengineContainer):
    """A container for the SOAP engine from quippy

    Attributes:
        SOAPengine (SOAP): the soap engine already set up

    """

    def __init__(self, SOAPengine, centerMask):
        super().__init__(SOAPengine, centerMask, "quippy")
        species = self.species
        nmax = self.nmax
        lmax = self.lmax

        slices = {}
        nextID = 0
        prev = 0
        fullmat = nmax * nmax * (lmax + 1)
        upperDiag = ((nmax + 1) * nmax) // 2 * (lmax + 1)
        for i in range(len(species) * nmax):
            for j in range(i, len(species)):
                key = species[i] + species[j]
                # addDim = (lmax + 1) * (
                # nmax * nmax if i != j else ((nmax + 1) * nmax) // 2
                # )
                if i == j:
                    nextID = prev + upperDiag
                else:
                    nextID = prev + fullmat
                slices[key] = slice(prev, nextID)
                prev = nextID

        self._addresses = getAddressesQuippyLikeDscribe(lmax, nmax, species)
        self._slices = slices

    @property
    def features(self):
        return self.SOAPengine.dimensions() - 1

    @property
    def nmax(self):
        return self.SOAPengine._quip_descriptor.descriptor_soap.n_max

    @property
    def lmax(self):
        return self.SOAPengine._quip_descriptor.descriptor_soap.l_max

    @property
    def rcut(self):
        return self.SOAPengine._quip_descriptor.descriptor_soap.cutoff

    @property
    def species(self):
        Z = self.SOAPengine._quip_descriptor.descriptor_soap.species_z
        return [chemical_symbols[i] for i in Z]

    @property
    def crossover(self) -> bool:
        return True

    def getLocation(self, specie1, specie2):
        """returns the slice where the two asked species are stored in the ouput array"""
        sp1, sp2 = orderByZ([specie1, specie2])
        return self._slices[sp1 + sp2]

    def __call__(self, atoms, **kwargs):
        if isinstance(atoms, ase.Atoms):
            atoms = [atoms]
        nat = len(self.centersMask_) if self.centersMask_ is not None else len(atoms[0])
        toret = numpy.empty((len(atoms), nat, self.features))
        for i, frame in enumerate(atoms):
            toret[i] = self.SOAPengine.calc(frame)["data"][:, self._addresses]
        return toret


def _getAtomMask(
    atomNames: "list[str]",
    SOAPatomMask: str = None,
    centersMask: Iterable = None,
) -> list:
    """creates the atom mask for :func:`getSoapEngine`

        this is an helper function for :func:`getSoapEngine`

    Args:
        atomNames (list[str]): The list of species present in the system
        SOAPatomMask (list[str], optional):
            the symbols of the atoms whose SOAP fingerprint will be calculated.
            Defaults to None.
        centersMask (Iterable, optional):
            the indexes of the atoms whose SOAP fingerprint will be calculated.
            Defaults to None.

    Raises:
        ValueError:
            raises an exeption if the user inputs both SOAPatomMask and centersMask

    Returns:
        list: the list of the center to work on
    """
    if SOAPatomMask is not None and centersMask is not None:
        raise ValueError(
            "getSoapEngine: You can't use both SOAPatomMask and centersMask"
        )
    if SOAPatomMask is not None:
        return centerMaskCreator(SOAPatomMask, atomNames)
    if centersMask is not None:
        return centersMask.copy()

    return None


def _makeSP(spArray: "list[str]") -> "tuple[list,str]":
    """creates the atom mask for using quippy in :func:`getSoapEngine`

        this is an helper function for :func:`getSoapEngine`

    Args:
        spArray (list[str]): the list of the name of the atoms

    Returns:
        tuple[list,str]:
            the numerical Z of the involved atoms, ad a comma separated list of
            the atom Z
    """
    spZ = [atomic_numbers[specie] for specie in spArray]
    return spZ, ", ".join([str(ii) for ii in spZ])


def getSoapEngine(
    atomNames: "list[str]",
    SOAPrcut: float,
    SOAPnmax: int,
    SOAPlmax: int,
    SOAPatomMask: str = None,
    centersMask: Iterable = None,
    SOAP_respectPBC: bool = True,
    SOAPkwargs: dict = None,
    useSoapFrom: KNOWNSOAPENGINES = "dscribe",
) -> SOAPengineContainer:
    """set up a soap engine with the given settings

    please visit the manual of the relative soap engine for the calculation
    parameters

    `SOAPatomMask` and `centersMask` are mutually exclusive:

    - `centerMask` is a list of atoms whose SOAP fingerprints will be calculate
    - `SOAPatomMask` is the list of species that will be included in the `centerMask`

    **NB**: if you use quippy, we impose it to not normalize the soap vector

    Args:
        atomNames (list[str]): The list of species present in the system
        SOAPrcut (float): the SOAP cut offf
        SOAPnmax (int, optional):
            The number of radial basis functions (option passed to the desired
            SOAP engine). Defaults to 8.
        SOAPlmax (int, optional):
            The maximum degree of spherical harmonics (option passed to the
            desired SOAP engine). Defaults to 8.
        SOAPatomMask (list[str], optional):
            the symbols of the atoms whose SOAP fingerprint will be calculated.
            Defaults to None.
        centersMask (Iterable, optional):
            the indexes of the atoms whose SOAP fingerprint will be calculated.
            Defaults to None.
        SOAP_respectPBC (bool, optional):
        Determines whether the system is considered to be periodic (option passed
            to the desired SOAP engine). Defaults to True.
        SOAPkwargs (dict, optional):
            additional keyword arguments to be passed to the SOAP engine.
            Defaults to {}.
        useSoapFrom (KNOWNSOAPENGINES, optional): This string determines the
            selected SOAP engine for the calculations. Defaults to "dscribe".

    Returns:
        SOAPengineContainer: the soap engine set up for the calcualations
    """
    if SOAPnmax <= 0:
        raise ValueError("SOAPnmax must be a positive non zero integer")
    if SOAPlmax < 0:
        raise ValueError("SOAPlmax must be a positive integer, or zero")

    # safely dict as with a default value:
    if SOAPkwargs is None:
        SOAPkwargs = {}
    species = orderByZ(list(set(atomNames)))

    useCentersMask = _getAtomMask(
        atomNames=atomNames, SOAPatomMask=SOAPatomMask, centersMask=centersMask
    )

    if useSoapFrom == "dscribe":
        if not HAVE_DSCRIBE:  # pragma: no cover
            raise ImportError("dscribe is not installed in your current environment")
        SOAPkwargs.update(
            {
                "species": species,
                "periodic": SOAP_respectPBC,
                "rcut": SOAPrcut,
                "nmax": SOAPnmax,
                "lmax": SOAPlmax,
            }
        )
        if "sparse" in SOAPkwargs.keys():
            SOAPkwargs["sparse"] = False
            warnings.warn("sparse output is not supported yet, forcing  dense output")
        return dscribeSOAPengineContainer(dscribeSOAP(**SOAPkwargs), useCentersMask)
    if useSoapFrom == "quippy":
        if not HAVE_QUIPPY:  # pragma: no cover
            raise ImportError("quippy-ase is not installed in your current environment")

        if useCentersMask is not None and SOAPatomMask is None:
            raise NotImplementedError(
                "WARNING: the quippy interface works only with SOAPatomMask"
            )
        SOAPkwargs.update(
            {
                "cutoff": SOAPrcut,
                "n_max": SOAPnmax,
                "l_max": SOAPlmax,
            }
        )
        if "atom_sigma" not in SOAPkwargs:
            SOAPkwargs["atom_sigma"] = 0.5
        # By default we impose quippy not to normalize the descriptor
        SOAPkwargs["normalise"] = "F"

        speciesZ, listOfTheZs = _makeSP(species)
        calculatedZs, listOftheCalcZs = speciesZ, listOfTheZs

        # TODO: Z and theZs personalized
        if SOAPatomMask is not None:
            calculatedZs, listOftheCalcZs = _makeSP(SOAPatomMask)

        settings = "soap"
        for key, value in SOAPkwargs.items():
            settings += f" {key}={value}"
        settings += f" n_species={len(speciesZ)} species_Z={{{listOfTheZs}}}"
        settings += f" n_Z={len(calculatedZs)} Z={{{listOftheCalcZs}}}"
        return quippySOAPengineContainer(Descriptor(settings), useCentersMask)

    raise NotImplementedError(f"{useSoapFrom} is not implemented yet")


# //from quippy.module_descriptors <-
# ============================= ===== =============== ==============================================
# Name                          Type  Value           Doc
# ============================= ===== =============== ==============================================
# cutoff                        None  //MANDATORY//   Cutoff for soap-type descriptors
# cutoff_transition_width       float 0.50            Cutoff transition width for soap-type
#                                                     descriptors
# cutoff_dexp                   int   0               Cutoff decay exponent
# cutoff_scale                  float 1.0             Cutoff decay scale
# cutoff_rate                   float 1.0             Inverse cutoff decay rate
# l_max                         None  PARAM_MANDATORY L_max(spherical harmonics basis band limit)
#                                                     for soap-type descriptors
# n_max                         None  PARAM_MANDATORY N_max(number of radial basis functions) for
#                                                     soap-type descriptors
# atom_gaussian_width           None  PARAM_MANDATORY Width of atomic Gaussians for soap-type
#                                                     descriptors
# central_weight                float 1.0             Weight of central atom in environment
# central_reference_all_species bool  F               Place a Gaussian reference for all atom
#                                                     species densities.
# average                       bool  F               Whether to calculate averaged SOAP -
#                                                     one descriptor per atoms object.
#                                                     If false(default) atomic SOAP is returned.
# diagonal_radial               bool  F
# covariance_sigma0             float 0.0             sigma_0 parameter in polynomial covariance
#                                                     function
# normalise                     bool  T               Normalise descriptor so magnitude is 1.
#                                                     In this case the kernel of two equivalent
#                                                     environments is 1.
# basis_error_exponent          float 10.0            10^(-basis_error_exponent) is the max
#                                                     difference between the target and the
#                                                     expanded function
# n_Z                           int   1               How many different types of central atoms to
#                                                     consider
# n_species                     int   1               Number of species for the descriptor
# species_Z                     None                  Atomic number of species
# xml_version                   int   1426512068      Version of GAP the XML potential file was
#                                                     created
# species_Z                     None  //MANDATORY//   Atomic number of species
# Z                             None  //MANDATORY//   Atomic numbers to be considered for central
#                                                     atom, must be a list
# ============================= ===== =============== ==============================================
