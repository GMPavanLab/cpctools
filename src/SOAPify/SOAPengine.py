import abc
import warnings
import numpy
from ase.data import atomic_numbers, chemical_symbols
import ase
from .utils import orderByZ, _getRSindex, getAddressesQuippyLikeDscribe
from typing import Iterable, Literal

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
    """given the list of types of atoms to select and the list od types of atoms in the topology returns the mask of the selected species

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
        return self.SOAPengine

    @property
    def SOAPenginekind(self):
        return self.SOAPenginekind_

    @property
    def centersMask(self):
        return self.centersMask_

    @property
    @abc.abstractmethod
    def features(self) -> int:  # pragma: no cover
        pass

    @property
    @abc.abstractmethod
    def nmax(self) -> int:  # pragma: no cover
        pass

    @property
    @abc.abstractmethod
    def lmax(self) -> int:  # pragma: no cover
        pass

    @property
    @abc.abstractmethod
    def rcut(self) -> int:  # pragma: no cover
        pass

    @property
    @abc.abstractmethod
    def species(self) -> list:  # pragma: no cover
        pass

    @property
    @abc.abstractmethod
    def crossover(self) -> bool:  # pragma: no cover
        pass

    @abc.abstractmethod
    def get_location(self, specie1, specie2):  # pragma: no cover
        """returns the slice where the two asked species are stored in the ouput array"""
        pass

    @abc.abstractmethod
    def __call__(self, atoms, **kwargs):  # pragma: no cover
        pass


class dscribeSOAPengineContainer(SOAPengineContainer):
    """A container for the SOAP engine from dscribe

    Attributes:
        SOAPengine (SOAP): the soap engine already set up

    """

    def __init__(self, SOAPengine, centerMask, SOAPengineKind):
        super().__init__(SOAPengine, centerMask, SOAPengineKind)

    @property
    def features(self):
        return self.SOAPengine.get_number_of_features()

    @property
    def nmax(self):
        if hasattr(self.SOAPengine, "_nmax"):
            return self.SOAPengine._nmax
        if hasattr(self.SOAPengine, "_n_max"):
            return self.SOAPengine._n_max

    @property
    def lmax(self):
        if hasattr(self.SOAPengine, "_lmax"):
            return self.SOAPengine._lmax
        if hasattr(self.SOAPengine, "_l_max"):
            return self.SOAPengine._l_max

    @property
    def rcut(self):
        if hasattr(self.SOAPengine, "_rcut"):
            return self.SOAPengine._rcut
        if hasattr(self.SOAPengine, "_r_cut"):
            return self.SOAPengine._r_cut

    @property
    def species(self):
        return self.SOAPengine.species

    @property
    def crossover(self) -> bool:
        return self.SOAPengine.crossover

    def get_location(self, specie1, specie2):
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

    def __init__(self, SOAPengine, centerMask, SOAPengineKind):
        super().__init__(SOAPengine, centerMask, SOAPengineKind)
        species = self.species
        nmax = self.nmax
        lmax = self.lmax

        slices = {}
        next = 0
        prev = 0
        fullmat = nmax * nmax * (lmax + 1)
        upperDiag = ((nmax + 1) * nmax) // 2 * (lmax + 1)
        for i in range(len(species) * nmax):
            for j in range(i, len(species)):
                key = species[i] + species[j]
                addDim = (lmax + 1) * (
                    nmax * nmax if i != j else ((nmax + 1) * nmax) // 2
                )
                if i == j:
                    next = prev + upperDiag
                else:
                    next = prev + fullmat
                slices[key] = slice(prev, next)
                prev = next

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

    def get_location(self, specie1, specie2):
        """returns the slice where the two asked species are stored in the ouput array"""
        a, b = orderByZ([specie1, specie2])
        return self._slices[a + b]

    def __call__(self, atoms, **kwargs):
        if isinstance(atoms, ase.Atoms):
            atoms = [atoms]
        nat = len(self.centersMask_) if self.centersMask_ is not None else len(atoms[0])
        toret = numpy.empty((len(atoms), nat, self.features))
        for i, frame in enumerate(atoms):
            toret[i] = self.SOAPengine.calc(frame)["data"][:, self._addresses]
        return toret


def getSoapEngine(
    atomNames: "list[str]",
    SOAPrcut: float,
    SOAPnmax: int,
    SOAPlmax: int,
    SOAPatomMask: str = None,
    centersMask: Iterable = None,
    SOAP_respectPBC: bool = True,
    SOAPkwargs: dict = {},
    useSoapFrom: KNOWNSOAPENGINES = "dscribe",
) -> SOAPengineContainer:
    """set up a soap engine with the given settings

    please visit the manual of the relative soap engine for the calculation parameters

    `SOAPatomMask` and `centersMask` are mutually exclusive:
    - `centerMask` is a list of atoms whose SOAP fingerprints will be calculate
    - `SOAPatomMask` is the list of species of atoms that will be used  create a `centerMask`

    Args:
        atomNames (list[str]): The list of species present in the system
        SOAPrcut (float): the SOAP cut offf
        SOAPnmax (int, optional): The number of radial basis functions (option passed to the desired SOAP engine). Defaults to 8.
        SOAPlmax (int, optional): The maximum degree of spherical harmonics (option passed to the desired SOAP engine). Defaults to 8.
        SOAPatomMask (list[str], optional): the symbols of the atoms whose SOAP fingerprint will be calculated (option passed to getSoapEngine). Defaults to None.
        centersMask (Iterable, optional): the indexes of the atoms whose SOAP fingerprint will be calculated (option passed getSoapEngine). Defaults to None.
        SOAP_respectPBC (bool, optional): Determines whether the system is considered to be periodic (option passed to the desired SOAP engine). Defaults to True.
        SOAPkwargs (dict, optional): additional keyword arguments to be passed to the SOAP engine. Defaults to {}.
        useSoapFrom (KNOWNSOAPENGINES, optional): This string determines the selected SOAP engine for the calculations. Defaults to "dscribe".

    Returns:
        SOAPengineContainer: the soap engine set up for the calcualations
    """
    if SOAPnmax <= 0:
        raise ValueError("SOAPnmax must be a positive non zero integer")
    if SOAPlmax < 0:
        raise ValueError("SOAPlmax must be a positive integer, or zero")
    # DAMNED pass by reference of python:
    # I need to make this a copy to avoid modifying the passed dictionary in the calling function
    mySOAPkwargs = SOAPkwargs.copy()
    species = list(set(atomNames))
    if SOAPatomMask is not None and centersMask is not None:
        raise ValueError(f"saponify: You can't use both SOAPatomMask and centersMask")
    centersMask_ = (
        centerMaskCreator(SOAPatomMask, atomNames)
        if SOAPatomMask is not None
        else centersMask.copy()
        if centersMask is not None
        else None
    )

    species = orderByZ(species)
    if useSoapFrom == "dscribe":
        if not HAVE_DSCRIBE:  # pragma: no cover
            raise ImportError("dscribe is not installed in your current environment")
        mySOAPkwargs.update(
            dict(
                species=species,
                periodic=SOAP_respectPBC,
                rcut=SOAPrcut,
                nmax=SOAPnmax,
                lmax=SOAPlmax,
            )
        )
        if "sparse" in mySOAPkwargs.keys():
            if mySOAPkwargs["sparse"]:
                mySOAPkwargs["sparse"] = False
                warnings.warn("sparse output is not supported yet, switching to dense")
        return dscribeSOAPengineContainer(
            dscribeSOAP(**mySOAPkwargs), centersMask_, "dscribe"
        )
    elif useSoapFrom == "quippy":
        if not HAVE_QUIPPY:  # pragma: no cover
            raise ImportError("quippy-ase is not installed in your current environment")
        """//from quippy.module_descriptors <-
        ============================= ===== =============== ===================================================
        Name                          Type  Value           Doc
        ============================= ===== =============== ===================================================
        cutoff                        None  PARAM_MANDATORY Cutoff for soap-type descriptors
        cutoff_transition_width       float 0.50            Cutoff transition width for soap-type descriptors
        cutoff_dexp                   int   0               Cutoff decay exponent
        cutoff_scale                  float 1.0             Cutoff decay scale
        cutoff_rate                   float 1.0             Inverse cutoff decay rate
        l_max                         None  PARAM_MANDATORY L_max(spherical harmonics basis band limit) for
                                                            soap-type descriptors
        n_max                         None  PARAM_MANDATORY N_max(number of radial basis functions) for
                                                            soap-type descriptors
        atom_gaussian_width           None  PARAM_MANDATORY Width of atomic Gaussians for soap-type
                                                            descriptors
        central_weight                float 1.0             Weight of central atom in environment
        central_reference_all_species bool  F               Place a Gaussian reference for all atom species
                                                            densities.
        average                       bool  F               Whether to calculate averaged SOAP - one
                                                            descriptor per atoms object. If false(default)
                                                            atomic SOAP is returned.
        diagonal_radial               bool  F
        covariance_sigma0             float 0.0             sigma_0 parameter in polynomial covariance
                                                            function
        normalise                     bool  T               Normalise descriptor so magnitude is 1. In this
                                                            case the kernel of two equivalent environments is
                                                            1.
        basis_error_exponent          float 10.0            10^(-basis_error_exponent) is the max difference
                                                            between the target and the expanded function
        n_Z                           int   1               How many different types of central atoms to
                                                            consider
        n_species                     int   1               Number of species for the descriptor
        species_Z                     None                  Atomic number of species
        xml_version                   int   1426512068      Version of GAP the XML potential file was created
        species_Z                     None  //MANDATORY//   Atomic number of species
        Z                             None  //MANDATORY//   Atomic numbers to be considered for central atom,
                                                            must be a list
        ============================= ===== =============== ===================================================
        """
        mySOAPkwargs.update(
            dict(
                cutoff=SOAPrcut,
                n_max=SOAPnmax,
                l_max=SOAPlmax,
            )
        )
        if "atom_sigma" not in mySOAPkwargs:
            mySOAPkwargs["atom_sigma"] = 0.5
        # By default we impose quippy not to normalize the descriptor
        mySOAPkwargs["normalise"] = "F"

        def _makeSP(spArray):
            sp_z = [atomic_numbers[specie] for specie in spArray]
            spString = str(sp_z[0])
            for sp in sp_z[1:]:
                spString += ", " + str(sp)
            return sp_z, spString

        species_z, thesps = _makeSP(species)
        Zs, theZs = species_z, thesps

        # TODO: Z and theZs personalized
        if SOAPatomMask is None and centersMask_ is not None:
            raise NotImplementedError(
                "WARNING: the quippy interface works only with SOAPatomMask"
            )
        if SOAPatomMask is not None:
            Zs, theZs = _makeSP(SOAPatomMask)

        settings = f"soap"
        for key, value in mySOAPkwargs.items():
            settings += f" {key}={value}"
        settings += f" n_species={len(species_z)} species_Z={{{thesps}}}"
        settings += f" n_Z={len(Zs)} Z={{{theZs}}}"
        return quippySOAPengineContainer(Descriptor(settings), centersMask_, "quippy")
    else:
        raise NotImplementedError(f"{useSoapFrom} is not implemented yet")
