import abc
import warnings
import numpy
from ase.data import atomic_numbers, chemical_symbols
import ase
from dscribe.descriptors import SOAP as dscribeSOAP
from quippy.descriptors import Descriptor
from .utils import orderByZ, _getRSindex, getAddressesQuippyLikeDscribe
from typing import Iterable


def centerMaskCreator(
    SOAPatomMask: "list[str]",
    symbols: "list[str]",
):
    return [i for i in range(len(symbols)) if symbols[i] in SOAPatomMask]


class SOAPengineContainer(abc.ABC):
    """A container for the SOAP engine

    Attributes:
        SOAPengine (SOAP): the soap engine already set up

    """

    def __init__(self, SOAPengine, centerMask):
        self.SOAPengine = SOAPengine
        self.centersMask_ = centerMask

    @property
    def engine(self):
        return self.SOAPengine

    @property
    def centersMask(self):
        return self.centersMask_

    @property
    @abc.abstractmethod
    def features(self):
        pass

    @property
    @abc.abstractmethod
    def nmax(self):
        pass

    @property
    @abc.abstractmethod
    def lmax(self):
        pass

    @property
    @abc.abstractmethod
    def rcut(self):
        pass

    @property
    @abc.abstractmethod
    def species(self) -> list:
        pass

    @property
    @abc.abstractmethod
    def crossover(self) -> bool:
        pass

    @abc.abstractmethod
    def get_location(self, specie1, specie2):
        pass

    @abc.abstractmethod
    def __call__(self, atoms, **kwargs):
        pass


class dscribeSOAPengineContainer(SOAPengineContainer):
    """A container for the SOAP engine from dscribe

    Attributes:
        SOAPengine (SOAP): the soap engine already set up

    """

    def __init__(self, SOAPengine, centerMask):
        super().__init__(SOAPengine, centerMask)

    @property
    def features(self):
        return self.SOAPengine.get_number_of_features()

    @property
    def nmax(self):
        return self.SOAPengine._nmax

    @property
    def lmax(self):
        return self.SOAPengine._lmax

    @property
    def rcut(self):
        return self.SOAPengine._rcut

    @property
    def species(self):
        return self.SOAPengine.species

    @property
    def crossover(self) -> bool:
        return self.SOAPengine.crossover

    def get_location(self, specie1, specie2):
        return self.SOAPengine.get_location((specie1, specie2))

    def __call__(self, atoms, **kwargs):
        return self.SOAPengine.create(atoms, **kwargs)


class quippySOAPengineContainer(SOAPengineContainer):
    """A container for the SOAP engine from quippy

    Attributes:
        SOAPengine (SOAP): the soap engine already set up

    """

    def __init__(self, SOAPengine, centerMask):
        super().__init__(SOAPengine, centerMask)
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
        a, b = orderByZ([specie1, specie2])
        return self._slices[a + b]

    def __call__(self, atoms, **kwargs):
        if isinstance(atoms, ase.Atoms):
            atoms = [atoms]
        nat = len(self.centersMask_) if self.centersMask_ is not None else len(atoms[0])
        toret = numpy.empty((len(atoms), nat, self.features))
        for i, frame in enumerate(atoms):
            toret[i] = self.SOAPengine.calc(frame)["data"][:, self._addresses]
        return numpy.array(toret)


def getSoapEngine(
    atomNames: "list[str]",
    SOAPrcut: float,
    SOAPnmax: int,
    SOAPlmax: int,
    SOAPatomMask: str = None,
    centersMask: Iterable = None,
    SOAP_respectPBC: bool = True,
    SOAPkwargs: dict = {},
    useSoapFrom: "str" = "dscribe",
) -> SOAPengineContainer:
    """Returns a soap engine already set up

    Returns:
        SOAP: the soap engine already set up
    """
    #DAMNED pass by reference of python:
    #I need to make this a copy to avoid modifying the passed dictionary in the calling function
    mySOAPkwargs = SOAPkwargs.copy()
    species = list(set(atomNames))
    if SOAPatomMask is not None and centersMask is not None:
        raise Exception(f"saponify: You can't use both SOAPatomMask and centersMask")
    if SOAPatomMask is not None:
        centersMask = centerMaskCreator(SOAPatomMask, atomNames)
    species = orderByZ(species)
    if useSoapFrom == "dscribe":
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
        return dscribeSOAPengineContainer(dscribeSOAP(**mySOAPkwargs), centersMask)
    elif useSoapFrom == "quippy":
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
        if SOAPatomMask is None and centersMask is not None:
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
        return quippySOAPengineContainer(Descriptor(settings), centersMask)
    else:
        raise NotImplementedError(f"{useSoapFrom} is not implemented yet")
