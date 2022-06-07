import warnings
import h5py
import abc

import time
import numpy
from typing import Iterable
from dscribe.descriptors import SOAP
from quippy.descriptors import Descriptor
from ase.data import atomic_numbers,chemical_symbols
import ase
from HDF5er import (
    HDF52AseAtomsChunckedwithSymbols as HDF2ase,
    isTrajectoryGroup,
)

__all__ = ["saponify", "saponifyGroup"]


class SOAPengineContainer(abc.ABC):
    """A container for the SOAP engine

    Attributes:
        SOAPengine (SOAP): the soap engine already set up

    """

    def __init__(self, SOAPengine):
        self.SOAPengine = SOAPengine

    @property
    def engine(self):
        return self.SOAPengine

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

    def __init__(self, SOAPengine):
        super().__init__(SOAPengine)

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

    def __init__(self, SOAPengine):
        super().__init__(SOAPengine)

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
        Z=self.SOAPengine._quip_descriptor.descriptor_soap.species_z
        return [chemical_symbols[i] for i in Z]
        return 

    @property
    def crossover(self) -> bool:
        return True

    def get_location(self, specie1, specie2):
        return self.SOAPengine.get_location((specie1, specie2))

    def calculate(self, atoms: ase.Atoms):
        d = self.SOAPengine.calc(atoms)
        return d["data"][:-1]

    def __call__(self, atoms, **kwargs):
        if isinstance(atoms, ase.Atoms):
            atoms = [atoms]

        toret = []
        for frame in atoms:
            toret.append(self.calculate(frame))
        return numpy.array(toret)


def getSoapEngine(
    species: "list[str]",
    SOAPrcut: float,
    SOAPnmax: int,
    SOAPlmax: int,
    SOAP_respectPBC: bool = True,
    SOAPkwargs: dict = {},
    useSoapFrom: "str" = "dscribe",
) -> SOAPengineContainer:
    """Returns a soap engine already set up

    Returns:
        SOAP: the soap engine already set up
    """
    if useSoapFrom == "dscribe":
        SOAPkwargs.update(
            dict(
                species=species,
                periodic=SOAP_respectPBC,
                rcut=SOAPrcut,
                nmax=SOAPnmax,
                lmax=SOAPlmax,
            )
        )
        if "sparse" in SOAPkwargs.keys():
            if SOAPkwargs["sparse"]:
                SOAPkwargs["sparse"] = False
                warnings.warn("sparse output is not supported yet, switching to dense")
        return dscribeSOAPengineContainer(SOAP(**SOAPkwargs))
    if useSoapFrom == "quippy":
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
        diagonal_radial               bool  F               Only return the n1=n2 elements of the power
                                                            spectrum.
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
        SOAPkwargs.update(
            dict(
                # species=species,
                # periodic=SOAP_respectPBC,
                cutoff=SOAPrcut,
                n_max=SOAPnmax,
                l_max=SOAPlmax,
            )
        )
        if "atom_sigma" not in SOAPkwargs:
            SOAPkwargs["atom_sigma"] = 0.5
        species_z = [atomic_numbers[specie] for specie in species]
        thesp = str(species_z[0])
        for sp in species_z[1:]:
            thesp += ", " + str(sp)

        settings = f"soap"
        for key, value in SOAPkwargs.items():
            settings += f" {key}={value}"
        settings += f" n_species={len(species_z)} species_Z={{{thesp}}}"
        settings += f" n_Z={len(species_z)} Z={{{thesp}}}"
        return quippySOAPengineContainer(Descriptor(settings))
    else:
        raise NotImplementedError(f"{useSoapFrom} is not implemented yet")


def saponifyWorker(
    trajGroup: h5py.Group,
    SOAPoutDataset: h5py.Dataset,
    soapEngine: SOAPengineContainer,
    centersMask: "list|None" = None,
    SOAPOutputChunkDim: int = 100,
    SOAPnJobs: int = 1,
):
    """Calculates the soap descriptor and store the result in the given dataset

    Args:
        trajGroup (h5py.Group): the grooput that contains the trajectory (must
        contain Box,Trajectory and Types subgroups)
        SOAPoutDataset (h5py.Dataset): The preformed dataset for storing the
        SOAP results
        soapEngine (SOAP): The soap engine already set up
        centersMask (list): the mask for the SOAP centers, already set up
        SOAPOutputChunkDim (int, optional): The dimension of the chunck of data
        in the SOAP results dataset. Defaults to 100.
        SOAPnJobs (int, optional): the number of concurrent SOAP calculations
        (option passed to dscribe's SOAP). Defaults to 1.
    """
    symbols = trajGroup["Types"].asstr()[:]
    SOAPoutDataset.attrs["l_max"] = soapEngine.lmax
    SOAPoutDataset.attrs["n_max"] = soapEngine.nmax
    SOAPoutDataset.attrs["r_cut"] = soapEngine.rcut
    SOAPoutDataset.attrs["species"] = soapEngine.species
    if centersMask is None:
        if "centersIndexes" in SOAPoutDataset.attrs:
            del centersMask.attrs["centersIndexes"]
    else:
        SOAPoutDataset.attrs.create("centersIndexes", centersMask)
        # print(centersMask)
        # print(SOAPoutDataset.attrs["centersIndexes"])
        # print(type(SOAPoutDataset.attrs["centersIndexes"]))

    nspecies = len(soapEngine.species)
    for i in range(nspecies):
        for j in range(nspecies):
            if soapEngine.crossover or (i == j):
                temp = soapEngine.get_location(
                    soapEngine.species[i], soapEngine.species[j]
                )
                SOAPoutDataset.attrs[
                    f"species_location_{soapEngine.species[i]}-{soapEngine.species[j]}"
                ] = (temp.start, temp.stop)

    for chunkTraj in trajGroup["Trajectory"].iter_chunks():
        chunkBox = (chunkTraj[0], slice(0, 6, 1))
        print(f'working on trajectory chunk "{chunkTraj}"')
        print(f'   and working on box chunk "{repr(chunkBox)}"')
        # load in memory a chunk of data
        atoms = HDF2ase(trajGroup, chunkTraj, chunkBox, symbols)
        jobchunk = min(SOAPOutputChunkDim, len(atoms))
        jobStart = 0
        jobEnd = jobStart + jobchunk
        while jobStart < len(atoms):
            t1 = time.time()
            frameStart = jobStart + chunkTraj[0].start
            FrameEnd = jobEnd + chunkTraj[0].start
            print(f"working on frames: [{frameStart}:{FrameEnd}]")
            # TODO: dscribe1.2.1 return (nat,nsoap) instead of (1,nat,nsoap) if we are analysing only ! frame!
            SOAPoutDataset[frameStart:FrameEnd] = soapEngine(
                atoms[jobStart:jobEnd],
                positions=[centersMask] * jobchunk,
                n_jobs=SOAPnJobs,
            )
            t2 = time.time()
            jobchunk = min(SOAPOutputChunkDim, len(atoms) - jobEnd)
            jobStart = jobEnd
            jobEnd = jobStart + jobchunk
            print(f"delta create= {t2-t1}")


def applySOAP(
    trajContainer: h5py.Group,
    SOAPoutContainer: h5py.Group,
    key: str,
    soapEngine: SOAPengineContainer,
    centersMask: "list|None" = None,
    SOAPOutputChunkDim: int = 100,
    SOAPnJobs: int = 1,
):
    NofFeatures = soapEngine.features
    symbols = trajContainer["Types"].asstr()[:]
    nCenters = len(symbols) if centersMask is None else len(centersMask)

    if key not in SOAPoutContainer.keys():
        SOAPoutContainer.create_dataset(
            key,
            (0, nCenters, NofFeatures),
            compression="gzip",
            compression_opts=9,
            chunks=(SOAPOutputChunkDim, nCenters, NofFeatures),
            maxshape=(None, nCenters, NofFeatures),
        )
    SOAPout = SOAPoutContainer[key]
    SOAPout.resize((len(trajContainer["Trajectory"]), nCenters, NofFeatures))
    saponifyWorker(
        trajContainer,
        SOAPout,
        soapEngine,
        centersMask,
        SOAPOutputChunkDim,
        SOAPnJobs,
    )


def saponifyGroup(
    trajContainers: "h5py.Group|h5py.File",
    SOAPoutContainers: "h5py.Group|h5py.File",
    SOAPrcut: float,
    SOAPnmax: int,
    SOAPlmax: int,
    SOAPOutputChunkDim: int = 100,
    SOAPnJobs: int = 1,
    SOAPatomMask: str = None,
    centersMask: Iterable = None,  # TODO: document this
    SOAP_respectPBC: bool = True,
    SOAPkwargs: dict = {},
    useSoapFrom: str = "dscribe",
):
    """From a trajectory stored in a group calculates and stores the SOAP
    descriptor in the given group/file

    Args:
        trajContainers (h5py.Group): The file/group that contains the trajectories
        SOAPoutContainers (h5py.Group): The file/group that will store the SOAP results
        SOAPOutputChunkDim (int, optional): The dimension of the chunck of data
        in the SOAP results dataset. Defaults to 100.
        SOAPnJobs (int, optional): the number of concurrent SOAP calculations
        (option passed to dscribe's SOAP). Defaults to 1.
        SOAPatomMask (str, optional): the symbols of the atoms whose SOAP
        fingerprint will be calculated (option passed to dscribe's SOAP). Defaults to None.
        SOAPrcut (float, optional): The cutoff for local region in angstroms.
        Should be bigger than 1 angstrom (option passed to dscribe's SOAP). Defaults to 8.0.
        SOAPnmax (int, optional): The number of radial basis functions (option
        passed to dscribe's SOAP). Defaults to 8.
        SOAPlmax (int, optional): The maximum degree of spherical harmonics
        (option passed to dscribe's SOAP). Defaults to 8.
        SOAP_respectPBC (bool, optional): Determines whether the system is
        considered to be periodic (option passed to dscribe's SOAP). Defaults to True.
        SOAPkwargs (dict, optional): additional keyword arguments to be passed to the SOAP engine. Defaults to {}.
    """
    soapEngine = None
    for key in trajContainers.keys():
        if isTrajectoryGroup(trajContainers[key]):
            traj = trajContainers[key]
            symbols = traj["Types"].asstr()[:]
            # TODO: unify the soap initialization with saponify
            if SOAPatomMask is not None and centersMask is not None:
                raise Exception(
                    f"saponifyGroup: You can't use both SOAPatomMask and centersMask"
                )
            if SOAPatomMask is not None:
                centersMask = [
                    i for i in range(len(symbols)) if symbols[i] in SOAPatomMask
                ]
            if soapEngine is None:
                soapEngine = getSoapEngine(
                    species=list(set(symbols)),
                    SOAPrcut=SOAPrcut,
                    SOAPnmax=SOAPnmax,
                    SOAPlmax=SOAPlmax,
                    SOAP_respectPBC=SOAP_respectPBC,
                    SOAPkwargs=SOAPkwargs,
                    useSoapFrom=useSoapFrom,
                )
            applySOAP(
                traj,
                SOAPoutContainers,
                key,
                soapEngine,
                centersMask,
                SOAPOutputChunkDim,
                SOAPnJobs,
            )


def saponify(
    trajContainer: "h5py.Group|h5py.File",
    SOAPoutContainer: "h5py.Group|h5py.File",
    SOAPrcut: float,
    SOAPnmax: int,
    SOAPlmax: int,
    SOAPOutputChunkDim: int = 100,
    SOAPnJobs: int = 1,
    SOAPatomMask: str = None,
    centersMask: Iterable = None,  # TODO: document this
    SOAP_respectPBC: bool = True,
    SOAPkwargs: dict = {},
    useSoapFrom: str = "dscribe",
):
    """Calculates the SOAP fingerprints for each atom in a given hdf5 trajectory

    This routine sets up a SOAP engine to calculate the SOAP fingerprints for all
    the atoms in a given trajectory. The user can choose the otpio

    Args:
        trajFname (str): The name of the hdf5 file in wich the trajectory is stored
        trajectoryGroupPath (str): the path of the group that contains the trajectory in trajFname
        outputFname (str): the name of the hdf5 file that will contain the ouput or the SOAP analysis
        exportDatasetName (str): the name of the dataset that will contain the SOAP
        results, it will be saved in the group called "SOAP"
        SOAPOutputChunkDim (int, optional): The dimension of the chunck of data
        in the SOAP results dataset. Defaults to 100.
        SOAPnJobs (int, optional): the number of concurrent SOAP calculations
        (option passed to dscribe's SOAP). Defaults to 1.
        SOAPatomMask (str, optional): the symbols of the atoms whose SOAP
        fingerprint will be calculated (option passed to dscribe's SOAP). Defaults to None.
        SOAPrcut (float, optional): The cutoff for local region in angstroms.
        Should be bigger than 1 angstrom (option passed to dscribe's SOAP). Defaults to 8.0.
        SOAPnmax (int, optional): The number of radial basis functions (option
        passed to dscribe's SOAP). Defaults to 8.
        SOAPlmax (int, optional): The maximum degree of spherical harmonics
        (option passed to dscribe's SOAP). Defaults to 8.
        SOAP_respectPBC (bool, optional): Determines whether the system is
        considered to be periodic (option passed to dscribe's SOAP). Defaults to True.
        SOAPkwargs (dict, optional): additional keyword arguments to be passed to the SOAP engine. Defaults to {}.
    """
    if isTrajectoryGroup(trajContainer):
        symbols = trajContainer["Types"].asstr()[:]
        if SOAPatomMask is not None and centersMask is not None:
            raise Exception(
                f"saponify: You can't use both SOAPatomMask and centersMask"
            )
        if SOAPatomMask is not None:
            centersMask = [i for i in range(len(symbols)) if symbols[i] in SOAPatomMask]
        soapEngine = getSoapEngine(
            species=list(set(symbols)),
            SOAPrcut=SOAPrcut,
            SOAPnmax=SOAPnmax,
            SOAPlmax=SOAPlmax,
            SOAP_respectPBC=SOAP_respectPBC,
            SOAPkwargs=SOAPkwargs,
            useSoapFrom=useSoapFrom,
        )
        exportDatasetName = trajContainer.name.split("/")[-1]
        applySOAP(
            trajContainer,
            SOAPoutContainer,
            exportDatasetName,
            soapEngine,
            centersMask,
            SOAPOutputChunkDim,
            SOAPnJobs,
        )
    else:
        raise Exception(f"saponify: The input object is not a trajectory group.")


if __name__ == "__main__":
    # this is an example script for Applying the SOAP analysis on a trajectory saved on an
    # HDF5 file formatted with our HDF5er and save the result in another HDF5 file
    with h5py.File("Water.hdf5", "r") as trajLoader, h5py.File(
        "WaterSOAP.hdf5", "a"
    ) as soapOffloader:
        saponify(
            trajLoader[f"Trajectories/1ns"],
            soapOffloader.require_group("SOAP"),
            SOAPatomMask="O",
            SOAPOutputChunkDim=100,
            SOAPnJobs=12,
        )
