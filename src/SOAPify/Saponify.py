import h5py
import time
from typing import Iterable


from HDF5er import (
    HDF52AseAtomsChunckedwithSymbols as HDF2ase,
    isTrajectoryGroup,
)
from .SOAPengine import SOAPengineContainer, getSoapEngine, KNOWNSOAPENGINES


def _saponifyWorker(
    trajGroup: h5py.Group,
    SOAPoutDataset: h5py.Dataset,
    soapEngine: SOAPengineContainer,
    SOAPOutputChunkDim: int = 100,
    SOAPnJobs: int = 1,
):
    """Calculates the soap descriptor and store the result in the given dataset

    Args:
        trajGroup (h5py.Group): the grooput that contains the trajectory (must contain "Box", "Trajectory" and "Types" datasets)
        SOAPoutDataset (h5py.Dataset): The preformed dataset for storing the SOAP results
        soapEngine (SOAPengineContainer): The soap engine already set up
        SOAPOutputChunkDim (int, optional): The dimension of the chunck of data in the SOAP results dataset. Defaults to 100.
        SOAPnJobs (int, optional): the number of concurrent SOAP calculations (option passed to the desired SOAP engine). Defaults to 1.
    """
    symbols = trajGroup["Types"].asstr()[:]
    SOAPoutDataset.attrs["SOAPengine"] = soapEngine.SOAPenginekind
    SOAPoutDataset.attrs["l_max"] = soapEngine.lmax
    SOAPoutDataset.attrs["n_max"] = soapEngine.nmax
    SOAPoutDataset.attrs["r_cut"] = soapEngine.rcut
    SOAPoutDataset.attrs["species"] = soapEngine.species
    if soapEngine.centersMask is None:
        if "centersIndexes" in SOAPoutDataset.attrs:
            del SOAPoutDataset.attrs["centersIndexes"]
    else:
        SOAPoutDataset.attrs.create("centersIndexes", soapEngine.centersMask)
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
                positions=[soapEngine.centersMask] * jobchunk,
                n_jobs=SOAPnJobs,
            )
            t2 = time.time()
            jobchunk = min(SOAPOutputChunkDim, len(atoms) - jobEnd)
            jobStart = jobEnd
            jobEnd = jobStart + jobchunk
            print(f"delta create= {t2-t1}")


def _applySOAP(
    trajContainer: h5py.Group,
    SOAPoutContainer: h5py.Group,
    key: str,
    soapEngine: SOAPengineContainer,
    SOAPOutputChunkDim: int = 100,
    SOAPnJobs: int = 1,
    doOverride: bool = False,
):
    """helper function: applies the soap engine to the given trajectory within the trajContainer

    Args:
        trajContainer (h5py.Group): The group or the file that contains the trajectory, must have the following dataset in '/': "Box", "Trajectory" and "Types"
        SOAPoutContainer (h5py.Group): The group where the dataset with the  SOAP fingerprints will be saved
        key (str): the name of the dataset to be saved, if exist will be overidden
        soapEngine (SOAPengineContainer): the contained of the soap engine
        SOAPOutputChunkDim (int, optional): The chunk of trajectory that will be loaded in memory to be calculated, if key is a new dataset will also be the size of the main chunck of data of the SOAP dataset . Defaults to 100.
        SOAPnJobs (int, optional): Number of concurrent SOAP calculations. Defaults to 1.
        doOverride (bool, optional): if False will raise and exception if the user ask to override an already existing DataSet. Defaults to False.
    """
    NofFeatures = soapEngine.features
    symbols = trajContainer["Types"].asstr()[:]
    nCenters = (
        len(symbols) if soapEngine.centersMask is None else len(soapEngine.centersMask)
    )

    if key in SOAPoutContainer.keys():
        if doOverride is False:
            raise ValueError(
                f"Are you sure that you want to override {SOAPoutContainer[key].name}?"
            )
        else:  # doOverride is True and key in SOAPoutContainer.keys():
            # check if deleting the dataset is necessary:
            oldshape = SOAPoutContainer[key].shape
            if oldshape[1] != nCenters or oldshape[2] != NofFeatures:
                del SOAPoutContainer[key]
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
    _saponifyWorker(
        trajContainer,
        SOAPout,
        soapEngine,
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
    SOAPatomMask: "list[str]" = None,
    centersMask: Iterable = None,  # TODO: document this
    SOAP_respectPBC: bool = True,
    SOAPkwargs: dict = dict(),
    useSoapFrom: KNOWNSOAPENGINES = "dscribe",
    doOverride: bool = False,
):
    """From a trajectory stored in a group calculates and stores the SOAP
    descriptor in the given group/file

    `SOAPatomMask` and `centersMask` are mutually exclusive (see :func:`SOAPify.SOAPengine.getSoapEngine`)

    Args:
        trajContainers (h5py.Group|h5py.File): The file/group that contains the trajectories
        SOAPoutContainers (h5py.Group|h5py.File): The file/group that will store the SOAP results
        SOAPrcut (float): The cutoff for local region in angstroms. Should be bigger than 1 angstrom (option passed to the desired SOAP engine). Defaults to 8.0.
        SOAPnmax (int): The number of radial basis functions (option passed to the desired SOAP engine). Defaults to 8.
        SOAPlmax (int): The maximum degree of spherical harmonics (option passed to the desired SOAP engine). Defaults to 8.
        SOAPOutputChunkDim (int, optional): The dimension of the chunck of data in the SOAP results dataset. Defaults to 100.
        SOAPnJobs (int, optional): the number of concurrent SOAP calculations (option passed to the desired SOAP engine). Defaults to 1.
        SOAPatomMask (list[str], optional): the symbols of the atoms whose SOAP fingerprint will be calculated (option passed to getSoapEngine). Defaults to None.
        centersMask (Iterable, optional): the indexes of the atoms whose SOAP fingerprint will be calculated (option passed getSoapEngine). Defaults to None.
        SOAP_respectPBC (bool, optional): Determines whether the system is considered to be periodic (option passed to the desired SOAP engine). Defaults to True.
        SOAPkwargs (dict, optional): additional keyword arguments to be passed to the selected SOAP engine. Defaults to {}.
        useSoapFrom (KNOWNSOAPENGINES, optional): This string determines the selected SOAP engine for the calculations. Defaults to "dscribe".
        doOverride (bool, optional): if False will raise and exception if the user ask to override an already existing DataSet. Defaults to False.
    """
    soapEngine = None
    print(f"using {useSoapFrom} to calculate SOAP")
    print(SOAPkwargs)
    for key in trajContainers.keys():
        if isTrajectoryGroup(trajContainers[key]):
            traj = trajContainers[key]
            symbols = traj["Types"].asstr()[:]
            if soapEngine is None:
                soapEngine = getSoapEngine(
                    atomNames=symbols,
                    SOAPrcut=SOAPrcut,
                    SOAPnmax=SOAPnmax,
                    SOAPlmax=SOAPlmax,
                    SOAPatomMask=SOAPatomMask,
                    centersMask=centersMask,
                    SOAP_respectPBC=SOAP_respectPBC,
                    SOAPkwargs=SOAPkwargs,
                    useSoapFrom=useSoapFrom,
                )
            _applySOAP(
                traj,
                SOAPoutContainers,
                key,
                soapEngine,
                SOAPOutputChunkDim,
                SOAPnJobs,
                doOverride=doOverride,
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
    doOverride: bool = False,
):
    """Calculates the SOAP fingerprints for each atom in a given hdf5 trajectory

    This routine sets up a SOAP engine to calculate the SOAP fingerprints for all
    the atoms in a given trajectory. The user can choose the otpio

    Args:
        trajFname (str): The name of the hdf5 file in wich the trajectory is stored
        trajectoryGroupPath (str): the path of the group that contains the trajectory in trajFname
        outputFname (str): the name of the hdf5 file that will contain the ouput or the SOAP analysis
        exportDatasetName (str): the name of the dataset that will contain the SOAP results, it will be saved in the group called "SOAP"
        SOAPOutputChunkDim (int, optional): The dimension of the chunck of data in the SOAP results dataset. Defaults to 100.
        SOAPnJobs (int, optional): the number of concurrent SOAP calculations (option passed to the desired SOAP engine). Defaults to 1.
        SOAPatomMask (str, optional): the symbols of the atoms whose SOAP fingerprint will be calculated (option passed to the desired SOAP engine). Defaults to None.
        SOAPrcut (float, optional): The cutoff for local region in angstroms. Should be bigger than 1 angstrom (option passed to the desired SOAP engine). Defaults to 8.0.
        SOAPnmax (int, optional): The number of radial basis functions (option passed to the desired SOAP engine). Defaults to 8.
        SOAPlmax (int, optional): The maximum degree of spherical harmonics (option passed to the desired SOAP engine). Defaults to 8.
        SOAP_respectPBC (bool, optional): Determines whether the system is considered to be periodic (option passed to the desired SOAP engine). Defaults to True.
        SOAPkwargs (dict, optional): additional keyword arguments to be passed to the SOAP engine. Defaults to {}.
        useSoapFrom (KNOWNSOAPENGINES, optional): This string determines the selected SOAP engine for the calculations. Defaults to "dscribe".
        doOverride (bool, optional): if False will raise and exception if the user ask to override an already existing DataSet. Defaults to False.
    """
    if isTrajectoryGroup(trajContainer):
        symbols = trajContainer["Types"].asstr()[:]
        soapEngine = getSoapEngine(
            atomNames=symbols,
            SOAPrcut=SOAPrcut,
            SOAPnmax=SOAPnmax,
            SOAPlmax=SOAPlmax,
            SOAPatomMask=SOAPatomMask,
            centersMask=centersMask,
            SOAP_respectPBC=SOAP_respectPBC,
            SOAPkwargs=SOAPkwargs,
            useSoapFrom=useSoapFrom,
        )
        exportDatasetName = trajContainer.name.split("/")[-1]
        _applySOAP(
            trajContainer,
            SOAPoutContainer,
            exportDatasetName,
            soapEngine,
            SOAPOutputChunkDim,
            SOAPnJobs,
            doOverride=doOverride,
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
