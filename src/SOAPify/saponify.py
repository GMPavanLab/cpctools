"""Submodule that contains the workhorse routines to apply the SOAP calculations
"""
import time
from typing import Iterable
import h5py
import numpy

from .HDF5er import (
    HDF52AseAtomsChunckedwithSymbols as HDF2ase,
    isTrajectoryGroup,
)
from .engine import SOAPengineContainer, getSoapEngine, KNOWNSOAPENGINES


def _saponifyWorker(
    trajGroup: h5py.Group,
    SOAPoutDataset: h5py.Dataset,
    soapEngine: SOAPengineContainer,
    SOAPOutputChunkDim: int = 100,
    SOAPnJobs: int = 1,
    verbose: bool = True,
):
    """Calculates the soap descriptor and store the result in the given dataset

    Args:
        trajGroup (h5py.Group):
            the group that contains the trajectory (must contain "Box",
            "Trajectory" and "Types" datasets)
        SOAPoutDataset (h5py.Dataset):
            The preformed dataset for storing the SOAP results
        soapEngine (SOAPengineContainer):
            The soap engine already set up
        SOAPOutputChunkDim (int, optional):
            The dimension of the chunck of data in the SOAP results dataset.
            Defaults to 100.
        SOAPnJobs (int, optional):
            the number of concurrent SOAP calculations (option passed to the
            desired SOAP engine). Defaults to 1.
        verbose (bool, optional):
            regulates the verbosity of the step by step operations.
            Defaults to True.
    """
    symbols = trajGroup["Types"].asstr()[:]
    SOAPoutDataset.attrs["SOAPengine"] = soapEngine.SOAPenginekind
    SOAPoutDataset.attrs["l_max"] = soapEngine.lmax
    SOAPoutDataset.attrs["n_max"] = soapEngine.nmax
    SOAPoutDataset.attrs["r_cut"] = soapEngine.rcut
    SOAPoutDataset.attrs["species"] = soapEngine.species
    # this should not be needed, given how the preparation of the dataset works
    # if soapEngine.centersMask is None:
    #     if "centersIndexes" in SOAPoutDataset.attrs:
    #         del SOAPoutDataset.attrs["centersIndexes"]
    # else:
    if soapEngine.centersMask is not None:
        SOAPoutDataset.attrs.create("centersIndexes", soapEngine.centersMask)
        # print(centersMask)
        # print(SOAPoutDataset.attrs["centersIndexes"])
        # print(type(SOAPoutDataset.attrs["centersIndexes"]))

    nspecies = len(soapEngine.species)
    for i in range(nspecies):
        for j in range(nspecies):
            if soapEngine.crossover or (i == j):
                temp = soapEngine.getLocation(
                    soapEngine.species[i], soapEngine.species[j]
                )
                SOAPoutDataset.attrs[
                    f"species_location_{soapEngine.species[i]}-{soapEngine.species[j]}"
                ] = (temp.start, temp.stop)

    for chunkTraj in trajGroup["Trajectory"].iter_chunks():
        chunkBox = (chunkTraj[0], slice(0, 6, 1))
        if verbose:
            print(f'working on trajectory chunk "{chunkTraj}"')
            print(f'   and working on box chunk "{repr(chunkBox)}"')
        # load in memory a chunk of data
        atoms = HDF2ase(trajGroup, chunkTraj, chunkBox, symbols)
        jobchunk = min(SOAPOutputChunkDim, len(atoms))
        jobStart = 0
        jobEnd = jobStart + jobchunk
        while jobStart < len(atoms):
            tStart = time.time()
            frameStart = jobStart + chunkTraj[0].start
            frameEnd = jobEnd + chunkTraj[0].start
            if verbose:
                print(f"working on frames: [{frameStart}:{frameEnd}]")
            # TODO: dscribe1.2.1 return (nat,nsoap) instead of (1,nat,nsoap) with 1 frame input!
            SOAPoutDataset[frameStart:frameEnd] = soapEngine(
                atoms[jobStart:jobEnd],
                positions=[soapEngine.centersMask] * jobchunk,
                n_jobs=SOAPnJobs,
            )
            tStop = time.time()
            jobchunk = min(SOAPOutputChunkDim, len(atoms) - jobEnd)
            jobStart = jobEnd
            jobEnd = jobStart + jobchunk
            if verbose:
                print(f"delta create= {tStop-tStart}")


def _applySOAP(
    trajContainer: h5py.Group,
    SOAPoutContainer: h5py.Group,
    key: str,
    soapEngine: SOAPengineContainer,
    SOAPOutputChunkDim: int = 100,
    SOAPnJobs: int = 1,
    doOverride: bool = False,
    verbose: bool = True,
    useType="float64",
):
    """helper function: applies the soap engine to the given trajectory within the trajContainer

    Args:
        trajContainer (h5py.Group):
            The group or the file that contains the trajectory, must have the
            following dataset in '/': "Box", "Trajectory" and "Types"
        SOAPoutContainer (h5py.Group)
             The group where the dataset with the  SOAP fingerprints will be saved
        key (str):
            the name of the dataset to be saved, if exist will be overidden
        soapEngine (SOAPengineContainer):
            the contained of the soap engine
        SOAPOutputChunkDim (int, optional):
            The chunk of trajectory that will be loaded in memory to be calculated,
            if key is a new dataset will also be the size of the main chunck of
            data of the SOAP dataset . Defaults to 100.
        SOAPnJobs (int, optional):
            Number of concurrent SOAP calculations. Defaults to 1.
        doOverride (bool, optional):
            if False will raise and exception if the user ask to override an
            already existing DataSet. Defaults to False.
        verbose (bool, optional):
            regulates the verbosity of the step by step operations.
            Defaults to True.
        useType (str,optional):
            The precision used to store the data. Defaults to "float64".
    """
    useType = numpy.dtype(useType)
    nOfFeatures = soapEngine.features
    symbols = trajContainer["Types"].asstr()[:]
    nCenters = (
        len(symbols) if soapEngine.centersMask is None else len(soapEngine.centersMask)
    )

    if key in SOAPoutContainer.keys():
        if doOverride is False:
            raise ValueError(
                f"Are you sure that you want to override {SOAPoutContainer[key].name}?"
            )
        # doOverride is True and key in SOAPoutContainer.keys():
        # check if deleting the dataset is necessary:
        oldshape = SOAPoutContainer[key].shape
        if oldshape[1] != nCenters or oldshape[2] != nOfFeatures:
            del SOAPoutContainer[key]
    if key not in SOAPoutContainer.keys():
        SOAPoutContainer.create_dataset(
            key,
            (0, nCenters, nOfFeatures),
            compression="gzip",
            compression_opts=9,
            chunks=(SOAPOutputChunkDim, nCenters, nOfFeatures),
            maxshape=(None, nCenters, nOfFeatures),
            dtype=useType,
        )
    SOAPout = SOAPoutContainer[key]
    SOAPout.resize((len(trajContainer["Trajectory"]), nCenters, nOfFeatures))
    _saponifyWorker(
        trajContainer,
        SOAPout,
        soapEngine,
        SOAPOutputChunkDim,
        SOAPnJobs,
        verbose=verbose,
    )


def saponifyMultipleTrajectories(
    trajContainers: "h5py.Group|h5py.File",
    SOAPoutContainers: "h5py.Group|h5py.File",
    SOAPrcut: float,
    SOAPnmax: int,
    SOAPlmax: int,
    SOAPOutputChunkDim: int = 100,
    SOAPnJobs: int = 1,
    SOAPatomMask: "list[str]" = None,
    centersMask: Iterable = None,
    SOAP_respectPBC: bool = True,
    SOAPkwargs: dict = None,
    useSoapFrom: KNOWNSOAPENGINES = "dscribe",
    doOverride: bool = False,
    verbose: bool = True,
    useType="float64",
):
    """Calculates and stores the SOAP descriptor for all of the trajectories in
    the given group/file

    `saponifyMultipleTrajectories` checks if any of the group contained in
    `trajContainers` is a "trajectory group"
    (see :func:`SOAPify.HDF5er.HDF5erUtils.isTrajectoryGroup`) and then calculates
    the soap fingerprints for that trajectory and saves the result in a dataset
    within the `SOAPoutContainers` group or file

    `SOAPatomMask` and `centersMask` are mutually exclusive (see
    :func:`SOAPify.engine.getSoapEngine`)

    Args:
        trajContainers (h5py.Group|h5py.File):
            The file/group that contains the trajectories
        SOAPoutContainers (h5py.Group|h5py.File)
            The file/group that will store the SOAP results
        SOAPrcut (float)
            The cutoff for local region in angstroms. Should be bigger than 1
            angstrom (option passed to the desired SOAP engine). Defaults to 8.0.
        SOAPnmax (int)
            The number of radial basis functions (option passed to the desired
            SOAP engine). Defaults to 8.
        SOAPlmax (int)
            The maximum degree of spherical harmonics (option passed to the
            desired SOAP engine). Defaults to 8.
        SOAPOutputChunkDim (int, optional)
            The dimension of the chunck of data in the SOAP results dataset.
            Defaults to 100.
        SOAPnJobs (int, optional)
            the number of concurrent SOAP calculations (option passed to the
            desired SOAP engine). Defaults to 1.
        SOAPatomMask (list[str], optional)
            the symbols of the atoms whose SOAP fingerprint will be calculated
            (option passed to getSoapEngine). Defaults to None.
        centersMask (Iterable, optional)
            the indexes of the atoms whose SOAP fingerprint will be calculated
            (option passed getSoapEngine). Defaults to None.
        SOAP_respectPBC (bool, optional)
            Determines whether the system is considered to be periodic (option
            passed to the desired SOAP engine). Defaults to True.
        SOAPkwargs (dict, optional)
            additional keyword arguments to be passed to the selected SOAP engine.
            Defaults to {}.
        useSoapFrom (KNOWNSOAPENGINES, optional)
            This string determines the selected SOAP engine for the calculations.
            Defaults to "dscribe".
        doOverride (bool, optional)
            if False will raise and exception if the user ask to override an
            already existing DataSet. Defaults to False.
        verbose (bool, optional):
            regulates the verbosity of the step by step operations.
            Defaults to True.
        useType (str,optional):
            The precision used to store the data. Defaults to "float64".
    """
    for key in trajContainers.keys():
        if isTrajectoryGroup(trajContainers[key]):
            saponifyTrajectory(
                trajContainer=trajContainers[key],
                SOAPoutContainer=SOAPoutContainers,
                SOAPrcut=SOAPrcut,
                SOAPnmax=SOAPnmax,
                SOAPlmax=SOAPlmax,
                SOAPOutputChunkDim=SOAPOutputChunkDim,
                SOAPnJobs=SOAPnJobs,
                SOAPatomMask=SOAPatomMask,
                centersMask=centersMask,
                SOAP_respectPBC=SOAP_respectPBC,
                SOAPkwargs=SOAPkwargs,
                useSoapFrom=useSoapFrom,
                doOverride=doOverride,
                verbose=verbose,
                useType=useType,
            )


def saponifyTrajectory(
    trajContainer: "h5py.Group|h5py.File",
    SOAPoutContainer: "h5py.Group|h5py.File",
    SOAPrcut: float,
    SOAPnmax: int,
    SOAPlmax: int,
    SOAPOutputChunkDim: int = 100,
    SOAPnJobs: int = 1,
    SOAPatomMask: str = None,
    centersMask: Iterable = None,
    SOAP_respectPBC: bool = True,
    SOAPkwargs: dict = None,
    useSoapFrom: KNOWNSOAPENGINES = "dscribe",
    doOverride: bool = False,
    verbose: bool = True,
    useType="float64",
):
    """Calculates the SOAP fingerprints for each atom in a given hdf5 trajectory

    Works exaclty as :func:`saponifyMultipleTrajectories` except for that it
    calculates the fingerprints only for the passed trajectory group
    (see :func:`SOAPify.HDF5er.HDF5erUtils.isTrajectoryGroup`).

    `SOAPatomMask` and `centersMask` are mutually exclusive (see
    :func:`SOAPify.engine.getSoapEngine`)

    Args:
        trajFname (str):
            The name of the hdf5 file in wich the trajectory is stored
        trajectoryGroupPath (str):
            the path of the group that contains the trajectory in trajFname
        outputFname (str):
            the name of the hdf5 file that will contain the ouput or the SOAP analysis
        exportDatasetName (str):
            the name of the dataset that will contain the SOAP results,
            it will be saved in the group called "SOAP"
        SOAPOutputChunkDim (int, optional):
            The dimension of the chunck of data in the SOAP results dataset.
            Defaults to 100.
        SOAPnJobs (int, optional):
            the number of concurrent SOAP calculations (option passed to the
            desired SOAP engine). Defaults to 1.
        SOAPatomMask (str, optional):
            the symbols of the atoms whose SOAP fingerprint will be calculated
            (option passed to the desired SOAP engine). Defaults to None.
        SOAPrcut (float, optional):
            The cutoff for local region in angstroms. Should be bigger than 1
            angstrom (option passed to the desired SOAP engine). Defaults to 8.0.
        SOAPnmax (int, optional):
            The number of radial basis functions (option passed to the desired
            SOAP engine). Defaults to 8.
        SOAPlmax (int, optional):
            The maximum degree of spherical harmonics (option passed to the
            desired SOAP engine). Defaults to 8.
        SOAP_respectPBC (bool, optional):
            Determines whether the system is considered to be periodic
            (option passed to the desired SOAP engine). Defaults to True.
        SOAPkwargs (dict, optional):
            additional keyword arguments to be passed to the SOAP engine.
            Defaults to {}.
        useSoapFrom (KNOWNSOAPENGINES, optional):
            This string determines the selected SOAP engine for the calculations.
            Defaults to "dscribe".
        doOverride (bool, optional):
            if False will raise and exception if the user ask to override an
            already existing DataSet. Defaults to False.
        verbose (bool, optional):
            regulates the verbosity of the step by step operations.
            Defaults to True.
        useType (str,optional):
            The precision used to store the data. Defaults to "float64".
    """
    if isTrajectoryGroup(trajContainer):
        print(f'using "{useSoapFrom}" to calculate SOAP for "{trajContainer.name}"')
        print("extra SOAP arguments:", SOAPkwargs)
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
            verbose=verbose,
            useType=useType,
        )
    else:
        raise ValueError("saponify: The input object is not a trajectory group.")
