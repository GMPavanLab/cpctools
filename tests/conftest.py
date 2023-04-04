import pytest
import SOAPify
import SOAPify.HDF5er as HDF5er
import numpy
import h5py
from numpy.random import randint
from .testSupport import (
    giveUniverse,
    giveUniverse_ChangingBox,
    giveUniverse_LongChangingBox,
    give_ico923,
    getUniverseWithWaterMolecules,
)


def __alph(k):
    "helper function to not overlap ref names in randomSOAPReferences"
    from string import ascii_lowercase as alph

    toret = ""
    while k >= len(alph):
        knew = k - len(alph)
        toret += alph[k - knew - 1]
        k -= len(alph)
    toret += alph[k]
    return toret


@pytest.fixture(scope="module", params=[2, 3, 4, 5, 6])
def randomSOAPReferences(request):
    toret = []
    totalLength = 0
    refDim = randint(2, high=7)
    for i in range(request.param):
        refLenght = randint(2, high=7)
        toret.append(
            SOAPify.SOAPReferences(
                [__alph(k) for k in range(totalLength, totalLength + refLenght)],
                randint(0, high=500, size=(refLenght, refDim)),
                8,
                8,
            )
        )
        totalLength += refLenght
    return toret


@pytest.fixture(
    scope="session",
    params=[
        giveUniverse,
        giveUniverse_ChangingBox,
        giveUniverse_LongChangingBox,
    ],
)
def input_universe(request):
    return request.param


@pytest.fixture(scope="session")
def hdf5_file(tmp_path_factory, input_universe):
    fourAtomsFiveFrames = input_universe((90.0, 90.0, 90.0))

    testFname = (
        tmp_path_factory.mktemp("data") / f"test{fourAtomsFiveFrames.myUsefulName}.hdf5"
    )

    HDF5er.MDA2HDF5(fourAtomsFiveFrames, testFname, "4Atoms5Frames", override=True)

    return testFname, fourAtomsFiveFrames


@pytest.fixture(
    scope="module",
    params=[
        SOAPify.SOAPclassification(
            [],
            numpy.array(
                # 0 never changes state
                # 1 change state at first frame
                # 2 alternates two states
                [
                    [0, 1, 1],
                    [0, 2, 2],
                    [0, 2, 1],
                    [0, 2, 2],
                    [0, 2, 1],
                    [0, 2, 2],
                ]
            ),
            ["state0", "state1", "state2"],
        ),
        SOAPify.SOAPclassification(
            [],
            numpy.array(
                # 0 never changes state
                # 1 change stare at first frame
                # 2 alternates two states
                # 3 has an error at some point
                [
                    [0, 1, 1, 1],
                    [0, 2, 2, 2],
                    [0, 2, 1, 1],
                    [0, 2, 2, -1],
                    [0, 2, 1, 1],
                    [0, 2, 2, 2],
                ]
            ),
            ["state0", "state1", "state2", "Errors"],
        ),
        SOAPify.SOAPclassification(  # big random "simulation"
            [],
            randint(0, high=4, size=(1000, 309)),
            ["state0", "state1", "state2", "state3"],
        ),
    ],
)
def input_mockedTrajectoryClassification(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        slice(None, None, None),  # no slice
        slice(1, None, 2),  # classic slice
        [0, 4],  # list-like slice
        [0],  # list-like slice - single atom
    ],
)
def input_framesSlice(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        "dscribe",
        "quippy",
    ],
)
def engineKind_fixture(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        ["C", "O", "H", "N"],
        ["H", "O"],
        ["H"],
    ],
)
def species_fixture(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[None, 1, 5, 10],
)
def inputNone_1_5_10(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[1, 5, 10],
)
def input1_5_10(request):
    return request.param


inputStrides = input1_5_10
inputWindows = input1_5_10
inputWindowsWithNone = inputNone_1_5_10
inputStridesWithNone = inputNone_1_5_10


@pytest.fixture(
    scope="module",
    params=[1, 4, 8],
)
def nMaxFixture(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[0, 4, 8],
)
def lMaxFixture(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[-1, 0, 1],
)
def input_intModify(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[(True, False), (True, True), (False, True), (False, False)],
)
def input_CreateParametersToExport(request):
    oneD, MultD = request.param

    class ParameterCreator:
        def __init__(self, doOneD, doMultyD):
            self.doOneD = doOneD
            self.doMultD = doMultyD
            self.rng = numpy.random.default_rng(12345)

        def __call__(self, frames, nat, frameSlice=slice(None)) -> dict:
            toret = dict()
            if self.doOneD:
                toret["OneD"] = self.rng.integers(0, 7, size=(frames, nat))[frameSlice]
            if self.doMultD:
                dataDim = self.rng.integers(2, 15)
                toret["MultD"] = self.rng.integers(0, 7, size=(frames, nat, dataDim))[
                    frameSlice
                ]
            return toret

        def __repr__(self) -> str:
            return f"ParameterCreator, doOneD:{self.doOneD}, doMultD:{self.doMultD}"

    return ParameterCreator(doOneD=oneD, doMultyD=MultD)


@pytest.fixture(scope="session")
def getReferencesConfs(tmp_path_factory):
    ico923 = give_ico923()

    referenceConfs = tmp_path_factory.mktemp("referencesNPs") / f"referencesConfs.hdf5"

    HDF5er.MDA2HDF5(ico923, referenceConfs, "ico923_6", override=True)
    with h5py.File(referenceConfs, "a") as workFile:
        SOAPify.saponifyMultipleTrajectories(
            trajContainers=workFile["Trajectories"],
            SOAPoutContainers=workFile.require_group("SOAP"),
            SOAPOutputChunkDim=1000,
            SOAPnJobs=1,
            SOAPrcut=4.48023312,
            SOAPnmax=4,
            SOAPlmax=4,
        )
    return referenceConfs


@pytest.fixture(scope="session")
def referencesTest(getReferencesConfs):
    FramesRequest = dict(
        ico923_6={
            "v_5f_ih": (0, 566),
            "e_(111)_ih": (0, 830),
            "e_(111)_vih": (0, 828),
            "s_(111)_ih": (0, 892),
            "s_(111)_eih": (0, 893),
            "ss_5f_ih": (0, 312),
            "ss_FCC_ih": (0, 524),
            "ss_HCP_ih": (0, 431),
            "b_5f_ih": (0, 1),
            "b_HCP_ih": (0, 45),
            "b_FCC_ih": (0, 127),
            "b_c_ih": (0, 0),
        },
    )
    referenceDict = dict()
    with h5py.File(getReferencesConfs, "r") as workFile:
        for k in FramesRequest:
            nmax = workFile[f"SOAP/{k}"].attrs["n_max"]
            lmax = workFile[f"SOAP/{k}"].attrs["l_max"]
            referenceDict[k] = SOAPify.createReferencesFromTrajectory(
                workFile[f"SOAP/{k}"], FramesRequest[k], nmax=nmax, lmax=lmax
            )
    return referenceDict, FramesRequest


@pytest.fixture(scope="session")
def referencesWater(tmp_path_factory):
    """Creates a base hdf5file to be used in various tests"""
    nMol = 27
    u = getUniverseWithWaterMolecules(nMol)

    fname = tmp_path_factory.mktemp("waterBase") / f"waterBase.hdf5"
    groupname = "waterBase"
    HDF5er.MDA2HDF5(u, fname, groupname, override=True)
    return fname, groupname, nMol


@pytest.fixture(scope="session")
def referencesSingleWater(tmp_path_factory):
    """Creates a base hdf5file to be used in various tests"""
    nMol = 1
    u = getUniverseWithWaterMolecules(nMol)

    fname = tmp_path_factory.mktemp("waterBase") / f"waterSingleMol.hdf5"
    groupname = "waterSingleMol"
    HDF5er.MDA2HDF5(u, fname, groupname, override=True)
    return fname, groupname, nMol


@pytest.fixture(scope="session")
def referencesWaterSOAP(referencesWater):
    """Creates a base hdf5file to be used in various tests"""
    confFile, groupName, nMol = referencesWater
    nmax = 8
    lmax = 8
    rcut = 10.0
    with h5py.File(confFile, "a") as f:
        soapGroup = f.require_group("SOAP")
        SOAPify.saponifyMultipleTrajectories(
            f["Trajectories"],
            soapGroup,
            rcut,
            nmax,
            lmax,
            useSoapFrom="dscribe",
        )

    return confFile, groupName, nMol


@pytest.fixture(scope="session")
def referencesTrajectory(tmp_path_factory):
    """Creates a base hdf5file to be used in various tests"""
    u = giveUniverse(repeatFrames=10)

    fname = tmp_path_factory.mktemp("trajBase") / f"trajBase.hdf5"
    groupname = "trajBase"
    HDF5er.MDA2HDF5(u, fname, groupname, override=True)
    return fname, groupname


@pytest.fixture(scope="session")
def referencesTrajectorySOAP(referencesTrajectory):
    """Creates a base hdf5file to be used in various tests"""
    confFile, groupName = referencesTrajectory
    nmax = 8
    lmax = 8
    rcut = 3.0
    with h5py.File(confFile, "a") as f:
        soapGroup = f.require_group("SOAP")
        SOAPify.saponifyMultipleTrajectories(
            f["Trajectories"],
            soapGroup,
            rcut,
            nmax,
            lmax,
            useSoapFrom="dscribe",
        )

    return confFile, groupName
