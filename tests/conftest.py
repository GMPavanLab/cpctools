import pytest
import SOAPify
import HDF5er
import numpy
from numpy.random import randint
from .testSupport import giveUniverse, giveUniverse_ChangingBox


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
                # 1 change stare at first frame
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
                # 3 as an error at some point
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
