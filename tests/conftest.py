import pytest
import SOAPify
import numpy
from numpy.random import randint


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

        def __call__(self, frames, nat) -> dict:
            toret = dict()
            if self.doOneD:
                toret["OneD"] = self.rng.integers(0, 7, size=(frames, nat))
            if self.doMultD:
                dataDim = self.rng.integers(2, 15)
                toret["MultD"] = self.rng.integers(0, 7, size=(frames, nat, dataDim))
            return toret

        def __repr__(self) -> str:
            return f"ParameterCreator, doOneD:{self.doOneD}, doMultD:{self.doMultD}"

    return ParameterCreator(doOneD=oneD, doMultyD=MultD)
