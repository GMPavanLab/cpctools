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
