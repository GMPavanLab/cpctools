import SOAPify
import numpy
import pytest
from numpy.random import randint
from numpy.testing import assert_array_equal
from SOAPify import SOAPReferences
import h5py

from SOAPify.SOAPClassify import SOAPclassification


def test_norm1D():
    a = numpy.array([2.0, 0.0])
    na = SOAPify.utils.normalizeArray(a)
    assert_array_equal(na, numpy.array([1.0, 0.0]))


def test_norm2D():
    a = numpy.array([[2.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    na = SOAPify.utils.normalizeArray(a)
    assert_array_equal(na, numpy.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))


def test_norm3D():
    a = numpy.array(
        [[[2.0, 0.0, 0.0], [0.0, 0.0, 3.0]], [[0.0, 2.0, 0.0], [0.0, 3.0, 0.0]]]
    )
    na = SOAPify.utils.normalizeArray(a)
    assert_array_equal(
        na,
        numpy.array(
            [[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]]
        ),
    )


def test_Saving_loadingOfSOAPreferences():
    a = SOAPReferences(["a", "b"], numpy.array([[0, 0], [1, 1]]), 4, 8)
    SOAPify.saveReferences(h5py.File("refSave.hdf5", "w"), "testRef", a)
    with h5py.File("refSave.hdf5", "r") as saved:
        bk = SOAPify.getReferencesFromDataset(saved["testRef"])
        assert len(a) == len(bk)
        assert_array_equal(bk.spectra, a.spectra)
        for i, j in zip(a.names, bk.names):
            assert i == j
        assert a.nmax == bk.nmax
        assert a.lmax == bk.lmax


def test_concatenationOfSOAPreferences():
    a = SOAPReferences(["a", "b"], [[0, 0], [1, 1]], 8, 8)
    b = SOAPReferences(["c", "d"], [[2, 2], [3, 4]], 8, 8)
    c = SOAPify.mergeReferences(a, b)
    assert len(c.names) == 4
    assert c.spectra.shape == (4, 2)
    assert_array_equal(a.spectra[0], c.spectra[0])
    assert_array_equal(a.spectra[1], c.spectra[1])
    assert_array_equal(b.spectra[0], c.spectra[2])
    assert_array_equal(b.spectra[1], c.spectra[3])


def test_concatenationOfSOAPreferencesLonger():
    a = SOAPReferences(["a", "b"], [[0, 0], [1, 1]], 8, 8)
    b = SOAPReferences(["c", "d"], [[2, 2], [3, 3]], 8, 8)
    d = SOAPReferences(["e", "f"], [[4, 4], [5, 5]], 8, 8)
    c = SOAPify.mergeReferences(a, b, d)
    assert len(c.names) == (len(a) + len(b) + len(d))
    assert c.spectra.shape == (6, 2)
    assert_array_equal(a.spectra[0], c.spectra[0])
    assert_array_equal(a.spectra[1], c.spectra[1])
    assert_array_equal(b.spectra[0], c.spectra[2])
    assert_array_equal(b.spectra[1], c.spectra[3])
    assert_array_equal(d.spectra[0], c.spectra[4])
    assert_array_equal(d.spectra[1], c.spectra[5])


def test_fillSOAPVectorFromdscribeSingleVector():
    nmax = 4
    lmax = 3
    a = randint(0, 10, size=int(((lmax + 1) * (nmax + 1) * nmax) / 2))
    b = SOAPify.fillSOAPVectorFromdscribe(a, lmax, nmax)
    assert b.shape[0] == (lmax + 1) * (nmax * nmax)
    limited = 0
    b = b.reshape((lmax + 1, nmax, nmax))
    for l in range(lmax + 1):
        for n in range(nmax):
            for np in range(n, nmax):
                assert b[l, n, np] == a[limited]
                assert b[l, np, n] == a[limited]
                limited += 1


def test_fillSOAPVectorFromdscribeArrayOfVector():
    nmax = 4
    lmax = 3
    a = randint(0, 10, size=(5, int(((lmax + 1) * (nmax + 1) * nmax) / 2)))
    b = SOAPify.fillSOAPVectorFromdscribe(a, lmax, nmax)
    assert b.shape[1] == (lmax + 1) * (nmax * nmax)

    for i in range(a.shape[0]):
        limited = 0
        c = b[i].reshape((lmax + 1, nmax, nmax))
        for l in range(lmax + 1):
            for n in range(nmax):
                for np in range(n, nmax):
                    assert c[l, n, np] == a[i, limited]
                    assert c[l, np, n] == a[i, limited]
                    limited += 1


@pytest.fixture(
    scope="module",
    params=[
        (
            SOAPify.SOAPclassification(
                [],
                numpy.array(
                    [
                        [0, 0, 1, 2, 2, 1],
                        [0, 0, 2, 2, 2, 2],
                        [0, 0, 2, 2, 2, 1],
                        [0, 0, 2, 2, 2, 2],
                    ]
                ),
                ["1", "2", "3"],
            ),
            # hand calculated:
            numpy.array(
                [
                    [6, 0, 0],
                    [0, 0, 3],
                    [0, 1, 8],
                ]
            ),
        ),
        (
            SOAPify.SOAPclassification(
                [],
                numpy.array(
                    [
                        [0, 0, 1, 2, 2, 1],
                        [0, 0, 2, 2, 2, 2],
                        [0, 0, 2, 2, 2, 1],
                        [0, 0, 2, 2, 2, -1],
                    ]
                ),
                ["1", "2", "3", "Errors"],
            ),  # hand calculated:
            numpy.array(
                [
                    [6, 0, 0, 0],
                    [0, 0, 2, 1],
                    [0, 1, 8, 0],
                    [0, 0, 0, 0],
                ]
            ),
        ),
    ],
)
def input_statesEvolution(request):
    return request.param


def test_transitionMatrix(input_statesEvolution):
    data: SOAPclassification = input_statesEvolution[0]
    expectedTmat = input_statesEvolution[1]
    tmat = SOAPify.transitionMatrixFromSOAPClassification(data)
    assert tmat.shape[0] == len(data.legend)

    assert_array_equal(tmat, expectedTmat)


def test_residenceTime(input_statesEvolution):
    data: SOAPclassification = input_statesEvolution[0]
    expectedResidenceTimes = [[] for i in range(len(data.legend))]
    for atomID in range(data.references.shape[1]):
        prevState = data.references[0, atomID]
        time = 0
        for frame in range(1, data.references.shape[0]):
            state = data.references[frame, atomID]
            if state != prevState:
                expectedResidenceTimes[prevState].append(time)
                time = 0
                prevState = state
            time += 1
        # the last state does not have an out transition, appendig negative time to make it clear
        expectedResidenceTimes[prevState].append(-time)

    for i in range(len(expectedResidenceTimes)):
        expectedResidenceTimes[i] = numpy.sort(numpy.array(expectedResidenceTimes[i]))

    ResidenceTimes = SOAPify.calculateResidenceTimes(data)
    for stateID in range(len(expectedResidenceTimes)):
        assert_array_equal(ResidenceTimes[stateID], expectedResidenceTimes[stateID])


def test_stateTracker(input_statesEvolution):
    data: SOAPclassification = input_statesEvolution[0]
    # code for derermining the events
    CURSTATE = SOAPify.TRACK_CURSTATE
    ENDSTATE = SOAPify.TRACK_ENDSTATE
    TIME = SOAPify.TRACK_EVENTTIME
    expectedEvents = []
    for atomID in range(data.references.shape[1]):
        atomTraj = data.references[:, atomID]
        # the array is [start state, state, end state,time]
        # wg
        event = numpy.array([atomTraj[0], atomTraj[0], atomTraj[0], 0], dtype=int)
        for frame in range(1, data.references.shape[0]):
            if atomTraj[frame] != event[CURSTATE]:
                event[ENDSTATE] = atomTraj[frame]
                expectedEvents.append(event)
                event = numpy.array(
                    [expectedEvents[-1][CURSTATE], atomTraj[frame], atomTraj[frame], 0],
                    dtype=int,
                )
            event[TIME] += 1
        # append the last event
        expectedEvents.append(event)

    events = SOAPify.trackStates(data)

    for event, expectedEvent in zip(events, expectedEvents):
        assert_array_equal(event, expectedEvent)


def test_residenceTimesFromTracking(input_statesEvolution):
    data = input_statesEvolution[0]
    events = SOAPify.trackStates(data)
    residenceTimesFromTracking = SOAPify.calculateResidenceTimes(data, events)
    expectedResidenceTimes = SOAPify.calculateResidenceTimes(data)

    for stateID in range(len(expectedResidenceTimes)):
        assert_array_equal(
            residenceTimesFromTracking[stateID], expectedResidenceTimes[stateID]
        )


def test_transitionMatrixFromTracking(input_statesEvolution):
    data = input_statesEvolution[0]
    events = SOAPify.trackStates(data)
    for event in events:
        print(event)
    transitionMatrixFromTracking = SOAPify.calculateTransitionMatrix(
        data, stride=1, statesTracker=events
    )
    expectedTmat = SOAPify.transitionMatrixFromSOAPClassification(data, stride=1)
    assert_array_equal(transitionMatrixFromTracking, expectedTmat)
