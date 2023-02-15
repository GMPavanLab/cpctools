import SOAPify
import numpy
from numpy.random import randint
from numpy.testing import assert_array_equal
from SOAPify import SOAPReferences
import h5py
from ase.data import atomic_numbers
from SOAPify.classify import SOAPclassification
import pytest


def test_atomicnumberOredring():
    species = ["O", "H"]
    assert SOAPify.orderByZ(species) == ["H", "O"]
    species = ["Au", "H", "C", "W", "O"]
    ordSpecies = SOAPify.orderByZ(species)
    for i in range(len(ordSpecies) - 1):
        assert atomic_numbers[ordSpecies[i]] < atomic_numbers[ordSpecies[i + 1]]


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


def test_normalizeMatrix():
    # this forces the test with an empy row
    mat = numpy.array([[0.0, 0.0, 0.0], [5.0, 5.0, 0.0], [4.0, 0.0, 0.0]])
    numpy.testing.assert_array_almost_equal(
        SOAPify.normalizeMatrix(mat),
        numpy.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [1.0, 0.0, 0.0]]),
    )
    rng = numpy.random.default_rng(12345)
    for i in range(5):
        size = rng.integers(2, 152)
        newMat = rng.random((size, size))
        testMat = newMat.copy()
        newMatNorm = SOAPify.normalizeMatrix(newMat)

        numpy.testing.assert_array_almost_equal(testMat, newMat)
        for row in range(newMatNorm.shape[0]):
            sum = numpy.sum(newMatNorm[row, :])
            if sum != 0.0:
                numpy.testing.assert_approx_equal(sum, 1.0, 8)


@pytest.fixture(
    scope="module",
    params=[-1, 0, 1],
)
def shiftFixture(request):
    return request.param


nshift = shiftFixture
lshift = shiftFixture


def test_concatenationSOAPreferencesCompatibility(
    nshift, lshift, lMaxFixture, nMaxFixture
):
    # lmax fixture can be 0, whit lshift -1 goes under 0, that is not  "matematically sensed",
    # but this only tests if nmax and lmax are equal in the merge references, and not if the vectors are senses
    a = SOAPReferences(["a", "b"], [[0, 0], [1, 1]], nMaxFixture, lMaxFixture)
    b = SOAPReferences(
        ["c", "d"], [[2, 2], [3, 3]], nMaxFixture + nshift, lMaxFixture + lshift
    )
    if lshift == 0 and nshift == 0:
        # this must not throw!
        try:
            SOAPify.mergeReferences(a, b)
        except ValueError:
            pytest.fail("nmax and lmax are not changed: should not throw")
        return
    with pytest.raises(ValueError):
        SOAPify.mergeReferences(a, b)


@pytest.fixture(
    scope="module",
    params=[-1, 0, 1],
)
def shiftFixture(request):
    return request.param


nshift = shiftFixture
lshift = shiftFixture


def test_concatenationSOAPreferencesCompatibility(
    nshift, lshift, lMaxFixture, nMaxFixture
):
    # lmax fixture can be 0, whit lshift -1 goes under 0, that is not  "matematically sensed",
    # but this only tests if nmax and lmax are equal in the merge references, and not if the vectors are senses
    a = SOAPReferences(["a", "b"], [[0, 0], [1, 1]], nMaxFixture, lMaxFixture)
    b = SOAPReferences(
        ["c", "d"], [[2, 2], [3, 3]], nMaxFixture + nshift, lMaxFixture + lshift
    )
    if lshift == 0 and nshift == 0:
        # this must not throw!
        try:
            SOAPify.mergeReferences(a, b)
        except ValueError:
            pytest.fail("nmax and lmax are not changed: should not throw")
        return
    with pytest.raises(ValueError):
        SOAPify.mergeReferences(a, b)


def test_concatenationOfSOAPreferencesNamedArgumumets():
    a = SOAPReferences(["a", "b"], [[0, 0], [1, 1]], 8, 8)
    b = SOAPReferences(["c", "d"], [[2, 2], [3, 4]], 8, 8)
    c = SOAPify.mergeReferences(a, b)
    assert len(c.names) == 4
    assert c.spectra.shape == (4, 2)
    assert_array_equal(a.spectra[0:2], c.spectra[0:2])
    assert_array_equal(b.spectra[0:2], c.spectra[2:4])


def test_concatenationOfSOAPreferencesLongerNamedArgumumets():
    a = SOAPReferences(["a", "b"], [[0, 0], [1, 1]], 8, 8)
    b = SOAPReferences(["c", "d"], [[2, 2], [3, 3]], 8, 8)
    d = SOAPReferences(["e", "f"], [[4, 4], [5, 5]], 8, 8)
    c = SOAPify.mergeReferences(a, b, d)
    assert len(c.names) == (len(a) + len(b) + len(d))
    assert c.spectra.shape == (6, 2)
    assert_array_equal(a.spectra[0:2], c.spectra[0:2])
    assert_array_equal(b.spectra[0:2], c.spectra[2:4])
    assert_array_equal(d.spectra[0:2], c.spectra[4:6])


def test_concatenationOfSOAPreferences(randomSOAPReferences):
    conc = SOAPify.mergeReferences(*randomSOAPReferences)
    refDim = randomSOAPReferences[0].spectra.shape[1]
    origTotalLenght = numpy.sum([len(a) for a in randomSOAPReferences])
    assert len(conc) == origTotalLenght
    assert conc.spectra.shape == (origTotalLenght, refDim)
    totalLength = 0
    for i, orig in enumerate(randomSOAPReferences):
        spectraL = orig.spectra.shape[0]
        assert_array_equal(
            orig.spectra, conc.spectra[totalLength : totalLength + spectraL]
        )
        assert_array_equal(orig.names, conc.names[totalLength : totalLength + spectraL])
        totalLength += spectraL


def test_fillSOAPVectorFromdscribeSingleVector(nMaxFixture, lMaxFixture):
    nmax = nMaxFixture
    lmax = lMaxFixture
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


def test_fillSOAPVectorFromdscribeArrayOfVector(nMaxFixture, lMaxFixture):
    nmax = nMaxFixture
    lmax = lMaxFixture
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


def test_fillSOAPVectorFromdscribeArrayOfVector_wrongerrors(nMaxFixture, lMaxFixture):
    nmax = nMaxFixture
    lmax = lMaxFixture
    expectedSize = (lmax + 1) * (nmax + 1) * nmax // 2
    # wrong lmax/nmax
    a = randint(0, 10, size=(5, 1 + expectedSize))
    with pytest.raises(Exception):
        SOAPify.fillSOAPVectorFromdscribe(a, lmax, nmax)
    # too much dimensions
    a = randint(0, 10, size=(3, 4, 5, expectedSize))
    with pytest.raises(Exception):
        SOAPify.fillSOAPVectorFromdscribe(a, lmax, nmax)


def test_fillSOAPVectorFromdscribeArrayOfVectorMultiSpecies(nMaxFixture, lMaxFixture):
    species = ["H", "O"]

    ncomb = 3
    nmax = nMaxFixture
    lmax = lMaxFixture
    nfeats = (lmax + 1) * nmax * nmax
    nfeatsreduced = int(((lmax + 1) * (nmax + 1) * nmax) / 2)
    nframes = 50
    natoms = 1000
    a = randint(
        0,
        10,
        size=(
            nframes,
            natoms,
            nfeats + 2 * nfeatsreduced,
        ),
    )
    speciesSlices = {
        "HH": slice(0, nfeatsreduced),
        "HO": slice(nfeatsreduced, nfeatsreduced + nfeats),
        "OO": slice(nfeatsreduced + nfeats, nfeats + 2 * nfeatsreduced),
    }
    b = SOAPify.fillSOAPVectorFromdscribe(a, lmax, nmax, species, speciesSlices)
    assert b.shape[2] == ncomb * nfeats
    slices = [
        slice(0, nfeats),
        slice(nfeats, 2 * nfeats),
        slice(2 * nfeats, 3 * nfeats),
    ]
    for frame in range(nframes):
        for i in range(natoms):
            limited = 0

            c = b[frame, i][slices[0]].reshape((lmax + 1, nmax, nmax))

            for l in range(lmax + 1):
                for n in range(nmax):
                    for np in range(n, nmax):
                        assert c[l, n, np] == a[frame, i, limited]
                        assert c[l, np, n] == a[frame, i, limited]
                        limited += 1

            c = b[frame, i][slices[1]].reshape((lmax + 1, nmax, nmax))
            for l in range(lmax + 1):
                for n in range(nmax):
                    for np in range(nmax):
                        assert c[l, n, np] == a[frame, i, limited]
                        limited += 1

            c = b[frame, i][slices[2]].reshape((lmax + 1, nmax, nmax))
            for l in range(lmax + 1):
                for n in range(nmax):
                    for np in range(n, nmax):
                        assert c[l, n, np] == a[frame, i, limited]
                        assert c[l, np, n] == a[frame, i, limited]
                        limited += 1


def test_transitionMatrix(input_mockedTrajectoryClassification):
    data: SOAPclassification = input_mockedTrajectoryClassification
    expectedTmat = numpy.zeros((len(data.legend), len(data.legend)))
    stride = 1
    for atomID in range(data.references.shape[1]):
        for frame in range(stride, data.references.shape[0]):
            expectedTmat[
                data.references[frame - stride, atomID],
                data.references[frame, atomID],
            ] += 1

    tmat = SOAPify.transitionMatrixFromSOAPClassification(data, stride=stride)
    assert tmat.shape[0] == len(data.legend)
    assert_array_equal(tmat, expectedTmat)

    tmatNorm = SOAPify.transitionMatrixFromSOAPClassificationNormalized(
        data, stride=stride
    )
    assert tmatNorm.shape[0] == len(data.legend)
    assert_array_equal(tmatNorm, SOAPify.normalizeMatrix(expectedTmat))


def test_residenceTime(input_mockedTrajectoryClassification):
    data: SOAPclassification = input_mockedTrajectoryClassification
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


def test_stateTracker(input_mockedTrajectoryClassification):
    data: SOAPclassification = input_mockedTrajectoryClassification
    # code for derermining the events
    CURSTATE = SOAPify.TRACK_CURSTATE
    ENDSTATE = SOAPify.TRACK_ENDSTATE
    TIME = SOAPify.TRACK_EVENTTIME
    expectedEvents = []
    for atomID in range(data.references.shape[1]):
        eventsperAtom = []
        atomTraj = data.references[:, atomID]
        # the array is [start state, state, end state,time]
        # wg
        event = numpy.array([atomTraj[0], atomTraj[0], atomTraj[0], 0], dtype=int)
        for frame in range(1, data.references.shape[0]):
            if atomTraj[frame] != event[CURSTATE]:
                event[ENDSTATE] = atomTraj[frame]
                eventsperAtom.append(event)
                event = numpy.array(
                    [eventsperAtom[-1][CURSTATE], atomTraj[frame], atomTraj[frame], 0],
                    dtype=int,
                )
            event[TIME] += 1
        # append the last event
        eventsperAtom.append(event)
        expectedEvents.append(eventsperAtom)

    events = SOAPify.trackStates(data)
    for atomID in range(data.references.shape[1]):
        for event, expectedEvent in zip(events[atomID], expectedEvents[atomID]):
            assert_array_equal(event, expectedEvent)


def test_residenceTimesFromTracking(input_mockedTrajectoryClassification):
    data = input_mockedTrajectoryClassification
    events = SOAPify.trackStates(data)
    residenceTimesFromTracking = SOAPify.calculateResidenceTimes(data, events)
    expectedResidenceTimes = SOAPify.calculateResidenceTimes(data)

    for stateID in range(len(expectedResidenceTimes)):
        assert_array_equal(
            residenceTimesFromTracking[stateID], expectedResidenceTimes[stateID]
        )
    assert isinstance(events[0], list)


def test_transitionMatrixFromTracking(input_mockedTrajectoryClassification):
    data = input_mockedTrajectoryClassification
    events = SOAPify.trackStates(data)
    for event in events:
        print(event)
    transitionMatrixFromTracking = SOAPify.calculateTransitionMatrix(
        data, stride=1, statesTracker=events
    )
    expectedTmat = SOAPify.transitionMatrixFromSOAPClassification(data, stride=1)
    standardTmat = SOAPify.calculateTransitionMatrix(data, stride=1)
    assert_array_equal(transitionMatrixFromTracking, expectedTmat)
    assert_array_equal(standardTmat, expectedTmat)
    assert isinstance(events[0], list)


def test_RemoveAtomIdentityFromEventTracker(input_mockedTrajectoryClassification):
    data = input_mockedTrajectoryClassification
    events = SOAPify.trackStates(data)
    newevents = SOAPify.RemoveAtomIdentityFromEventTracker(events)
    # verify that nothing is changed:
    assert isinstance(events[0], list)
    assert isinstance(events, list)
    assert isinstance(events[0][0], numpy.ndarray)
    # verify that the copy is correct:
    assert isinstance(newevents, list)
    count = 0
    for atomID in range(data.references.shape[1]):
        for event in events[atomID]:
            assert_array_equal(event, newevents[count])
            assert isinstance(newevents[atomID], numpy.ndarray)
            count += 1


def test_centerMaskCreator():
    symbols = ["C", "O", "H", "H", "N"] * 5
    for SOAPatomMask in [["O"], ["H"], ["N", "O"]]:
        mask = [i for i in range(len(symbols)) if symbols[i] in SOAPatomMask]
        getMask = SOAPify.centerMaskCreator(SOAPatomMask, symbols)
        assert_array_equal(mask, getMask)
