"""tests for SOAPify.transitions"""
from numpy.testing import assert_array_equal
import numpy
import numba
import pytest
import SOAPify
from SOAPify.classify import SOAPclassification


@numba.jit
def isSorted(a):
    for i in range(a.size - 1):
        if a[i + 1] < a[i]:
            return False
    return True


def test_transitionMatrix(
    input_mockedTrajectoryClassification, inputStrides, inputWindowsWithNone
):
    data: SOAPclassification = input_mockedTrajectoryClassification
    expectedTmat = numpy.zeros((len(data.legend), len(data.legend)))
    stride = inputStrides
    window = inputWindowsWithNone
    print(data.references.shape[0], f"{window=}, {stride=}")

    def isInvalidCombination(dataLen, stride, window):
        return (window is not None and window < stride) or (
            (window is not None and window > dataLen) or stride > dataLen
        )

    if isInvalidCombination(data.references.shape[0], stride, window):
        with pytest.raises(ValueError) as excinfo:
            SOAPify.transitionMatrixFromSOAPClassification(
                data, stride=stride, window=window
            )
        if window is not None and window < stride:
            assert "window must be bigger" in str(excinfo.value)
            return
        if (
            window is not None and window > data.references.shape[0]
        ) or stride > data.references.shape[0]:
            assert "window must be smaller" in str(excinfo.value)
            return
    tmat = SOAPify.transitionMatrixFromSOAPClassification(
        data, stride=stride, window=window
    )
    # transitionMatrixFromSOAPClassification is called by transitionMatrixFromSOAPClassificationNormalized
    tmatNorm = SOAPify.transitionMatrixFromSOAPClassificationNormalized(
        data, stride=stride, window=window
    )
    # hand calculation of the expected quantities
    if window is None:
        window = stride
    print(data.references.shape[0], f"{window=}, {stride=}")
    for atomID in range(data.references.shape[1]):
        for frame in range(window, data.references.shape[0], stride):
            expectedTmat[
                data.references[frame - window, atomID],
                data.references[frame, atomID],
            ] += 1

    assert tmat.shape[0] == len(data.legend)
    assert_array_equal(tmat, expectedTmat)

    assert tmatNorm.shape[0] == len(data.legend)
    assert_array_equal(tmatNorm, SOAPify.normalizeMatrixByRow(expectedTmat))


def test_residenceTimeBehaviourNoStateChanges():
    """The Residence time must pass this tests first:

    a non changing data should return a negative number
    equal to the total number of frames"""
    threeStates: list = ["state0", "state1", "state2"]

    for nFrames in range(4, 8):
        dataNC: SOAPclassification = SOAPclassification(
            [], numpy.array([[0]] * nFrames), threeStates
        )
        residenceTimes = SOAPify.calculateResidenceTimesFromClassification(
            dataNC, stride=1, window=1
        )
        print(*residenceTimes)
        for i, rt in enumerate(residenceTimes):
            if i == 0:
                assert len(rt) == 1
            else:
                assert len(rt) == 0
        assert residenceTimes[0][0] == -nFrames


def test_residenceTimeBehaviourAlternatingStates():
    """The Residence time must pass this tests first

    an alternating states should return a series of ones , with a -1 for
    starting and ending frame
    """
    threeStates: list = ["state0", "state1", "state2"]
    nFrames = 12
    data: SOAPclassification = SOAPclassification(
        [], numpy.array([[0], [1]] * (nFrames // 2)), threeStates
    )
    residenceTimes = SOAPify.calculateResidenceTimesFromClassification(
        data, stride=1, window=1
    )
    print(*[numpy.sum(numpy.abs(rt)) for rt in residenceTimes])
    print(*[numpy.abs(rt) for rt in residenceTimes])
    print(*residenceTimes)
    total = numpy.sum([numpy.sum(numpy.abs(rt)) for rt in residenceTimes])
    assert total == nFrames

    expectedTotals = [(nFrames // 2), (nFrames // 2), 0]
    for i, rt in enumerate(residenceTimes):
        assert_array_equal(numpy.abs(rt), [1] * expectedTotals[i])
        assert isSorted(rt)


def test_residenceTimeBehaviourAlternatingStatesOdd():
    """The Residence time must pass this tests first

    an alternating states should return a series of twos , with a -2 for
    starting and ending frame
    """
    threeStates: list = ["state0", "state1", "state2"]
    nFrames = 12
    data: SOAPclassification = SOAPclassification(
        [], numpy.array([[0], [1]] * (nFrames // 2) + [[0]]), threeStates
    )
    residenceTimes = SOAPify.calculateResidenceTimesFromClassification(
        data, stride=1, window=1
    )
    print(*[numpy.sum(numpy.abs(rt)) for rt in residenceTimes])
    print(*[numpy.abs(rt) for rt in residenceTimes])
    print(*residenceTimes)
    total = numpy.sum([numpy.sum(numpy.abs(rt)) for rt in residenceTimes])
    assert total == nFrames + 1

    expectedTotals = [(nFrames // 2) + 1, (nFrames // 2), 0]
    for i, rt in enumerate(residenceTimes):
        assert_array_equal(numpy.abs(rt), [1] * expectedTotals[i])
        assert isSorted(rt)


def test_residenceTimeBehaviourAlternatingStatesTwoInTwo():
    """The Residence time must pass this tests first

    an alternating states should return a series of ones , with a -1 for
    starting and ending frame
    """
    threeStates: list = ["state0", "state1", "state2"]
    nFrames = 12
    data: SOAPclassification = SOAPclassification(
        [], numpy.array([[0], [0], [1], [1]] * (nFrames // 4)), threeStates
    )
    residenceTimes = SOAPify.calculateResidenceTimesFromClassification(
        data, stride=1, window=1
    )
    print(*[numpy.sum(numpy.abs(rt)) for rt in residenceTimes])
    print(*[numpy.abs(rt) for rt in residenceTimes])
    print(*residenceTimes)
    total = numpy.sum([numpy.sum(numpy.abs(rt)) for rt in residenceTimes])
    assert total == nFrames

    for i, rt in enumerate(residenceTimes):
        if i == 2:
            assert_array_equal(rt, [])
        else:
            assert_array_equal(numpy.abs(rt), [2] * (nFrames // 4))
        assert isSorted(rt)


def _expectedTotalFrames(nFrames, window, stride):
    expectedTotal = 0
    for i in range(0, window, stride):
        tframes = nFrames - i
        expectedTotalToSum = tframes
        # if window is not a multiple of nFrames total we expect that the count will
        if tframes % window != 0:
            expectedTotalToSum = tframes // window + 1
            expectedTotalToSum *= window
        expectedTotal += expectedTotalToSum
    return expectedTotal


def test_residenceTimeBehaviourWindowAndTrajectory():
    """The Residence time must pass this tests first

    Th resulting time of the simulation will be different,
    and should be proportional to the given window
    """
    threeStates: list = ["state0", "state1", "state2"]
    nFrames = 12
    data: SOAPclassification = SOAPclassification(
        [], numpy.array([[0]] * (nFrames)), threeStates
    )
    for window in range(1, nFrames + 1):
        stride = window

        residenceTimes = SOAPify.calculateResidenceTimesFromClassification(
            data, stride=stride, window=window
        )
        # print(*[numpy.sum(numpy.abs(rt)) for rt in residenceTimes])
        # print(*[numpy.abs(rt) for rt in residenceTimes])
        print(*residenceTimes)
        total = numpy.sum([numpy.sum(numpy.abs(rt)) for rt in residenceTimes])
        assert total == _expectedTotalFrames(nFrames, window, stride)


def test_residenceTimeBehaviourWindowAndStrideAndTrajectory():
    """The Residence time must pass this tests first

    Th resulting time of the simulation will be different,
    and should be proportional to the given window
    """
    threeStates: list = ["state0", "state1", "state2"]
    nFrames = 12
    data: SOAPclassification = SOAPclassification(
        [], numpy.array([[0]] * (nFrames)), threeStates
    )
    for window in range(1, nFrames + 1):
        for stride in range(1, window + 1):
            residenceTimes = SOAPify.calculateResidenceTimesFromClassification(
                data, stride=stride, window=window
            )
            # print(*[numpy.sum(numpy.abs(rt)) for rt in residenceTimes])
            # print(*[numpy.abs(rt) for rt in residenceTimes])
            print(*residenceTimes)
            total = numpy.sum([numpy.sum(numpy.abs(rt)) for rt in residenceTimes])

            assert total == _expectedTotalFrames(nFrames, window, stride)


def test_residenceTime(
    input_mockedTrajectoryClassification, inputStridesWithNone, inputWindows
):
    data: SOAPclassification = input_mockedTrajectoryClassification
    stride = inputStridesWithNone
    window = inputWindows
    print(data.references.shape[0], f"{window=}, {stride=}")

    def isInvalidCombination(dataLen, stride, window):
        return (stride is not None and window < stride) or (
            (stride is not None and stride > dataLen) or window > dataLen
        )

    if isInvalidCombination(data.references.shape[0], stride, window):
        with pytest.raises(ValueError) as excinfo:
            SOAPify.calculateResidenceTimesFromClassification(
                data, stride=stride, window=window
            )
        if stride is not None and window < stride:
            assert "window must be bigger" in str(excinfo.value)
            pytest.skip("Exception thrown correctly")
        if (
            stride is not None and stride > data.references.shape[0]
        ) or window > data.references.shape[0]:
            assert "window must be smaller" in str(excinfo.value)
            pytest.skip("Exception thrown correctly")

    residenceTimes = SOAPify.calculateResidenceTimesFromClassification(
        data, stride=stride, window=window
    )
    # hand calculation of the expected quantities

    if stride is None:
        stride = window

    expectedResidenceTimes = [[] for i in range(len(data.legend))]
    # TOD decide better if start with time=0 and where to increment the frame count
    for iframe in range(0, window, stride):
        for atomID in range(data.references.shape[1]):
            prevState = data.references[iframe, atomID]
            first = True
            time = 0
            for frame in range(iframe + window, data.references.shape[0], window):
                state = data.references[frame, atomID]
                time += window
                if state != prevState:
                    expectedResidenceTimes[prevState].append(
                        time if not first else -time
                    )
                    first = False
                    time = 0
                    prevState = state

            # the last state does not have an out transition, appendig negative time to make it clear
            expectedResidenceTimes[prevState].append(-time - window)

    for i in range(len(expectedResidenceTimes)):
        expectedResidenceTimes[i] = numpy.sort(numpy.array(expectedResidenceTimes[i]))
    total = numpy.sum([numpy.sum(numpy.abs(rt)) for rt in residenceTimes])
    nFrames = data.references.shape[0]
    expectedTotal = _expectedTotalFrames(nFrames, window, stride)
    expectedTotal *= data.references.shape[1]
    assert total == expectedTotal

    for stateID in range(len(expectedResidenceTimes)):
        print(stateID, residenceTimes[stateID], expectedResidenceTimes[stateID])
        assert_array_equal(residenceTimes[stateID], expectedResidenceTimes[stateID])
        assert isSorted(residenceTimes[stateID])
