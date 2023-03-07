"""tests for SOAPify.transitions"""
from numpy.testing import assert_array_equal
import numpy
import pytest
import SOAPify
from SOAPify.classify import SOAPclassification


# ,inputWindows
def test_transitionMatrix(
    input_mockedTrajectoryClassification, inputStrides, inputWindows
):
    data: SOAPclassification = input_mockedTrajectoryClassification
    expectedTmat = numpy.zeros((len(data.legend), len(data.legend)))
    stride = inputStrides
    window = inputWindows
    print(data.references.shape[0], window, stride)
    if window is not None and window < stride:
        with pytest.raises(ValueError) as excinfo:
            SOAPify.transitionMatrixFromSOAPClassification(
                data, stride=stride, window=window
            )
            assert "window must be bigger" in str(excinfo.value)
        return
    if (
        window is not None and window > data.references.shape[0]
    ) or stride > data.references.shape[0]:
        with pytest.raises(ValueError):
            SOAPify.transitionMatrixFromSOAPClassification(
                data, stride=stride, window=window
            )
            assert "window must be smaller" in str(excinfo.value)
        return

    if window is None:
        window = stride
    print(data.references.shape[0], f"{window=}, {stride=}")
    for atomID in range(data.references.shape[1]):
        for frame in range(window, data.references.shape[0], stride):
            expectedTmat[
                data.references[frame - window, atomID],
                data.references[frame, atomID],
            ] += 1

    tmat = SOAPify.transitionMatrixFromSOAPClassification(
        data, stride=stride, window=window
    )
    assert tmat.shape[0] == len(data.legend)
    assert_array_equal(tmat, expectedTmat)

    tmatNorm = SOAPify.transitionMatrixFromSOAPClassificationNormalized(
        data, stride=stride, window=window
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
