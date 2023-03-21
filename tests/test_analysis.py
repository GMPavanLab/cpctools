"""tests for analysis"""
import numpy
from numpy.testing import assert_array_almost_equal
import pytest
import SOAPify.analysis as analysis
import SOAPify
import h5py


def test_tempoSOAP(referencesTrajectorySOAP, inputWindows):
    window = inputWindows
    stride = window
    confFile, groupName = referencesTrajectorySOAP

    with h5py.File(confFile, "r") as f:
        t = f[f"/SOAP/{groupName}"]
        fillArgs = {
            "soapFromdscribe": t[:],
            "lMax": t.attrs["l_max"],
            "nMax": t.attrs["n_max"],
        }
        fillArgs["atomTypes"], fillArgs["atomicSlices"] = SOAPify.getSlicesFromAttrs(
            t.attrs
        )

    def isInvalidCombination(dataLen, stride, window):
        return (stride is not None and window < stride) or (
            (stride is not None and stride >= dataLen) or window >= dataLen
        )

    if isInvalidCombination(fillArgs["soapFromdscribe"].shape[0], stride, window):
        with pytest.raises(ValueError) as excinfo:
            analysis.tempoSOAP(
                fillArgs["soapFromdscribe"], stride=stride, window=window
            )
        if stride is not None and window < stride:
            assert "window must be bigger" in str(excinfo.value)
            pytest.skip("Exception thrown correctly")
        if (
            stride is not None and stride >= fillArgs["soapFromdscribe"].shape[0]
        ) or window >= fillArgs["soapFromdscribe"].shape[0]:
            assert "window must be smaller" in str(excinfo.value)
            pytest.skip("Exception thrown correctly")

    SOAPTraj = SOAPify.fillSOAPVectorFromdscribe(**fillArgs)

    SOAPTraj = SOAPify.normalizeArray(SOAPTraj)

    # 2. Get tSOAP (SOAP distance from frame t+1 and frame t-1)

    # 3. DERIVATA IN AVANTI
    expectedTimedSOAP = numpy.zeros((SOAPTraj.shape[0] - window, SOAPTraj.shape[1]))

    for frame in range(window, SOAPTraj.shape[0]):
        for molecule in range(0, SOAPTraj.shape[1]):
            x = SOAPTraj[frame, molecule, :]
            y = SOAPTraj[frame - window, molecule, :]
            distance = SOAPify.simpleSOAPdistance(x, y)
            expectedTimedSOAP[
                frame - window, molecule
            ] = distance  # fill the matrix (each molecule for each frame)

    expectedDeltaTimedSOAP = []
    for molecule in range(0, expectedTimedSOAP.shape[1]):
        derivative = numpy.diff(expectedTimedSOAP[:, molecule])
        expectedDeltaTimedSOAP.append(derivative)

    timedSOAP, deltaTimedSOAP = analysis.tempoSOAP(
        SOAPTraj, stride=stride, window=window
    )
    print(timedSOAP, expectedTimedSOAP)
    assert_array_almost_equal(timedSOAP, expectedTimedSOAP)

    assert_array_almost_equal(deltaTimedSOAP, expectedDeltaTimedSOAP)
