"""tests for analysis"""
import numpy
from numpy.testing import assert_array_almost_equal
import pytest
import SOAPify.analysis as analysis
import SOAPify
import h5py
import MDAnalysis


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


def test_tempoSOAPsimple(referencesTrajectorySOAP, inputWindows):
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

    expectedTimedSOAP, expectedDeltaTimedSOAP = analysis.tempoSOAP(
        SOAPTraj, stride=stride, window=window
    )
    timedSOAP, deltaTimedSOAP = analysis.tempoSOAPsimple(
        SOAPTraj, stride=stride, window=window
    )
    print(timedSOAP, expectedTimedSOAP)
    assert_array_almost_equal(timedSOAP, expectedTimedSOAP)

    assert_array_almost_equal(deltaTimedSOAP, expectedDeltaTimedSOAP)


def test_countNeighboursForLENS(hdf5_file):
    # this is the original version by Martina Crippa
    t = slice(0, 1001, 1)
    XYZ_UNI = hdf5_file[1]
    COFF = 16.0
    INIT = t.start
    END = t.stop
    STRIDE = t.step

    beads = XYZ_UNI.select_atoms("all")
    cont_list = list()
    # loop over traj
    for frame, ts in enumerate(XYZ_UNI.trajectory[INIT:END:STRIDE]):
        nsearch = MDAnalysis.lib.NeighborSearch.AtomNeighborSearch(
            beads, box=XYZ_UNI.dimensions
        )
        cont_list.append([nsearch.search(i, COFF, level="A") for i in beads])

    mylist_sum = analysis.listNeighboursAlongTrajectory()

    for NNlistOrig, myNNList in zip(cont_list, mylist_sum):
        assert len(NNlistOrig) == len(myNNList)
        for atomGroupNN, myAtomGroup in zip(NNlistOrig, myNNList):
            atomsID = numpy.sort([at.ix for at in atomGroupNN])
            myatomsID = numpy.sort([at.ix for at in myAtomGroup])
            assert_array_almost_equal(atomsID, myatomsID)


def test_emulateLENS():
    list_sum = analysis.listNeighboursAlongTrajectory()
    # this is the original version by Martina Crippa
    # def local_dynamics(list_sum):
    particle = [i for i in range(numpy.shape(list_sum)[1])]
    ncont_tot = list()
    nn_tot = list()
    num_tot = list()
    den_tot = list()
    for p in particle:
        ncont = list()
        nn = list()
        num = list()
        den = list()
        for frame in range(len(list_sum)):
            if frame == 0:
                ncont.append(0)
                nn.append(0)
            else:
                # se il set di primi vicini cambia totalmente, l'intersezione è lunga 1 ovvero la bead self
                # vale anche se il numero di primi vicini prima e dopo cambia
                if (
                    len(list(set(list_sum[frame - 1][p]) & set(list_sum[frame][p])))
                    == 1
                ):
                    # se non ho NN lens è 0
                    if (
                        len(list(set(list_sum[frame - 1][p]))) == 1
                        and len(set(list_sum[frame][p])) == 1
                    ):
                        ncont.append(0)
                        nn.append(0)
                        num.append(0)
                        den.append(0)
                    # se ho NN lo metto 1
                    else:
                        ncont.append(1)
                        nn.append(len(list_sum[frame][p]) - 1)
                        num.append(1)
                        den.append(
                            len(list_sum[frame - 1][p])
                            - 1
                            + len(list_sum[frame][p])
                            - 1
                        )
                else:
                    # contrario dell'intersezione fra vicini al frame f-1 e al frame f
                    c_diff = set(list_sum[frame - 1][p]).symmetric_difference(
                        set(list_sum[frame][p])
                    )
                    ncont.append(
                        len(c_diff)
                        / (
                            len(list_sum[frame - 1][p])
                            - 1
                            + len(list_sum[frame][p])
                            - 1
                        )
                    )
                    nn.append(len(list_sum[frame][p]) - 1)
                    num.append(len(c_diff))
                    den.append(
                        len(list_sum[frame - 1][p]) - 1 + len(list_sum[frame][p]) - 1
                    )
        num_tot.append(num)
        den_tot.append(den)
        ncont_tot.append(ncont)
        nn_tot.append(nn)
    # return ncont_tot, nn_tot, num_tot, den_tot
