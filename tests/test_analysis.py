"""tests for analysis"""
import numpy
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
import SOAPify.analysis as analysis
import SOAPify
import h5py
import MDAnalysis
from .testSupport import is_sorted, fewFrameUniverse

# NB: the trueFalse fixture for the backward settings is here to anticipate the
# implementation of that feature


@pytest.mark.parametrize("window", [1, 2, 5, 7, 10])
@pytest.mark.parametrize("backward", [True, False])
def test_timeSOAP(referencesTrajectorySOAP, window, backward):
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
            analysis.timeSOAP(fillArgs["soapFromdscribe"], stride=stride, window=window)
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
            # TODO:
            # if backward:
            # else:
            expectedTimedSOAP[
                frame - window, molecule
            ] = distance  # fill the matrix (each molecule for each frame)

    expectedDeltaTimedSOAP = []
    for molecule in range(0, expectedTimedSOAP.shape[1]):
        derivative = numpy.diff(expectedTimedSOAP[:, molecule])
        expectedDeltaTimedSOAP.append(derivative)

    timedSOAP, deltaTimedSOAP = analysis.timeSOAP(
        SOAPTraj, stride=stride, window=window, backward=backward
    )
    # print(timedSOAP, expectedTimedSOAP)
    print(deltaTimedSOAP.shape, timedSOAP.shape)
    print(
        numpy.asarray(expectedDeltaTimedSOAP).shape,
        numpy.asarray(expectedTimedSOAP).shape,
    )
    assert_array_almost_equal(timedSOAP, expectedTimedSOAP)

    assert_array_almost_equal(deltaTimedSOAP, expectedDeltaTimedSOAP)


@pytest.mark.parametrize("window", [1, 2, 5, 7, 10])
@pytest.mark.parametrize("backward", [True, False])
def test_timeSOAPsimple(referencesTrajectorySOAP, window, backward):
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
            analysis.timeSOAP(fillArgs["soapFromdscribe"], stride=stride, window=window)
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

    expectedTimedSOAP, expectedDeltaTimedSOAP = analysis.timeSOAP(
        SOAPTraj, stride=stride, window=window, backward=backward
    )
    timedSOAP, deltaTimedSOAP = analysis.timeSOAPsimple(
        SOAPTraj, stride=stride, window=window, backward=backward
    )
    print(timedSOAP, expectedTimedSOAP)
    assert_array_almost_equal(timedSOAP, expectedTimedSOAP)

    assert_array_almost_equal(deltaTimedSOAP, expectedDeltaTimedSOAP)


@pytest.mark.parametrize("window", [1, 2, 5, 7, 10])
@pytest.mark.parametrize("backward", [True, False])
def test_getTimeSOAPsimple(referencesTrajectorySOAP, window, backward):
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
            analysis.timeSOAP(fillArgs["soapFromdscribe"], stride=stride, window=window)
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

    expectedTimedSOAP, expectedDeltaTimedSOAP = analysis.timeSOAPsimple(
        SOAPTraj, stride=stride, window=window, backward=backward
    )
    with h5py.File(confFile, "r") as f:
        timedSOAP, deltaTimedSOAP = analysis.getTimeSOAPSimple(
            f[f"/SOAP/{groupName}"], stride=stride, window=window, backward=backward
        )
    print(timedSOAP, expectedTimedSOAP)
    assert_array_almost_equal(timedSOAP, expectedTimedSOAP)

    assert_array_almost_equal(deltaTimedSOAP, expectedDeltaTimedSOAP)


@pytest.fixture(
    scope="module",
    params=[1, 2],
)
def input1_2(request):
    return request.param


def test_countNeighboursForLENS(hdf5_file, input1_2):
    # this is the original version by Martina Crippa

    inputUniverse: MDAnalysis.Universe = hdf5_file[1]
    wantedSlice = slice(0, len(inputUniverse.trajectory) // input1_2, 1)
    COFF = 10.0
    INIT = wantedSlice.start
    END = wantedSlice.stop
    STRIDE = wantedSlice.step

    beads = inputUniverse.select_atoms("all")
    cont_list = list()
    # loop over traj
    for frame, ts in enumerate(inputUniverse.trajectory[INIT:END:STRIDE]):
        nsearch = MDAnalysis.lib.NeighborSearch.AtomNeighborSearch(
            beads, box=inputUniverse.dimensions
        )
        cont_list.append([nsearch.search(i, COFF, level="A") for i in beads])
    for selection in [inputUniverse, beads]:
        myNNlistPerFrame = analysis.listNeighboursAlongTrajectory(
            selection, cutOff=COFF, trajSlice=wantedSlice
        )

        assert len(myNNlistPerFrame) == len(cont_list)
        for NNlistOrig, myNNList in zip(cont_list, myNNlistPerFrame):
            assert len(NNlistOrig) == len(myNNList)
            for atomGroupNN, myatomsID in zip(NNlistOrig, myNNList):
                atomsID = numpy.sort([at.ix for at in atomGroupNN])
                assert is_sorted(myatomsID)
                assert_array_equal(atomsID, myatomsID)


def lensIsZeroFixtures():
    # no change in NN
    return (
        fewFrameUniverse(
            trajectory=[
                [[0, 0, 0], [0, 0, 1], [5, 5, 5], [5, 5, 6]],
                [[0, 0, 0], [0, 0, 1], [5, 5, 5], [5, 5, 6]],
            ],
            dimensions=[10, 10, 10, 90, 90, 90],
        ),
        [0] * 4,
    )


def lensIsZeroNNFixtures():
    # Zero NN
    return (
        fewFrameUniverse(
            trajectory=[
                [[0, 0, 0], [5, 5, 5]],
                [[0, 0, 0], [5, 5, 5]],
            ],
            dimensions=[10, 10, 10, 90, 90, 90],
        ),
        [0] * 2,
    )


def lensIsOneFixtures():
    # all NN changes
    return (
        fewFrameUniverse(
            trajectory=[
                [[0, 0, 0], [0, 0, 1], [5, 5, 5], [0, 1, 0]],
                [[0, 0, 0], [5, 5, 6], [5, 5, 5], [5, 6, 5]],
            ],
            dimensions=[10, 10, 10, 90, 90, 90],
        ),
        [1] * 4,
    )


getUNI = {
    "LENSISZERO": lensIsZeroFixtures(),
    "LENSISZERONN": lensIsZeroNNFixtures(),
    "LENSISONE": lensIsOneFixtures(),
}


@pytest.fixture(scope="module", params=["LENSISZERO", "LENSISZERONN", "LENSISONE"])
def lensFixtures(request):
    return getUNI[request.param]


def test_specialLENS(lensFixtures):
    expected = lensFixtures[1]
    universe = lensFixtures[0]
    COFF = 1.1
    nnListPerFrame = analysis.listNeighboursAlongTrajectory(universe, cutOff=COFF)
    myncontTot, mynnTot, mynumTot, mydenTot = analysis.neighbourChangeInTime(
        nnListPerFrame
    )

    assert_array_equal(myncontTot[:, 0], [0] * myncontTot.shape[0])
    assert_array_equal(myncontTot[:, 1], expected)

    for frame in [0, 1]:
        for atom in universe.atoms:
            atomId = atom.ix
            assert mynnTot[atomId, frame] == len(nnListPerFrame[frame][atomId]) - 1
    for frame in [1]:
        for atom in universe.atoms:
            atomId = atom.ix

            assert (
                mydenTot[atomId, frame]
                == len(nnListPerFrame[frame][atomId])
                + len(nnListPerFrame[frame - 1][atomId])
                - 2
            )
            assert (
                mynumTot[atomId, frame]
                == numpy.setxor1d(
                    nnListPerFrame[frame][atomId], nnListPerFrame[frame - 1][atomId]
                ).shape[0]
            )


def test_emulateLENS(hdf5_file, input1_2):
    inputUniverse: MDAnalysis.Universe = hdf5_file[1]
    wantedSlice = slice(0, len(inputUniverse.trajectory) // input1_2, 1)
    COFF = 4.0
    nnListPerFrame = analysis.listNeighboursAlongTrajectory(
        inputUniverse, cutOff=COFF, trajSlice=wantedSlice
    )
    # this is the original version by Martina Crippa
    # def local_dynamics(list_sum):
    particle = [i for i in range(numpy.shape(nnListPerFrame)[1])]
    ncont_tot = list()
    nn_tot = list()
    num_tot = list()
    den_tot = list()
    for p in particle:
        ncont = list()
        nn = list()
        num = list()
        den = list()
        for frame in range(len(nnListPerFrame)):
            if frame == 0:
                ncont.append(0)
                # modifications by Daniele:
                # needed to give the nn counts on the first nn
                nn.append(len(nnListPerFrame[frame][p]) - 1)
                # needed to give same lenght the all on the lists
                num.append(0)
                den.append(0)
                # END modification
                # ORIGINAL:nn.append(0)
            else:
                # if the nn set chacne totally set LENS to 1: the nn list contains
                # the atom, hence the  various ==1 and -1

                # se il set di primi vicini cambia totalmente, l'intersezione è lunga 1 ovvero la bead self
                # vale anche se il numero di primi vicini prima e dopo cambia
                if (
                    len(
                        list(
                            set(nnListPerFrame[frame - 1][p])
                            & set(nnListPerFrame[frame][p])
                        )
                    )
                    == 1
                ):
                    # se non ho NN lens è 0
                    if (
                        len(list(set(nnListPerFrame[frame - 1][p]))) == 1
                        and len(set(nnListPerFrame[frame][p])) == 1
                    ):
                        ncont.append(0)
                        nn.append(0)
                        num.append(0)
                        den.append(0)
                    # se ho NN lo metto 1
                    else:
                        ncont.append(1)
                        nn.append(len(nnListPerFrame[frame][p]) - 1)
                        # changed by daniele
                        # needed to make num/den=1
                        num.append(
                            len(nnListPerFrame[frame - 1][p])
                            - 1
                            + len(nnListPerFrame[frame][p])
                            - 1
                        )
                        # END modification
                        # ORGINAL: num.append(1)
                        den.append(
                            len(nnListPerFrame[frame - 1][p])
                            - 1
                            + len(nnListPerFrame[frame][p])
                            - 1
                        )
                else:
                    # contrario dell'intersezione fra vicini al frame f-1 e al frame f
                    c_diff = set(nnListPerFrame[frame - 1][p]).symmetric_difference(
                        set(nnListPerFrame[frame][p])
                    )
                    ncont.append(
                        len(c_diff)
                        / (
                            len(nnListPerFrame[frame - 1][p])
                            - 1
                            + len(nnListPerFrame[frame][p])
                            - 1
                        )
                    )
                    nn.append(len(nnListPerFrame[frame][p]) - 1)
                    num.append(len(c_diff))
                    den.append(
                        len(nnListPerFrame[frame - 1][p])
                        - 1
                        + len(nnListPerFrame[frame][p])
                        - 1
                    )
        num_tot.append(num)
        den_tot.append(den)
        ncont_tot.append(ncont)
        nn_tot.append(nn)
    # return ncont_tot, nn_tot, num_tot, den_tot
    myncontTot, mynnTot, mynumTot, mydenTot = analysis.neighbourChangeInTime(
        nnListPerFrame
    )

    assert len(myncontTot) == len(ncont_tot)
    assert len(mynnTot) == len(nn_tot)
    assert len(mynumTot) == len(num_tot)
    assert len(mydenTot) == len(den_tot)

    # lens Value
    for atomData, wantedAtomData in zip(myncontTot, ncont_tot):
        assert_array_almost_equal(atomData, wantedAtomData)
    # NN count
    for atomData, wantedAtomData in zip(mynnTot, nn_tot):
        assert_array_almost_equal(atomData, wantedAtomData)
    # LENS numerator
    for atomData, wantedAtomData in zip(mynumTot, num_tot):
        assert_array_almost_equal(atomData, wantedAtomData)
    # LENS denominator
    for atomData, wantedAtomData in zip(mydenTot, den_tot):
        assert_array_almost_equal(atomData, wantedAtomData)
