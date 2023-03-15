import SOAPify
import numpy
from numpy.testing import (
    assert_array_equal,
    assert_array_almost_equal,
    assert_almost_equal,
)
import h5py


def test_creatingReferencesFromTrajectoryAndSavingThem(
    getReferencesConfs, tmp_path_factory
):
    FramesRequest = {
        "ss_5f_ih": (0, 312),
        "b_5f_ih": (0, 1),
        "b_c_ih": (0, 0),
    }
    k = "ico923_6"
    with h5py.File(getReferencesConfs, "r") as workFile:
        nmax = workFile[f"SOAP/{k}"].attrs["n_max"]
        lmax = workFile[f"SOAP/{k}"].attrs["l_max"]
        references = SOAPify.createReferencesFromTrajectory(
            workFile[f"SOAP/{k}"], FramesRequest, nmax=nmax, lmax=lmax
        )
        referencesNN = SOAPify.createReferencesFromTrajectory(
            workFile[f"SOAP/{k}"],
            FramesRequest,
            nmax=nmax,
            lmax=lmax,
            doNormalize=False,
        )
        for key in FramesRequest:
            assert key in references.names
            assert key in referencesNN.names
            frame = FramesRequest[key][0]
            atomID = FramesRequest[key][1]
            where = references.names.index(key)
            whereNN = referencesNN.names.index(key)
            assert_array_almost_equal(
                references.spectra[where],
                SOAPify.normalizeArray(
                    SOAPify.fillSOAPVectorFromdscribe(
                        workFile[f"SOAP/{k}"][frame, atomID], lmax, nmax
                    )
                ),
            )
            assert_array_almost_equal(
                referencesNN.spectra[whereNN],
                SOAPify.fillSOAPVectorFromdscribe(
                    workFile[f"SOAP/{k}"][frame, atomID], lmax, nmax
                ),
            )

    for r in [references, referencesNN]:
        assert len(r) == len(FramesRequest)
        assert r.spectra.shape[0] == len(FramesRequest)
        assert r.lmax == lmax
        assert r.nmax == nmax


def test_savingAndLoadingReferences(tmp_path_factory, referencesTest):
    references, FramesRequest = referencesTest
    referenceSave = tmp_path_factory.mktemp("referencesNPs") / f"referencesTest.hdf5"

    with h5py.File(referenceSave, "w") as refFile:
        g = refFile.require_group("NPReferences")
        for k in references:
            nmax = references[k].nmax
            lmax = references[k].lmax
            SOAPify.saveReferences(g, k, references[k])
            assert len(g[k]) == len(references[k])
            assert g[k].attrs["lmax"] == lmax
            assert g[k].attrs["nmax"] == nmax
            names = list(g[k].attrs["names"])
            for n, n1 in zip(g[k].attrs["names"], references[k].names):
                assert n == n1
            for key in references[k].names:
                assert key in g[k].attrs["names"]
                whereRef = references[k].names.index(key)
                where = names.index(key)
                assert whereRef == where
                assert_array_almost_equal(
                    references[k].spectra[whereRef],
                    g[k][where],
                )
    with h5py.File(referenceSave, "r") as refFile:
        for k in references:
            print(k, f"NPReferences/{k}")
            refTest = SOAPify.getReferencesFromDataset(refFile[f"NPReferences/{k}"])
            assert_array_almost_equal(refTest.spectra, references[k].spectra)
            assert refTest.names == references[k].names
            assert refTest.nmax == references[k].nmax
            assert refTest.lmax == references[k].lmax


def test_distance(nMaxFixture, lMaxFixture):
    rng = numpy.random.default_rng(12345)
    dataSize = rng.integers(10, 100)
    spSize = max(dataSize // nMaxFixture, 1)
    dim = nMaxFixture * (lMaxFixture + 1)

    data = rng.random((dataSize, dim))
    spectra = rng.random((spSize, dim))

    def dCalc(x, y):
        return numpy.linalg.norm(x - y)

    distances = numpy.empty((dataSize, spSize))
    for i in range(dataSize):
        for j in range(spSize):
            distances[i, j] = dCalc(data[i], spectra[j])
    distancesCalculated = SOAPify.getDistanceBetween(data, spectra, dCalc)
    assert_array_almost_equal(distances, distancesCalculated)


def test_distanceFromRefs(getReferencesConfs, referencesTest):
    referenceDict, FramesRequest = referencesTest
    with h5py.File(getReferencesConfs, "r") as f:
        ds = f["SOAP/ico923_6"]
        nmax = ds.attrs["n_max"]
        lmax = ds.attrs["l_max"]
        ndists = len(referenceDict["ico923_6"])
        nat = ds.shape[1]
        data = SOAPify.normalizeArray(
            SOAPify.fillSOAPVectorFromdscribe(ds[:], lMax=lmax, nMax=nmax)
        )
        distancesCalculated = SOAPify.getDistanceBetween(
            data.reshape(-1, data.shape[-1]),
            referenceDict["ico923_6"].spectra,
            SOAPify.SOAPdistanceNormalized,
        )
        distances = SOAPify.getDistancesFromRef(
            ds,
            referenceDict["ico923_6"],
            SOAPify.SOAPdistanceNormalized,
            doNormalize=True,
        )
        distancesNormalized = SOAPify.getDistancesFromRefNormalized(
            ds,
            referenceDict["ico923_6"],
        )
        assert_array_almost_equal(distances, distancesNormalized)
        assert_array_almost_equal(
            distances, distancesCalculated.reshape(1, nat, ndists)
        )

        assert distances.shape == (
            ds.shape[0],
            ds.shape[1],
            len(referenceDict["ico923_6"]),
        )

        for name in FramesRequest["ico923_6"].keys():
            centerID = FramesRequest["ico923_6"][name][1]
            frameID = FramesRequest["ico923_6"][name][0]
            centerIDRefs = referenceDict["ico923_6"].names.index(name)
            assert_almost_equal(distances[frameID, centerID, centerIDRefs], 0.0)


def test_classifyShortcut(getReferencesConfs, referencesTest):
    referenceDict, _ = referencesTest
    k = "ico923_6"
    with h5py.File(getReferencesConfs, "r") as f:
        ds = f["SOAP/ico923_6"]
        nat = ds.shape[1]
        nmax = ds.attrs["n_max"]
        lmax = ds.attrs["l_max"]
        data = SOAPify.normalizeArray(
            SOAPify.fillSOAPVectorFromdscribe(ds[:], lMax=lmax, nMax=nmax)
        )
        distancesCalculated = SOAPify.getDistanceBetween(
            data.reshape(-1, data.shape[-1]),
            referenceDict[k].spectra,
            SOAPify.SOAPdistanceNormalized,
        )
        minimumDistID = numpy.argmin(distancesCalculated, axis=-1).reshape(-1, nat)
        minimumDist = numpy.amin(distancesCalculated, axis=-1).reshape(-1, nat)

        classification = SOAPify.applyClassification(
            ds, referenceDict[k], SOAPify.SOAPdistanceNormalized, doNormalize=True
        )
        assert_array_almost_equal(minimumDist, classification.distances)
        assert_array_equal(minimumDistID, classification.references)
