import pytest
import SOAPify
import numpy
from numpy.testing import assert_array_equal
import h5py
import HDF5er
from .testSupport import getUniverseWithWaterMolecules


# def test_test(referencesIco923NP):
#    pass


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
            numpy.testing.assert_array_almost_equal(
                references.spectra[where],
                SOAPify.normalizeArray(
                    SOAPify.fillSOAPVectorFromdscribe(
                        workFile[f"SOAP/{k}"][frame, atomID], lmax, nmax
                    )
                ),
            )
            numpy.testing.assert_array_almost_equal(
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
    referenceDict = tmp_path_factory.mktemp("referencesNPs") / f"referencesTest.hdf5"
    with h5py.File(referenceDict, "w") as refFile:
        g = refFile.require_group("NPReferences")
        SOAPify.saveReferences(g, k, references)
        assert len(g[k]) == len(references)
        assert g[k].attrs["lmax"] == lmax
        assert g[k].attrs["nmax"] == nmax
        names = list(g[k].attrs["names"])
        for n, n1 in zip(g[k].attrs["names"], references.names):
            assert n == n1
        for key in references.names:
            assert key in g[k].attrs["names"]
            whereRef = references.names.index(key)
            where = names.index(key)
            assert whereRef == where
            numpy.testing.assert_array_almost_equal(
                references.spectra[whereRef],
                g[k][where],
            )


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
    numpy.testing.assert_array_almost_equal(distances, distancesCalculated)


def test_distanceFromRefs(getReferencesConfs, referencesTest):
    referenceDict, FramesRequest = referencesTest
    with h5py.File(getReferencesConfs, "r") as f:
        ds = f["SOAP/ico923_6"]
        nmax = ds.attrs["n_max"]
        lmax = ds.attrs["l_max"]
        ndists = len(referenceDict["ico923_6"])
        nat = ds.shape[1]
        data = SOAPify.normalizeArray(
            SOAPify.fillSOAPVectorFromdscribe(ds[:], l_max=lmax, n_max=nmax)
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
        numpy.testing.assert_array_almost_equal(
            distances, distancesCalculated.reshape(1, nat, ndists)
        )
        centerID = FramesRequest["ico923_6"]["b_c_ih"][1]
        centerIDRefs = referenceDict["ico923_6"].names.index("b_c_ih")
        assert distances.shape == (
            ds.shape[0],
            ds.shape[1],
            len(referenceDict["ico923_6"]),
        )
        # assert numpy.min(distances) == 0
        print(referenceDict["ico923_6"].names)
        print(centerID, centerIDRefs)
        print(distances[0, 0, 11])
        myid = numpy.argmin(distances[0, :, 11], axis=-1)
        print(myid, numpy.min(distances))
        print(
            SOAPify.SOAPdistanceNormalized(
                referenceDict["ico923_6"].spectra[11],
                data[0, 0],
            ),
            numpy.dot(referenceDict["ico923_6"].spectra[11], data[0, 0]),
            distances.dtype,
            data.dtype,
            ds.dtype,
        )
        assert distances[0, centerID, centerIDRefs] == 0
