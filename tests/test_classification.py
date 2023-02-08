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
