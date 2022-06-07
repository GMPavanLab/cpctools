import SOAPify
from SOAPify.Saponify import getSoapEngine
import numpy
from numpy.testing import assert_array_equal
import h5py
import MDAnalysis as mda
import HDF5er
from testSupport import getUniverseWithWaterMolecules
import pytest


@pytest.fixture(
    scope="module",
    params=[
        "dscribe",
        "quippy",
    ],
)
def engineKind_fixture(request):
    return request.param


def test_askEngine(engineKind_fixture):
    nMol = 1
    SOAPrcut = 10.0
    n_max = 4
    l_max = 4
    species = ["H", "O"]
    engine = getSoapEngine(
        species=species,
        SOAPrcut=SOAPrcut,
        SOAPnmax=n_max,
        SOAPlmax=l_max,
        SOAP_respectPBC=True,
        SOAPkwargs={},
        useSoapFrom=engineKind_fixture,
    )
    nsp = len(species)
    mixes = nsp * (nsp - 1) // 2
    fullmat = n_max * n_max * (l_max + 1)
    upperDiag = ((n_max + 1) * n_max) // 2 * (l_max + 1)
    assert engine.features == nsp * upperDiag + mixes * fullmat
    assert engine.nmax == n_max
    assert engine.lmax == l_max
    assert engine.rcut == SOAPrcut
    assert engine.species == species
    if engineKind_fixture == "dscribe":
        assert engine.get_location("H", "H") == slice(0, upperDiag)
        assert engine.get_location("H", "O") == slice(upperDiag, upperDiag + fullmat)
        assert engine.get_location("O", "H") == slice(
            upperDiag, upperDiag + fullmat
        )  # redundant
        assert engine.get_location("O", "O") == slice(
            upperDiag + fullmat, 2 * upperDiag + fullmat
        )
        # check if the engine is accessible
        assert engine.engine._nmax == n_max
