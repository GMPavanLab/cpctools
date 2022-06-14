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


@pytest.fixture(
    scope="module",
    params=[
        ["C", "O", "H", "N"],
        ["H", "O"],
        ["H"],
    ],
)
def species_fixture(request):
    return request.param


def test_askEngine(engineKind_fixture, species_fixture):
    nMol = 1
    SOAPrcut = 10.0
    n_max = 4
    l_max = 4
    species = SOAPify.orderByZ(species_fixture)
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
        # check if the engine is accessible
        assert engine.engine._nmax == n_max
    if engineKind_fixture == "quippy":
        keys = list(engine._addresses.keys())
        for i in range(len(species)):
            for j in range(i, len(species)):
                key = species[i] + species[j]
                assert key in keys
                assert len(engine._addresses[key]) == upperDiag if i == j else fullmat
                assert len(engine._addresses[key]) == len(
                    numpy.unique(engine._addresses[key])
                )

        assert len(keys) == nsp + mixes
        for i in range(len(keys)):
            for j in range(i, len(keys)):
                # check if the engine that the addresses id are not repeated
                assert numpy.isin(
                    engine._addresses[keys[i]],
                    engine._addresses[keys[j]],
                    invert=i != j,
                ).all()

    prev = 0
    next = 0
    # the engine wrapper must return the same slice for every engine
    for i in range(nsp):
        for j in range(i, nsp):
            if i == j:
                next = prev + upperDiag
            else:
                next = prev + fullmat
            assert engine.get_location(species[i], species[j]) == slice(prev, next)
            prev = next
