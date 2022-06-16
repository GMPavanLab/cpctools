from operator import invert
import SOAPify
from SOAPify.Saponify import getSoapEngine
import numpy
from numpy.testing import assert_array_equal
from ase.data import atomic_numbers
from testSupport import engineKind_fixture, species_fixture, nMaxFixture, lMaxFixture


def test_askEngine(engineKind_fixture, species_fixture, nMaxFixture, lMaxFixture):
    nMol = 1
    SOAPrcut = 10.0
    n_max = nMaxFixture
    l_max = lMaxFixture
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
        keys = list(engine._slices.keys())
        for i in range(len(species)):
            for j in range(i, len(species)):
                key = species[i] + species[j]
                assert key in keys
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


def test_noCrossoverSOAPMAPPINGDscribe(species_fixture, nMaxFixture, lMaxFixture):
    nmax = nMaxFixture
    lmax = lMaxFixture
    species = SOAPify.orderByZ(species_fixture)
    nsp = len(species)
    # check if the support array are created correctly
    pdscribe = SOAPify.getdscribeSOAPMapping(lmax, nmax, species, crossover=False)
    assert len(pdscribe) == nsp * (lmax + 1) * (nmax * (nmax + 1)) // 2
    assert len(pdscribe) == len(numpy.unique(pdscribe))
    i = 0
    for Z in species:
        for Zp in species:
            if Z != Zp:
                continue
            for l in range(lmax + 1):
                for n in range(nmax):
                    for np in range(nmax):
                        if (np, atomic_numbers[Zp]) >= (n, atomic_numbers[Z]):
                            if atomic_numbers[Z] >= atomic_numbers[Zp]:
                                assert pdscribe[i] == (f"{l}_{Z}{n}_{Zp}{np}")
                            else:
                                assert pdscribe[i] == (f"{l}_{Zp}{np}_{Z}{n}")
                            i += 1


def test_reorderQuippyLikeDscribe(species_fixture, nMaxFixture, lMaxFixture):
    nmax = nMaxFixture
    lmax = lMaxFixture
    species = SOAPify.orderByZ(species_fixture)
    nsp = len(species)
    # chech if the support array are created correctly
    pdscribe = SOAPify.getdscribeSOAPMapping(lmax, nmax, species)
    assert len(pdscribe) == (lmax + 1) * ((nmax * nsp) * (nmax * nsp + 1)) // 2
    assert len(pdscribe) == len(numpy.unique(pdscribe))
    pquippy = SOAPify.getquippySOAPMapping(lmax, nmax, species)
    assert len(pquippy) == (lmax + 1) * ((nmax * nsp) * (nmax * nsp + 1)) // 2
    assert len(pquippy) == len(numpy.unique(pquippy))
    # pquippy and pdscribe must contain the same elements
    assert len(pquippy) == len(pdscribe)
    assert numpy.isin(pdscribe, pquippy).all()
    i = 0
    for Z in species:
        for Zp in species:
            for l in range(lmax + 1):
                for n in range(nmax):
                    for np in range(nmax):
                        if (np, atomic_numbers[Zp]) >= (n, atomic_numbers[Z]):
                            if atomic_numbers[Z] >= atomic_numbers[Zp]:
                                assert pdscribe[i] == (f"{l}_{Z}{n}_{Zp}{np}")
                            else:
                                assert pdscribe[i] == (f"{l}_{Zp}{np}_{Z}{n}")
                            i += 1

    rs_index = SOAPify.utils._getRSindex(nmax, species)
    assert rs_index.shape == (2, len(species) * nmax)
    i = 0
    for ia in range(len(species) * nmax):
        na = rs_index[0, ia]
        i_species = species[rs_index[1, ia]]
        for jb in range(ia + 1):  # ia is  in the range
            nb = rs_index[0, jb]
            j_species = species[rs_index[1, jb]]
            # if(this%diagonal_radial .and. a /= b) cycle
            for l in range(lmax + 1):
                if atomic_numbers[j_species] >= atomic_numbers[i_species]:
                    assert pquippy[i] == f"{l}_{j_species}{nb}_{i_species}{na}"
                else:
                    assert pquippy[i] == f"{l}_{i_species}{na}_{j_species}{nb}"
                i += 1

    reorderIdexes = SOAPify.getAddressesQuippyLikeDscribe(lmax, nmax, species)
    assert len(reorderIdexes) == len(pdscribe)
    assert len(reorderIdexes) == len(numpy.unique(reorderIdexes))

    t = numpy.array(pquippy)[reorderIdexes]
    assert_array_equal(t, pdscribe)
