import SOAPify
import numpy
from numpy.testing import assert_array_equal
import h5py
import MDAnalysis as mda
import HDF5er
from testSupport import getUniverseWithWaterMolecules


def test_MultiAtomicSoapify():
    nMol = 27
    u = getUniverseWithWaterMolecules(nMol)
    HDF5er.MDA2HDF5(u, "testH2O.hdf5", "testH2O", override=True)
    n_max = 4
    l_max = 4
    rcut = 10.0
    with h5py.File("testH2O.hdf5", "a") as f:
        soapGroup = f.require_group("SOAP")
        trajGroup = f["Trajectories/testH2O"]
        SOAPify.saponify(
            trajGroup, soapGroup, rcut, n_max, l_max, useSoapFrom="dscribe"
        )
        assert soapGroup["testH2O"].attrs["n_max"] == n_max
        assert soapGroup["testH2O"].attrs["l_max"] == l_max
        assert "O" in soapGroup["testH2O"].attrs["species"]
        assert "H" in soapGroup["testH2O"].attrs["species"]
        assert numpy.abs(soapGroup["testH2O"].attrs["r_cut"] - rcut) < 1e-8
        assert "centersIndexes" not in soapGroup["testH2O"].attrs
        assert (
            soapGroup[f"testH2O"].shape[-1]
            == (1 + l_max) * n_max * n_max
            + 2 * (1 + l_max) * ((n_max + 1) * n_max) // 2
        )

        soapGroup = f.require_group("SOAPOxygen")
        SOAPify.saponifyGroup(
            f["Trajectories"],
            soapGroup,
            10.0,
            n_max,
            l_max,
            SOAPatomMask=["O"],
            useSoapFrom="dscribe",
        )
        assert soapGroup["testH2O"].attrs["n_max"] == n_max
        assert soapGroup["testH2O"].attrs["l_max"] == l_max
        assert "O" in soapGroup["testH2O"].attrs["species"]
        assert "H" in soapGroup["testH2O"].attrs["species"]
        assert numpy.abs(soapGroup["testH2O"].attrs["r_cut"] - rcut) < 1e-8
        assert_array_equal(
            soapGroup["testH2O"].attrs["centersIndexes"], [i * 3 for i in range(nMol)]
        )


def test_MultiAtomicSoap():
    nMol = 27
    u = getUniverseWithWaterMolecules(nMol)
    HDF5er.MDA2HDF5(u, "testH2O.hdf5", "testH2O", override=True)
    n_max = 4
    l_max = 4
    rcut = 10.0
    with h5py.File("testH2O.hdf5", "a") as f:
        soapGroup = f.require_group("SOAP")
        SOAPify.saponifyGroup(
            f["Trajectories"], soapGroup, rcut, n_max, l_max, useSoapFrom="dscribe"
        )
        assert soapGroup["testH2O"].attrs["n_max"] == n_max
        assert soapGroup["testH2O"].attrs["l_max"] == l_max
        assert "O" in soapGroup["testH2O"].attrs["species"]
        assert "H" in soapGroup["testH2O"].attrs["species"]
        assert numpy.abs(soapGroup["testH2O"].attrs["r_cut"] - rcut) < 1e-8
        assert "centersIndexes" not in soapGroup["testH2O"].attrs
        assert (
            soapGroup[f"testH2O"].shape[-1]
            == (1 + l_max) * n_max * n_max
            + 2 * (1 + l_max) * ((n_max + 1) * n_max) // 2
        )

        soapGroup = f.require_group("SOAPOxygen")
        SOAPify.saponifyGroup(
            f["Trajectories"],
            soapGroup,
            10.0,
            n_max,
            l_max,
            SOAPatomMask=["O"],
            useSoapFrom="dscribe",
        )
        assert soapGroup["testH2O"].attrs["n_max"] == n_max
        assert soapGroup["testH2O"].attrs["l_max"] == l_max
        assert "O" in soapGroup["testH2O"].attrs["species"]
        assert "H" in soapGroup["testH2O"].attrs["species"]
        assert numpy.abs(soapGroup["testH2O"].attrs["r_cut"] - rcut) < 1e-8
        assert_array_equal(
            soapGroup["testH2O"].attrs["centersIndexes"], [i * 3 for i in range(nMol)]
        )


def test_slicesNo():
    nMol = 1
    u = getUniverseWithWaterMolecules(nMol)
    HDF5er.MDA2HDF5(u, "testH2O.hdf5", "testH2O", override=True)
    n_max = 4
    l_max = 4
    upperDiag = (l_max + 1) * ((n_max) * (n_max + 1)) // 2
    fullmat = n_max * n_max * (l_max + 1)
    rcut = 10.0
    with h5py.File("testH2O.hdf5", "a") as f:
        soapGroup = f.require_group("SOAP")
        SOAPify.saponifyGroup(
            f["Trajectories"], soapGroup, rcut, n_max, l_max, useSoapFrom="dscribe"
        )
        species, slices = SOAPify.getSlicesFromAttrs(f["SOAP/testH2O"].attrs)
        assert "O" in species
        assert "H" in species
        assert slices["H" + "H"] == slice(0, upperDiag)
        assert slices["H" + "O"] == slice(upperDiag, upperDiag + fullmat)
        assert slices["O" + "H"] == slice(upperDiag, upperDiag + fullmat)  # redundant
        assert slices["O" + "O"] == slice(upperDiag + fullmat, 2 * upperDiag + fullmat)
        fullSpectrum = SOAPify.fillSOAPVectorFromdscribe(
            f["SOAP/testH2O"][:], l_max, n_max, species, slices
        )
        assert fullSpectrum.shape[-1] == 3 * fullmat


def test_MultiAtomicSoapkwargs():
    nMol = 27
    u = getUniverseWithWaterMolecules(nMol)
    HDF5er.MDA2HDF5(u, "testH2O.hdf5", "testH2O", override=True)
    n_max = 4
    l_max = 4
    rcut = 10.0
    upperDiag = (l_max + 1) * ((n_max) * (n_max + 1)) // 2
    fullmat = n_max * n_max * (l_max + 1)
    with h5py.File("testH2O.hdf5", "a") as f:
        soapGroup = f.require_group("SOAPNoCrossover")
        SOAPify.saponifyGroup(
            f["Trajectories"],
            soapGroup,
            rcut,
            n_max,
            l_max,
            SOAPkwargs={"crossover": False},
            useSoapFrom="dscribe",
        )
        assert soapGroup["testH2O"].attrs["n_max"] == n_max
        assert soapGroup["testH2O"].attrs["l_max"] == l_max
        assert "O" in soapGroup["testH2O"].attrs["species"]
        assert "H" in soapGroup["testH2O"].attrs["species"]
        assert numpy.abs(soapGroup["testH2O"].attrs["r_cut"] - rcut) < 1e-8
        assert "centersIndexes" not in soapGroup["testH2O"].attrs
        species, slices = SOAPify.getSlicesFromAttrs(soapGroup["testH2O"].attrs)
        print(slices)
        assert "O" in species
        assert "H" in species
        assert slices["H" + "H"] == slice(0, upperDiag)
        assert "HO" not in slices.keys()
        assert "OH" not in slices.keys()
        assert slices["O" + "O"] == slice(upperDiag, 2 * upperDiag)
        assert 22
        for gname, args in [
            ("SOAPinner", {"average": "inner"}),
            ("SOAPouter", {"average": "outer"}),
        ]:
            soapGroup = f.require_group(gname)
            SOAPify.saponifyGroup(
                f["Trajectories"],
                soapGroup,
                rcut,
                n_max,
                l_max,
                SOAPkwargs=args,
                useSoapFrom="dscribe",
            )
            assert soapGroup["testH2O"].attrs["n_max"] == n_max
            assert soapGroup["testH2O"].attrs["l_max"] == l_max
            assert "O" in soapGroup["testH2O"].attrs["species"]
            assert "H" in soapGroup["testH2O"].attrs["species"]
            assert numpy.abs(soapGroup["testH2O"].attrs["r_cut"] - rcut) < 1e-8
            assert "centersIndexes" not in soapGroup["testH2O"].attrs

            assert soapGroup[f"testH2O"].shape[-1] == 2 * upperDiag + fullmat
            species, slices = SOAPify.getSlicesFromAttrs(soapGroup["testH2O"].attrs)
            print(slices)
            assert "O" in species
            assert "H" in species
            assert slices["H" + "H"] == slice(0, upperDiag)
            assert slices["H" + "O"] == slice(upperDiag, upperDiag + fullmat)
            assert slices["O" + "H"] == slice(
                upperDiag, upperDiag + fullmat
            )  # redundant
            assert slices["O" + "O"] == slice(
                upperDiag + fullmat, 2 * upperDiag + fullmat
            )

        soapGroup = f.require_group("SOAPsparse")
        SOAPify.saponifyGroup(
            f["Trajectories"],
            soapGroup,
            rcut,
            n_max,
            l_max,
            SOAPkwargs={"sparse": True},
            useSoapFrom="dscribe",
        )
        assert soapGroup["testH2O"].attrs["n_max"] == n_max
        assert soapGroup["testH2O"].attrs["l_max"] == l_max
        assert "O" in soapGroup["testH2O"].attrs["species"]
        assert "H" in soapGroup["testH2O"].attrs["species"]
        assert numpy.abs(soapGroup["testH2O"].attrs["r_cut"] - rcut) < 1e-8
        assert "centersIndexes" not in soapGroup["testH2O"].attrs
        upperDiag = int((l_max + 1) * (n_max) * (n_max + 1) / 2)
        assert soapGroup[f"testH2O"].shape[-1] == 2 * upperDiag + fullmat
        species, slices = SOAPify.getSlicesFromAttrs(soapGroup["testH2O"].attrs)
        print(slices)
        assert "O" in species
        assert "H" in species
        assert slices["H" + "H"] == slice(0, upperDiag)
        assert slices["H" + "O"] == slice(upperDiag, upperDiag + fullmat)
        assert slices["O" + "H"] == slice(upperDiag, upperDiag + fullmat)  # redundant
        assert slices["O" + "O"] == slice(upperDiag + fullmat, 2 * upperDiag + fullmat)

        soapGroup = f.require_group("SOAPOxygen")
        SOAPify.saponifyGroup(
            f["Trajectories"],
            soapGroup,
            10.0,
            n_max,
            l_max,
            SOAPatomMask=["O"],
            useSoapFrom="dscribe",
        )
        assert soapGroup["testH2O"].attrs["n_max"] == n_max
        assert soapGroup["testH2O"].attrs["l_max"] == l_max
        assert "O" in soapGroup["testH2O"].attrs["species"]
        assert "H" in soapGroup["testH2O"].attrs["species"]
        assert numpy.abs(soapGroup["testH2O"].attrs["r_cut"] - rcut) < 1e-8
        assert_array_equal(
            soapGroup["testH2O"].attrs["centersIndexes"], [i * 3 for i in range(nMol)]
        )
