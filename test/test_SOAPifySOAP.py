import SOAPify
import numpy
from numpy.random import randint
from numpy.testing import assert_array_equal
from SOAPify import SOAPReferences
import h5py
import MDAnalysis as mda
import HDF5er


def getUniverseWithWaterMolecules(n_residues=10):
    # following the tutorial at https://userguide.mdanalysis.org/stable/examples/constructing_universe.html
    n_atoms = n_residues * 3
    # create resindex list
    resindices = numpy.repeat(range(n_residues), 3)
    assert len(resindices) == n_atoms
    # all water molecules belong to 1 segment
    segindices = [0] * n_residues
    # create the Universe
    sol = mda.Universe.empty(
        n_atoms,
        n_residues=n_residues,
        atom_resindex=resindices,
        residue_segindex=segindices,
        trajectory=True,
    )  # necessary for adding coordinates
    sol.add_TopologyAttr("name", ["O", "H1", "H2"] * n_residues)
    sol.add_TopologyAttr("type", ["O", "H", "H"] * n_residues)
    sol.add_TopologyAttr("resname", ["SOL"] * n_residues)
    sol.add_TopologyAttr("resid", list(range(1, n_residues + 1)))
    sol.add_TopologyAttr("segid", ["SOL"])

    # coordinates obtained by building a molecule in the program IQMol
    h2o = numpy.array(
        [
            [0, 0, 0],  # oxygen
            [0.95908, -0.02691, 0.03231],  # hydrogen
            [-0.28004, -0.58767, 0.70556],  # hydrogen
        ]
    )  # hydrogen

    grid_size = numpy.ceil(numpy.cbrt(n_residues))
    spacing = 8
    coordinates = []

    # translating h2o coordinates around a grid
    for i in range(n_residues):
        x = spacing * (i % grid_size)
        y = spacing * ((i // grid_size) % grid_size)
        z = spacing * (i // (grid_size * grid_size))

        xyz = numpy.array([x, y, z])

        coordinates.extend(h2o + xyz.T)
    sol.dimensions = [
        grid_size * spacing,
        grid_size * spacing,
        grid_size * spacing,
        90,
        90,
        90,
    ]

    coord_array = numpy.array(coordinates)
    assert coord_array.shape == (n_atoms, 3)
    sol.atoms.positions = coord_array
    bonds = []
    for o in range(0, n_atoms, 3):
        bonds.extend([(o, o + 1), (o, o + 2)])

    sol.add_TopologyAttr("bonds", bonds)

    return sol


def test_MultiAtomicSoap():
    nMol = 27
    u = getUniverseWithWaterMolecules(nMol)
    HDF5er.MDA2HDF5(u, "testH2O.hdf5", "testH2O", override=True)
    n_max = 4
    l_max = 4
    rcut = 10.0
    with h5py.File("testH2O.hdf5", "a") as f:
        soapGroup = f.require_group("SOAP")
        SOAPify.saponifyGroup(f["Trajectories"], soapGroup, rcut, n_max, l_max)
        assert soapGroup["testH2O"].attrs["n_max"] == n_max
        assert soapGroup["testH2O"].attrs["l_max"] == l_max
        assert "O" in soapGroup["testH2O"].attrs["species"]
        assert "H" in soapGroup["testH2O"].attrs["species"]
        assert numpy.abs(soapGroup["testH2O"].attrs["r_cut"] - rcut) < 1e-8
        assert "centersIndexes" not in soapGroup["testH2O"].attrs
        soapGroup = f.require_group("SOAPOxygen")
        SOAPify.saponifyGroup(
            f["Trajectories"], soapGroup, 10.0, n_max, l_max, SOAPatomMask=["O"]
        )
        assert soapGroup["testH2O"].attrs["n_max"] == n_max
        assert soapGroup["testH2O"].attrs["l_max"] == l_max
        assert "O" in soapGroup["testH2O"].attrs["species"]
        assert "H" in soapGroup["testH2O"].attrs["species"]
        assert numpy.abs(soapGroup["testH2O"].attrs["r_cut"] - rcut) < 1e-8
        assert_array_equal(
            soapGroup["testH2O"].attrs["centersIndexes"], [i * 3 for i in range(nMol)]
        )
        HDF5er.saveXYZfromTrajGroup("testH2O.xyz", f["Trajectories/testH2O"])
