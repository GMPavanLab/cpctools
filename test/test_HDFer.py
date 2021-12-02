from MDAnalysis import coordinates
import HDF5er
import h5py
import pytest
import MDAnalysis
from MDAnalysis.topology.MinimalParser import Topology
from MDAnalysis.coordinates.memory import MemoryReader
import numpy

# from MDAnalysis.tests.datafiles import


def giveUniverse() -> MDAnalysis.Universe:
    traj = numpy.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
            [[0.1, 0.1, 0.1], [1.1, 1.1, 1.1], [2.1, 2.1, 2.1], [3.1, 3.1, 3.1]],
            [[0.2, 0.2, 0.2], [1.2, 1.2, 1.2], [2.2, 2.2, 2.2], [3.2, 3.2, 3.2]],
            [[0.3, 0.3, 0.3], [1.3, 1.3, 1.3], [2.3, 2.3, 2.3], [3.3, 3.3, 3.3]],
            [[0.4, 0.4, 0.4], [1.4, 1.4, 1.4], [2.4, 2.4, 2.4], [3.4, 3.4, 3.4]],
        ]
    )
    u = MDAnalysis.Universe.empty(4, trajectory=True)
    u.add_TopologyAttr("type", ["H", "H", "H", "H"])

    u.trajectory.set_array(traj, "fac")
    return u


def test_MDA2HDF5():
    # Given an MDA Universe :
    fourAtomsFiveFrames = giveUniverse()
    HDF5er.MDA2HDF5(fourAtomsFiveFrames, "test.hdf5", "4Atoms5Frames", override=True)

    with h5py.File("test.hdf5", "r") as hdf5test:
        # this checks also that the group has been created
        group = hdf5test["Trajectories/4Atoms5Frames"]
        # this checks also that the dataset has been created
        nat = len(group["Types"])
        for i, f in enumerate(fourAtomsFiveFrames.trajectory):
            assert nat == len(fourAtomsFiveFrames.atoms)
            for atomID in range(nat):
                for coord in range(3):
                    assert (
                        group["Trajectory"][i, atomID, coord]
                        - fourAtomsFiveFrames.atoms.positions[atomID, coord]
                        < 1e-8
                    )


# def test_ase():
#    # This creates or overwite the test file:
#    fourAtomsFiveFrames = giveUniverse()
#    HDF5er.MDA2HDF5(fourAtomsFiveFrames, "test.hdf5", "4Atoms5Frames", override=True)
#    aseTraj = HDF5er.HDF52AseAtomsChunckedwithSymbols()
