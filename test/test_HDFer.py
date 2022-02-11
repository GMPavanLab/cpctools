import HDF5er
import h5py
import pytest
import MDAnalysis
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

    u.add_TopologyAttr("type", ["H"] * 4)
    u.atoms.positions = traj[0]
    u.trajectory = MDAnalysis.coordinates.memory.MemoryReader(
        traj,
        order="fac",
        # this tests the non orthogonality of the box
        dimensions=numpy.array([[6.0, 6.0, 6.0, 90, 60, 90]] * traj.shape[0]),
    )
    return u


def test_MDA2HDF5():
    # Given an MDA Universe :
    fourAtomsFiveFrames = giveUniverse()
    attributes = {"ts": "1ps", "anotherAttr": "anotherAttrVal"}
    HDF5er.MDA2HDF5(
        fourAtomsFiveFrames,
        "test.hdf5",
        "4Atoms5Frames",
        override=True,
        attrs=attributes,
    )
    # verify:
    with h5py.File("test.hdf5", "r") as hdf5test:
        # this checks also that the group has been created
        group = hdf5test["Trajectories/4Atoms5Frames"]
        for key in attributes.keys():
            assert group.attrs[key] == attributes[key]
        # this checks also that the dataset has been created
        nat = len(group["Types"])
        assert len(group["Trajectory"]) == len(fourAtomsFiveFrames.trajectory)
        for i, f in enumerate(fourAtomsFiveFrames.trajectory):
            assert nat == len(fourAtomsFiveFrames.atoms)
            for atomID in range(nat):
                for coord in range(3):
                    assert (
                        group["Trajectory"][i, atomID, coord]
                        - fourAtomsFiveFrames.atoms.positions[atomID, coord]
                        < 1e-8
                    )


def test_MDA2HDF5Sliced():
    # Given an MDA Universe :
    fourAtomsFiveFrames = giveUniverse()
    attributes = {"ts": "1ps", "anotherAttr": "anotherAttrVal"}
    sl = slice(0, None, 2)
    HDF5er.MDA2HDF5(
        fourAtomsFiveFrames,
        "test.hdf5",
        "4Atoms3Frames",
        override=True,
        attrs=attributes,
        trajslice=sl,
    )
    with h5py.File("test.hdf5", "r") as hdf5test:
        # this checks also that the group has been created
        group = hdf5test["Trajectories/4Atoms3Frames"]
        for key in attributes.keys():
            assert group.attrs[key] == attributes[key]
        # this checks also that the dataset has been created
        nat = len(group["Types"])
        assert len(group["Trajectory"]) == 3
        for i, f in enumerate(fourAtomsFiveFrames.trajectory[sl]):
            assert nat == len(fourAtomsFiveFrames.atoms)
            for atomID in range(nat):
                for coord in range(3):
                    assert (
                        group["Trajectory"][i, atomID, coord]
                        - fourAtomsFiveFrames.atoms.positions[atomID, coord]
                        < 1e-8
                    )


def test_ase():
    # this test has been writtend After the function has been written
    # This creates or overwite the test file:
    fourAtomsFiveFrames = giveUniverse()
    HDF5er.MDA2HDF5(fourAtomsFiveFrames, "test.hdf5", "4Atoms5Frames", override=True)
    with h5py.File("test.hdf5", "r") as hdf5test:
        group = hdf5test["Trajectories/4Atoms5Frames"]
        aseTraj = HDF5er.HDF52AseAtomsChunckedwithSymbols(
            group, slice(5), slice(5), ["H", "H", "H", "H"]
        )
        for i, d in enumerate([6.0, 0.0, 0.0]):
            assert aseTraj[0].cell[0][i] - d < 1e-8
        for i, d in enumerate([0.0, 6.0, 0.0]):
            assert aseTraj[0].cell[1][i] - d < 1e-8
        for i, d in enumerate(6.0 * numpy.array([0.5, 0.0, numpy.sqrt(3.0) / 2])):
            assert aseTraj[0].cell[2][i] - d < 1e-8
