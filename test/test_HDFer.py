import HDF5er
import h5py
import pytest
import MDAnalysis
import numpy
from MDAnalysis.lib.mdamath import triclinic_vectors


def giveUniverse(angles: set = (90.0, 90.0, 90.0)) -> MDAnalysis.Universe:
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
        dimensions=numpy.array(
            [[6.0, 6.0, 6.0, angles[0], angles[1], angles[2]]] * traj.shape[0]
        ),
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
        assert len(group["Trajectory"]) == 5
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

    # this test has been writtend After the function has been written
    # This creates or overwite the test file:


def test_MDA2HDF5Box():
    fourAtomsFiveFrames = giveUniverse((90, 90, 90))
    HDF5er.MDA2HDF5(
        fourAtomsFiveFrames, "test.hdf5", "4Atoms5Frames_toOver", override=True
    )
    with h5py.File("test.hdf5", "r") as hdf5test:
        group = hdf5test["Trajectories/4Atoms5Frames_toOver"]
        aseTraj = HDF5er.HDF52AseAtomsChunckedwithSymbols(
            group, slice(5), slice(5), ["H", "H", "H", "H"]
        )
        for j, array in enumerate(triclinic_vectors([6.0, 6.0, 6.0, 90, 90, 90])):
            for i, d in enumerate(array):
                assert aseTraj[0].cell[j][i] - d < 1e-7
    for angles in [
        (90, 60, 90),
        (60, 60, 60),
        (50, 60, 90),
        (90, 90, 45),
        (80, 60, 90),
    ]:
        print(angles)
        fourAtomsFiveFramesSkew = giveUniverse(angles)
        HDF5er.MDA2HDF5(
            fourAtomsFiveFramesSkew, "test.hdf5", "4Atoms5Frames_toOver", override=True
        )
        with h5py.File("test.hdf5", "r") as hdf5test:
            group = hdf5test["Trajectories/4Atoms5Frames_toOver"]
            aseTraj = HDF5er.HDF52AseAtomsChunckedwithSymbols(
                group, slice(5), slice(5), ["H", "H", "H", "H"]
            )
            for j, array in enumerate(
                triclinic_vectors(
                    [6.0, 6.0, 6.0, angles[0], angles[1], angles[2]],
                    dtype=numpy.float64,
                )
            ):
                for i, d in enumerate(array):
                    assert aseTraj[0].cell[j][i] - d < 1e-7


def test_copiMDA2HDF52xyz():
    fourAtomsFiveFrames = giveUniverse((90, 90, 90))
    HDF5er.MDA2HDF5(fourAtomsFiveFrames, "test.hdf5", "4Atoms5Frames", override=True)
    with h5py.File("test.hdf5", "r") as hdf5test:
        group = hdf5test["Trajectories/4Atoms5Frames_toOver"]
        stringData = HDF5er.getXYZfromTrajGroup(group)
        lines = stringData.splitlines()
        assert int(lines[0]) == len(fourAtomsFiveFrames.atoms)
