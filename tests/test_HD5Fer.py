import HDF5er
import h5py
import numpy
from MDAnalysis.lib.mdamath import triclinic_vectors
from .testSupport import giveUniverse


def test_istTrajectoryGroupCheck():
    # Given an HDF5 group
    with h5py.File("test.hdf5", "w") as hdf5test:
        group = hdf5test.create_group("Trajectories/4Atoms3Frames")
        # empty group
        assert not HDF5er.isTrajectoryGroup(group)
        group.create_dataset("Trajectory", data=numpy.zeros((5, 4, 3)))
        assert not HDF5er.isTrajectoryGroup(group)
        group.create_dataset("Types", data=numpy.zeros((4)))
        assert not HDF5er.isTrajectoryGroup(group)
        group.create_dataset("Box", data=numpy.zeros((5, 3)))
        assert HDF5er.isTrajectoryGroup(group)
    with h5py.File("test.hdf5", "w") as hdf5test:
        group = hdf5test.create_group("Trajectories/4Atoms3Frames")
        # empty group
        assert not HDF5er.isTrajectoryGroup(group)
        # now trajectory is set as a group
        group.create_group("Trajectory")
        group.create_dataset("Types", data=numpy.zeros((4)))
        group.create_dataset("Box", data=numpy.zeros((5, 3)))
        assert not HDF5er.isTrajectoryGroup(group)


def test_MDA2HDF5(input_framesSlice):
    # Given an MDA Universe :
    fourAtomsFiveFrames = giveUniverse()
    attributes = {"ts": "1ps", "anotherAttr": "anotherAttrVal"}
    # sl = slice(0, None, 2)
    sl = input_framesSlice
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
        assert HDF5er.isTrajectoryGroup(group)
        for key in attributes.keys():
            assert group.attrs[key] == attributes[key]
        # this checks also that the dataset has been created
        nat = len(group["Types"])
        assert len(group["Trajectory"]) == len(fourAtomsFiveFrames.trajectory[sl])
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


