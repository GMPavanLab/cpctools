"""Test for HDF5er"""
import SOAPify.HDF5er as HDF5er
import h5py
import numpy
from numpy.testing import assert_array_almost_equal


def test_istTrajectoryGroupCheck(tmp_path):
    fname = tmp_path / "test.hdf5"
    # Given an HDF5 group
    with h5py.File(fname, "w") as hdf5test:
        group = hdf5test.create_group("Trajectories/4Atoms3Frames")
        # empty group
        assert not HDF5er.isTrajectoryGroup(group)
        group.create_dataset("Trajectory", data=numpy.zeros((5, 4, 3)))
        assert not HDF5er.isTrajectoryGroup(group)
        assert not HDF5er.isTrajectoryGroup(group["Trajectory"])

        group.create_dataset("Types", data=numpy.zeros((4)))
        assert not HDF5er.isTrajectoryGroup(group)
        assert not HDF5er.isTrajectoryGroup(group["Types"])

        group.create_dataset("Box", data=numpy.zeros((5, 3)))
        # now group contains a Box. a Types and  and a Trajectory Datasets
        assert HDF5er.isTrajectoryGroup(group)
        assert not HDF5er.isTrajectoryGroup(group["Box"])

    with h5py.File(fname, "w") as hdf5test:
        group = hdf5test.create_group("Trajectories/4Atoms3Frames")
        # empty group
        assert not HDF5er.isTrajectoryGroup(group)
        # now trajectory is set as a group
        group.create_group("Trajectory")
        group.create_dataset("Types", data=numpy.zeros((4)))
        group.create_dataset("Box", data=numpy.zeros((5, 3)))
        assert not HDF5er.isTrajectoryGroup(group)


def test_MDA2HDF5(input_framesSlice, input_universe, tmp_path):
    # Given an MDA Universe :
    fourAtomsFiveFrames = input_universe()
    attributes = {"ts": "1ps", "anotherAttr": "anotherAttrVal"}
    # sl = slice(0, None, 2)
    sl = input_framesSlice
    testFname = (
        f"test{fourAtomsFiveFrames.myUsefulName}{input_framesSlice}_p{len}.hdf5".replace(
            ", ", "_"
        )
        .replace("(", "-")
        .replace(")", "-")
        .replace("[", "-")
        .replace("]", "-")
    )
    fname = tmp_path / testFname
    HDF5er.MDA2HDF5(
        fourAtomsFiveFrames,
        fname,
        "4Atoms3Frames",
        override=True,
        attrs=attributes,
        trajslice=sl,
    )
    with h5py.File(fname, "r") as hdf5test:
        # this checks also that the group has been created
        group = hdf5test["Trajectories/4Atoms3Frames"]
        assert HDF5er.isTrajectoryGroup(group)
        for key in attributes.keys():
            assert group.attrs[key] == attributes[key]
        # this checks also that the datasets has been created
        nat = len(group["Types"])
        assert len(group["Trajectory"]) == len(fourAtomsFiveFrames.trajectory[sl])
        assert len(group["Box"]) == len(fourAtomsFiveFrames.trajectory[sl])
        for i, f in enumerate(fourAtomsFiveFrames.trajectory[sl]):
            assert nat == len(fourAtomsFiveFrames.atoms)
            for atomID in range(nat):
                for coord in range(3):
                    assert (
                        group["Trajectory"][i, atomID, coord]
                        - fourAtomsFiveFrames.atoms.positions[atomID, coord]
                        < 1e-8
                    )
            assert_array_almost_equal(
                group["Box"][i][:], fourAtomsFiveFrames.dimensions
            )
            print(i, group["Box"][i][:], fourAtomsFiveFrames.dimensions)
        print(group["Box"].shape, group["Trajectory"].chunks)


def test_MDA2HDF5Box(input_universe, tmp_path):
    for angles in [
        (90, 90, 90),
        (90, 60, 90),
        (60, 60, 60),
        (50, 60, 90),
        (90, 90, 45),
        (80, 60, 90),
    ]:
        print(angles)
        fourAtomsFiveFrames = input_universe(angles)
        testFname = (
            f"test{fourAtomsFiveFrames.myUsefulName}_p{len}.hdf5".replace(", ", "_")
            .replace("(", "-")
            .replace(")", "-")
            .replace("[", "-")
            .replace("]", "-")
        )
        fname = tmp_path / testFname
        name = f"4Atoms5Frames_{angles[0]}_{angles[1]}_{angles[2]}"
        HDF5er.MDA2HDF5(fourAtomsFiveFrames, fname, name, override=True)
        with h5py.File(fname, "r") as hdf5test:
            group = hdf5test[f"Trajectories/{name}"]
            for i, f in enumerate(fourAtomsFiveFrames.trajectory):
                print(i, fourAtomsFiveFrames.dimensions, group["Box"][i])
                for original, control in zip(
                    fourAtomsFiveFrames.dimensions, group["Box"][i]
                ):
                    assert (original - float(control)) < 1e-7
