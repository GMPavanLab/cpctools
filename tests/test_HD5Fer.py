import HDF5er
import h5py
import pytest
import numpy
from MDAnalysis.lib.mdamath import triclinic_vectors
from io import StringIO
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


def test_copyMDA2HDF52xyz1DData(input_framesSlice):

    angles = (75.0, 60.0, 90.0)
    fourAtomsFiveFrames = giveUniverse(angles)
    latticeVector = triclinic_vectors(
        [6.0, 6.0, 6.0, angles[0], angles[1], angles[2]]
    ).flatten()
    rng = numpy.random.default_rng(12345)
    OneDDataOrig = rng.integers(
        0, 7, size=(len(fourAtomsFiveFrames.trajectory), len(fourAtomsFiveFrames.atoms))
    )
    OneDData = OneDDataOrig[input_framesSlice]
    HDF5er.MDA2HDF5(fourAtomsFiveFrames, "test.hdf5", "4Atoms5Frames", override=True)
    with h5py.File("test.hdf5", "r") as hdf5test:
        group = hdf5test["Trajectories/4Atoms5Frames"]
        stringData = StringIO()
        HDF5er.getXYZfromTrajGroup(
            stringData,
            group,
            framesToExport=input_framesSlice,
            OneDData=OneDData,
        )

        lines = stringData.getvalue().splitlines()
        nat = int(lines[0])
        assert int(lines[0]) == len(fourAtomsFiveFrames.atoms)
        assert lines[2].split()[0] == fourAtomsFiveFrames.atoms.types[0]
        for frame, traj in enumerate(fourAtomsFiveFrames.trajectory[input_framesSlice]):
            frameID = frame * (nat + 2)
            assert int(lines[frameID]) == nat
            t = lines[frameID + 1].split(" Lattice=")
            Lattice = t[1].replace('"', "").split()
            Properties = t[0].split(":")

            assert "OneDData" in Properties
            for original, control in zip(latticeVector, Lattice):
                assert (original - float(control)) < 1e-7
            for atomID in range(len(fourAtomsFiveFrames.atoms)):
                thisline = frameID + 2 + atomID
                assert (
                    lines[thisline].split()[0]
                    == fourAtomsFiveFrames.atoms.types[atomID]
                )
                assert len(lines[thisline].split()) == 5
                assert int((lines[thisline].split()[-1])) == OneDData[frame, atomID]
                for i in range(3):
                    assert (
                        float(lines[thisline].split()[i + 1])
                        == fourAtomsFiveFrames.atoms.positions[atomID][i]
                    )


def test_copyMDA2HDF52xyzMultiDData(input_framesSlice):
    angles = (75.0, 60.0, 90.0)
    fourAtomsFiveFrames = giveUniverse(angles)
    latticeVector = triclinic_vectors(
        [6.0, 6.0, 6.0, angles[0], angles[1], angles[2]]
    ).flatten()
    rng = numpy.random.default_rng(12345)
    dataDim = rng.integers(2, 15)
    MultiDData_original = rng.integers(
        0,
        7,
        size=(
            len(fourAtomsFiveFrames.trajectory),
            len(fourAtomsFiveFrames.atoms),
            dataDim,
        ),
    )
    MultiDData = MultiDData_original[input_framesSlice]
    HDF5er.MDA2HDF5(fourAtomsFiveFrames, "test.hdf5", "4Atoms5Frames", override=True)
    with h5py.File("test.hdf5", "r") as hdf5test:
        group = hdf5test["Trajectories/4Atoms5Frames"]
        stringData = StringIO()
        HDF5er.getXYZfromTrajGroup(
            stringData,
            group,
            framesToExport=input_framesSlice,
            MultiDData=MultiDData,
        )
        lines = stringData.getvalue().splitlines()
        nat = int(lines[0])
        assert int(lines[0]) == len(fourAtomsFiveFrames.atoms)
        assert lines[2].split()[0] == fourAtomsFiveFrames.atoms.types[0]
        for frame, traj in enumerate(fourAtomsFiveFrames.trajectory[input_framesSlice]):
            frameID = frame * (nat + 2)
            assert int(lines[frameID]) == nat
            t = lines[frameID + 1].split(" Lattice=")
            Lattice = t[1].replace('"', "").split()
            Properties = t[0].split(":")
            assert "MultiDData" in Properties
            for original, control in zip(latticeVector, Lattice):
                assert (original - float(control)) < 1e-7
            for atomID in range(len(fourAtomsFiveFrames.atoms)):
                thisline = frameID + 2 + atomID
                assert (
                    lines[thisline].split()[0]
                    == fourAtomsFiveFrames.atoms.types[atomID]
                )
                assert len(lines[thisline].split()) == (4 + dataDim)
                for i, d in enumerate(MultiDData[frame, atomID]):
                    assert int((lines[thisline].split()[4 + i])) == d
                for i in range(3):
                    assert (
                        float(lines[thisline].split()[i + 1])
                        == fourAtomsFiveFrames.atoms.positions[atomID][i]
                    )


def test_copyMDA2HDF52xyzAllFrameProperty(input_framesSlice):
    angles = (75.0, 60.0, 90.0)
    fourAtomsFiveFrames = giveUniverse(angles)
    latticeVector = triclinic_vectors(
        [6.0, 6.0, 6.0, angles[0], angles[1], angles[2]]
    ).flatten()
    HDF5er.MDA2HDF5(fourAtomsFiveFrames, "test.hdf5", "4Atoms5Frames", override=True)
    with h5py.File("test.hdf5", "r") as hdf5test:
        group = hdf5test["Trajectories/4Atoms5Frames"]
        stringData = StringIO()
        HDF5er.getXYZfromTrajGroup(
            stringData,
            group,
            framesToExport=input_framesSlice,
            allFramesProperty='Origin="-1 -1 -1"',
        )

        lines = stringData.getvalue().splitlines()
        nat = int(lines[0])
        assert int(lines[0]) == len(fourAtomsFiveFrames.atoms)
        assert lines[2].split()[0] == fourAtomsFiveFrames.atoms.types[0]
        for frame, traj in enumerate(fourAtomsFiveFrames.trajectory[input_framesSlice]):
            frameID = frame * (nat + 2)
            assert int(lines[frameID]) == nat
            t = lines[frameID + 1].split(" Lattice=")
            Lattice = t[1].replace('"', "").split()
            assert "Origin" in lines[frameID + 1]
            t = lines[frameID + 1].split(" Origin=")[1].split('"')[1]
            for v in t.split(" "):
                assert int(v) == -1
            for original, control in zip(latticeVector, Lattice):
                assert (original - float(control)) < 1e-7
            for atomID in range(len(fourAtomsFiveFrames.atoms)):
                thisline = frameID + 2 + atomID
                assert (
                    lines[thisline].split()[0]
                    == fourAtomsFiveFrames.atoms.types[atomID]
                )
                assert len(lines[thisline].split()) == 4
                for i in range(3):
                    assert (
                        float(lines[thisline].split()[i + 1])
                        == fourAtomsFiveFrames.atoms.positions[atomID][i]
                    )


def test_copyMDA2HDF52xyzPerFrameProperty(input_framesSlice):
    angles = (75.0, 60.0, 90.0)
    fourAtomsFiveFrames = giveUniverse(angles)
    latticeVector = triclinic_vectors(
        [6.0, 6.0, 6.0, angles[0], angles[1], angles[2]]
    ).flatten()
    HDF5er.MDA2HDF5(fourAtomsFiveFrames, "test.hdf5", "4Atoms5Frames", override=True)
    perFrameProperties = numpy.array([f'Originpf="-{i} -{i} -{i}"' for i in range(5)])[
        input_framesSlice
    ]

    with h5py.File("test.hdf5", "r") as hdf5test:
        group = hdf5test["Trajectories/4Atoms5Frames"]
        stringData = StringIO()
        HDF5er.getXYZfromTrajGroup(
            stringData,
            group,
            framesToExport=input_framesSlice,
            perFrameProperties=perFrameProperties,
        )

        lines = stringData.getvalue().splitlines()
        nat = int(lines[0])
        assert int(lines[0]) == len(fourAtomsFiveFrames.atoms)
        assert lines[2].split()[0] == fourAtomsFiveFrames.atoms.types[0]
        for frame, traj in enumerate(fourAtomsFiveFrames.trajectory[input_framesSlice]):
            frameID = frame * (nat + 2)
            assert int(lines[frameID]) == nat
            t = lines[frameID + 1].split(" Lattice=")
            Lattice = t[1].replace('"', "").split()
            assert "Originpf" in lines[frameID + 1]
            t = lines[frameID + 1].split(" Originpf=")[1].split('"')[1]
            to = perFrameProperties[frame].split("Originpf=")[1].split('"')[1]
            for saved, orig in zip(t.split(" "), to.split(" ")):
                assert int(saved) == int(orig)
            for original, control in zip(latticeVector, Lattice):
                assert (original - float(control)) < 1e-7
            for atomID in range(len(fourAtomsFiveFrames.atoms)):
                thisline = frameID + 2 + atomID
                assert (
                    lines[thisline].split()[0]
                    == fourAtomsFiveFrames.atoms.types[atomID]
                )
                assert len(lines[thisline].split()) == 4
                for i in range(3):
                    assert (
                        float(lines[thisline].split()[i + 1])
                        == fourAtomsFiveFrames.atoms.positions[atomID][i]
                    )


def test_copyMDA2HDF52xyz_error1D():
    angles = (75.0, 60.0, 90.0)
    fourAtomsFiveFrames = giveUniverse(angles)
    latticeVector = triclinic_vectors(
        [6.0, 6.0, 6.0, angles[0], angles[1], angles[2]]
    ).flatten()
    rng = numpy.random.default_rng(12345)
    OneDData = rng.integers(
        0,
        7,
        size=(len(fourAtomsFiveFrames.trajectory), len(fourAtomsFiveFrames.atoms) + 1),
    )

    HDF5er.MDA2HDF5(fourAtomsFiveFrames, "test.hdf5", "4Atoms5Frames", override=True)
    with h5py.File("test.hdf5", "r") as hdf5test, pytest.raises(ValueError):
        group = hdf5test["Trajectories/4Atoms5Frames"]
        stringData = StringIO()
        HDF5er.getXYZfromTrajGroup(stringData, group, OneDData=OneDData)


def test_copyMDA2HDF52xyz_error2D():
    angles = (75.0, 60.0, 90.0)
    fourAtomsFiveFrames = giveUniverse(angles)
    latticeVector = triclinic_vectors(
        [6.0, 6.0, 6.0, angles[0], angles[1], angles[2]]
    ).flatten()
    rng = numpy.random.default_rng(12345)
    TwoDData = rng.integers(
        0,
        7,
        size=(
            len(fourAtomsFiveFrames.trajectory),
            len(fourAtomsFiveFrames.atoms) + 1,
            2,
        ),
    )
    HDF5er.MDA2HDF5(fourAtomsFiveFrames, "test.hdf5", "4Atoms5Frames", override=True)
    with h5py.File("test.hdf5", "r") as hdf5test, pytest.raises(ValueError):
        group = hdf5test["Trajectories/4Atoms5Frames"]
        stringData = StringIO()
        HDF5er.getXYZfromTrajGroup(stringData, group, TwoDData=TwoDData)


def test_copyMDA2HDF52xyz_wrongD():
    angles = (75.0, 60.0, 90.0)
    fourAtomsFiveFrames = giveUniverse(angles)
    latticeVector = triclinic_vectors(
        [6.0, 6.0, 6.0, angles[0], angles[1], angles[2]]
    ).flatten()
    rng = numpy.random.default_rng(12345)
    WrongDData = rng.integers(
        0,
        7,
        size=(
            len(fourAtomsFiveFrames.trajectory),
            len(fourAtomsFiveFrames.atoms),
            2,
            4,
        ),
    )
    HDF5er.MDA2HDF5(fourAtomsFiveFrames, "test.hdf5", "4Atoms5Frames", override=True)
    with h5py.File("test.hdf5", "r") as hdf5test, pytest.raises(ValueError):
        group = hdf5test["Trajectories/4Atoms5Frames"]
        stringData = StringIO()
        HDF5er.getXYZfromTrajGroup(stringData, group, WrongDData=WrongDData)


def test_copyMDA2HDF52xyz_wrongTrajlen():
    angles = (75.0, 60.0, 90.0)
    fourAtomsFiveFrames = giveUniverse(angles)
    latticeVector = triclinic_vectors(
        [6.0, 6.0, 6.0, angles[0], angles[1], angles[2]]
    ).flatten()
    rng = numpy.random.default_rng(12345)
    WrongDData = rng.integers(
        0,
        7,
        size=(
            len(fourAtomsFiveFrames.trajectory) + 5,
            len(fourAtomsFiveFrames.atoms),
            2,
        ),
    )
    HDF5er.MDA2HDF5(fourAtomsFiveFrames, "test.hdf5", "4Atoms5Frames", override=True)
    with h5py.File("test.hdf5", "r") as hdf5test, pytest.raises(ValueError):
        group = hdf5test["Trajectories/4Atoms5Frames"]
        stringData = StringIO()
        HDF5er.getXYZfromTrajGroup(stringData, group, WrongDData=WrongDData)
