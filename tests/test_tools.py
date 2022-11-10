import HDF5er
import h5py
import pytest
import numpy
from io import StringIO
from .testSupport import (
    giveUniverse,
    checkStringDataFromUniverse,
    getUniverseWithWaterMolecules,
)


def test_MDA2EXYZ(input_framesSlice):
    angles = (75.0, 60.0, 90.0)
    fourAtomsFiveFrames = giveUniverse(angles)
    rng = numpy.random.default_rng(12345)
    OneDDataOrig = rng.integers(
        0, 7, size=(len(fourAtomsFiveFrames.trajectory), len(fourAtomsFiveFrames.atoms))
    )
    dataDim = rng.integers(2, 15)
    OneDData = OneDDataOrig[input_framesSlice]
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
    stringData = StringIO()
    HDF5er.getXYZfromMDA(
        stringData,
        fourAtomsFiveFrames,
        framesToExport=input_framesSlice,
        OneDData=OneDData,
        MultiDData=MultiDData,
    )

    checkStringDataFromUniverse(
        stringData,
        fourAtomsFiveFrames,
        input_framesSlice,
        OneDData=OneDData,
        MultiDData=MultiDData,
    )


def test_MDA2EXYZ_selection():
    input_framesSlice = slice(None)
    myUniverse = getUniverseWithWaterMolecules()
    print(
        len(myUniverse.trajectory),
    )
    mySel = myUniverse.select_atoms("type O")
    rng = numpy.random.default_rng(12345)
    OneDDataOrig = rng.integers(
        0, 7, size=(len(myUniverse.trajectory), len(mySel.atoms))
    )
    dataDim = rng.integers(2, 15)
    OneDData = OneDDataOrig[input_framesSlice]
    MultiDData_original = rng.integers(
        0,
        7,
        size=(
            len(myUniverse.trajectory),
            len(mySel.atoms),
            dataDim,
        ),
    )
    MultiDData = MultiDData_original[input_framesSlice]
    stringData = StringIO()
    HDF5er.getXYZfromMDA(
        stringData,
        mySel,
        framesToExport=input_framesSlice,
        OneDData=OneDData,
        MultiDData=MultiDData,
    )

    checkStringDataFromUniverse(
        stringData,
        mySel,
        input_framesSlice,
        OneDData=OneDData,
        MultiDData=MultiDData,
    )


def test_copyMDA2HDF52xyz1DData(input_framesSlice):
    angles = (75.0, 60.0, 90.0)
    fourAtomsFiveFrames = giveUniverse(angles)
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

        checkStringDataFromUniverse(
            stringData,
            fourAtomsFiveFrames,
            input_framesSlice,
            OneDData=OneDData,
        )


def test_copyMDA2HDF52xyzMultiDData(input_framesSlice):
    angles = (75.0, 60.0, 90.0)
    fourAtomsFiveFrames = giveUniverse(angles)
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
        checkStringDataFromUniverse(
            stringData,
            fourAtomsFiveFrames,
            input_framesSlice,
            MultiDData=MultiDData,
        )


def test_copyMDA2HDF52xyzAllFrameProperty(input_framesSlice):
    angles = (75.0, 60.0, 90.0)
    fourAtomsFiveFrames = giveUniverse(angles)
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

        checkStringDataFromUniverse(
            stringData,
            fourAtomsFiveFrames,
            input_framesSlice,
            allFramesProperty='Origin="-1 -1 -1"',
        )


def test_copyMDA2HDF52xyzPerFrameProperty(input_framesSlice):
    angles = (75.0, 60.0, 90.0)
    fourAtomsFiveFrames = giveUniverse(angles)
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
        checkStringDataFromUniverse(
            stringData,
            fourAtomsFiveFrames,
            input_framesSlice,
            perFrameProperties=perFrameProperties,
        )


def test_copyMDA2HDF52xyz_error1D():
    angles = (75.0, 60.0, 90.0)
    fourAtomsFiveFrames = giveUniverse(angles)
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
