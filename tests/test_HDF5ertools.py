import SOAPify.HDF5er as HDF5er
import h5py
import pytest
import numpy
from numpy.testing import assert_array_almost_equal, assert_array_equal
from io import StringIO
from .testSupport import (
    giveUniverse,
    giveUniverse_ChangingBox,
    checkStringDataFromUniverse,
    getUniverseWithWaterMolecules,
    checkStringDataFromHDF5,
    __PropertiesFinder,
)


def test_MDA2EXYZ(input_framesSlice, input_CreateParametersToExport, input_universe):
    angles = (75.0, 60.0, 90.0)
    fourAtomsFiveFrames = input_universe(angles)
    additionalParameters = input_CreateParametersToExport(
        frames=len(fourAtomsFiveFrames.trajectory),
        nat=len(fourAtomsFiveFrames.atoms),
        frameSlice=input_framesSlice,
    )

    stringData = StringIO()
    HDF5er.getXYZfromMDA(
        stringData,
        fourAtomsFiveFrames,
        framesToExport=input_framesSlice,
        **additionalParameters,
    )

    checkStringDataFromUniverse(
        stringData,
        fourAtomsFiveFrames,
        input_framesSlice,
        **additionalParameters,
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


def test_copyMDA2HDF52xyz(input_framesSlice, input_CreateParametersToExport, hdf5_file):
    testFname = hdf5_file[0]
    fourAtomsFiveFrames = hdf5_file[1]
    additionalParameters = input_CreateParametersToExport(
        frames=len(fourAtomsFiveFrames.trajectory),
        nat=len(fourAtomsFiveFrames.atoms),
        frameSlice=input_framesSlice,
    )
    print(testFname)
    with h5py.File(testFname, "r") as hdf5test:
        group = hdf5test["Trajectories/4Atoms5Frames"]
        print("Box", group["Box"][:])
        stringData = StringIO()
        HDF5er.getXYZfromTrajGroup(
            stringData, group, framesToExport=input_framesSlice, **additionalParameters
        )

        checkStringDataFromHDF5(
            stringData, group, input_framesSlice, **additionalParameters
        )


def test_writeMDA2HDF52xyz(
    tmp_path_factory, input_framesSlice, input_CreateParametersToExport, hdf5_file
):
    testFname = hdf5_file[0]
    fourAtomsFiveFrames = hdf5_file[1]
    additionalParameters = input_CreateParametersToExport(
        frames=len(fourAtomsFiveFrames.trajectory),
        nat=len(fourAtomsFiveFrames.atoms),
        frameSlice=input_framesSlice,
    )
    print(str(testFname), type(testFname))
    outFname = tmp_path_factory.mktemp("tempXYZ") / testFname.name.replace(
        ".hdf5", ".xyz"
    )
    with h5py.File(testFname, "r") as hdf5test:
        group = hdf5test["Trajectories/4Atoms5Frames"]
        print("Box", group["Box"][:])

        HDF5er.saveXYZfromTrajGroup(
            outFname, group, framesToExport=input_framesSlice, **additionalParameters
        )
        with open(outFname, "r") as file:
            checkStringDataFromHDF5(
                file, group, input_framesSlice, **additionalParameters
            )


def test_copyMDA2HDF52xyzAllFrameProperty(input_framesSlice, tmp_path):
    angles = (75.0, 60.0, 90.0)
    fourAtomsFiveFrames = giveUniverse(angles)
    fname = tmp_path / "test.hdf5"
    HDF5er.MDA2HDF5(fourAtomsFiveFrames, fname, "4Atoms5Frames", override=True)
    with h5py.File(fname, "r") as hdf5test:
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


def test_copyMDA2HDF52xyzPerFrameProperty(input_framesSlice, tmp_path):
    angles = (75.0, 60.0, 90.0)
    fourAtomsFiveFrames = giveUniverse(angles)
    nframes = len(fourAtomsFiveFrames.trajectory)
    fname = tmp_path / "test.hdf5"
    HDF5er.MDA2HDF5(fourAtomsFiveFrames, fname, "4Atoms5Frames", override=True)
    perFrameProperties = numpy.array(
        [f'Originpf="-{i} -{i} -{i}"' for i in range(nframes)]
    )[input_framesSlice]

    with h5py.File(fname, "r") as hdf5test:
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


def test_copyMDA2HDF52xyz_error1D(tmp_path):
    angles = (75.0, 60.0, 90.0)
    fourAtomsFiveFrames = giveUniverse(angles)
    nframes = len(fourAtomsFiveFrames.trajectory)
    rng = numpy.random.default_rng(12345)
    OneDData = rng.integers(
        0,
        7,
        size=(len(fourAtomsFiveFrames.trajectory), len(fourAtomsFiveFrames.atoms) + 1),
    )
    fname = tmp_path / "test.hdf5"
    HDF5er.MDA2HDF5(fourAtomsFiveFrames, fname, "4Atoms5Frames", override=True)
    with h5py.File(fname, "r") as hdf5test, pytest.raises(ValueError):
        group = hdf5test["Trajectories/4Atoms5Frames"]
        stringData = StringIO()
        HDF5er.getXYZfromTrajGroup(stringData, group, OneDData=OneDData)
        assert (2 + nframes) * nframes == len(str(stringData).splitlines())


def test_copyMDA2HDF52xyz_error2D(tmp_path):
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
    fname = tmp_path / "test.hdf5"
    HDF5er.MDA2HDF5(fourAtomsFiveFrames, fname, "4Atoms5Frames", override=True)
    with h5py.File(fname, "r") as hdf5test, pytest.raises(ValueError):
        group = hdf5test["Trajectories/4Atoms5Frames"]
        stringData = StringIO()
        HDF5er.getXYZfromTrajGroup(stringData, group, TwoDData=TwoDData)


def test_copyMDA2HDF52xyz_wrongD(tmp_path):
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
    fname = tmp_path / "test.hdf5"
    HDF5er.MDA2HDF5(fourAtomsFiveFrames, fname, "4Atoms5Frames", override=True)
    with h5py.File(fname, "r") as hdf5test, pytest.raises(ValueError):
        group = hdf5test["Trajectories/4Atoms5Frames"]
        stringData = StringIO()
        HDF5er.getXYZfromTrajGroup(stringData, group, WrongDData=WrongDData)


def test_copyMDA2HDF52xyz_wrongTrajlen(tmp_path):
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
    fname = tmp_path / "test.hdf5"
    HDF5er.MDA2HDF5(fourAtomsFiveFrames, fname, "4Atoms5Frames", override=True)
    with h5py.File(fname, "r") as hdf5test, pytest.raises(ValueError):
        group = hdf5test["Trajectories/4Atoms5Frames"]
        stringData = StringIO()
        HDF5er.getXYZfromTrajGroup(stringData, group, WrongDData=WrongDData)


@pytest.fixture(
    scope="module",
    params=[None, 'Origin="-1 -1 -1"', 'Title="Graph"'],
)
def input_afp(request):
    return request.param


def test_headerPreparation(input_CreateParametersToExport, input_afp):
    nat = 10
    nframes = 5
    testData = input_CreateParametersToExport(nframes, nat)
    header: str = HDF5er.HDF5To.__prepareHeaders(
        testData, nframes=nframes, nat=nat, allFramesProperty=input_afp
    )
    headerProperties: str = __PropertiesFinder.search(header).group(1)
    headerProperties = headerProperties.split(":")
    for k in testData:
        assert k in headerProperties
        numOfData = 1
        if len(testData[k].shape) > 2:
            numOfData = testData[k].shape[2]
        MapPos = headerProperties.index(k)
        assert numOfData == int(headerProperties[MapPos + 2])
    # the fixture can report a None:
    if input_afp is None:
        headerbis: str = HDF5er.HDF5To.__prepareHeaders(
            testData, nframes=nframes, nat=nat
        )
        assert headerbis == header
    else:
        assert input_afp in header
    assert nat == int(header[: header.find("\n")])
    assert header[-1] != "\n"


def test_headerPreparation_failures(input_CreateParametersToExport):
    nat = 10
    nframes = 5
    testData = input_CreateParametersToExport(nframes, nat)
    perFrameProperties = numpy.array(
        [f'Originpf="-{i} -{i} -{i}"' for i in range(nframes + 1)]
    )
    with pytest.raises(ValueError):
        headerbis: str = HDF5er.HDF5To.__prepareHeaders(
            testData, nframes=nframes, nat=nat, perFrameProperties=perFrameProperties
        )


@pytest.fixture
def diffFrames(input_intModify):
    return input_intModify


@pytest.fixture
def diffNat(input_intModify):
    return input_intModify


def test_headerPreparation_errors(input_CreateParametersToExport, diffFrames, diffNat):
    nat = 10
    nframes = 5
    testData = input_CreateParametersToExport(nframes, nat)
    if len(testData) == 0 or (diffFrames == 0 and diffNat == 0):
        return

    with pytest.raises(ValueError):
        HDF5er.HDF5To.__prepareHeaders(
            testData, nframes=nframes + diffFrames, nat=nat + diffNat
        )


def test_HDF52Universe(
    input_framesSlice,
    hdf5_file,
):
    testFname = hdf5_file[0]

    print(testFname)
    with h5py.File(testFname, "r") as hdf5test:
        group = hdf5test["Trajectories/4Atoms5Frames"]
        print("Box", group["Box"][:])
        newUniverse = HDF5er.createUniverseFromSlice(group, input_framesSlice)
        for frameTraj, frameBox, ts in zip(
            group["Trajectory"][input_framesSlice],
            group["Box"][input_framesSlice],
            newUniverse.trajectory,
        ):
            assert_array_almost_equal(frameTraj, newUniverse.atoms.positions)
            assert_array_almost_equal(frameBox, newUniverse.dimensions)

        assert_array_equal(group["Types"].asstr(), newUniverse.atoms.types)
