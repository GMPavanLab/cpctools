import HDF5er
import h5py
import pytest
import numpy
from io import StringIO
from .testSupport import (
    giveUniverse,
    giveUniverse_ChangingBox,
    checkStringDataFromUniverse,
    getUniverseWithWaterMolecules,
    __PropertiesFinder,
)


@pytest.fixture(
    scope="module",
    params=[
        giveUniverse,
        # giveUniverse_ChangingBox,
    ],
)
def input_universe(request):
    return request.param


def test_MDA2EXYZ(input_framesSlice, input_CreateParametersToExport, input_universe):
    angles = (75.0, 60.0, 90.0)
    fourAtomsFiveFrames = input_universe(angles)
    additionalParameters = input_CreateParametersToExport(
        len(fourAtomsFiveFrames.trajectory), len(fourAtomsFiveFrames.atoms)
    )
    # making data coherent with the input_framesSlice
    for k in additionalParameters:
        additionalParameters[k] = additionalParameters[k][input_framesSlice]

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


def test_copyMDA2HDF52xyz(
    input_framesSlice, input_CreateParametersToExport, input_universe
):
    angles = (75.0, 60.0, 90.0)
    fourAtomsFiveFrames = input_universe(angles)
    additionalParameters = input_CreateParametersToExport(
        len(fourAtomsFiveFrames.trajectory), len(fourAtomsFiveFrames.atoms)
    )
    # making data coherent with the input_framesSlice
    for k in additionalParameters:
        additionalParameters[k] = additionalParameters[k][input_framesSlice]

    HDF5er.MDA2HDF5(fourAtomsFiveFrames, "test.hdf5", "4Atoms5Frames", override=True)
    with h5py.File("test.hdf5", "r") as hdf5test:
        group = hdf5test["Trajectories/4Atoms5Frames"]
        stringData = StringIO()
        HDF5er.getXYZfromTrajGroup(
            stringData, group, framesToExport=input_framesSlice, **additionalParameters
        )

        checkStringDataFromUniverse(
            stringData, fourAtomsFiveFrames, input_framesSlice, **additionalParameters
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
