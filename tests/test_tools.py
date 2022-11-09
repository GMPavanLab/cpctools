import HDF5er
import h5py
import pytest
import numpy
from MDAnalysis.lib.mdamath import triclinic_vectors
from io import StringIO
from .testSupport import giveUniverse, checkStringDataFromUniverse


def test_MDA2EXYZ(input_framesSlice):
    angles = (75.0, 60.0, 90.0)
    fourAtomsFiveFrames = giveUniverse(angles)
    rng = numpy.random.default_rng(12345)
    OneDDataOrig = rng.integers(
        0, 7, size=(len(fourAtomsFiveFrames.trajectory), len(fourAtomsFiveFrames.atoms))
    )

    OneDData = OneDDataOrig[input_framesSlice]
    stringData = StringIO()
    HDF5er.getXYZfromMDA(
        stringData,
        fourAtomsFiveFrames,
        framesToExport=input_framesSlice,
        OneDData=OneDData,
    )

    checkStringDataFromUniverse(
        stringData,
        fourAtomsFiveFrames,
        input_framesSlice,
        OneDData=OneDData,
    )
