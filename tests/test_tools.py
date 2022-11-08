import HDF5er
import h5py
import pytest
import numpy
from MDAnalysis.lib.mdamath import triclinic_vectors
from io import StringIO
from .testSupport import giveUniverse


def test_MDA2EXYZ(input_framesSlice):
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
    stringData = StringIO()
    HDF5er.getXYZfromMDA(
        stringData,
        fourAtomsFiveFrames,
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
            assert lines[thisline].split()[0] == fourAtomsFiveFrames.atoms.types[atomID]
            assert len(lines[thisline].split()) == 5
            assert int((lines[thisline].split()[-1])) == OneDData[frame, atomID]
            for i in range(3):
                assert (
                    float(lines[thisline].split()[i + 1])
                    == fourAtomsFiveFrames.atoms.positions[atomID][i]
                )
