import HDF5er
import pytest
import MDAnalysis
from MDAnalysis.topology.MinimalParser import MinimalParser
from MDAnalysis.coordinates.memory import MemoryReader
import numpy

# from MDAnalysis.tests.datafiles import


def giveUniverse() -> MDAnalysis.Universe:
    traj = numpy.array(
        [
            [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
            [[0.1, 0.1, 0.1], [1.1, 1.1, 1.1], [2.1, 2.1, 2.1]],
            [[0.2, 0.2, 0.2], [1.2, 1.2, 1.2], [2.2, 2.2, 2.2]],
        ]
    )
    p = MinimalParser(filename="")  # , n_atoms=3)
    p.parse(n_atoms=3)
    u = MDAnalysis.Universe(p, traj, format=MemoryReader, order="fac")
    u.trajectory
    for frame in u.trajectory:
        u.atoms.positions()
    return u


def test_MDA2HDF5():
    # Given an MDA Universe:
    u = giveUniverse()
