"""Simple submodule with support functions for SOAPify.HDF5er"""
from sys import getsizeof
from numpy import ndarray
import h5py

from MDAnalysis import Universe as mdaUniverse


def isTrajectoryGroup(trajGroup: h5py.Group) -> bool:
    """Checks if the given group is a trajectory group

    Args:
        trajGroup (h5py.Group): the group to check

    Returns:
        bool: True if the group is a trajectory group, False otherwise
    """

    if (
        isinstance(trajGroup, h5py.Group)
        and "Trajectory" in trajGroup.keys()
        and "Types" in trajGroup.keys()
        and "Box" in trajGroup.keys()
    ):
        check = True
        for key in ["Trajectory", "Types", "Box"]:
            check &= isinstance(trajGroup[key], h5py.Dataset)
        return check
    return False


def exportChunk2HDF5(
    trajFolder: h5py.Group,
    intervalStart: int,
    intervalEnd: int,
    boxes: "list[list[list[3*float]]]|ndarray",
    coordinates: "list|ndarray",
):
    """Export a chunk of atom information in an h5py.Group

    Args:
        trajFolder (h5py.Group):
            the group in the hdf5 file
        intervalStart (int):
            the first frame to save within the group
        intervalEnd (int):
            the last frame to save within the group
        boxes (list):
            the list of the boxes, frame per frame (for each frame 6 floats
            representing the dimension and the angles of the box)
        coordinates (list):
            the list of the coordinates, frame per frame (for each frame a
            list that contains the coordinates of all atoms in that frame)
    """

    # TODO: put an if and /or decide if this is an expansion or an ovewrite
    trajFolder["Box"].resize((intervalEnd, 6))
    trajFolder["Trajectory"].resize((intervalEnd, len(coordinates[0]), 3))
    print(
        f"[{intervalStart}:{intervalEnd}]",
        len(coordinates),
        len(trajFolder["Box"][intervalStart:intervalEnd]),
        f"chunk of {getsizeof(coordinates)} B",
    )

    trajFolder["Box"][intervalStart:intervalEnd] = boxes
    trajFolder["Trajectory"][intervalStart:intervalEnd] = coordinates


def patchBoxFromTopology(hdf5TrajFile: str, topologyFile: str):  # pragma: no cover
    """Patch the non orthogonal box in the trajectory.

    This was a workaround for a __now solved__ bug in MDA, it can be used to change
    the box in a trajectory
    NOTE: this will only change the box, it will not modify the coordinates!
    NOTE: this function is not test covered and will be likely go deprecated soon

    Args:
        hdf5TrajFile (str):
            The name of the hdf5 file with the trajectories to patch
        topologyFile (str):
            The LAMMPS data file with the wnated box
    """
    universe = mdaUniverse(topologyFile, atom_style="id type x y z")
    with h5py.File(hdf5TrajFile, "a") as workFile:
        for key in workFile["Trajectories"]:
            tgroup = workFile[f"Trajectories/{key}"]
            tgroup["Box"][:] = [universe.dimensions] * tgroup["Box"].shape[0]
