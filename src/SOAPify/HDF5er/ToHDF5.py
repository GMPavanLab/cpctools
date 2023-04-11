"""This submodule contains some function to import date to the hdf5 files"""
import warnings
import h5py
import numpy
from MDAnalysis import Universe as mdaUniverse, AtomGroup as mdaAtomGroup
from deprecated import deprecated
from ase.io import iread as aseIRead
from ase.io import read as aseRead
from .HDF5erUtils import exportChunk2HDF5


def universe2HDF5(
    mdaTrajectory: "mdaUniverse | mdaAtomGroup",
    trajFolder: h5py.Group,
    trajChunkSize: int = 100,
    trajslice: slice = slice(None),
    useType="float64",
):
    """Uploads an mda.Universe or an mda.AtomGroup to a h5py.Group in an hdf5 file

    Args:
        MDAUniverseOrSelection (MDAnalysis.Universe or MDAnalysis.AtomGroup):
            the container with the trajectory data
        trajFolder (h5py.Group):
            the group in which store the trajectory in the hdf5 file
        trajChunkSize (int, optional):
            The desired dimension of the chunks of data that are stored in the hdf5 file.
            Defaults to 100.
        useType (str,optional):
            The precision used to store the data. Defaults to "float64".
    """

    atoms = mdaTrajectory.atoms
    universe = mdaTrajectory.universe
    nat = len(atoms)
    useType = numpy.dtype(useType)
    if "Types" not in list(trajFolder.keys()):
        trajFolder.create_dataset("Types", (nat), compression="gzip", data=atoms.types)

    if "Trajectory" not in list(trajFolder.keys()):
        trajFolder.create_dataset(
            "Trajectory",
            (0, nat, 3),
            compression="gzip",
            chunks=(trajChunkSize, nat, 3),
            maxshape=(None, nat, 3),
            dtype=useType,
        )

    if "Box" not in list(trajFolder.keys()):
        trajFolder.create_dataset(
            "Box",
            (0, 6),
            compression="gzip",
            chunks=True,
            maxshape=(None, 6),
            dtype=useType,
        )

    frameNum = 0
    first = 0
    boxes = []
    atomicframes = []
    for _ in universe.trajectory[trajslice]:
        boxes.append(universe.dimensions.copy())
        atomicframes.append(atoms.positions)
        frameNum += 1
        if frameNum % trajChunkSize == 0:
            exportChunk2HDF5(trajFolder, first, frameNum, boxes, atomicframes)

            first = frameNum
            boxes = []
            atomicframes = []

    # in the case that there are some dangling frames
    if frameNum != first:
        exportChunk2HDF5(trajFolder, first, frameNum, boxes, atomicframes)


def MDA2HDF5(
    mdaTrajectory: "mdaUniverse | mdaAtomGroup",
    targetHDF5File: str,
    groupName: str,
    trajChunkSize: int = 100,
    override: bool = False,
    attrs: dict = None,
    trajslice: slice = slice(None),
    useType="float64",
):
    """Creates an HDF5 trajectory groupfrom an mda trajectory

        Opens or creates the given HDF5 file, request the user's chosen group,
        then uploads an mda.Universe or an mda.AtomGroup to a h5py.Group in an
        hdf5 file

        **WARNING**: in the HDF5 file if the chosen group is already present it
        will be overwritten by the new data

    Args:
        MDAUniverseOrSelection (MDAnalysis.Universe or MDAnalysis.AtomGroup):
            the container with the trajectory data
        targetHDF5File (str):
            the name of HDF5 file
        groupName (str):
            the name of the group in wich save the trajectory data within the
            `targetHDF5File`
        trajChunkSize (int, optional):
            The desired dimension of the chunks of data that are stored in the
            hdf5 file. Defaults to 100.
        override (bool, optional):
            If true the hdf5 file will be completely overwritten.
            Defaults to False.
        useType (str,optional):
            The precision used to store the data. Defaults to "float64".
    """
    with h5py.File(targetHDF5File, "w" if override else "a") as newTraj:
        trajGroup = newTraj.require_group(f"Trajectories/{groupName}")
        universe2HDF5(
            mdaTrajectory,
            trajGroup,
            trajChunkSize=trajChunkSize,
            trajslice=trajslice,
            useType=useType,
        )
        if attrs:
            for key in attrs.keys():
                trajGroup.attrs.create(key, attrs[key])


@deprecated('xyz2hdf5Converter is "legacy code" **not covered by unit tests**')
def xyz2hdf5Converter(
    xyzName: str, boxfilename: str, group: h5py.Group
):  # pragma: no cover
    """Generate an HDF5 trajectory from an xyz file and a box file

        This function reads an xyz file with ase and then export it to an
        trajectory in and hdf5 file,
        the user should pass the group within the hdf5file to this function

        **NB**: this is "legacy code" **not covered by unit tests**, use with caution

    Args:
        xyzName (str):
            the filename of the xyz trajaectory
        boxfilename (str):
            the filename of the  per frame box dimensions
        group (h5py.Group):
            the group within the hdf5 file where the trajectroy will be saved
    """
    # TODO: convert use exportChunk2HDF5
    warnings.warn(
        'this is untested "legacy code", use with caution', PendingDeprecationWarning
    )
    frame = aseRead(xyzName)
    nat = len(frame.get_positions())
    if "Types" not in list(group.keys()):
        group.create_dataset(
            "Types", (nat), compression="gzip", data=frame.get_chemical_symbols()
        )
    if "Trajectory" not in list(group.keys()):
        group.create_dataset(
            "Trajectory",
            (0, nat, 3),
            compression="gzip",
            chunks=(10, nat, 3),
            maxshape=(None, nat, 3),
            dtype=numpy.float64,
        )
    if "Box" not in list(group.keys()):
        group.create_dataset(
            "Box",
            (0, 6),
            compression="gzip",
            chunks=True,
            maxshape=(None, 6),
            dtype=numpy.float64,
        )
    xyz = aseIRead(xyzName)
    with open(boxfilename, "r") as bf:
        frameNum = 0
        first = 0
        boxes = []
        atomicframes = []
        for box, frame in zip(bf, xyz):
            t = box.split()
            boxInfo = numpy.array(
                [float(t[0]), float(t[1]), float(t[2]), 90.0, 90.0, 90.0]
            )
            boxes.append(boxInfo)
            atomicframes.append(frame.get_positions())
            frameNum += 1
            if frameNum % 20 == 0:
                group["Box"].resize((frameNum, 6))
                group["Trajectory"].resize((frameNum, nat, 3))
                print(
                    f"[{first}-{frameNum}]",
                    len(boxes),
                    len(group["Box"][first:frameNum]),
                )

                group["Box"][first:frameNum] = boxes
                group["Trajectory"][first:frameNum] = atomicframes

                first = frameNum
                boxes = []
                atomicframes = []

        if frameNum != first:
            group["Box"].resize((frameNum, 6))
            group["Trajectory"].resize((frameNum, nat, 3))
            print(
                f"[{first}-{frameNum}]",
                len(boxes),
                len(group["Box"][first:frameNum]),
            )
            group["Box"][first:frameNum] = boxes
            group["Trajectory"][first:frameNum] = atomicframes
