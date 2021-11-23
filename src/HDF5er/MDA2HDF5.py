import MDAnalysis
import h5py
from .HDF5erUtils import exportChunk2HDF5


def Universe2HDF5(
    MDAUniverseOrSelection: "MDAnalysis.Universe | MDAnalysis.AtomGroup",
    trajFolder: h5py.Group,
    trajChunkSize: int = 100,
):
    """Uploads an mda.Universe or an mda.AtomGroup to a h5py.Group in an hdf5 file

    Args:
        MDAUniverseOrSelection (MDAnalysis.Universe or MDAnalysis.AtomGroup): the container with the trajectory data
        trajFolder (h5py.Group): the group in which store the trajectory in the hdf5 file
        trajChunkSize (int, optional): The desired dimension of the chunks of data that are stored in the hdf5 file. Defaults to 100.
    """

    atoms = MDAUniverseOrSelection.atoms
    universe = (
        MDAUniverseOrSelection
        if MDAUniverseOrSelection is MDAnalysis.Universe
        else MDAUniverseOrSelection.universe
    )
    nat = len(atoms)

    if "Types" not in list(trajFolder.keys()):
        trajFolder.create_dataset("Types", (nat), compression="gzip", data=atoms.types)

    if "Trajectory" not in list(trajFolder.keys()):
        trajFolder.create_dataset(
            "Trajectory",
            (0, nat, 3),
            compression="gzip",
            chunks=(trajChunkSize, nat, 3),
            maxshape=(None, nat, 3),
        )

    if "Box" not in list(trajFolder.keys()):
        trajFolder.create_dataset(
            "Box", (0, 6), compression="gzip", chunks=True, maxshape=(None, 6)
        )

    frameNum = 0
    first = 0
    boxes = []
    atomicframes = []
    for frame in universe.trajectory:
        boxes.append(universe.dimensions)
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
    MDAUniverseOrSelection: "MDAnalysis.Universe | MDAnalysis.AtomGroup",
    targetHDF5File: str,
    groupName: str,
    trajChunkSize: int = 100,
):
    """Opens or creates the given HDF5 file, request the user's chosen group, then uploads an mda.Universe or an mda.AtomGroup to a h5py.Group in an hdf5 file

        **WARNING**: in the HDF5 file if the chosen group is already present it will be overwritten by the new data

    Args:
        MDAUniverseOrSelection (MDAnalysis.Universe or MDAnalysis.AtomGroup): the container with the trajectory data
        targetHDF5File (str): the name of HDF5 file
        groupName (str): the name of the group in wich save the trajectory data within the `targetHDF5File`
        trajChunkSize (int, optional): The desired dimension of the chunks of data that are stored in the hdf5 file. Defaults to 100.
    """
    with h5py.File(targetHDF5File, "a") as newTraj:
        trajGroup = newTraj.require_group(f"Trajectories/{groupName}")
        Universe2HDF5(MDAUniverseOrSelection, trajGroup, trajChunkSize=trajChunkSize)
