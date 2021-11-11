import MDAnalysis
import h5py
from MDAnalysis import Universe as mdaUniverse
import numpy as np
from sys import getsizeof
from ase.io import iread as aseIRead
from ase.io import read as aseRead
from ase import Atoms as aseAtoms

__all__ = ["MDA2HDF5", "Universe2HDF5"]


def exportChunk2HDF5(
    trajFolder: h5py.Group, intervalStart: int, intervalEnd: int, boxes, coordinates
):
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


def Universe2HDF5(
    universe: mdaUniverse,
    trajFolder: h5py.Group,
    trajChunkSize: int = 100,
    selection: MDAnalysis.AtomGroup = None,
):
    """Upload an Universe or a selection to a group in an hdf5 file

    Args:
        universe (mdaUniverse): [description]
        trajFolder (h5py.Group): [description]
        trajChunkSize (int, optional): [description]. Defaults to 100.
        selection (MDAnalysis.AtomGroup, optional): [description]. Defaults to None.
    """
    atoms = selection.atoms if selection is not None else universe.atoms
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


# todo: converto use exportChunk2HDF5
def xyz2hdf5Converter(xyzName: str, boxfilename: str, group: h5py.Group):
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
        )
    if "Box" not in list(group.keys()):
        group.create_dataset(
            "Box", (0, 6), compression="gzip", chunks=True, maxshape=(None, 6)
        )
    xyz = aseIRead(xyzName)
    with open(boxfilename, "r") as bf:
        frameNum = 0
        first = 0
        boxes = []
        atomicframes = []
        for box, frame in zip(bf, xyz):
            t = box.split()
            boxInfo = np.array(
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


def HDF52AseAtomsChunckedwithSymbols(
    traj: h5py.Group,
    chunkTraj: "tuple[slice]",
    chunkBox: "tuple[slice]",
    symbols: "list[str]",
) -> "list[aseAtoms]":
    atoms = []

    for frame, box in zip(traj["Trajectory"][chunkTraj], traj["Box"][chunkBox]):
        theBox = [[box[0], 0, 0], [0, box[1], 0], [0, 0, box[2]]]
        # celldisp = -box[0:3] / 2
        atoms.append(
            aseAtoms(
                symbols=symbols,
                positions=frame,
                cell=theBox,
                pbc=True,
                # celldisp=celldisp,
            )
        )
    return atoms


def MDA2HDF5(
    universe: mdaUniverse,
    targetHDF5File: str,
    groupName: str,
    trajChunkSize: int = 100,
    selection: MDAnalysis.AtomGroup = None,
):
    """Opens or creates the given HDF5 file, creates there a requested group and writes in it the Universe or the selection

    Args:
        universe (mdaUniverse): [description]
        targetHDF5File (str): [description]
        groupName (str): [description]
        trajChunkSize (int, optional): [description]. Defaults to 100.
        selection (MDAnalysis.AtomGroup, optional): [description]. Defaults to None.
    """
    with h5py.File(targetHDF5File, "a") as newTraj:
        trajGroup = newTraj.require_group(f"Trajectories/{groupName}")
        Universe2HDF5(
            universe, trajGroup, trajChunkSize=trajChunkSize, selection=selection
        )
