#!/usr/bin/env python
import MDAnalysis
import h5py
from MDAnalysis import Universe as mdaUniverse
import numpy as np
from sys import getsizeof
from ase.io import iread as aseIRead
from ase.io import read as aseRead
from ase import Atoms as aseAtoms


def MDA2HDF5(
    universe: mdaUniverse,
    targetFile: str,
    name: str,
    trajChunkSize: int = 100,
    selection: MDAnalysis.AtomGroup = None,
):

    newTraj = h5py.File(targetFile, "a")
    trajGroup = newTraj.require_group(f"Trajectories/{name}")
    Universe2HDF5(universe, trajGroup, trajChunkSize=trajChunkSize, selection=selection)
    newTraj.close()


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
def xyz2hdf5Converter(xyzName, boxfilename, trajFolder):
    frame = aseRead(xyzName)
    nat = len(frame.get_positions())
    if "Types" not in list(trajFolder.keys()):
        trajFolder.create_dataset(
            "Types", (nat), compression="gzip", data=frame.get_chemical_symbols()
        )
    if "Trajectory" not in list(trajFolder.keys()):
        trajFolder.create_dataset(
            "Trajectory",
            (0, nat, 3),
            compression="gzip",
            chunks=(10, nat, 3),
            maxshape=(None, nat, 3),
        )
    if "Box" not in list(trajFolder.keys()):
        trajFolder.create_dataset(
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
                trajFolder["Box"].resize((frameNum, 6))
                trajFolder["Trajectory"].resize((frameNum, nat, 3))
                print(
                    f"[{first}-{frameNum}]",
                    len(boxes),
                    len(trajFolder["Box"][first:frameNum]),
                )

                trajFolder["Box"][first:frameNum] = boxes
                trajFolder["Trajectory"][first:frameNum] = atomicframes

                first = frameNum
                boxes = []
                atomicframes = []

        if frameNum != first:
            trajFolder["Box"].resize((frameNum, 6))
            trajFolder["Trajectory"].resize((frameNum, nat, 3))
            print(
                f"[{first}-{frameNum}]",
                len(boxes),
                len(trajFolder["Box"][first:frameNum]),
            )
            trajFolder["Box"][first:frameNum] = boxes
            trajFolder["Trajectory"][first:frameNum] = atomicframes


def HDF52AseAtomsChunckedwithSymbols(
    traj, chunkTraj, chunkBox, symbols
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
