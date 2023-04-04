"""This submodule gives the user some function to extract data from the hdf5 files"""
from typing import IO, List
import re
import MDAnalysis
import h5py
from ase import Atoms as aseAtoms
import numpy
from MDAnalysis.lib.mdamath import triclinic_vectors

__all__ = [
    "getXYZfromTrajGroup",
    "saveXYZfromTrajGroup",
    "HDF52AseAtomsChunckedwithSymbols",
    "getXYZfromMDA",
    "createUniverseFromSlice",
]


# TODO: using slices is not the best compromise here
# TODO: maybe it is better to make this and iterator/generator
def HDF52AseAtomsChunckedwithSymbols(
    groupTraj: h5py.Group,
    chunkTraj: "tuple[slice]",
    chunkBox: "tuple[slice]",
    symbols: "list[str]",
) -> "list[aseAtoms]":
    """generates an ase trajectory from an hdf5 trajectory

    Args:
        groupTraj (h5py.Group):
            the group within the hdf5 file where the trajectroy is stored
        chunkTraj (tuple[slice]):
            the list of the chunks of the trajectory within the given group
        chunkBox (tuple[slice]):
            the list of the chunks of the frame boxes within the given group
        symbols (list[str]):
            the list of the name of the atoms


    Returns:
        list[ase.Atoms]: the trajectory stored in the given group
    """
    atoms = []

    for frame, box in zip(
        groupTraj["Trajectory"][chunkTraj], groupTraj["Box"][chunkBox]
    ):
        # theBox = [[box[0], 0, 0], [0, box[1], 0], [0, 0, box[2]]]
        # celldisp = -box[0:3] / 2
        atoms.append(
            aseAtoms(
                symbols=symbols,
                positions=frame,
                cell=box,
                pbc=True,
                # celldisp=celldisp,
            )
        )
    return atoms


def __prepareHeaders(
    additionalColumns: dict,
    nframes: int,
    nat: int,
    allFramesProperty: str = None,
    perFrameProperties: "list[str]" = None,
) -> str:
    """Generates the static part of the header and perform sanity checks on optional data

    Args:
        additionalColumns (dict):
            the dictionary of the values of the additional columns
        nframes (int):
            the total number of frame, used for sanity checks
        nat (int):
            the total number of atoms, used for sanity checks (n is fixed)
        allFramesProperty (str, optional):
            the property common to all frames.
            Defaults to None.
        perFrameProperties (list[str], optional):
            list of properties that changes frame per frame.
            Defaults to None.

    Raises:
        ValueError:
            if additionalColumns is ill-formed
        ValueError:
            if perFrameProperties is ill-formed

    Returns:
        str: the static part of the header
    """
    additional = ""
    for key in additionalColumns:
        shapeOfData = additionalColumns[key].shape
        dim = shapeOfData[2] if len(shapeOfData) == 3 else 1
        if (  # wrong shape of the array
            (len(shapeOfData) != 2 and len(shapeOfData) != 3)
            # different data from number of frames
            or shapeOfData[0] != nframes
            # wrong number of atoms
            or shapeOfData[1] != nat
        ):
            raise ValueError(
                'Extra data passed to "getXYZfromTrajGroup" do not has the right dimensions'
                + f"\n(Trajectory shape:{(nframes,nat)}, data {key} shape:{shapeOfData})"
            )
        additional += ":" + key + ":R:" + str(dim)
    if perFrameProperties is not None:
        if len(perFrameProperties) != nframes:
            raise ValueError(
                "perFrameProperties do not have the same lenght of the trajectory"
            )
    return f"{nat}\nProperties=species:S:1:pos:R:3{additional} {allFramesProperty}"


def getXYZfromTrajGroup(
    filelike: IO,
    group: h5py.Group,
    framesToExport: "List | slice" = slice(None),
    allFramesProperty: str = "",
    perFrameProperties: "list[str]" = None,
    **additionalColumns,
) -> None:
    """generate an xyz-file in a IO object from a trajectory group in an hdf5

        the additionalColumns arguments are added as extra columns to the file,
        they must be numpy.ndarray with shape (nofFrames,NofAtoms) for 1D data
        or (nofFrames,NofAtoms,Nvalues) for multidimensional data
        this will add one or more columns to the xyz file, named after the keyword
        argument

    Args:
        filelike (IO):
            the IO destination, can be a file
        group (h5py.Group):
            the trajectory group
        frames (List or slice or None, optional):
            the frames to export. Defaults to None.
        allFramesProperty (str, optional):
            A comment string that will be present in all of the frames.
            Defaults to "".
        perFrameProperties (list[str], optional):
            A list of comment.
            Defaults to None.
        additionalColumns():
            the additional columns to add to the file: each new keyword arguments
            will add a column to the xyz file
    """

    atomtypes = group["Types"].asstr()
    boxes: h5py.Dataset = group["Box"][framesToExport]
    coordData: h5py.Dataset = group["Trajectory"][framesToExport]

    trajlen: int = coordData.shape[0]
    nat: int = coordData.shape[1]

    header: str = __prepareHeaders(
        additionalColumns, nframes=trajlen, nat=nat, allFramesProperty=allFramesProperty
    )

    for frameIndex in range(trajlen):
        coord = coordData[frameIndex, :]
        data = __writeAframe(
            header,
            nat,
            atomtypes,
            coord,
            boxes[frameIndex],
            perFrameProperties[frameIndex] if perFrameProperties is not None else None,
            # this may create a bottleneck
            **{k: additionalColumns[k][frameIndex] for k in additionalColumns},
        )
        filelike.write(data)


# TODO: it is slow with huge files
def saveXYZfromTrajGroup(
    filename: str,
    group: h5py.Group,
    framesToExport: "List | slice" = slice(None),
    allFramesProperty: str = "",
    perFrameProperties: "list[str]" = None,
    **additionalColumns,
) -> None:
    """Saves "filename" as an xyz file

    see saveXYZfromTrajGroup this calls getXYZfromTrajGroup and treats the inputs in the same way

    Args:
        filename (str):
            name of the file
        group (h5py.Group):
            the trajectory group
        additionalColumsn():
            the additional columns to add to the file
        allFramesProperty (str, optional):
            A comment string that will be present in all of the frames.
            Defaults to "".
        perFrameProperties (list[str], optional):
            A list of comment.
            Defaults to None.
    """
    with open(filename, "w") as file:
        getXYZfromTrajGroup(
            file,
            group,
            framesToExport,
            allFramesProperty,
            perFrameProperties,
            **additionalColumns,
        )


def getXYZfromMDA(
    filelike: IO,
    trajToExport: "MDAnalysis.Universe | MDAnalysis.AtomGroup",
    framesToExport: "List | slice" = slice(None),
    allFramesProperty: str = "",
    perFrameProperties: "list[str]" = None,
    **additionalColumns,
) -> None:
    """generate an xyz-file in a IO object from an MDA trajectory

        the additionalColumns arguments are added as extra columns to the file,
        they must be numpy.ndarray with shape (nofFrames,NofAtoms) for 1D data
        or (nofFrames,NofAtoms,Nvalues) for multidimensional data
        this will add one or more columns to the xyz file

    Args:
        filelike (IO):
            the IO destination, can be a file
        group (MDAnalysis.Universe | MDAnalysis.AtomGroup):
            the universe or the selection of atoms to export
        frames (List or slice or None, optional):
            the frames to export.
            Defaults to None.
        allFramesProperty (str, optional):
            A comment string that will be present in all of the frames.
            Defaults to "".
        perFrameProperties (list[str], optional):
            A list of comment.
            Defaults to None.
        additionalColumns():
            the additional columns to add to the file: each new keyword arguments
            will add a column to the xyz file
    """

    atoms = trajToExport.atoms
    universe = trajToExport.universe
    atomtypes = atoms.types
    coordData: "MDAnalysis.Universe | MDAnalysis.AtomGroup" = universe.trajectory[
        framesToExport
    ]

    trajlen: int = len(coordData)
    nat: int = len(atoms)

    header: str = __prepareHeaders(
        additionalColumns, nframes=trajlen, nat=nat, allFramesProperty=allFramesProperty
    )
    for frameIndex, _ in enumerate(universe.trajectory[framesToExport]):
        coord = atoms.positions
        data = __writeAframe(
            header,
            nat,
            atomtypes,
            coord,
            universe.dimensions,
            perFrameProperties[frameIndex] if perFrameProperties is not None else None,
            # this may create a bottleneck
            **{k: additionalColumns[k][frameIndex] for k in additionalColumns},
        )
        filelike.write(data)


def __writeAframe(
    header,
    nat,
    atomtypes,
    coord,
    boxDimensions,
    perFrameProperty: str = None,
    **additionalColumns,
) -> str:
    data = f"{header}"
    data += f" {perFrameProperty}" if perFrameProperty is not None else ""
    theBox = triclinic_vectors(boxDimensions)
    data += f' Lattice="{theBox[0][0]} {theBox[0][1]} {theBox[0][2]} '
    data += f"{theBox[1][0]} {theBox[1][1]} {theBox[1][2]} "
    data += f'{theBox[2][0]} {theBox[2][1]} {theBox[2][2]}"'
    data += "\n"

    for atomID in range(nat):
        data += (
            f"{atomtypes[atomID]} {coord[atomID,0]} {coord[atomID,1]} {coord[atomID,2]}"
        )
        for key in additionalColumns:
            # this removes the brackets from the data if the dimensions are >1
            data += " " + re.sub("( \[|\[|\])", "", str(additionalColumns[key][atomID]))
        data += "\n"
    return data


def createUniverseFromSlice(
    trajectoryGroup: h5py.Group, useSlice=slice(None)
) -> MDAnalysis.Universe:
    """Creates a MDanalysis.Universe from a trajectory group

    Args:
        trajectoryGroup (h5py.Group):
            the given trajectory group
        useSlice (_type_, optional):
            the asked slice from wich create an universe.
            Defaults to slice(None).

    Returns:
        MDAnalysis.Universe:
            an universe containing the wnated part of the trajectory
    """
    # TODO: also add a slice for the atoms
    traj = trajectoryGroup["Trajectory"]
    box = trajectoryGroup["Box"]
    atomNames = trajectoryGroup["Types"].asstr()
    nAt = traj.shape[1]
    # TODO add names
    toRet = MDAnalysis.Universe.empty(
        n_atoms=nAt,
        n_residues=nAt,
        n_segments=1,
        atom_resindex=numpy.arange(nAt),
        residue_segindex=[1] * nAt,
        trajectory=True,
    )
    toRet.add_TopologyAttr("type", atomNames)
    toRet.load_new(
        traj[useSlice],
        format=MDAnalysis.coordinates.memory.MemoryReader,
        dimensions=box[useSlice],
    )

    return toRet
