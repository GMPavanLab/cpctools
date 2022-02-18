import h5py
from ase import Atoms as aseAtoms
from MDAnalysis.lib.mdamath import triclinic_vectors
import re

__all__ = [
    "getXYZfromTrajGroup",
    "saveXYZfromTrajGroup",
    "HDF52AseAtomsChunckedwithSymbols",
]

# TODO: using slices is not the best compromise here
def HDF52AseAtomsChunckedwithSymbols(
    groupTraj: h5py.Group,
    chunkTraj: "tuple[slice]",
    chunkBox: "tuple[slice]",
    symbols: "list[str]",
) -> "list[aseAtoms]":
    """generates an ase trajectory from an hdf5 trajectory

    Args:
        groupTraj (h5py.Group): the group within the hdf5 file where the trajectroy is stored
        chunkTraj (tuple[slice]): the list of the chunks of the trajectory within the given group
        chunkBox (tuple[slice]): the list of the chunks of the frame boxes within the given group
        symbols (list[str]): the list of the name of the atoms


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


def getXYZfromTrajGroup(group: h5py.Group, **additionalColumns) -> str:
    """generate an xyz-style string from a trajectory group in an hdf5

        The string generated can be then exported to a file,
        the additionalColumns arguments are added as extra columns to the file,
        they must be numpy.ndarray with shape (nofFrames,NofAtoms) for 1D data
        or (nofFrames,NofAtoms,Nvalues) for multidimensional data
        this will add one or more columns to the xyz file

    Args:
        group (h5py.Group): the trajectory group
        additionalColumsn(): the additional columns to add to the file

    Returns:
        str: the content of the xyz file
    """
    data = ""
    atomtypes = group["Types"].asstr()

    boxes = group["Box"]
    coord = group["Trajectory"]
    trajlen = coord.shape[0]
    nat = coord.shape[1]
    additional = ""
    for key in additionalColumns:
        shapeOfData = additionalColumns[key].shape
        dim = shapeOfData[2] if len(shapeOfData) == 3 else 1
        if (  # wrong shape of the array
            (len(shapeOfData) != 2 and len(shapeOfData) != 3)
            # wrong number of frames
            or shapeOfData[0] != trajlen
            # wrong number of atoms
            or shapeOfData[1] != nat
        ):
            raise ValueError(
                'Extra data passed to "getXYZfromTrajGroup" do not has the right dimensions'
            )
        additional += ":" + key + ":R:" + str(dim)

    header: str = f"{nat}\nProperties=species:S:1:pos:R:3{additional} Lattice="
    for frame in range(trajlen):
        data += f"{header}"
        theBox = triclinic_vectors(boxes[frame])
        data += f'"{theBox[0][0]} {theBox[0][1]} {theBox[0][2]} '
        data += f"{theBox[1][0]} {theBox[1][1]} {theBox[1][2]} "
        data += f'{theBox[2][0]} {theBox[2][1]} {theBox[2][2]}"\n'
        for atomID in range(nat):
            data += f"{atomtypes[atomID]} {coord[frame,atomID,0]} {coord[frame,atomID,1]} {coord[frame,atomID,2]}"
            for key in additionalColumns:
                # this removes the brackets from the data if the dimensions are >1
                data += " " + re.sub(
                    "( \[|\[|\])", "", str(additionalColumns[key][frame, atomID])
                )
            data += "\n"
    return data


def saveXYZfromTrajGroup(filename: str, group: h5py.Group, **additionalColumns) -> None:
    """Saves "filename" as an xyz file see
       saveXYZfromTrajGroup this calls getXYZfromTrajGroup and treats the inputs in the same way

    Args:
        filename (str): name of the file
        group (h5py.Group): the trajectory group
        additionalColumsn(): the additional columns to add to the file
    """
    dataStr = getXYZfromTrajGroup(group, **additionalColumns)
    with open(filename, "w") as file:
        file.write(dataStr)