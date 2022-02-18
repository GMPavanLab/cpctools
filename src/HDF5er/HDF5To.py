import h5py
from ase import Atoms as aseAtoms
from MDAnalysis.lib.mdamath import triclinic_vectors

__all__ = ["getXYZfromTrajGroup", "HDF52AseAtomsChunckedwithSymbols"]

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


def getXYZfromTrajGroup(group: h5py.Group) -> str:
    """generate an xyz-style string from a trajectory group in an hdf5

    Args:
        group (h5py.Group): the trajectory group

    Returns:
        str: the content of the xyz file
    """
    data = ""
    atomtypes = group["Types"].asstr()

    boxes = group["Box"]
    coord = group["Trajectory"]
    trajlen = coord.shape[0]
    nat = coord.shape[1]

    header: str = f"{nat}\nProperties=species:S:1:pos:R:3:Lattice="
    for frame in range(trajlen):
        data += f"{header}"
        theBox = triclinic_vectors(boxes[frame])
        data += f'"{theBox[0][0]} {theBox[0][1]} {theBox[0][2]} '
        data += f"{theBox[1][0]} {theBox[1][1]} {theBox[1][2]} "
        data += f'{theBox[2][0]} {theBox[2][1]} {theBox[2][2]}"\n'
        for atomID in range(nat):
            data += f"{atomtypes[atomID]} {coord[frame,atomID,0]} {coord[frame,atomID,1]} {coord[frame,atomID,2]}\n"
    return data
