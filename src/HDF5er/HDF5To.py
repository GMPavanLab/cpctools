import h5py
from ase import Atoms as aseAtoms

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
    data = "4\n\nH 1 1 1\n"
    atomtypes = group["Types"]
    nat = atomtypes.shape[0]
    boxes = group["Box"]
    coord = group["Trajectory"]
    trajlen = coord.shape[0]
    for frame in range(trajlen):
        data += f"{nat}\n\n"
        for atomID in range(nat):
            data += f"{atomtypes[atomID]} {coord[frame,0]} {coord[frame,1]} {coord[frame,2]}\n"
    return data
