from ase.io import iread as aseIRead
from ase.io import read as aseRead
from ase import Atoms as aseAtoms
import h5py

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
