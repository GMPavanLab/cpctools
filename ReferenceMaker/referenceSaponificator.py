import h5py
import numpy
from MDAnalysis import Universe as mdaUniverse
from dscribe.descriptors import SOAP as dscribeSOAP
from ase import Atoms as aseAtoms
from .Masks_Ico5083 import *
from .Masks_th4116 import *
from .Masks_dhfat3049 import *


class mda2AseIterator:
    """
    Wraps an MDAnalysis.Universe and creates an iterator for passing the frames to SOAP
    """

    def __init__(self, univ: mdaUniverse, AtomName: str) -> None:
        self.universe = univ
        self.nat = len(self.universe.atoms)
        self.centersNames = [AtomName] * self.nat
        self.atomFormula = "".join(self.centersNames)

    def __iter__(self):
        for timestep in self.universe.trajectory:
            yield aseAtoms(
                self.atomFormula,
                positions=self.universe.atoms.positions,
                cell=self.universe.dimensions,
                pbc=True,
            )


def fingerprintMaker(
    structure: str,
    SOAPrcut: float,
    atomsMask: "numpy.array|None" = None,
    atomkind: str = "Au",
    SOAPpbc: bool = True,
    SOAPnmax: int = 8,
    SOAPlmax: int = 8,
    SOAPnJobs: int = 1,
) -> numpy.array:
    """
    create the SOAP fingerprint for a configuration of atoms or for the atoms whose
    indexes are specified in the atomsMask

    Args:
        structure (str): the name of the file with the atomsDescription
        SOAPrcut (float): the cutoff range of SOAP
        atomsMask (numpy.array, optional): the indexes of the atoms to consider for the
            fingerprint, if None all atoms are considered. Defaults to None.
        atomkind (str, optional): the kind of the atoms. Defaults to "Au".
        SOAPpbc (bool, optional): if set to True SOAP will consider the PBC. Defaults to True.
        SOAPnmax (int, optional): SOAP parameter. Defaults to 8.
        SOAPlmax (int, optional): SOAP parameter. Defaults to 8.
        SOAPnJobs (int, optional): number of jobs to use with SOAP. Defaults to 1.

    Returns:
        numpy.array: the fingerprint of the configuration the shape is (1,1,nFeatures),
            with nFeatures the number of features calculated by SOAP
    """

    u = mdaUniverse(structure, atom_style="id type x y z")

    framesIterator = mda2AseIterator(u, atomkind)

    aseAtoms = [t for t in framesIterator]
    soapEngine = dscribeSOAP(
        SOAPrcut, SOAPnmax, SOAPlmax, species=[atomkind], periodic=SOAPpbc
    )

    res = soapEngine.create(
        aseAtoms,
        positions=[atomsMask] * len(aseAtoms) if atomsMask is not None else None,
        n_jobs=SOAPnJobs,
    )
    fingerprint = numpy.mean(res[0], axis=0)
    fingerprint = numpy.array([fingerprint])
    return fingerprint


# wfpipes.
def RefCreator(
    rcuts, groupname, targetFile: h5py.File, structureFile, masks, PBC: bool = False
):
    for rcut in rcuts:
        outgroup = targetFile.require_group(f"{groupname}/R{rcut}")
        outgroup.attrs["rcut"] = rcut
        outgroup.attrs["lmax"] = 8
        outgroup.attrs["nmax"] = 8
        for mask in masks:
            structure = structureFile

            fp = fingerprintMaker(structure, rcut, mask["mask"], SOAPpbc=PBC)

            dataset = outgroup.require_dataset(
                mask["name"], shape=fp.shape, data=fp, dtype=fp.dtype
            )
            dataset[:] = fp


def referenceSaponificator(rcuts, referencesFile):
    with h5py.File(referencesFile, "a") as file:
        # for rcut in [6.0, 8.0, 10.0]:
        for rcut in rcuts:
            outgroup = file.require_group(f"Bulk/R{rcut}")
            outgroup.attrs["rcut"] = rcut
            outgroup.attrs["lmax"] = 8
            outgroup.attrs["nmax"] = 8
            for structure in [
                "bcc.data",
                "fcc.data",
                "hcp.data",
                "hcp.data",
                "sc.data",
            ]:

                fp = fingerprintMaker(structure, rcut)
                name = structure.rsplit(sep=".", maxsplit=1)[0]
                ds = outgroup.require_dataset(name, shape=fp.shape, dtype=fp.dtype)
                ds[:] = fp

        RefCreator(
            rcuts,
            "Ico5083",
            file,
            "ico5083.data",
            [
                {"name": "vertexes_ico", "mask": vertexes_ico},
                {"name": "edges_ico", "mask": edges_ico},
                {"name": "face111_ico", "mask": face111_ico},
                {"name": "fiveFoldedAxis_ico", "mask": fiveFoldedAxis_ico},
            ],
            PBC=False,
        )
        RefCreator(
            rcuts,
            "Th4116",
            file,
            "th4116.data",
            [
                {"name": "face111_th", "mask": face111_th},
                {"name": "face001_th", "mask": face001_th},
                {"name": "vertexes_th", "mask": vertexes_th},
                {"name": "edges_th", "mask": edges_th},
            ],
            PBC=False,
        )

        RefCreator(
            rcuts,
            "Dhfat3049",
            file,
            "dhfat3049.data",
            [
                # {"name": "face111_dh", "mask": face111_th},
                {"name": "concave_dh", "mask": concave_deca},
                {"name": "fiveFoldedAxis_dh", "mask": fiveFoldedAxis_deca},
            ],
            PBC=False,
        )


if __name__ == "__main__":
    rcuts = [2.9, 3.0, 5.8, 6.0]
    referencesFile = "AuReferences.hdf5"
    referenceSaponificator(
        rcuts=[2.9, 3.0, 5.8, 6.0], referencesFile="AuReferences.hdf5"
    )
