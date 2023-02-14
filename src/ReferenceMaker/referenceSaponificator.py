import h5py
import numpy
from MDAnalysis import Universe as mdaUniverse
from MDAnalysis import AtomGroup as mdaAtomGroup
from dscribe.descriptors import SOAP as dscribeSOAP
from ase import Atoms as aseAtoms
from .ico5083 import (
    maskVertexes_ico,
    maskEdges_ico,
    maskFace111_ico,
    maskFiveFoldedAxis_ico,
)
from .th4116 import maskFace111_th, maskFace001_th, maskVertexes_th, maskEdges_th
from .dhfat3049 import maskConcave_dh, maskFiveFoldedAxis_dh
from dataclasses import dataclass

# __all__ = ["fingerprintMaker", "referenceSaponificator"]


@dataclass
class radiusInfo:
    """Helper class: contains the name and the value of the rcut for a SOAP calculation"""

    rcut: float  #: the value of the rcut
    name: str  #: the name assigned to the rcut


def mda2AseIterator(
    MDAContainer: "mdaUniverse|mdaAtomGroup", AtomName: str
) -> aseAtoms:
    """Wraps a MDAnalysis.Universe or a MDAnalysis.AtomGroup and creates an iterator for passing the frames to SOAP in ase Atoms format

    Args:
        MDAContainer (MDAnalysis.Universe | MDAnalysis.AtomGroup): a MDAnalysis.Universe or a MDAnalysis.AtomGroup with the desired system
        AtomName (str): The name of the atoms (MDA can store atom names that ase won't support)

    Yields:
        ase.Atoms: a frame converted ftom the iterator
    """

    nat = len(MDAContainer.atoms)
    centersNames = [AtomName] * nat
    atomFormula = "".join(centersNames)
    for timestep in MDAContainer.universe.trajectory:
        yield aseAtoms(
            atomFormula,
            positions=MDAContainer.atoms.positions,
            cell=MDAContainer.universe.dimensions,
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
    creates the SOAP fingerprint for a configuration of atoms or for the atoms whose
    indexes are specified in the atomsMask

    Args:
        structure (str): the name of the file with the atomsDescription
        SOAPrcut (float): the cutoff range of SOAP
        atomsMask (numpy.array, optional): the indexes of the atoms to consider for the fingerprint, if None all atoms are considered. Defaults to None.
        atomkind (str, optional): the kind of the atoms. Defaults to "Au".
        SOAPpbc (bool, optional): if set to True SOAP will consider the PBC. Defaults to True.
        SOAPnmax (int, optional): SOAP parameter. Defaults to 8.
        SOAPlmax (int, optional): SOAP parameter. Defaults to 8.
        SOAPnJobs (int, optional): number of jobs to use with SOAP. Defaults to 1.

    Returns:
        numpy.array: the fingerprint of the configuration the shape is (1,1,nFeatures), with nFeatures the number of features calculated by SOAP
    """

    u = mdaUniverse(structure, atom_style="id type x y z")

    aseAtoms = [t for t in mda2AseIterator(u, atomkind)]
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


def export2hdf5(SOAPfingerPrint: numpy.ndarray, hdf5Group: h5py.Group, dataName: str):
    """Creates or overwite the required dataset in the given group, and then stores in it the fingerprint

    Args:
        SOAPfingerPrint (numpy.ndarray): The soap fingerprint to store
        hdf5Group (h5py.Group): the group where to save or overwrite the dataset
        dataName (str): the name of the dataset to create or overwrite
    """
    dataset = hdf5Group.require_dataset(
        dataName,
        shape=SOAPfingerPrint.shape,
        data=SOAPfingerPrint,
        dtype=SOAPfingerPrint.dtype,
    )
    dataset[:] = SOAPfingerPrint


def RefCreator(
    rcut: radiusInfo,
    groupname: str,
    targetFile: h5py.File,
    structureFile: str,
    masks: "list[dict[str,numpy.ndarray]]",
    PBC: bool = False,
    SOAPlmax: int = 8,
    SOAPnmax: int = 8,
):
    """Creates a series of SOAP fingerprints and store them in a subgroup in the requested group

    RefCreator creates the requested subgroup within the HDF5 file with the following attributes:

    * **rcutName** the shortcut name given to the cutoff (such as *2R* for 2 time atomic radius or *LattUn* for "Lattice Unit")
    * **rcut** the value of the cutoff used
    * **lmax** the lmax SOAP parameter user
    * **nmax** the nmax SOAP parameter user

    the subgroup will be named after *rcutName*: *groupname/rcutName*

    Args:
        rcut (radiusInfo): The rcut radius informations
        groupname (str): the name of the group that will store the fingerprints
        targetFile (h5py.File): the hdf5file that will store the fingerprints
        structureFile (str): the name of the file with the configuration
        masks (list[dict[str,numpy.ndarray]]): list containing the dictionaries
            with the mask of atoms to consider for the fingerprints the dictionaries
            must contain a str value 'name' and a ndarray 'mask' of indexes called
        PBC (bool, optional): tells if SOAP should consider the PBC of the system. Defaults to False.
    """
    outgroup = targetFile.require_group(f"{groupname}/{rcut.name}")
    outgroup.attrs["rcutName"] = rcut.name
    outgroup.attrs["rcut"] = rcut.rcut
    outgroup.attrs["lmax"] = SOAPlmax
    outgroup.attrs["nmax"] = SOAPnmax
    for mask in masks:
        structure = structureFile
        export2hdf5(
            fingerprintMaker(
                structure,
                rcut.rcut,
                mask["mask"],
                SOAPpbc=PBC,
                SOAPlmax=SOAPlmax,
                SOAPnmax=SOAPnmax,
            ),
            outgroup,
            mask["name"],
        )


def referenceSaponificator(
    rcuts: "list[radiusInfo]",
    referencesFileName: str,
    kind: str,
    SOAPlmax: int = 8,
    SOAPnmax: int = 8,
):
    """generates the SOAP fingerprints for the reference systems.

    We propose ar reference systems:

    * Bulk:

      * simple cubic
      * body centered cubic
      * face centered cubic
      * hexagonal close packed

    * Ico5083:

      * atoms on the vertexes
      * atoms on the edges
      * 111 face atoms
      * atoms in the five folded axes

    * Dhfat3049:

      * atoms in the concave stuctures
      * atoms in the five folded axis

    * Th4116:

      * 111 face atoms
      * 001 face atoms
      * atoms on the vertexes
      * atoms on the edges

    Args:
        rcuts (list[radiusInfo]): the list of the radous info to use fo calculate the fingerprints
        referencesFile (str): the name of the reference file to create or overwite
        kind (str): the prefix of the data files that will be used as base for calculating the SOAP fingerprints
    """
    with h5py.File(referencesFileName, "a") as referenceFile:
        for rcut in rcuts:
            outgroup = referenceFile.require_group(f"Bulk/{rcut.name}")
            outgroup.attrs["rcutName"] = rcut.name
            outgroup.attrs["rcut"] = rcut.rcut
            outgroup.attrs["lmax"] = SOAPlmax
            outgroup.attrs["nmax"] = SOAPnmax
            for structure in [
                f"{kind}_bcc.data",
                f"{kind}_fcc.data",
                f"{kind}_hcp.data",
                f"{kind}_sc.data",
            ]:
                structureName = structure.rsplit(sep=".", maxsplit=1)[0]
                structureName = structureName.rsplit(sep="_", maxsplit=1)[1]
                export2hdf5(
                    fingerprintMaker(
                        structure, rcut.rcut, SOAPlmax=SOAPlmax, SOAPnmax=SOAPnmax
                    ),
                    outgroup,
                    structureName,
                )

            RefCreator(
                rcut,
                "Ico5083",
                referenceFile,
                f"{kind}_ico5083.data",
                [
                    {"name": "vertexes_ico", "mask": maskVertexes_ico},
                    {"name": "edges_ico", "mask": maskEdges_ico},
                    {"name": "face111_ico", "mask": maskFace111_ico},
                    {"name": "fiveFoldedAxis_ico", "mask": maskFiveFoldedAxis_ico},
                ],
                PBC=False,
                SOAPlmax=SOAPlmax,
                SOAPnmax=SOAPnmax,
            )
            RefCreator(
                rcut,
                "Th4116",
                referenceFile,
                f"{kind}_th4116.data",
                [
                    {"name": "face111_th", "mask": maskFace111_th},
                    {"name": "face001_th", "mask": maskFace001_th},
                    {"name": "vertexes_th", "mask": maskVertexes_th},
                    {"name": "edges_th", "mask": maskEdges_th},
                ],
                PBC=False,
                SOAPlmax=SOAPlmax,
                SOAPnmax=SOAPnmax,
            )

            RefCreator(
                rcut,
                "Dhfat3049",
                referenceFile,
                f"{kind}_dhfat3049.data",
                [
                    # {"name": "face111_dh", "mask": face111_th},
                    {"name": "concave_dh", "mask": maskConcave_dh},
                    {"name": "fiveFoldedAxis_dh", "mask": maskFiveFoldedAxis_dh},
                ],
                PBC=False,
                SOAPlmax=SOAPlmax,
                SOAPnmax=SOAPnmax,
            )


if __name__ == "__main__":
    rcuts = [2.9, 3.0, 5.8, 6.0]
    referencesFile = "AuReferences.hdf5"
    referenceSaponificator(
        rcuts=[2.9, 3.0, 5.8, 6.0], referencesFileName="AuReferences.hdf5"
    )
