import h5py
import numpy as np
from ase import Atoms as aseAtoms
from dscribe.descriptors import SOAP
from HDF5er import HDF52AseAtomsChunckedwithSymbols as HDF2ase
import time

# this is an example script for Applying the SOAP analysis on a trajectory saved on an
# HDF5 file formatted with our HDF5er and save the result in another HDF5 file


def saponify(
    trajectoryGroupName: str,
    exportGroupName: str,
    trajFname: str,
    outputFname: str,
    chunkdim: int = 100,
    SOAPnJobs: int = 8,
    SOAPatomMask: str = None,
    rcut: float = 8.0,
    nmax: int = 8,
    lmax: int = 8,
    PBC: bool = True,
):
    """[summary]

    Args:
        trajectoryGroupName (str): [description]
        trajFname (str): [description]
        outputFname (str): [description]
        chunkdim (int, optional): [description]. Defaults to 100.
        SOAPnJobs (int, optional): [description]. Defaults to 8.
        SOAPatomMask (str, optional): A plain string with the names of the atoms whose
            SOAP will be saved. Defaults to None.
    """

    with h5py.File(trajFname, "r") as trajLoader, h5py.File(
        outputFname, "a"
    ) as soapOffloader:
        traj = trajLoader[f"Trajectories/{trajectoryGroupName}"]

        symbols = traj["Types"].asstr()[:]
        # we are getting only the SOAP results of the oxigens from each water molecule in
        # this simulation
        centersMask = None
        if SOAPatomMask is not None:
            centersMask = [i for i in range(len(symbols)) if symbols[i] in SOAPatomMask]

        species = list(set(symbols))
        soapEngine = SOAP(
            species=species,
            periodic=PBC,
            rcut=rcut,
            nmax=nmax,
            lmax=lmax,
            average="off",
        )

        NofFeatures = soapEngine.get_number_of_features()

        soapDir = soapOffloader.require_group("SOAP")
        nCenters = len(symbols) if centersMask is None else len(centersMask)
        if exportGroupName not in soapDir.keys():
            soapDir.create_dataset(
                exportGroupName,
                (0, nCenters, NofFeatures),
                compression="gzip",
                compression_opts=9,
                chunks=(100, nCenters, NofFeatures),
                maxshape=(None, nCenters, NofFeatures),
            )
        SOAPout = soapDir[exportGroupName]
        SOAPout.resize((len(traj["Trajectory"]), nCenters, NofFeatures))

        for chunkTraj in traj["Trajectory"].iter_chunks():
            chunkBox = (chunkTraj[0], slice(0, 6, 1))
            print(f'working on trajectory chunk "{chunkTraj}"')
            print(f'   and working on box chunk "{repr(chunkBox)}"')
            # load in memory a chunk of data
            atoms = HDF2ase(traj, chunkTraj, chunkBox, symbols)

            jobchunk = min(chunkdim, len(atoms))
            jobStart = 0
            jobEnd = jobStart + jobchunk
            while jobStart < len(atoms):
                t1 = time.time()
                frameStart = jobStart + chunkTraj[0].start
                FrameEnd = jobEnd + chunkTraj[0].start
                print(f"working on frames: [{frameStart}:{FrameEnd}]")
                SOAPout[frameStart:FrameEnd] = soapEngine.create(
                    atoms[jobStart:jobEnd],
                    positions=[centersMask] * jobchunk,
                    n_jobs=SOAPnJobs,
                )
                t2 = time.time()
                jobchunk = min(chunkdim, len(atoms) - jobEnd)
                jobStart = jobEnd
                jobEnd = jobStart + jobchunk
                print(f"delta create= {t2-t1}")


if __name__ == "__main__":
    saponify(
        "1ns",
        "Water.hdf5",
        "WaterSOAP.hdf5",
        SOAPatomMask="O",
        chunkdim=100,
        SOAPnJobs=12,
    )
