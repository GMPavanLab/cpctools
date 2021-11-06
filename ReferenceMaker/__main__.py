from .referenceSaponificator import referenceSaponificator
from lammps import lammps
import numpy

pair_coeff = "pair_coeff	1 1 2.88 10.35	4.178	0.210	1.818	4.07293506	4.9883063257983666"
atomMass = 196.96655
diameter = 2.88
referencesFile = "AuReferences.hdf5"
latticebcc = 2 * diameter / numpy.sqrt(3.0)
latticefcc = diameter * numpy.sqrt(2.0)

for d in [
    {"name": "bcc", "latticeCMD": f"lattice bcc {latticebcc}"},
    {"name": "fcc", "latticeCMD": f"lattice fcc {latticefcc}"},
    {"name": "hcp", "latticeCMD": f"lattice hcp {diameter}"},
    {"name": "sc", "latticeCMD": f"lattice sc {diameter}"},
    # "npMinimizer.in",
]:
    input = d["name"]
    with lammps() as lmp:

        lmp.commands_list(
            [
                "units           metal",
                "atom_style      atomic",
                "boundary	p p p",
                d["latticeCMD"],
                "region myreg block 0 8 0 8 0 8",
                "create_box      1 myreg",
                "create_atoms    1 box",
                f"mass 1 {atomMass}",
                "pair_style	smatb/single",
                pair_coeff,
                "neighbor	8.0 bin",
                "neigh_modify	every 1 delay 0 check yes",
                "fix            boxmin all box/relax iso 1.0",
                "minimize       1.0e-8 1.0e-10 10000 100000",
                "unfix boxmin",
                "minimize       1.0e-8 1.0e-10 10000 100000",
                f"write_data     {input}.data",
            ]
        )
        # print(lmp.extract_box())
        # print(0,lmp.extract_atom("id")[0],[lmp.extract_atom("x")[0][i] for i in [0, 1, 2]],)
        ## TODO extract box and atoms data and pass them to ase, so writing data files is not necessary
for np in ["dhfat3049", "ico5083", "th4116"]:
    with lammps() as lmp:

        lmp.commands_list(
            [
                "units           metal",
                "atom_style      atomic",
                "boundary	p p p",
                f"read_data       {np}.reference",
                f"mass 1 {atomMass}",
                "pair_style	smatb/single",
                pair_coeff,
                "neighbor	8.0 bin",
                "neigh_modify	every 1 delay 0 check yes",
                "minimize       1.0e-8 1.0e-10 10000 100000",
                f"write_data {np}.data",
            ]
        )

referenceSaponificator(rcuts=[2.9, 3.0, 5.8, 6.0], referencesFile=referencesFile)
