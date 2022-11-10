import MDAnalysis
import numpy
import re
from io import StringIO
from MDAnalysis.lib.mdamath import triclinic_vectors

__PropertiesFinder = re.compile('Properties="{0,1}(.*?)"{0,1}(?: |$)', flags=0)
__LatticeFinder = re.compile('Lattice="(.*?)"', flags=0)


def checkStringDataFromUniverse(
    stringData: StringIO,
    myUniverse: "MDAnalysis.Universe | MDAnalysis.AtomGroup",
    frameSlice: slice,
    allFramesProperty: str = None,
    perFrameProperties: list = None,
    **passedValues,
):
    universe = myUniverse.universe
    atoms = myUniverse.atoms
    lines = stringData.getvalue().splitlines()
    nat = int(lines[0])
    assert int(lines[0]) == len(atoms)
    assert lines[2].split()[0] == atoms.types[0]
    for frame, traj in enumerate(universe.trajectory[frameSlice]):
        frameID = frame * (nat + 2)
        assert int(lines[frameID]) == nat
        Lattice = __LatticeFinder.search(lines[frameID + 1]).group(1).split()
        Properties = __PropertiesFinder.search(lines[frameID + 1]).group(1).split(":")
        WhereIsTheProperty = dict()
        for name in passedValues.keys():
            assert name in Properties
            mapPos = Properties.index(name)
            WhereIsTheProperty[name] = numpy.sum(
                [int(k) for k in Properties[2:mapPos:3]]
            )
        numberOfproperties = int(numpy.sum([int(k) for k in Properties[2::3]]))

        universeBox = triclinic_vectors(myUniverse.dimensions).flatten()
        for original, control in zip(universeBox, Lattice):
            assert (original - float(control)) < 1e-7
        if allFramesProperty is not None:
            assert allFramesProperty in lines[frameID + 1]
        if perFrameProperties is not None:
            assert perFrameProperties[frame] in lines[frameID + 1]
        for atomID in range(len(myUniverse.atoms)):
            thisline = lines[frameID + 2 + atomID]
            print(thisline)
            assert thisline.split()[0] == myUniverse.atoms.types[atomID]
            assert len(thisline.split()) == numberOfproperties
            for name in passedValues.keys():
                if len(passedValues[name].shape) == 2:
                    assert (
                        int((thisline.split()[WhereIsTheProperty[name]]))
                        == passedValues[name][frame, atomID]
                    )
                else:
                    for i, d in enumerate(passedValues[name][frame, atomID]):
                        assert (
                            int((thisline.split()[WhereIsTheProperty[name] + i])) == d
                        )

            for i in range(3):
                assert (
                    float(thisline.split()[i + 1])
                    == myUniverse.atoms.positions[atomID][i]
                )


def getUniverseWithWaterMolecules(n_residues=10):
    # following the tutorial at https://userguide.mdanalysis.org/stable/examples/constructing_universe.html
    n_atoms = n_residues * 3
    # create resindex list
    resindices = numpy.repeat(range(n_residues), 3)
    assert len(resindices) == n_atoms
    # all water molecules belong to 1 segment
    segindices = [0] * n_residues
    # create the Universe
    sol = MDAnalysis.Universe.empty(
        n_atoms,
        n_residues=n_residues,
        atom_resindex=resindices,
        residue_segindex=segindices,
        trajectory=True,
    )  # necessary for adding coordinates
    sol.add_TopologyAttr("name", ["O", "H1", "H2"] * n_residues)
    sol.add_TopologyAttr("type", ["O", "H", "H"] * n_residues)
    sol.add_TopologyAttr("resname", ["SOL"] * n_residues)
    sol.add_TopologyAttr("resid", list(range(1, n_residues + 1)))
    sol.add_TopologyAttr("segid", ["SOL"])

    # coordinates obtained by building a molecule in the program IQMol
    h2o = numpy.array(
        [
            [0, 0, 0],  # oxygen
            [0.95908, -0.02691, 0.03231],  # hydrogen
            [-0.28004, -0.58767, 0.70556],  # hydrogen
        ]
    )

    grid_size = numpy.ceil(numpy.cbrt(n_residues))
    spacing = 8
    coordinates = []

    # translating h2o coordinates around a grid
    for i in range(n_residues):
        x = spacing * (i % grid_size)
        y = spacing * ((i // grid_size) % grid_size)
        z = spacing * (i // (grid_size * grid_size))

        xyz = numpy.array([x, y, z])

        coordinates.extend(h2o + xyz.T)
    sol.dimensions = [
        grid_size * spacing,
        grid_size * spacing,
        grid_size * spacing,
        90,
        90,
        90,
    ]

    coord_array = numpy.array(coordinates)
    assert coord_array.shape == (n_atoms, 3)
    sol.atoms.positions = coord_array
    bonds = []
    for o in range(0, n_atoms, 3):
        bonds.extend([(o, o + 1), (o, o + 2)])

    sol.add_TopologyAttr("bonds", bonds)

    return sol


def giveUniverse(angles: set = (90.0, 90.0, 90.0)) -> MDAnalysis.Universe:
    traj = numpy.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
            [[0.1, 0.1, 0.1], [1.1, 1.1, 1.1], [2.1, 2.1, 2.1], [3.1, 3.1, 3.1]],
            [[0.2, 0.2, 0.2], [1.2, 1.2, 1.2], [2.2, 2.2, 2.2], [3.2, 3.2, 3.2]],
            [[0.3, 0.3, 0.3], [1.3, 1.3, 1.3], [2.3, 2.3, 2.3], [3.3, 3.3, 3.3]],
            [[0.4, 0.4, 0.4], [1.4, 1.4, 1.4], [2.4, 2.4, 2.4], [3.4, 3.4, 3.4]],
        ]
    )
    u = MDAnalysis.Universe.empty(
        4, trajectory=True, atom_resindex=[0, 0, 0, 0], residue_segindex=[0]
    )

    u.add_TopologyAttr("type", ["H"] * 4)
    u.atoms.positions = traj[0]
    u.trajectory = MDAnalysis.coordinates.memory.MemoryReader(
        traj,
        order="fac",
        # this tests the non orthogonality of the box
        dimensions=numpy.array(
            [[6.0, 6.0, 6.0, angles[0], angles[1], angles[2]]] * traj.shape[0]
        ),
    )
    return u
