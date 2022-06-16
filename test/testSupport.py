import MDAnalysis
import numpy
import pytest
import SOAPify
from numpy.random import randint


@pytest.fixture(
    scope="module",
    params=[
        SOAPify.SOAPclassification(
            [],
            numpy.array(
                # 0 never changes state
                # 1 change stare at first frame
                # 2 alternates two states
                [
                    [0, 1, 1],
                    [0, 2, 2],
                    [0, 2, 1],
                    [0, 2, 2],
                    [0, 2, 1],
                    [0, 2, 2],
                ]
            ),
            ["state0", "state1", "state2"],
        ),
        SOAPify.SOAPclassification(
            [],
            numpy.array(
                # 0 never changes state
                # 1 change stare at first frame
                # 2 alternates two states
                # 3 as an error at some point
                [
                    [0, 1, 1, 1],
                    [0, 2, 2, 2],
                    [0, 2, 1, 1],
                    [0, 2, 2, -1],
                    [0, 2, 1, 1],
                    [0, 2, 2, 2],
                ]
            ),
            ["state0", "state1", "state2", "Errors"],
        ),
        SOAPify.SOAPclassification(  # big random "simulation"
            [],
            randint(0, high=4, size=(1000, 309)),
            ["state0", "state1", "state2", "state3"],
        ),
    ],
)
def input_mockedTrajectoryClassification(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        slice(None, None, None),  # no slice
        slice(1, None, 2),  # classic slice
        [0, 4],  # list-like slice
    ],
)
def input_framesSlice(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        "dscribe",
        "quippy",
    ],
)
def engineKind_fixture(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        ["C", "O", "H", "N"],
        ["H", "O"],
        ["H"],
    ],
)
def species_fixture(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[1, 2, 3, 4, 5, 6],
)
def nMaxFixture(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[0, 1, 2, 3, 4, 5, 6],
)
def lMaxFixture(request):
    return request.param


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
