"""submodule containing the command line interfaces of SOAPify"""
from argparse import ArgumentParser


def getDictFromList(nameAndValues: list) -> dict:
    """generates a dictionary from a list of couples [name, value]

    Args:
        nameAndValues (list):
            the list of couples [name, value]

    Returns:
        dict:
            the dictionary, for example {name0:value0, name1:value1}
    """
    myDict = {}
    for attr in nameAndValues:
        myDict[attr[0]] = attr[1]
    return myDict


def createTrajectory():
    """Creates or overide an hdf5 file containing a given trajectory

    if you are creating ahdf5 file from a data+dump from a soap simulation
    remember to add `-u atom_style "id type x y z"` to the arguments

    defaults trajChunkSize=1000"""
    from MDAnalysis import Universe as mdaUniverse

    from SOAPify.HDF5er import MDA2HDF5
    from os import path

    parser = ArgumentParser(description=createTrajectory.__doc__)
    parser.add_argument("topology", help="the topology file")
    parser.add_argument("hdf5File", help="the file where to putput the trajectory")
    parser.add_argument(
        "-t",
        "--trajectory",
        help="the trajectory file(s)",
        metavar="trajectory",
        action="append",
    )
    parser.add_argument(
        "-n", "--name", metavar="name", help="the name of the trajectory to save"
    )
    parser.add_argument(
        "--types",
        metavar="atomNames",
        help="list of the atoms names",
        nargs="+",
        dest="atomTypes",
    )
    parser.add_argument(
        "-a",
        "--attribute",
        nargs=2,
        action="append",
        dest="extraAttributes",
        metavar=("name", "value"),
        help="extra attributes to store in the trajectory",
    )
    parser.add_argument(
        "-u",
        "--universe-options",
        nargs=2,
        action="append",
        dest="universeOPTs",
        metavar=("name", "value"),
        help="extra option to pass to the MDA universe",
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="just print the output without making any action",
    )
    args = parser.parse_args()
    print(args)
    # arguments
    trajectoryFiles = args.trajectory
    filename = args.hdf5File
    topo = args.topology
    name = args.name
    if name is None:
        name = path.basename(topo)
        name = name.split(".")[0]

    print(
        f'from topology "{topo}"',
        f'and trajectory "{trajectoryFiles}":',
        "and creating a new trajectory in",
    )

    print(f'"{filename}/Trajectories/{name}"')
    extraAttrs = None
    if args.extraAttributes:
        extraAttrs = getDictFromList(args.extraAttributes)
        print("extra attributes:", extraAttrs)
    universeOptions = {}
    if args.universeOPTs:
        universeOptions = getDictFromList(args.universeOPTs)

    if args.dry_run:
        exit()

    u = mdaUniverse(topo, *trajectoryFiles, **universeOptions)
    if args.atomTypes:
        ntypes = len(args.atomTypes)
        if len(u.atoms) % ntypes != 0:
            raise ValueError(
                f"The number of atom types is not compatible with the number"
                f" of atoms:{len(u.atoms)} % {ntypes} = {len(u.atoms) % ntypes}"
            )
        u.atoms.types = args.atomTypes * (len(u.atoms) // ntypes)

    MDA2HDF5(u, filename, name, trajChunkSize=1000, attrs=extraAttrs)
    # TODO: implement this:
    # from MDAnalysis import transformations
    # ref = mdaUniverse(topo, atom_style="id type x y z")
    # u.trajectory.add_transformations(transformations.fit_rot_trans(u, ref))
    # MDA2HDF5(u, name + "_fitted.hdf5", f"{name}_fitted", trajChunkSize=1000))
