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
        default=[],
    )
    parser.add_argument(
        "-n",
        "--name",
        metavar="name",
        help="the name of the trajectory to save",
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
        help="extra attributes to store in the trajectory, saved as strings",
    )
    parser.add_argument(
        "-u",
        "--universe-options",
        nargs=2,
        action="append",
        dest="universeOPTs",
        metavar=("name", "value"),
        help="extra option to pass to the MDA universe, compatible only with string"
        " values, use the python script if you need to pass more other settings",
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="just print the output without making any action",
    )
    args = parser.parse_args()
    # print(args)
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


def traj2SOAP():
    """Given an hdf5 file containing trajectories calculates SOAP of the contained trajectories


    default SOAP engine is dscribe"""

    from SOAPify import saponifyTrajectory
    from SOAPify.HDF5er import isTrajectoryGroup
    import h5py

    parser = ArgumentParser(description=traj2SOAP.__doc__)
    parser.add_argument(
        "trajFile",
        help="the file containing the trajectories",
    )
    parser.add_argument(
        "-s",
        "--SOAPFile",
        help="The file were to save the SOAP fingerprints, if not specified"
        ' creates a "SOAP" group in within the trajFile',
    )
    parser.add_argument(
        "-g",
        "--group",
        default="SOAP",
        help="the name of the group where to store the SOAP fingerprints,"
        ' if not specified is "SOAP"',
        dest="SOAPgroup",
    )
    parser.add_argument(
        "-t",
        "--trajectory",
        default="Trajectories",
        help="Specify the group containing the trajectories groups or the"
        " trajectory group tha you want to calculate the SOAP fingerprints",
    )
    parser.add_argument(
        "-e",
        "--engine",
        choices=["dscribe", "quippy"],
        default="dscribe",
        help="the engine used to calculate SOAP",
    )
    parser.add_argument(
        "-l",
        "--lMax",
        type=int,
        default=8,
        help="the lmax parameter, defaults to 8",
    )
    parser.add_argument(
        "-n",
        "--nMax",
        type=int,
        default=8,
        help="the nmax parameter, defaults to 8",
    )
    parser.add_argument(
        "-r",
        "--rCut",
        type=float,
        default=10.0,
        help="the rcut parameter, defaults to 10.0 Å",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="the number of jobs to use, defaults to 1",
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="just print the output without making any action",
    )
    args = parser.parse_args()
    # print(args)

    trajGroupLocation = args.trajectory

    def worker(
        group: "h5py.Group|h5py.Dataset", soapFile: h5py.File, SOAPgroup: str = "SOAP"
    ):
        print(
            f"\"{group.name}\" is {'' if isTrajectoryGroup(group) else 'not '}a trajectory group"
        )
        if isTrajectoryGroup(group):
            if args.dry_run:
                return
            saponifyTrajectory(
                trajContainer=group,
                SOAPoutContainer=soapFile.require_group(SOAPgroup),
                SOAPOutputChunkDim=1000,
                SOAPnJobs=args.jobs,
                SOAPrcut=args.rCut,
                SOAPnmax=args.nMax,
                SOAPlmax=args.lMax,
                useSoapFrom=args.engine,
            )

    SOAPFile = args.SOAPFile
    SOAPgroup = args.SOAPgroup
    if args.SOAPFile is None:
        SOAPFile = args.trajFile
    with h5py.File(
        args.trajFile, "r" if SOAPFile != args.trajFile else "a"
    ) as workFile, h5py.File(SOAPFile, "a") as SOAPFile:
        trajGroup = workFile[trajGroupLocation]
        if isTrajectoryGroup(trajGroup):
            worker(trajGroup, SOAPFile, SOAPgroup)
        else:
            for _, group in trajGroup.items():
                worker(group, SOAPFile, SOAPgroup)
