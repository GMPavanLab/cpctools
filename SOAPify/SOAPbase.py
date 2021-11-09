import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from ase.io import read
from dscribe.descriptors import SOAP

# import scipy


def get_axes(L, max_col=3):
    cols = L if L <= max_col else max_col
    rows = int(L / max_col) + int(L % max_col != 0)
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    ax = ax.flatten()
    return fig, ax


# --------------------------------------------------------------------
### Kernel (unit normalized)


def simpleKernelSoap(x, y):
    """
    Soap Kernel
    """

    # return 1 - scipy.spatial.distance.cosine(x, y)
    return np.dot(x, y) / (la.norm(x) * la.norm(y))


def simpleSOAPdistance(x, y):
    """
    Distance based on Soap Kernel.
    """
    try:
        return np.sqrt(2.0 - 2.0 * simpleKernelSoap(x, y))
    except FloatingPointError:
        return 0.0


def KernelSoap(x, y, n):
    """
    Soap Kernel
    """

    # return (1 - scipy.spatial.distance.cosine(x, y)) ** n
    return (np.dot(x, y) / (la.norm(x) * la.norm(y))) ** n


def SOAPdistance(x, y, n=1):
    """
    Distance based on Soap Kernel.
    """
    try:
        return np.sqrt(2.0 - 2.0 * KernelSoap(x, y, n))
    except FloatingPointError:
        return 0


def KL(p, q):
    """
    Kullback-Leibler divergence
    """
    return np.sum(p * np.log(p / q))


def JS(p, q):
    """
    Jensen–Shannon divergence
    """
    return jensenshannon(np.exp(p), np.exp(q))


def SOAPify(
    inputFile,
    boxFile,
    rcut,
    systemIndexes=[0],
    nmax=8,
    lmax=8,
    SOAPaverage=False,
):
    system = read(inputFile, index=":", format="xyz")
    print("system read.")
    systembox = np.array([np.loadtxt(boxFile)])[0]
    print("Adding PBC information...")
    for i in range(len(system)):
        system[i].set_cell(systembox[i])
        system[i].set_pbc([1, 1, 1])

    species = list(set(system[0].get_chemical_symbols()))

    average = "inner" if SOAPaverage else "off"

    soapDSCAVE = SOAP(
        species=species, periodic=True, rcut=rcut, nmax=nmax, lmax=lmax, average=average
    )

    print("SOAPing system...")
    soapSystem = soapDSCAVE.create(
        system, positions=[systemIndexes for x in range(501)]
    )
    return soapSystem


if __name__ == "__main__":
    from sys import argv

    inputFile = argv[1]
    boxFile = argv[2]
    rcut = argv[3]
    outFolder = "references_traj_"
    soapAverage = False
    if len(argv) > 4:
        t = argv[4].lower()
        if argv[4] == "true":
            soapAverage = True

    if len(argv) > 5:
        outFolder = argv[5]

    soapRes = SOAPify(
        inputFile,
        boxFile,
        rcut,
        [0],
        8,
        8,
        soapAverage,
    )

    np.savez_compressed(outFolder + "/system_soap.npz", soapRes)
