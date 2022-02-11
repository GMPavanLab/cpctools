# SOAPify, HDF5er and ReferenceMaker

This sequence of commands will setup the environment:

```bash
python3 -m venv ./venv --prompt SOAPenv
source ./venv/bin/activate
pip install -r requirements
```

## SOAPify

This package contains a toolbox to calculate the [SOAP fingerprints](https://doi.org/10.1103/PhysRevB.87.184115) of a system of atoms.

## HDF5er

This package contains a small toolbox to create [hdf5 files](https://www.hdfgroup.org/) with [h5py](https://www.h5py.org/) from trajectory and topology files. The format we use **do not** align with [h5md](https://www.nongnu.org/h5md/h5md.html).

## ReferenceMaker

The ReferenceMaker package contains a set of function that can create a reference file to be used with the SOAPify package.

ReferenceMaker function can be called with custom made scripts, but the user can create a list of SOAP references with the following:

```bash
python3 -m ReferenceMaker
```

The command will create a file called "XxReferences.hdf5" (with Xx is the chemical symbol of the chosen metal in the prompt from the command `python3 -m ReferenceMaker`) that contains the fingerprints of the following structures:

- bulk: sc,bcc,hcp,fcc
- th4116: vertexes, edges, 001 faces, 111 faces
- ico5083: vertexes, edges, 111 faces, five folded axis
- dh3049: concave atom, five folded axis

To use the automatic procedure the user needs to install lammps as a python package so that lammps is avaiable to the newly created virtual environment, following the guide on the [lammps site](https://docs.lammps.org/Python_install.html)
