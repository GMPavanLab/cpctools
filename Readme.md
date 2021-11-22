# SOAPify and ReferenceMaker

This sequence of commands will setup the environment:
```
python3 -m venv ./venv
source ./venv/bin/activate
pip install -r requirements
```
You need to install lammps as a python package so that lammps is avaiable to the newly created virtual environment, following the guide on the [lammps site](https://docs.lammps.org/Python_install.html)

ReferenceMaker can create a list of SOAP references with the following:
```
python3 -m ReferenceMaker
```
The command will create a file called "XxReferences.hdf5" (with Xx is the chemical symbol of the chosen metal in the frompt from the command `python3 -m ReferenceMaker`) that contains the fingerprints of the following structures:

 - bulk: sc,bcc,hcp,fcc
 - th4116: vertexes, edges, 001 faces, 111 faces
 - ico5083: vertexes, edges, 111 faces, five folded axis
 - dh3049: concave atom, five folded axis