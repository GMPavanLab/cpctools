# SOAPify, HDF5er and ReferenceMaker

[![Powered by MDAnalysis](https://img.shields.io/badge/powered%20by-MDAnalysis-orange.svg?logoWidth=16&logo=data:image/x-icon;base64,AAABAAEAEBAAAAEAIAAoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJD+XwCY/fEAkf3uAJf97wGT/a+HfHaoiIWE7n9/f+6Hh4fvgICAjwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACT/yYAlP//AJ///wCg//8JjvOchXly1oaGhv+Ghob/j4+P/39/f3IAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJH8aQCY/8wAkv2kfY+elJ6al/yVlZX7iIiI8H9/f7h/f38UAAAAAAAAAAAAAAAAAAAAAAAAAAB/f38egYF/noqAebF8gYaagnx3oFpUUtZpaWr/WFhY8zo6OmT///8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgICAn46Ojv+Hh4b/jouJ/4iGhfcAAADnAAAA/wAAAP8AAADIAAAAAwCj/zIAnf2VAJD/PAAAAAAAAAAAAAAAAICAgNGHh4f/gICA/4SEhP+Xl5f/AwMD/wAAAP8AAAD/AAAA/wAAAB8Aov9/ALr//wCS/Z0AAAAAAAAAAAAAAACBgYGOjo6O/4mJif+Pj4//iYmJ/wAAAOAAAAD+AAAA/wAAAP8AAABhAP7+FgCi/38Axf4fAAAAAAAAAAAAAAAAiIiID4GBgYKCgoKogoB+fYSEgZhgYGDZXl5e/m9vb/9ISEjpEBAQxw8AAFQAAAAAAAAANQAAADcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAjo6Mb5iYmP+cnJz/jY2N95CQkO4pKSn/AAAA7gAAAP0AAAD7AAAAhgAAAAEAAAAAAAAAAACL/gsAkv2uAJX/QQAAAAB9fX3egoKC/4CAgP+NjY3/c3Nz+wAAAP8AAAD/AAAA/wAAAPUAAAAcAAAAAAAAAAAAnP4NAJL9rgCR/0YAAAAAfX19w4ODg/98fHz/i4uL/4qKivwAAAD/AAAA/wAAAP8AAAD1AAAAGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALGxsVyqqqr/mpqa/6mpqf9KSUn/AAAA5QAAAPkAAAD5AAAAhQAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADkUFBSuZ2dn/3V1df8uLi7bAAAATgBGfyQAAAA2AAAAMwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB0AAADoAAAA/wAAAP8AAAD/AAAAWgC3/2AAnv3eAJ/+dgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9AAAA/wAAAP8AAAD/AAAA/wAKDzEAnP3WAKn//wCS/OgAf/8MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIQAAANwAAADtAAAA7QAAAMAAABUMAJn9gwCe/e0Aj/2LAP//AQAAAAAAAAAA)](https://www.mdanalysis.org) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch) [![License - MIT](https://img.shields.io/badge/license-MIT-9400d3.svg)](https://spdx.org/licenses/)[![Documentation Status](https://readthedocs.org/projects/soapify/badge/?version=latest)](https://soapify.readthedocs.io/en/latest/?badge=latest)

SOAPify is python 3.8/3.9 library aimed at simplify the analysis of Molecular Dynamics simulation using the Smooth Overlap of Atomic Position (SOAP) in context that includes the time along the geometrical informations of the frames of the simulation.

SOAPify uses `h5py` for caching the results of the various analysis.

SOAPify offers a suite for a (basic) state analysis for the simulation.

## How To Install

To set up the environment and install _SOAPify_ run the following in the repository folder:

```bash
python3 -m venv ./venv --prompt SOAPify
source ./venv/bin/activate
pip install --upgrade pip 
pip install .
```

If you want to use _dscribe_ or _quippy_ for calculating the SOAP features you should install them separately:

```bash
pip install "dscribe >1.2.0 <=1.2.2"
pip install quippy-ase
```

(PyPI support is incoming!)

Now with a (very basic) [documentation](https://gmpavanlab.github.io/SOAPify/SOAPify.html) of the latest version pushed to the main branch!

A more complete history of the documetation is avaiable on [read the docs](https://soapify.readthedocs.io/en/latest/), with storage of the old

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
