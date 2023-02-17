# cpctools: SOAPify and SOAPify.HDF5er

![PyPI - License](https://img.shields.io/pypi/l/cpctools)
[![PyPI](https://img.shields.io/pypi/v/cpctools)](https://pypi.org/project/cpctools/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cpctools)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)
[![Documentation Status](https://readthedocs.org/projects/soapify/badge/?version=latest)](https://soapify.readthedocs.io/en/latest/?badge=latest)

[![Powered by MDAnalysis](https://img.shields.io/badge/powered%20by-MDAnalysis-orange.svg?logoWidth=16&logo=data:image/x-icon;base64,AAABAAEAEBAAAAEAIAAoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJD+XwCY/fEAkf3uAJf97wGT/a+HfHaoiIWE7n9/f+6Hh4fvgICAjwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACT/yYAlP//AJ///wCg//8JjvOchXly1oaGhv+Ghob/j4+P/39/f3IAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJH8aQCY/8wAkv2kfY+elJ6al/yVlZX7iIiI8H9/f7h/f38UAAAAAAAAAAAAAAAAAAAAAAAAAAB/f38egYF/noqAebF8gYaagnx3oFpUUtZpaWr/WFhY8zo6OmT///8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgICAn46Ojv+Hh4b/jouJ/4iGhfcAAADnAAAA/wAAAP8AAADIAAAAAwCj/zIAnf2VAJD/PAAAAAAAAAAAAAAAAICAgNGHh4f/gICA/4SEhP+Xl5f/AwMD/wAAAP8AAAD/AAAA/wAAAB8Aov9/ALr//wCS/Z0AAAAAAAAAAAAAAACBgYGOjo6O/4mJif+Pj4//iYmJ/wAAAOAAAAD+AAAA/wAAAP8AAABhAP7+FgCi/38Axf4fAAAAAAAAAAAAAAAAiIiID4GBgYKCgoKogoB+fYSEgZhgYGDZXl5e/m9vb/9ISEjpEBAQxw8AAFQAAAAAAAAANQAAADcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAjo6Mb5iYmP+cnJz/jY2N95CQkO4pKSn/AAAA7gAAAP0AAAD7AAAAhgAAAAEAAAAAAAAAAACL/gsAkv2uAJX/QQAAAAB9fX3egoKC/4CAgP+NjY3/c3Nz+wAAAP8AAAD/AAAA/wAAAPUAAAAcAAAAAAAAAAAAnP4NAJL9rgCR/0YAAAAAfX19w4ODg/98fHz/i4uL/4qKivwAAAD/AAAA/wAAAP8AAAD1AAAAGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALGxsVyqqqr/mpqa/6mpqf9KSUn/AAAA5QAAAPkAAAD5AAAAhQAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADkUFBSuZ2dn/3V1df8uLi7bAAAATgBGfyQAAAA2AAAAMwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB0AAADoAAAA/wAAAP8AAAD/AAAAWgC3/2AAnv3eAJ/+dgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9AAAA/wAAAP8AAAD/AAAA/wAKDzEAnP3WAKn//wCS/OgAf/8MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIQAAANwAAADtAAAA7QAAAMAAABUMAJn9gwCe/e0Aj/2LAP//AQAAAAAAAAAA)](https://www.mdanalysis.org)

SOAPify is python 3.8/3.9/3.10 library aimed at simplify the analysis of Molecular Dynamics simulation using the Smooth Overlap of Atomic Position (SOAP) in context that includes the time along the geometrical informations of the frames of the simulation.

SOAPify uses `h5py` for caching the results of the various analysis.

SOAPify offers a suite for a (basic) state analysis for the simulation.

## How To Install

To install SOAPify just type:
```bash
pip install cpctools
```
cpc stands for **C**omputational **P**hysical **C**hemistry.

If you want to use _dscribe_ or _quippy_ for calculating the SOAP features you should install them separately:

```bash
pip install "dscribe<=1.2.2,>1.2.0"
pip install "quippy-ase==0.9.10"
```

### Installing the latest version
We alwyays recomment to install ypur code in a dedicated environment:

```bash
python3 -m venv /path/to/new/venv --prompt SOAPify
source /path/to/new/venv/bin/activate
pip install --upgrade pip
```

Then to install SOAPify you can simply go to the repository directory and run the following:
```bash
cd /path/to/SOAPify/directory
pip install .
```
or if you do not want to download the repo, you can have pip install from source:
```bash
pip install 'cpctools @ git+https://github.com/GMPavanLab/SOAPify.git'
```
Or if you desire an older version you can install from a tag:
```bash
pip install 'cpctools @ git+https://github.com/GMPavanLab/SOAPify.git@0.0.6'
```

We have a (very basic) [documentation](https://gmpavanlab.github.io/SOAPify/SOAPify.html) of the latest version avaiable on the github pages.

A more complete history of the documentation is avaiable on [read the docs](https://soapify.readthedocs.io/en/latest/). There you can consult the documentation for each avaiable version of the package.


## SOAPify

This package contains:
 - a toolbox to calculate the [SOAP fingerprints](https://doi.org/10.1103/PhysRevB.87.184115) of a system of atoms. The principal aim is to simplify the setup of the calculation. This toolbox depends on `dscribe` and `quippy` and 'unifies' the output of the two codes.
 - a toolbox to calculate the distances between SOAP fingerprints
 - a simple analysis tool for trajectories of classified atoms


## SOAPify.HDF5er

This package is a toolbox to create [hdf5 files](https://www.hdfgroup.org/) with [h5py](https://www.h5py.org/) from trajectory and topology files. The format we use **do not** align with [h5md](https://www.nongnu.org/h5md/h5md.html).

SOAPify.HDF5er contains a tool for exporting the trajectories from the hdf5 file to [extended xyz](https://www.ovito.org/docs/current/reference/file_formats/input/xyz.html#extended-xyz-format) format, compatible with [ovito](https://www.ovito.org/)
