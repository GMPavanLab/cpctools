# SOAPify and ReferenceMaker

This sequence of commands will prepare the environment.
```
pithon3 -m venv ./venv
source ./venv/bin/activate
pip install -r requirements
```

In order to launch the ReferenceMaker to create a list of SOAP references with
```
python3 -m ReferenceMaker
```
You need to install lammps as a python package avaiable to the newly created virtual environment, following the guide on the lammps [site](https://docs.lammps.org/Python_install.html)