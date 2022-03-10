# Changelog

## Changes From 0.0.2a

- Only for Monospecies: added a small utility (fillSOAPVectorFromdscribe) that returns the correct SOAP vector from the simplified one from dscribe
- Added a utility for normalize SOAP vectors
- Added createReferencesFromTrajectory that creates a variables that stores SOAP references
- set up a way to classify with soap with a different method thant the original idea
- the new references now can be loaded/unloaded on an hdf5 file
- added a patch for hdf5 imported files: workaround for mda not loading correctly non orthogonal boxes from lammps dumps

## Changes From 0.0.2

- Added the possiblity to export xyz files from the hdf5 trajectories, also with extra data columns
- Improved performance for getXYZfromTrajGroup

## Changes From 0.0.1

- Tests
- Changed default imports for HDF5er
- Adding override option to HDF5er.MDA2HDF5
- Added 3rd neighbours in the References
- Added attributes to HDF5er.MDA2HDF5
- In the referenceMaker: added the possibility to choose lmax and nmax for SOAP
- Added the possibility to export to hdf5 slices of the trajectories
