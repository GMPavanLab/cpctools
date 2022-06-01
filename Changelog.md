# Changelog

## Changes From 0.0.3

- `getXYZfromTrajGroup` now accept IO objects as inputs
- **WARNING**: broken interface for `getXYZfromTrajGroup`, now it needs 3 inputs and the first is a file-like object
- `saveXYZfromTrajGroup` and `getXYZfromTrajGroup` now can export comments per frames
- `transitionMatrixFromSOAPClassification` now creates matrix with shape  `(n,n)` and no more `(n+1,n+1)`, where `n` is the lenght of the legend. The user will now need to address the errors in classification, if needed
- added `calculateResidenceTimesFromClassification` for calculating the residence times of the states during the MD
- added `trackStates` for calculating the history of the evolution of the states in the MD
- the result of `trackStates` can be used for calculating the residence times and the transition matrices
- Now when appliyng soap, the created dataset will be given attributes that describe the parameters used for its creation
- Removed some default values in function from Saponify and fillSOAPVectorFromdscribe
- fillSOAPVectorFromdscribe now can fill soap vectors from multispecies calculations
- changed slightly `saponifyGroup` and `saponify`: now they accept dscribe SOAP options as a dictionary, but not the sparse option
- Now HDFTo.getXYZfromTrajGroup accepts slices as an option to export the trajectory
- **WARNING**: broken interface for saponify
- `isTrajectoryGroup` added to HDF5er to check if a group contain a trajectory in our format

## Changes From 0.0.2a

- Only for Monospecies systems: added a small utility (`fillSOAPVectorFromdscribe`) that returns the complete SOAP vector from the simplified one from dscribe
- Added a utility for normalize SOAP vectors
- Added `createReferencesFromTrajectory` that creates a variables that stores SOAP references
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
