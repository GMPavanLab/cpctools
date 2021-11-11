#!/usr/bin/env python

import HDF5er
from MDAnalysis import Universe as mdaUniverse


# In[2]:


# Load universe
def hdf5erWater(gro, xtc, title, hdf5file):
    universe = mdaUniverse(gro, xtc)
    selection = universe.select_atoms("not name MW4")
    HDF5er.MDA2HDF5(universe, hdf5file, title, trajChunkSize=1000, selection=selection)


# In[3]:


hdf5erWater("water_1ns/ref.gro", "water_1ns/run_1ns.xtc", "1ns", "WaterNew.hdf5")
hdf5erWater("water_1ns/ref.gro", "water_1ns/run_100ns.xtc", "100ns", "WaterNew.hdf5")
