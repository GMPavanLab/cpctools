import h5py
import numpy as np
from SOAPify import classifyWithSOAP, transitionMatrix

with h5py.File("WaterSOAP.hdf5", "r") as trajLoader, h5py.File(
    "ReferenceWater.hdf5", "r"
) as refFile:
    SOAPtraj = trajLoader["SOAP/100ns"]

    data = classifyWithSOAP(SOAPtraj, refFile, ["Waters/R10", "Ices/R10"])

    for stride in [1, 10, 100, 1000, 10000]:
        t = transitionMatrix(data, stride)
        np.savetxt(f"matTransStride{stride}.dat", t)
