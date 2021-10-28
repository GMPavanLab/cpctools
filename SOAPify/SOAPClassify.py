import h5py
import numpy as np
from SOAPbase import SOAPdistance
from dataclasses import dataclass


@dataclass
class SOAPclassification:
    distances: "np.ndarray[float]"
    references: "np.ndarray[int]"
    legend: "list[str]"


def classifyWithSOAP(
    SOAPTrajData: h5py.Dataset, hdf5FileReference: h5py.File, referenceAddresses: list
) -> SOAPclassification:
    spectra, legend = loadRefs(hdf5FileReference, referenceAddresses)
    nframes = SOAPTrajData.shape[0]
    nat = SOAPTrajData.shape[1]
    distances = np.zeros((nframes, nat), np.dtype(float))
    references = np.zeros((nframes, nat), np.dtype(int))
    n = 0
    # nchunks=len(T1.iter_chunks())
    for chunk in SOAPTrajData.iter_chunks():
        print(f"working on chunk {n}, {chunk}")
        n += 1
        for chunkID, frame in enumerate(SOAPTrajData[chunk]):
            frameID = chunk[0].start + chunkID
            for atomID, atom in enumerate(frame):
                distances[frameID, atomID] = 1000.0
                references[frameID, atomID] = -1
                for j in range(len(spectra)):
                    try:
                        tdist = SOAPdistance(atom, spectra[j])
                        if tdist < distances[frameID, atomID]:
                            distances[frameID, atomID] = tdist
                            references[frameID, atomID] = j
                    except:
                        print(f"at {j}:")
                        print(f"spectra: {spectra[j]}")
                        print(f"atom: {atomID}")
    return SOAPclassification(distances, references, legend)


def loadRefs(hdf5FileReference: h5py.File, referenceAddresses: list):
    spectra = []  # np.zeros((0, 0), dtype=np.dtype(float))
    legend = []
    for address in referenceAddresses:
        data = hdf5FileReference[address]
        if type(data) is h5py.Group:
            for refName in data.keys():
                dataset = hdf5FileReference[f"{address}/{refName}"]
                legend.append(refName)
                spectra.append(np.mean(dataset[:], axis=0))

        elif type(data) is h5py.Dataset:
            legend.append(data.name.rsplit("/")[-1])
            spectra.append(np.mean(data[:], axis=0))
        else:
            raise TypeError
    spectra = np.array(spectra)
    return spectra, legend

def loadLegend(hdf5FileReference: h5py.File, referenceAddresses: list):
    legend = []
    for address in referenceAddresses:
        data = hdf5FileReference[address]
        if isinstance(data, h5py.Group):
            for refName in data.keys():
                dataset = hdf5FileReference[f"{address}/{refName}"]
                legend.append(refName)

        else:  # assuming isinstance(data, h5py.Dataset)
            legend.append(data.name.rsplit("/")[-1])
    return legend


def transitionMatrixFromSOAPClassification(
    data: SOAPclassification, stride: int = 1
) -> "np.ndarray[float]":
    nframes = len(data.references)
    nat = len(data.references[0])
    # +1 in case of errors
    nclasses = len(data.legend) + 1
    transMat = np.zeros((nclasses, nclasses), np.dtype(float))

    for frameID in range(stride, nframes, 1):
        for atomID in range(0, nat):
            classFrom = data.references[frameID - stride][atomID]
            classTo = data.references[frameID][atomID]
            transMat[classFrom, classTo] += 1
    return transMat


def normalizeMatrix(transMat):
    for row in range(transMat.shape[0]):
        sum = np.sum(transMat[row, :])
        if sum != 0:
            transMat[row, :] /= sum
    return transMat


def transitionMatrixFromSOAPClassificationNormalized(
    data: SOAPclassification, stride: int = 1
) -> "np.ndarray[float]":
    transMat = transitionMatrixFromSOAPClassification(data, stride)
    return normalizeMatrix(transMat)

def Transitions(legend:"list[str]",references:np.ndarray):
    residenceList={}
    for atomID in range(len( references[0])):
        startTime=0
        stateFrom=references[0][atomID]
        stateTo=-1
        for time,frame in enumerate(references[1:]):
            if stateFrom!=frame[atomID]:
                stateTo=frame[atomID]
                numberOfFrames=time+1-startTime
                if stateFrom not in residenceList.keys():
                    residenceList[stateFrom]={}    
                if stateTo not in residenceList[stateFrom].keys():
                    residenceList[stateFrom][stateTo]=[]    
                residenceList[stateFrom][stateTo].append(numberOfFrames)
                stateFrom=frame[atomID]
                startTime=time+1


    return residenceList


if __name__ == "__main__":
    import h5py
    import numpy as np

    trajLoader = h5py.File("WaterSOAP__.hdf5", "r")
    SOAPtraj = trajLoader["SOAP/1ns"]
    refFile = h5py.File("ReferenceWater.hdf5", "r")
    data = classifyWithSOAP(SOAPtraj, refFile, ["Waters/R10/tip4p2005", "Ices/R10"])
    print(data)
