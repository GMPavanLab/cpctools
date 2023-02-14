from .SOAPClassify import SOAPclassification
import numpy as np


# TODO add stride/window selection
def transitionMatrixFromSOAPClassification(
    data: SOAPclassification, stride: int = 1
) -> "np.ndarray[float]":
    """Generates the unnormalized matrix of the transitions from a :func:`classifyWithSOAP`

        The matrix is organized in the following way:
        for each atom in each frame we increment by one the cell whose row is the
        state at the frame `n-stride` and the column is the state at the frame `n`
        If the classification includes an error with a `-1` values the user should add an 'error' class in the legend

    Args:
        data (SOAPclassification): the results of the soapClassification from :func:`classifyWithSOAP`
        stride (int): the stride in frames between each state confrontation. Defaults to 1.
        of the groups that contain the references in hdf5FileReference
    Returns:
        np.ndarray[float]: the unnormalized matrix of the transitions
    """
    nframes = len(data.references)
    nat = len(data.references[0])

    nclasses = len(data.legend)
    transMat = np.zeros(
        (nclasses, nclasses),
        # dtype=np.float64,
    )

    for frameID in range(stride, nframes, 1):
        for atomID in range(0, nat):
            classFrom = data.references[frameID - stride][atomID]
            classTo = data.references[frameID][atomID]
            transMat[classFrom, classTo] += 1
    return transMat


def normalizeMatrix(transMat: "np.ndarray[float]") -> "np.ndarray[float]":
    """normalizes a matrix that is an ouput of :func:`transitionMatrixFromSOAPClassification`

    The matrix is normalized with the criterion that the sum of each **row** is `1`

    Args:
        np.ndarray[float]: the unnormalized matrix of the transitions

    Returns:
        np.ndarray[float]: the normalized matrix of the transitions
    """
    toRet = transMat.copy()
    for row in range(transMat.shape[0]):
        sum = np.sum(toRet[row, :])
        if sum != 0:
            toRet[row, :] /= sum
    return toRet


# TODO add stride/window selection
def transitionMatrixFromSOAPClassificationNormalized(
    data: SOAPclassification, stride: int = 1
) -> "np.ndarray[float]":
    """Generates the normalized matrix of the transitions from a :func:`classifyWithSOAP` and normalize it

        The matrix is organized in the following way:
        for each atom in each frame we increment by one the cell whose row is the
        state at the frame `n-stride` and the column is the state at the frame `n`

        The matrix is normalized with the criterion that the sum of each **row** is `1`

    Args:
        data (SOAPclassification): the results of the soapClassification from :func:`classifyWithSOAP`
        stride (int): the stride in frames between each state confrontation. Defaults to 1.
        of the groups that contain the references in hdf5FileReference
    Returns:
        np.ndarray[float]: the normalized matrix of the transitions
    """
    transMat = transitionMatrixFromSOAPClassification(data, stride)
    return normalizeMatrix(transMat)


TRACK_PREVSTATE = 0
TRACK_CURSTATE = 1
TRACK_ENDSTATE = 2
TRACK_EVENTTIME = 3


def _createStateTracker(
    prevState: int, curState: int, endState: int, eventTime: int = 0
) -> np.ndarray:
    return np.array([prevState, curState, endState, eventTime], dtype=int)


# TODO add stride/window here
def trackStates(classification: SOAPclassification) -> list:
    nofFrames = classification.references.shape[0]
    nofAtoms = classification.references.shape[1]
    stateHistory = []
    # should I use a dedicated class?
    for atomID in range(nofAtoms):
        statesPerAtom = []
        atomTraj = classification.references[:, atomID]
        # TODO: this can be made concurrent per atom

        # the array is [start state, state, end state,time]
        # when PREVSTATE and CURSTATE are the same the event is the first event for the atom in the simulation
        # when ENDSTATE and CURSTATE are the same the event is the last event for the atom in the simulation
        stateTracker = _createStateTracker(
            prevState=atomTraj[0],
            curState=atomTraj[0],
            endState=atomTraj[0],
            eventTime=0,
        )
        for frame in range(1, nofFrames):
            if atomTraj[frame] != stateTracker[TRACK_CURSTATE]:
                stateTracker[TRACK_ENDSTATE] = atomTraj[frame]
                statesPerAtom.append(stateTracker)
                stateTracker = _createStateTracker(
                    prevState=stateTracker[TRACK_CURSTATE],
                    curState=atomTraj[frame],
                    endState=atomTraj[frame],
                )

            stateTracker[TRACK_EVENTTIME] += 1
        # append the last event
        statesPerAtom.append(stateTracker)
        stateHistory.append(statesPerAtom)
    return stateHistory


# TODO add stride here?
def calculateResidenceTimesFromClassification(
    classification: SOAPclassification,
) -> np.ndarray:
    nofFrames = classification.references.shape[0]
    nofAtoms = classification.references.shape[1]
    residenceTimes = [[] for i in range(len(classification.legend))]
    for atomID in range(nofAtoms):
        atomTraj = classification.references[:, atomID]
        time = 0
        state = atomTraj[0]
        for frame in range(1, nofFrames):
            if atomTraj[frame] != state:
                residenceTimes[state].append(time)
                state = atomTraj[frame]
                time = 0
            time += 1
        # the last state does not have an out transition, appendig negative time to make it clear
        residenceTimes[state].append(-time)

    for i in range(len(residenceTimes)):
        residenceTimes[i] = np.sort(np.array(residenceTimes[i]))

    return residenceTimes


def RemoveAtomIdentityFromEventTracker(statesTracker: list) -> list:
    if isinstance(statesTracker[0], list):
        t = []
        for tracks in statesTracker:
            t += tracks
        statesTracker = t
    return statesTracker


def getResidenceTimesFromStateTracker(statesTracker: list, legend: list) -> np.ndarray:
    states = RemoveAtomIdentityFromEventTracker(statesTracker)

    residenceTimes = [[] for i in range(len(legend))]
    for event in states:
        residenceTimes[event[TRACK_CURSTATE]].append(
            event[TRACK_EVENTTIME]
            if event[TRACK_ENDSTATE] != event[TRACK_CURSTATE]
            else -event[TRACK_EVENTTIME]
        )
    for i in range(len(residenceTimes)):
        residenceTimes[i] = np.sort(np.array(residenceTimes[i]))
    return residenceTimes


def calculateResidenceTimes(
    data: SOAPclassification, statesTracker: list = None
) -> np.ndarray:
    if not statesTracker:
        return calculateResidenceTimesFromClassification(data)
    else:
        return getResidenceTimesFromStateTracker(statesTracker, data.legend)


def transitionMatrixFromStateTracker(statesTracker: list, legend: list) -> np.ndarray:
    states = RemoveAtomIdentityFromEventTracker(statesTracker)

    nclasses = len(legend)
    transMat = np.zeros(
        (nclasses, nclasses),
        # dtype=np.float64,
    )
    # print(len(states), states[0], file=sys.stderr)
    for event in states:
        transMat[event[TRACK_CURSTATE], event[TRACK_CURSTATE]] += (
            event[TRACK_EVENTTIME] - 1
        )

        # the transition matrix is genetated with:
        #   classFrom = data.references[frameID - stride][atomID]
        #   classTo = data.references[frameID][atomID]
        transMat[event[TRACK_PREVSTATE], event[TRACK_CURSTATE]] += 1
    return transMat


# TODO add stride/window selection
def calculateTransitionMatrix(
    data: SOAPclassification, stride: int = 1, statesTracker: list = None
) -> np.ndarray:
    if not statesTracker:
        return transitionMatrixFromSOAPClassification(data, stride)
    else:
        return transitionMatrixFromStateTracker(statesTracker, data.legend)
