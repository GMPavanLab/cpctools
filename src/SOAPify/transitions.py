from .classify import SOAPclassification
import numpy


# TODO add stride/window selection
def transitionMatrixFromSOAPClassification(
    data: SOAPclassification, stride: int = 1
) -> "numpy.ndarray[float]":
    """Generates the unnormalized matrix of the transitions from a :func:`classify`

        see :func:`calculateTransitionMatrix` for a detailed description of an unnormalized transition matrix

    Args:
        data (SOAPclassification): the results of the soapClassification from :func:`classify`
        stride (int): the stride in frames between each state confrontation. Defaults to 1.
        of the groups that contain the references in hdf5FileReference
    Returns:
        numpy.ndarray[float]: the unnormalized matrix of the transitions
    """
    nframes = len(data.references)
    nat = len(data.references[0])

    nclasses = len(data.legend)
    transMat = numpy.zeros((nclasses, nclasses), dtype=numpy.float64)

    for frameID in range(stride, nframes, 1):
        for atomID in range(0, nat):
            classFrom = data.references[frameID - stride][atomID]
            classTo = data.references[frameID][atomID]
            transMat[classFrom, classTo] += 1
    return transMat


def normalizeMatrix(transMat: "numpy.ndarray[float]") -> "numpy.ndarray[float]":
    """normalizes a matrix that is an ouput of :func:`transitionMatrixFromSOAPClassification`

    The matrix is normalized with the criterion that the sum of each **row** is `1`

    Args:
        numpy.ndarray[float]: the unnormalized matrix of the transitions

    Returns:
        numpy.ndarray[float]: the normalized matrix of the transitions
    """
    toRet = transMat.copy()
    for row in range(transMat.shape[0]):
        sum = numpy.sum(toRet[row, :])
        if sum != 0:
            toRet[row, :] /= sum
    return toRet


# TODO add stride/window selection
def transitionMatrixFromSOAPClassificationNormalized(
    data: SOAPclassification, stride: int = 1
) -> "numpy.ndarray[float]":
    """Generates the normalized matrix of the transitions from a :func:`classify` and normalize it

        The matrix is organized in the following way:
        for each atom in each frame we increment by one the cell whose row is the
        state at the frame `n-stride` and the column is the state at the frame `n`

        The matrix is normalized with the criterion that the sum of each **row** is `1`

    Args:
        data (SOAPclassification): the results of the soapClassification from :func:`classify`
        stride (int): the stride in frames between each state confrontation. Defaults to 1.
        of the groups that contain the references in hdf5FileReference
    Returns:
        numpy.ndarray[float]: the normalized matrix of the transitions
    """
    transMat = transitionMatrixFromSOAPClassification(data, stride)
    return normalizeMatrix(transMat)


#: the index of the component of the statetracker that stores the previous state
TRACK_PREVSTATE = 0
#: the index of the component of the statetracker that stores the current state
TRACK_CURSTATE = 1
#: the index of the component of the statetracker that stores the next state
TRACK_ENDSTATE = 2
#: the index of the component of the statetracker that stores the duration of the state, in frames
TRACK_EVENTTIME = 3


def _createStateTracker(
    prevState: int, curState: int, endState: int, eventTime: int = 0
) -> numpy.ndarray:
    """Compile the given collection of data in a state tracker

    Args:
        prevState (int): the id of the previous state
        curState (int): the id of the current state
        endState (int): the id of the next state
        eventTime (int, optional): the duration (in frames) of this event. Defaults to 0.

    Returns:
        numpy.ndarray: the state tracker
    """
    return numpy.array([prevState, curState, endState, eventTime], dtype=int)


# TODO add stride/window here
def trackStates(classification: SOAPclassification) -> list:
    """Creates an ordered list of events for each atom in the classified trajectory
    each event is a numpy.array with four compontents: the previous state, the current state, the final state and the duration of the current state

    Args:
        classification (SOAPclassification): the classified trajectory

    Returns:
        list: ordered list of events for each atom in the classified trajectory
    """
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
) -> "list[numpy.ndarray]":
    """Given a SOAPclassification, calculate the resindence time for each element of the classification.
        The residence time is how much an atom stays in a determined state: this function calculates the redidence time for each atom and for each state,
        and returns an ordered list of residence times for each state (hence losing the atom identity)

    Args:
        classification (SOAPclassification): the classified trajectory

    Returns:
        list[numpy.ndarray]: an ordered list of the residence times for each state
    """

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
        residenceTimes[i] = numpy.sort(numpy.array(residenceTimes[i]))

    return residenceTimes


def RemoveAtomIdentityFromEventTracker(statesTracker: list) -> list:
    """Merge all of the list of stateTracker into a single lists, by removing the information of what atom did a given event.


    Args:
        statesTracker (list): a list of list of state trackers, organized by atoms

    Returns:
        list: the list of stateTracker organized only by states
    """
    if isinstance(statesTracker[0], list):
        t = []
        for tracks in statesTracker:
            t += tracks
        return t
    return statesTracker


def getResidenceTimesFromStateTracker(
    statesTracker: list, legend: list
) -> "list[numpy.ndarray]":
    """Given a state tracker and the list of the states returns the list of residence times per state

    Args:
        statesTracker (list): a list of list of state trackers, organized by atoms, or a list of state trackers
        legend (list): the list of states

    Returns:
        list[numpy.ndarray]: an ordered list of the residence times for each state
    """
    states = RemoveAtomIdentityFromEventTracker(statesTracker)

    residenceTimes = [[] for i in range(len(legend))]
    for event in states:
        residenceTimes[event[TRACK_CURSTATE]].append(
            event[TRACK_EVENTTIME]
            if event[TRACK_ENDSTATE] != event[TRACK_CURSTATE]
            else -event[TRACK_EVENTTIME]
        )
    for i in range(len(residenceTimes)):
        residenceTimes[i] = numpy.sort(numpy.array(residenceTimes[i]))
    return residenceTimes


def calculateResidenceTimes(
    data: SOAPclassification, statesTracker: list = None
) -> "list[numpy.ndarray]":
    """Given a classification (and the state tracker) generates a ordered list of residence times per state

    Args:
        data (SOAPclassification): the classified trajectory, is statesTracker is passed will be used fog getting the legend of the states
        statesTracker (list, optional): a list of list of state trackers, organized by atoms, or a list of state trackers. Defaults to None.

    Returns:
        list[numpy.ndarray]: an ordered list of the residence times for each state
    """

    if not statesTracker:
        return calculateResidenceTimesFromClassification(data)
    else:
        return getResidenceTimesFromStateTracker(statesTracker, data.legend)


def transitionMatrixFromStateTracker(
    statesTracker: list, legend: list
) -> numpy.ndarray:
    """Generates the unnormalized matrix of the transitions from a statesTracker

    see :func:`calculateTransitionMatrix` for a detailed description of an unnormalized transition matrix

    Args:
        statesTracker (list): a list of list of state trackers, organized by atoms, or a list of state trackers
        legend (list): the list of states

    Returns:
        numpy.ndarray[float]: the unnormalized matrix of the transitions
    """
    states = RemoveAtomIdentityFromEventTracker(statesTracker)

    nclasses = len(legend)
    transMat = numpy.zeros((nclasses, nclasses), dtype=numpy.float64)
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
) -> numpy.ndarray:
    """Generates the unnormalized matrix of the transitions from a :func:`classify` of from a statesTracker

        The matrix is organized in the following way:
        for each atom in each frame we increment by one the cell whose row is the
        state at the frame `n-stride` and the column is the state at the frame `n`
        If the classification includes an error with a `-1` values the user should add an 'error' class in the legend

    Args:
        data (SOAPclassification): the results of the soapClassification from :func:`classify`
        stride (int): the stride in frames between each state confrontation. Defaults to 1.
        statesTracker (list, optional): _description_. Defaults to None.

    Returns:
        numpy.ndarray[float]: the unnormalized matrix of the transitions
    """

    if not statesTracker:
        return transitionMatrixFromSOAPClassification(data, stride)
    else:
        return transitionMatrixFromStateTracker(statesTracker, data.legend)
