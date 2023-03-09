"""submodules with the stateTracker workflow"""
import numpy
from ..classify import SOAPclassification


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


def trackStates(
    classification: SOAPclassification, window: int = 1, stride: "int|None" = None
) -> list:
    """Creates an ordered list of events for each atom in the classified trajectory

    each tracker is composed of events, each events is and array of foir components:
    `[start state, state, end state,time]`:

        - if `PREVSTATE == CURSTATE` is the first event for the atom
        - if `ENDSTATE == CURSTATE` is the last event for the atom

    Args:
        classification (SOAPclassification):
            the classified trajectory
        stride (int):
            the stride in frames between each window. Defaults to 1.
    Returns:
        list: ordered list of events for each atom in the classified trajectory
    """

    if stride is None:
        stride = window
    if stride > window:
        raise ValueError("the window must be bigger than the stride")

    if (
        window > classification.references.shape[0]
        or stride > classification.references.shape[0]
    ):
        raise ValueError("stride and window must be smaller than simulation lenght")

    nofFrames = classification.references.shape[0]
    nofAtoms = classification.references.shape[1]
    stateHistory = [None for _ in range(nofAtoms)]
    # should I use a dedicated class?
    # TODO: this can be made concurrent per atom
    for atomID in range(nofAtoms):
        statesPerAtom = []
        atomTraj = classification.references[:, atomID]
        for iframe in range(0, window, stride):
            # the array is [start state, state, end state,time]
            # if PREVSTATE == CURSTATE the event is the first event for the atom
            # if ENDSTATE == CURSTATE the event is the last event for the atom
            stateTracker = _createStateTracker(
                prevState=atomTraj[iframe],
                curState=atomTraj[iframe],
                endState=atomTraj[iframe],
                eventTime=0,
            )
            for frame in range(window + iframe, nofFrames, window):
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
        stateHistory[atomID] = statesPerAtom
    return stateHistory


def removeAtomIdentityFromEventTracker(statesTracker: list) -> list:
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
    states = removeAtomIdentityFromEventTracker(statesTracker)

    residenceTimes = [[] for _ in range(len(legend))]
    for event in states:
        residenceTimes[event[TRACK_CURSTATE]].append(
            event[TRACK_EVENTTIME]
            if event[TRACK_ENDSTATE] != event[TRACK_CURSTATE]
            else -event[TRACK_EVENTTIME]
        )

    for i, rts in enumerate(residenceTimes):
        residenceTimes[i] = numpy.sort(numpy.array(rts))

    return residenceTimes


def transitionMatrixFromStateTracker(
    statesTracker: list, legend: list
) -> numpy.ndarray:
    """Generates the unnormalized matrix of the transitions from a statesTracker

    see :func:`calculateTransitionMatrix` for a detailed description of an
    unnormalized transition matrix

    Args:
        statesTracker (list):
            a list of list of state trackers, organized by atoms,
            or a list of state trackers
        legend (list):
            the list of states

    Returns:
        numpy.ndarray[float]: the unnormalized matrix of the transitions
    """
    states = removeAtomIdentityFromEventTracker(statesTracker)

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
