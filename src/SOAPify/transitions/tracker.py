"""
This submodule contains the function to build and work with the 'state trackers':
a state tracker is a list of 4-element arrays whose components
represent an 'event':
[previous state ID, current state ID, next state ID, the duration of the event]

In the module there is some logic to help to the user to understand if the event
is a '*first one*' or a '*last one*' (aka the first/las seen in the simulation for
each atom): a *first one* will have `previous state ID = current state ID` and 
a *last one* will have `next state ID = current state ID`
"""
import numpy

from ..classify import SOAPclassification


#: the index of the component of the statetracker with the previous state
TRACK_PREVSTATE = 0
#: the index of the component of the statetracker with the current state
TRACK_CURSTATE = 1
#: the index of the component of the statetracker with the next state
TRACK_ENDSTATE = 2
#: the index of the component of the statetracker with the duration of the state, in frames
TRACK_EVENTTIME = 3


def _createStateTracker(
    prevState: int, curState: int, endState: int, eventTime: int = 0
) -> numpy.ndarray:
    """Compile the given collection of data in a state tracker

    Args:
        prevState (int): the id of the previous state
        curState (int): the id of the current state
        endState (int): the id of the next state
        eventTime (int, optional):
            the duration (in frames) of this event. Defaults to 0.

    Returns:
        numpy.ndarray: the state tracker
    """
    return numpy.array([prevState, curState, endState, eventTime], dtype=int)


# TODO add stride/window here
def trackStates(classification: SOAPclassification) -> list:
    """Creates a list of event that track states

        Creates an ordered list of events for each atom in the classified
        trajectory each event is a numpy.array with four compontents:
        the previous state, the current state, the final state and the duration
        of the current state

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
        # when PREVSTATE and CURSTATE are the same the event is the first event
        #   for the atom in the simulation
        # when ENDSTATE and CURSTATE are the same the event is the last event
        #   for the atom in the simulation
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


def removeAtomIdentityFromEventTracker(statesTracker: list) -> list:
    """Remove the atomic infromation from a list of trackers

        Merge all of the list of stateTracker into a single lists,
        by removing the information of what atom did a given event.


    Args:
        statesTracker (list):
            a list of list of state trackers, organized by atoms

    Returns:
        list: the list of stateTracker organized only by states
    """
    if isinstance(statesTracker[0], list):
        tracker = []
        for tracks in statesTracker:
            tracker += tracks
        return tracker
    return statesTracker


def getResidenceTimesFromStateTracker(
    statesTracker: list, legend: list
) -> "list[numpy.ndarray]":
    """Calculates the resindence times from the events.

        Given a state tracker and the list of the states returns the list of
        residence times per state

    Args:
        statesTracker (list):
            a list of list of state trackers, organized by atoms,
            or a list of state trackers
        legend (list): the list of states

    Returns:
        list[numpy.ndarray]:
        an ordered list of the residence times for each state
    """
    states = removeAtomIdentityFromEventTracker(statesTracker)

    residenceTimes = [[] for i in range(len(legend))]
    for event in states:
        residenceTimes[event[TRACK_CURSTATE]].append(
            event[TRACK_EVENTTIME]
            if event[TRACK_ENDSTATE] != event[TRACK_CURSTATE]
            else -event[TRACK_EVENTTIME]
        )

    return [
        numpy.sort(numpy.array(residenceTimes[i])) for i in range(len(residenceTimes))
    ]


def transitionMatrixFromStateTracker(
    statesTracker: list, legend: list
) -> numpy.ndarray:
    """Generates the unnormalized matrix of the transitions

    see :func:`SOAPify.transitions.calculateTransitionMatrix` for a detailed description of an
    unnormalized transition matrix

    Args:
        statesTracker (list):
            a list of list of state trackers, organized by atoms, or a list of
            state trackers
        legend (list): the list of states

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
