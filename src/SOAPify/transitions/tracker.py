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


class StateTracker:
    """A contained for the state trackers"""

    stateHistory: list
    window_: int
    stride_: int

    def __init__(self, nat: int, window: int, stride: int) -> None:
        """_summary_

        Args:
            nat (int): _description_
            window (int): _description_
            stride (int): _description_
        """
        self.stateHistory = [None for _ in range(nat)]
        self.window_ = window
        self.stride_ = stride

    @property
    def window(self) -> int:
        """_summary_

        Returns:
            int: _description_
        """
        return self.window_

    @property
    def stride(self) -> int:
        """_summary_

        Returns:
            int: _description_
        """
        return self.stride_

    def __len__(self) -> int:
        """_summary_

        Returns:
            int: _description_
        """
        return len(self.stateHistory)

    def __getitem__(self, key):
        """_summary_

        Args:
            key (_type_): the addres of the list of the asked atom
        """
        return self.stateHistory[key]

    def __setitem__(self, key, data):
        """_summary_

        Args:
            key (_type_): the addres of the list of the asked atom
        """
        self.stateHistory[key] = data

    def __iter__(self):
        """iterate thought the stored list of events"""
        return iter(self.stateHistory)


def _createEvent(
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


def trackStates(
    classification: SOAPclassification, window: int = 1, stride: "int|None" = None
) -> StateTracker:
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
        StateTracker: ordered list of events for each atom in the classified trajectory
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

    stateHistory = StateTracker(nofAtoms, window, stride)
    # TODO: this can be made concurrent per atom
    for atomID in range(nofAtoms):
        statesPerAtom = []
        atomTraj = classification.references[:, atomID]
        for iframe in range(0, window, stride):
            # the array is [start state, state, end state,time]
            # if PREVSTATE == CURSTATE the event is the first event for the atom
            # if ENDSTATE == CURSTATE the event is the last event for the atom
            stateTracker = _createEvent(
                prevState=atomTraj[iframe],
                curState=atomTraj[iframe],
                endState=atomTraj[iframe],
                eventTime=0,
            )
            for frame in range(window + iframe, nofFrames, window):
                stateTracker[TRACK_EVENTTIME] += window
                if atomTraj[frame] != stateTracker[TRACK_CURSTATE]:
                    stateTracker[TRACK_ENDSTATE] = atomTraj[frame]
                    statesPerAtom.append(stateTracker)
                    stateTracker = _createEvent(
                        prevState=stateTracker[TRACK_CURSTATE],
                        curState=atomTraj[frame],
                        endState=atomTraj[frame],
                    )

            # append the last event
            stateTracker[TRACK_EVENTTIME] += window
            statesPerAtom.append(stateTracker)
        stateHistory[atomID] = statesPerAtom
    return stateHistory


def removeAtomIdentityFromEventTracker(statesTracker: StateTracker) -> StateTracker:
    """Merge all of the list of stateTracker into a single lists, by removing the information of what atom did a given event.


    Args:
        statesTracker (list):
            a list of list of state trackers, organized by atoms
        statesTracker (list):
            a list of list of state trackers, organized by atoms

    Returns:
        StateTracker: a state tracker
    """
    if len(statesTracker) > 1:
        newST = StateTracker(
            1, window=statesTracker.window, stride=statesTracker.stride
        )
        newST.stateHistory[0] = []
        for tracks in statesTracker.stateHistory:
            newST.stateHistory[0] += tracks
        return newST
    return statesTracker


def getResidenceTimesFromStateTracker(
    statesTracker: StateTracker, legend: list
) -> "list[numpy.ndarray]":
    """Calculates the resindence times from the events.

        Given a state tracker and the list of the states returns the list of
        residence times per state

    Args:
        statesTracker (list):

            a list of list of state trackers, organized by atoms,
            or a list of state trackers
        legend (list):
            the list of states

    Returns:
        list[numpy.ndarray]:
        an ordered list of the residence times for each state
    """
    states = removeAtomIdentityFromEventTracker(statesTracker)

    residenceTimes = [[] for _ in range(len(legend))]
    # using states[0] because we have removed the atom identities
    for event in states[0]:
        residenceTimes[event[TRACK_CURSTATE]].append(
            event[TRACK_EVENTTIME]
            if (
                event[TRACK_ENDSTATE] != event[TRACK_CURSTATE]
                and event[TRACK_PREVSTATE] != event[TRACK_CURSTATE]
            )
            else -event[TRACK_EVENTTIME]
        )

    for i, rts in enumerate(residenceTimes):
        residenceTimes[i] = numpy.sort(numpy.array(rts))

    return residenceTimes


def transitionMatrixFromStateTracker(
    statesTracker: StateTracker, legend: list
) -> numpy.ndarray:
    """Generates the unnormalized matrix of the transitions

    see :func:`calculateTransitionMatrix` for a detailed description of an
    unnormalized transition matrix

    Args:
        statesTracker (StateTracker):
            a StateTracker
        legend (list):
            the list of the name of the states

    Returns:
        numpy.ndarray[float]: the unnormalized matrix of the transitions
    """
    states = removeAtomIdentityFromEventTracker(statesTracker)

    nclasses = len(legend)
    transMat = numpy.zeros((nclasses, nclasses), dtype=numpy.float64)
    window = statesTracker.window
    # using states[0] because we have removed the atom identities
    for event in states[0]:
        transMat[event[TRACK_CURSTATE], event[TRACK_CURSTATE]] += (
            event[TRACK_EVENTTIME] // window - 1
        )
        # the transition matrix is genetated with:
        #   classFrom = data.references[frameID - stride][atomID]
        #   classTo = data.references[frameID][atomID]
        if event[TRACK_PREVSTATE] != event[TRACK_CURSTATE]:
            transMat[event[TRACK_PREVSTATE], event[TRACK_CURSTATE]] += 1
    return transMat
