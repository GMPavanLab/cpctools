"""test for the stateTracker utility"""
from numpy.testing import assert_array_equal
import SOAPify
import numpy
import pytest
from SOAPify.classify import SOAPclassification
from .test_transitions import _expectedTotalFrames

################################################################################
################################################################################
###The tests here assume that the tests in test_transitions are all successful##
################################################################################
################################################################################

PREVSTATE = SOAPify.TRACK_PREVSTATE
CURSTATE = SOAPify.TRACK_CURSTATE
ENDSTATE = SOAPify.TRACK_ENDSTATE
TIME = SOAPify.TRACK_EVENTTIME


def test_classStateTracker(
    input_mockedTrajectoryClassification, inputStrides, inputWindows
):
    """test for the StateTracker class"""
    nat = input_mockedTrajectoryClassification.references.shape[1]
    stride = inputStrides
    window = inputWindows
    states = SOAPify.StateTracker(nat, stride=stride, window=window)
    assert states.window == window
    assert states.stride == stride
    assert len(states) == nat


def test_stateTrackerBehaviourNoStateChanges():
    """The StateTracker time must pass this tests first:

    There should be only one statewith leght equal to the trajectory"""
    threeStates: list = ["state0", "state1", "state2"]

    for nFrames in range(4, 8):
        dataNC: SOAPclassification = SOAPclassification(
            [], numpy.array([[0]] * nFrames), threeStates
        )
        tracker = SOAPify.trackStates(dataNC, stride=1, window=1)
        # 1 atom
        assert len(tracker) == 1
        assert len(tracker[0]) == 1
        assert tracker[0][0][PREVSTATE] == tracker[0][0][CURSTATE]
        assert tracker[0][0][ENDSTATE] == tracker[0][0][CURSTATE]
        assert tracker[0][0][CURSTATE] == 0
        assert tracker[0][0][TIME] == nFrames


def test_stateTrackerBehaviourAlternatingStates():
    """The StateTracker time must pass this tests first

    an alternating states should return a series of events of lenght one
    """

    threeStates: list = ["state0", "state1", "state2"]
    nFrames = 12
    data: SOAPclassification = SOAPclassification(
        [], numpy.array([[0], [1]] * (nFrames // 2)), threeStates
    )
    tracker = SOAPify.trackStates(data, stride=1, window=1)
    # 1 atom
    assert len(tracker) == 1
    assert len(tracker[0]) == nFrames
    totalTime = 0
    for event in tracker[0]:
        print(event)
        assert event[TIME] == 1
        totalTime += event[TIME]
    assert totalTime == nFrames
    # ensuringht the behaviour
    assert tracker[0][0][PREVSTATE] == tracker[0][0][CURSTATE]
    assert tracker[0][0][ENDSTATE] != tracker[0][0][CURSTATE]
    assert tracker[0][-1][PREVSTATE] != tracker[0][-1][CURSTATE]
    assert tracker[0][-1][ENDSTATE] == tracker[0][-1][CURSTATE]


def test_stateTrackerBehaviourAlternatingStatesOdd():
    """The StateTracker time must pass this tests first

    an alternating states should return a series of events of lenght one
    """

    threeStates: list = ["state0", "state1", "state2"]
    nFrames = 13
    # the // makes the 13 behave like a 12 as a integer division
    data: SOAPclassification = SOAPclassification(
        [], numpy.array([[0], [1]] * (nFrames // 2) + [[0]]), threeStates
    )
    tracker = SOAPify.trackStates(data, stride=1, window=1)
    # 1 atom
    assert len(tracker) == 1
    assert len(tracker[0]) == nFrames
    totalTime = 0
    for event in tracker[0]:
        print(event)
        assert event[TIME] == 1
        totalTime += event[TIME]
    assert totalTime == nFrames
    # ensuringht the behaviour
    assert tracker[0][0][PREVSTATE] == tracker[0][0][CURSTATE]
    assert tracker[0][0][ENDSTATE] != tracker[0][0][CURSTATE]
    assert tracker[0][-1][PREVSTATE] != tracker[0][-1][CURSTATE]
    assert tracker[0][-1][ENDSTATE] == tracker[0][-1][CURSTATE]


def test_stateTrackerBehaviourAlternatingStatesTwoInTwo():
    """The StateTracker time must pass this tests first

    an alternating states should return a series of events of lenght two
    """
    threeStates: list = ["state0", "state1", "state2"]
    nFrames = 12
    data: SOAPclassification = SOAPclassification(
        [], numpy.array([[0], [0], [1], [1]] * (nFrames // 4)), threeStates
    )
    tracker = SOAPify.trackStates(data, stride=1, window=1)
    # 1 atom
    assert len(tracker) == 1
    assert len(tracker[0]) == nFrames // 2
    totalTime = 0
    for event in tracker[0]:
        print(event)
        assert event[TIME] == 2
        totalTime += event[TIME]
    assert totalTime == nFrames
    # ensuringht the behaviour
    assert tracker[0][0][PREVSTATE] == tracker[0][0][CURSTATE]
    assert tracker[0][0][ENDSTATE] != tracker[0][0][CURSTATE]
    assert tracker[0][-1][PREVSTATE] != tracker[0][-1][CURSTATE]
    assert tracker[0][-1][ENDSTATE] == tracker[0][-1][CURSTATE]


def test_stateTrackerBehaviourWindowAndTrajectory():
    """The StateTracker time must pass this tests first

    Th resulting time of the simulation will be different,
    and should be proportional to the given window
    """
    threeStates: list = ["state0", "state1", "state2"]
    nFrames = 12
    data: SOAPclassification = SOAPclassification(
        [], numpy.array([[0]] * (nFrames)), threeStates
    )
    for window in range(1, nFrames + 1):
        stride = window
        tracker = SOAPify.trackStates(data, stride=stride, window=window)
        assert len(tracker) == 1
        assert len(tracker[0]) == 1
        print(*tracker[0])
        total = numpy.sum([rt[TIME] for rt in tracker[0]])
        assert total == _expectedTotalFrames(nFrames, window, stride)


def test_stateTrackerBehaviourWindowAndStrideAndTrajectory():
    """The StateTracker time must pass this tests first

    Th resulting time of the simulation will be different,
    and should be proportional to the given window
    """
    threeStates: list = ["state0", "state1", "state2"]
    nFrames = 12
    data: SOAPclassification = SOAPclassification(
        [], numpy.array([[0]] * (nFrames)), threeStates
    )
    for window in range(1, nFrames + 1):
        for stride in range(1, window + 1):
            tracker = SOAPify.trackStates(data, stride=stride, window=window)
            assert len(tracker) == 1
            assert len(tracker[0]) == len(range(0, window, stride))
            print(*tracker[0])
            total = numpy.sum([rt[TIME] for rt in tracker[0]])
            assert total == _expectedTotalFrames(nFrames, window, stride)


def test_stateTracker(
    input_mockedTrajectoryClassification, inputStridesWithNone, inputWindows
):
    data: SOAPclassification = input_mockedTrajectoryClassification
    stride = inputStridesWithNone
    window = inputWindows

    def isInvalidCombination(dataLen, stride, window):
        return (stride is not None and window < stride) or (
            (stride is not None and stride > dataLen) or window > dataLen
        )

    if isInvalidCombination(data.references.shape[0], stride, window):
        with pytest.raises(ValueError) as excinfo:
            SOAPify.trackStates(data, stride=stride, window=window)
        if stride is not None and window < stride:
            assert "window must be bigger" in str(excinfo.value)
            pytest.skip("Exception thrown correctly")
        if (
            stride is not None and stride > data.references.shape[0]
        ) or window > data.references.shape[0]:
            assert "window must be smaller" in str(excinfo.value)
            pytest.skip("Exception thrown correctly")
    events = SOAPify.trackStates(data, stride=stride, window=window)

    # hand calculation of the expected quantities
    if stride is None:
        stride = window
    # code for determining the events
    expectedEvents = [None for _ in range(data.references.shape[1])]
    for atomID in range(data.references.shape[1]):
        eventsperAtom = []
        atomTraj = data.references[:, atomID]

        for iframe in range(0, window, stride):
            # the array is [start state, state, end state,time]
            event = numpy.array(
                [
                    atomTraj[iframe],
                    atomTraj[iframe],
                    atomTraj[iframe],
                    0,
                ],
                dtype=int,
            )
            for frame in range(iframe + window, data.references.shape[0], window):
                event[TIME] += window
                if atomTraj[frame] != event[CURSTATE]:
                    event[ENDSTATE] = atomTraj[frame]
                    eventsperAtom.append(event)
                    event = numpy.array(
                        [
                            eventsperAtom[-1][CURSTATE],
                            atomTraj[frame],
                            atomTraj[frame],
                            0,
                        ],
                        dtype=int,
                    )

            # append the last event
            event[TIME] += window
            eventsperAtom.append(event)
        expectedEvents[atomID] = eventsperAtom

    assert len(events) == len(expectedEvents)
    print(len(events), len(expectedEvents), data.references.shape[1])
    print(len(events[0]), len(expectedEvents[0]))

    for atomID in range(data.references.shape[1]):
        assert len(events[atomID]) == len(expectedEvents[atomID])
        assert_array_equal(events[atomID], expectedEvents[atomID])


def test_transitionMatrixFromTracking(
    input_mockedTrajectoryClassification, inputStrides, inputWindows
):
    """Testing "same input-different procedure" for calculating transition matrices"""
    data = input_mockedTrajectoryClassification
    # the None entries are verified in other tests
    stride = inputStrides
    window = inputWindows

    if (
        window > data.references.shape[0]
        or stride > data.references.shape[0]
        or stride > window
    ):
        pytest.skip("failing condition are tested separately")
    # calculate tmat from data and stride and window
    expectedTmat = SOAPify.transitionMatrixFromSOAPClassification(
        data, stride=stride, window=window
    )
    # calculate tmat from data and stride and window, using the shortcut
    standardTmat = SOAPify.calculateTransitionMatrix(
        data,
        stride=stride,
        window=window,
    )
    assert_array_equal(standardTmat, expectedTmat)
    # calculate tmat from tracker
    events = SOAPify.trackStates(data, window=window, stride=stride)
    transitionMatrixFromTracking = SOAPify.calculateTransitionMatrix(
        data, statesTracker=events
    )
    assert_array_equal(transitionMatrixFromTracking, expectedTmat)


def test_residenceTimesFromTracking(
    input_mockedTrajectoryClassification, inputStrides, inputWindows
):
    """Testing "same input-different procedure" for calculating residence times"""
    data = input_mockedTrajectoryClassification
    # the None entries are verified in other tests
    stride = inputStrides
    window = inputWindows
    if (
        window > data.references.shape[0]
        or stride > data.references.shape[0]
        or stride > window
    ):  # will fail, so returns
        pytest.skip("failing condition are tested separately")

    # calculate RTs from data and stride and window
    expectedResidenceTimes = SOAPify.calculateResidenceTimesFromClassification(
        data, window=window, stride=stride
    )
    # calculate RTs from data and stride and window with shortcut
    shortcutResidenceTimes = SOAPify.calculateResidenceTimes(
        data, window=window, stride=stride
    )
    # calculate RTs from events
    events = SOAPify.trackStates(data, window=window, stride=stride)
    residenceTimesFromTracking = SOAPify.calculateResidenceTimes(data, events)

    for rtFromTrack, rtShortcut, rtExpected in zip(
        residenceTimesFromTracking,
        shortcutResidenceTimes,
        expectedResidenceTimes,
    ):
        assert_array_equal(rtFromTrack, rtExpected)
        assert_array_equal(rtShortcut, rtExpected)


def test_RemoveAtomIdentityFromEventTracker(input_mockedTrajectoryClassification):
    data = input_mockedTrajectoryClassification
    events = SOAPify.trackStates(data)
    newevents = SOAPify.removeAtomIdentityFromEventTracker(events)
    # verify that nothing is changed:
    assert isinstance(events[0], list)
    assert isinstance(events, SOAPify.StateTracker)
    assert isinstance(events[0][0], numpy.ndarray)
    # verify that the copy is correct:
    assert isinstance(newevents, SOAPify.StateTracker)
    count = 0
    for atomID in range(data.references.shape[1]):
        for event in events[atomID]:
            assert_array_equal(event, newevents[0][count])
            assert isinstance(newevents[0][count], numpy.ndarray)
            count += 1
    # if we pass to the function a list of tracker
    otherevents = SOAPify.removeAtomIdentityFromEventTracker(newevents)
    # nothing should happen
    assert otherevents == newevents
