from types import SimpleNamespace

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from balltrack.basketball import association_box,head_regions


def test_bent_player_crown_is_rejected_but_ball_below_hand_is_not():
    points=np.array([[946,392,.6],[947,390,.16],[946,389,.66],[939,386,.06],[941,385,.81]])
    center,radius=head_regions(points,[866,370,956,470])[0]
    assert np.linalg.norm(np.array([946,380])-center)<radius
    assert np.linalg.norm(np.array([934,420])-center)>radius


def test_no_reliable_face_does_not_invent_head_region():
    assert head_regions(np.zeros((17,3)),[0,0,100,100])==[]


def test_small_fast_ball_reconnects_after_gap_without_drawing_fake_box():
    tracker=BYTETracker(SimpleNamespace(track_high_thresh=.25,track_low_thresh=.1,
        new_track_thresh=.35,track_buffer=45,match_thresh=.99,fuse_score=False),frame_rate=30)
    first=np.array([930,470,950,490],dtype=float)
    initial=tracker.update(Boxes(np.array([[*association_box(first),.8,0]],np.float32),(720,1280)))
    identity=int(initial[0,4])
    for _ in range(10):
        empty=tracker.update(Boxes(np.empty((0,6),np.float32),(720,1280)))
        assert len(empty)==0
    actual=np.array([950,420,970,440],dtype=float)
    restored=tracker.update(Boxes(np.array([[*association_box(actual),.8,0]],np.float32),(720,1280)))
    assert int(restored[0,4])==identity
    assert np.array_equal(actual,[950,420,970,440])
