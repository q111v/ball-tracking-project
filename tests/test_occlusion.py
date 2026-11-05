import numpy as np
from balltrack.occlusion import ShortGapPredictor

I = np.eye(2, 3)


def test_ballistic_short_gap_is_estimated_without_observation_updates():
    p = ShortGapPredictor(0.3)
    for f in range(6):
        p.update(f, 30, I, (1, [100 + 4 * f, 100 - 5 * f + 0.6 * f * f], 20))
    for f in range(6, 12):
        result = p.update(f, 30, I, None, (540, 960))
        assert result is not None
        assert np.allclose(
            result["center"], [100 + 4 * f, 100 - 5 * f + 0.6 * f * f], atol=1e-5
        )
    assert len(p.history) == 6


def test_prediction_stops_at_horizon_and_requires_observations():
    p = ShortGapPredictor(0.2)
    assert p.update(0, 30, I, None) is None
    for f in range(6):
        p.update(f, 30, I, (1, [100 + f, 100], 20))
    assert p.update(11, 30, I, None) is not None
    assert p.update(12, 30, I, None) is None


def test_camera_transform_and_identity_change():
    p = ShortGapPredictor()
    for f in range(4):
        p.update(f, 30, I, (1, [100 + f, 100], 20))
    matrix = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 3.0]])
    r = p.update(4, 30, matrix, None)
    assert np.allclose(r["center"], [114, 103])
    p.update(5, 30, I, (2, [300, 300], 20))
    assert len(p.history) == 1 and not p.path
    assert p.update(6, 30, I, None) is None


def test_unreliable_camera_and_out_of_frame_stop_prediction():
    p = ShortGapPredictor()
    for f in range(6):
        p.update(f, 30, I, (1, [90 + 2 * f, 20], 20))
    assert p.update(6, 30, I, None, (50, 100)) is None
    assert p.update(7, 30, None, None) is None
    assert not p.history


def test_reacquisition_disagreement_discards_predicted_connection():
    p = ShortGapPredictor()
    for f in range(6):
        p.update(f, 30, I, (1, [100 + f, 100], 20))
    assert p.update(6, 30, I, None) is not None
    p.update(7, 30, I, (1, [600, 400], 20))
    assert not p.path
    assert np.allclose(p.history[-1][1], [600, 400])


def test_short_segments_still_form_a_dashed_line():
    from collections import deque

    predictor = ShortGapPredictor()
    predictor.path = deque((i, np.array([10 + 3 * i, 50.0])) for i in range(11))
    image = np.zeros((100, 100, 3), np.uint8)
    predictor.draw(image, None)
    assert image[50, 14].sum() > 0
    assert image[50, 20].sum() < image[50, 14].sum() * 0.5
