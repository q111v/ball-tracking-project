from pathlib import Path

import numpy as np

from balltrack.video import cache_read, cache_write, chunk_capacity
from balltrack.temporal import select_trajectory


def observation(x, y=100, confidence=0.8):
    return (
        np.array([x - 10, y - 10, x + 10, y + 10], float),
        confidence,
        "test_detection",
    )


def test_rejects_isolated_high_confidence_jump_to_a_head():
    forward = [observation(100 + 3 * i, confidence=0.5) for i in range(12)]
    reverse = list(forward)
    reverse[5] = observation(400, 300, 0.99)
    chosen = select_trajectory(forward, reverse)
    assert np.allclose(chosen[5][0], forward[5][0])


def test_missing_in_both_passes_remains_missing():
    forward = [
        observation(100 + 3 * i) if i not in (4, 5, 6) else None for i in range(12)
    ]
    selected = select_trajectory(forward, forward)
    assert selected[4:7] == [None, None, None]


def test_backward_evidence_fills_a_real_detector_gap():
    forward = [observation(100 + 3 * i) if i != 5 else None for i in range(12)]
    backward = [None] * 12
    backward[5] = observation(115)
    selected = select_trajectory(forward, backward)
    assert selected[5] is backward[5]


def test_frame_cache_supports_unicode_and_preserves_pixels(tmp_path):
    image = np.random.default_rng(3).integers(0, 256, size=(30, 40, 3), dtype=np.uint8)
    folder = tmp_path / "篮球测试"
    folder.mkdir()
    path = folder / "帧.png"
    cache_write(path, image)
    assert np.array_equal(image, cache_read(path))


def test_cache_capacity_scales_down_for_4k_video():
    shape = (2160, 3840, 3)
    count = chunk_capacity(shape)
    assert 1 <= count < 120
    assert count * np.prod(shape) <= 384 * 1024 * 1024


def test_manual_target_cannot_be_overridden_by_confident_wrong_reverse_track():
    from balltrack.temporal import select_anchored_trajectory

    forward = [observation(100 + 3 * i, confidence=0.7) for i in range(20)]
    reverse = [observation(500, 400, 0.99) for i in range(20)]
    result = select_anchored_trajectory(forward, reverse)
    assert all(a is b for a, b in zip(result, forward))


def test_manual_gap_rejects_unrelated_reverse_detection():
    from balltrack.temporal import select_anchored_trajectory

    forward = [observation(100 + 3 * i) if i != 5 else None for i in range(12)]
    reverse = [None] * 12
    reverse[5] = observation(500, 400, 0.99)
    assert select_anchored_trajectory(forward, reverse)[5] is None
