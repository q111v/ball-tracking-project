import cv2
import numpy as np
from balltrack.motion import CameraMotion
from balltrack.display_trail import DisplayTrail


def texture():
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
    return cv2.GaussianBlur(image, (3, 3), 0)


def test_background_translation_and_zoom():
    image = texture()
    expected = cv2.getRotationMatrix2D((320, 180), 1, 1.025)
    expected[:, 2] += [8, 4]
    moved = cv2.warpAffine(image, expected, (640, 360))
    camera = CameraMotion()
    assert camera.update(image)[1] == "initial"
    matrix, state = camera.update(moved)
    assert state == "ok"
    points = np.array([[80, 100, 1], [320, 180, 1], [550, 260, 1]], float)
    assert np.max(np.linalg.norm(points @ matrix.T - points @ expected.T, axis=1)) < 1


def test_cut_and_low_texture_are_rejected():
    camera = CameraMotion()
    camera.update(np.zeros((360, 640, 3), np.uint8))
    assert camera.update(np.full((360, 640, 3), 255, np.uint8))[1] == "cut"
    assert camera.update(np.full((360, 640, 3), 255, np.uint8))[1] == "unreliable"


def test_cumulative_transform_and_no_change_to_measurement():
    trail = DisplayTrail(seconds=0.7, smooth=1)
    raw = np.array([100.0, 100.0])
    trail.update(0, 30, None, 1, raw)
    matrix = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 3.0]])
    trail.update(1, 30, matrix, 1, [115, 105])
    assert np.allclose(trail.points[0][2], [110, 103])
    assert np.allclose(raw, [100, 100])
    trail.update(2, 30, matrix, 1, [128, 110])
    assert np.allclose(trail.points[0][2], [120, 106])


def test_cut_gap_jump_and_identity_break_old_segments():
    trail = DisplayTrail()
    identity = np.eye(2, 3)
    trail.update(0, 30, None, 1, [10, 10])
    trail.update(1, 30, identity, 1, [12, 10])
    assert len(trail.points) == 2
    trail.update(2, 30, None, 1, [14, 10])
    assert len(trail.points) == 1
    trail.update(4, 30, identity, 1, [18, 10])
    assert len(trail.points) == 1
    trail.update(5, 30, identity, 2, [20, 10])
    assert len(trail.points) == 1
    trail.update(6, 30, identity, 2, [400, 100])
    assert len(trail.points) == 1


def test_expiration_and_fading_drawing():
    trail = DisplayTrail(seconds=0.5, smooth=1)
    identity = np.eye(2, 3)
    image = np.zeros((100, 160, 3), np.uint8)
    for frame in range(30):
        trail.update(frame, 30, identity, 1, [10 + frame * 3, 40])
    assert len(trail.points) <= 16
    trail.draw(image, 29, 30)
    assert image[40, 90].sum() > image[40, 50].sum() > 0
    trail.update(50, 30, identity, -1, None)
    assert not trail.points


def test_motion_maps_back_to_original_resolution():
    image = cv2.resize(texture(), (1280, 720))
    expected = np.array([[1.0, 0.0, 16.0], [0.0, 1.0, 8.0]])
    moved = cv2.warpAffine(image, expected, (1280, 720))
    camera = CameraMotion()
    camera.update(image)
    matrix, state = camera.update(moved)
    assert state == "ok"
    assert np.allclose(matrix[:, 2], expected[:, 2], atol=1)


def test_moving_foreground_is_excluded():
    image = texture()
    moved = cv2.warpAffine(image, np.float32([[1, 0, 7], [0, 1, 3]]), (640, 360))
    rng = np.random.default_rng(81)
    moved[110:260, 240:360] = rng.integers(0, 256, (150, 120, 3), dtype=np.uint8)
    camera = CameraMotion()
    heads = [(np.array([300, 125]), 30)]
    camera.update(image, heads)
    matrix, state = camera.update(moved, heads)
    assert state == "ok"
    assert np.allclose(matrix[:, 2], [7, 3], atol=1)
