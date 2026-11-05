import cv2
import numpy as np
from balltrack.head_memory import HeadMemory


def test_semantic_head_is_retained_during_one_pose_miss():
    rng = np.random.default_rng(9)
    image = np.zeros((180, 240, 3), np.uint8)
    image[60:100, 70:110] = rng.integers(30, 230, size=(40, 40, 3), dtype=np.uint8)
    memory = HeadMemory()
    memory.update(image, [(np.array([90.0, 80.0]), 18)])
    moved = cv2.warpAffine(image, np.float32([[1, 0, 5], [0, 1, 2]]), (240, 180))
    heads = memory.update(moved, [])
    assert heads
    assert np.linalg.norm(heads[0][0] - [95, 82]) < 2


def test_blank_frame_does_not_support_a_hallucinated_head():
    rng = np.random.default_rng(9)
    image = rng.integers(0, 255, size=(180, 240, 3), dtype=np.uint8)
    memory = HeadMemory()
    memory.update(image, [(np.array([90.0, 80.0]), 18)])
    assert memory.update(np.zeros_like(image), []) == []
