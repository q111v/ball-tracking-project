from types import SimpleNamespace
import numpy as np
from balltrack.closeup import CloseupDetector, suppress_equipment


def detector():
    return CloseupDetector(
        SimpleNamespace(names={0: "ball", 1: "rim", 2: "net"}), "cpu", [0]
    )


def test_net_duplicate_rejected_but_small_ball_near_basket_remains():
    net = np.array([100, 100, 140, 140], float)
    fake = (net.copy(), 0.8, "closeup_full")
    ball = (np.array([106, 108, 118, 120], float), 0.7, "closeup_full")
    assert suppress_equipment([fake, ball], [net]) == [ball]


def test_strong_global_ball_reacquired_after_wrong_prediction():
    d = detector()
    d.diameter = 20
    ball = (np.array([400, 80, 420, 100], float), 0.95, "closeup_full")
    choice = d.choose(np.zeros((540, 960, 3), np.uint8), [ball], np.array([100, 400]))
    assert choice is not None


def test_weak_roi_shoe_is_rejected():
    d = detector()
    image = np.zeros((540, 960, 3), np.uint8)
    shoe = (np.array([100, 400, 130, 430], float), 0.2, "closeup_roi")
    assert d.choose(image, [shoe], np.array([110, 410])) is None


def test_no_observation_is_invented_and_elapsed_motion_used():
    d = detector()
    d.center = np.array([100.0, 100.0])
    d.velocity = np.array([4.0, 0.0])
    d.misses = 2
    predictions = []
    d.predict = lambda *args: ([], [])

    def choose(image, candidates, prediction):
        predictions.append(prediction.copy())
        return None

    d.choose = choose
    assert d.observe(np.zeros((540, 960, 3), np.uint8)) is None
    assert np.allclose(predictions[0], [112, 100])
    assert d.misses == 3


def test_equipment_memory_survives_one_missing_semantic_detection():
    d = detector()
    d.camera = SimpleNamespace(update=lambda image: (np.eye(2, 3), "ok"))
    net = np.array([100, 100, 140, 140], float)
    outputs = iter([([], [net]), ([], [])])
    d.predict = lambda *args: next(outputs)
    d.observe(np.zeros((540, 960, 3), np.uint8))
    d.observe(np.zeros((540, 960, 3), np.uint8))
    assert len(d.equipment) == 1
    assert np.allclose(d.equipment[0], net)


def test_equipment_memory_moves_with_camera_and_expires():
    d = detector()
    matrix = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 2.0]])
    d.camera = SimpleNamespace(update=lambda image: (matrix, "ok"))
    d.equipment_memory = [(np.array([100, 100, 140, 140], float), 0)]
    d.predict = lambda *args: ([], [])
    image = np.zeros((540, 960, 3), np.uint8)
    d.observe(image)
    assert np.allclose(d.equipment[0], [105, 102, 145, 142])
    for _ in range(15):
        d.observe(image)
    assert d.equipment == []


def test_head_core_is_rejected_even_with_high_ball_confidence():
    d = detector()
    d.heads = [(np.array([200.0, 200.0]), 30)]
    head = (np.array([185, 185, 215, 215], float), 0.98, "closeup_full")
    ball = (np.array([235, 245, 265, 275], float), 0.95, "closeup_full")
    chosen = d.choose(np.zeros((540, 960, 3), np.uint8), [head, ball], None)
    assert chosen is not None and np.allclose(chosen[1], ball[0])


def test_person_crop_recovers_ball_missing_in_full_frame():
    d = detector()
    d.pose = object()
    d.people = [np.array([200, 100, 300, 400])]
    calls = []
    box = np.array([310, 300, 340, 330], float)

    def predict(image, region, global_search, image_size=None):
        calls.append(region)
        return ([], []) if global_search else ([(box, 0.8, "closeup_roi")], [])

    d.predict = predict
    value = d.observe(np.zeros((540, 960, 3), np.uint8), heads=[])
    assert value is not None and value[2] == "closeup_person"
    assert any(region != (0, 0, 960, 540) for region in calls)


def test_closeup_face_geometry_does_not_mask_a_ball_next_to_head():
    from balltrack.closeup import closeup_heads

    points = np.zeros((17, 3))
    points[:5] = [
        [200, 200, 0.9],
        [196, 197, 0.9],
        [204, 197, 0.9],
        [193, 200, 0.8],
        [207, 200, 0.8],
    ]
    heads = closeup_heads(points, [150, 180, 250, 450])
    assert heads and heads[0][1] < 20
    d = detector()
    d.heads = heads
    ball = (np.array([216, 188, 236, 208], float), 0.8, "closeup_full")
    assert d.choose(np.zeros((540, 960, 3), np.uint8), [ball], None) is not None


def test_ball_hue_averaging_handles_red_wraparound():
    import cv2

    hsv = np.full((40, 40, 3), [179, 160, 80], np.uint8)
    hsv[:, :20, 0] = 1
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    color = CloseupDetector.appearance(image, np.array([0, 0, 40, 40]))
    assert min(color[0], 180 - color[0]) < 5


def test_moderate_confidence_cannot_jump_to_distant_body_part():
    d = detector()
    d.diameter = 30
    d.airborne = True
    wrong = (np.array([440, 360, 470, 390], float), 0.4, "closeup_full")
    assert (
        d.choose(np.zeros((540, 960, 3), np.uint8), [wrong], np.array([380, 160]))
        is None
    )


def test_holding_phase_can_reacquire_strong_global_ball():
    d = detector()
    d.diameter = 30
    d.airborne = False
    ball = (np.array([400, 80, 420, 100], float), 0.7, "closeup_full")
    assert (
        d.choose(np.zeros((540, 960, 3), np.uint8), [ball], np.array([100, 400]))
        is not None
    )
