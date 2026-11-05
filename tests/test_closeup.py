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
    ball = (np.array([400, 80, 420, 100], float), 0.75, "closeup_full")
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
    d=detector()
    d.camera=SimpleNamespace(update=lambda image:(np.eye(2,3),'ok'))
    net=np.array([100,100,140,140],float)
    outputs=iter([([], [net]),([],[])])
    d.predict=lambda *args:next(outputs)
    d.observe(np.zeros((540,960,3),np.uint8))
    d.observe(np.zeros((540,960,3),np.uint8))
    assert len(d.equipment)==1
    assert np.allclose(d.equipment[0],net)


def test_equipment_memory_moves_with_camera_and_expires():
    d=detector()
    matrix=np.array([[1.,0.,5.],[0.,1.,2.]])
    d.camera=SimpleNamespace(update=lambda image:(matrix,'ok'))
    d.equipment_memory=[(np.array([100,100,140,140],float),0)]
    d.predict=lambda *args:([],[])
    image=np.zeros((540,960,3),np.uint8)
    d.observe(image)
    assert np.allclose(d.equipment[0],[105,102,145,142])
    for _ in range(15):d.observe(image)
    assert d.equipment==[]
