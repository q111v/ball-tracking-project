from balltrack.scene import scene_from_people


def test_large_single_person_selects_closeup():
    assert scene_from_people([[300, 140, 480, 500]], 540) == "closeup"


def test_many_small_players_select_broadcast():
    boxes = [[i * 100, 300, i * 100 + 50, 430] for i in range(6)]
    assert scene_from_people(boxes, 720) == "broadcast"


def test_no_people_defaults_to_conservative_broadcast():
    assert scene_from_people([], 720) == "broadcast"
