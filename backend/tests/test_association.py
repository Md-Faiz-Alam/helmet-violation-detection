from ridesafe.association import AssociationRule, associate, offsets, on_vehicle

BIKE = (100, 200, 200, 300)  # 100 x 100 box


def head_at(cx, cy, size=20):
    return (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)


def test_offsets_are_relative_to_vehicle():
    dx, dy, ratio = offsets(head_at(150, 170), BIKE)
    assert (round(dx, 2), round(dy, 2), round(ratio, 2)) == (0.0, -0.3, 0.2)


def test_rider_head_above_bike_is_on_it():
    assert on_vehicle(head_at(150, 170), BIKE)


def test_pedestrian_beside_bike_is_not():
    assert not on_vehicle(head_at(320, 170), BIKE)   # dx = 1.7


def test_head_far_above_or_below_is_not():
    assert not on_vehicle(head_at(150, 0), BIKE)     # dy = -2.0
    assert not on_vehicle(head_at(150, 290), BIKE)   # dy = 0.9 (at wheel height)


def test_huge_head_over_small_bike_is_not():
    assert not on_vehicle(head_at(150, 170, size=150), BIKE)


def test_associate_picks_closest_and_ignores_weak_vehicles():
    near = ((100, 200, 200, 300), 0.9)
    other = ((180, 200, 280, 300), 0.9)
    weak = ((100, 200, 200, 300), 0.1)
    heads = [head_at(150, 170), head_at(600, 170)]
    assert associate(heads, [other, near]) == [1, -1]
    assert associate(heads[:1], [weak], AssociationRule(vehicle_conf=0.25)) == [-1]


def test_rider_and_pillion_share_a_vehicle():
    assert associate([head_at(140, 170), head_at(165, 160)], [(BIKE, 0.9)]) == [0, 0]
