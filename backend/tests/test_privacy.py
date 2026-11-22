"""Blurring for the public sample (ridesafe.privacy): the geometry and the blur itself, without models."""

import numpy as np

import cv2

from ridesafe.privacy import blur_boxes, face_region, neighbours_union, pad_text, plate_shaped


def test_face_region_keeps_the_top_of_the_head():
    x1, y1, x2, y2 = face_region((100, 100, 200, 200))
    assert y1 == 135  # the top 35% (helmet or hair) stays visible
    assert y2 > 200 and x1 < 100 and x2 > 200  # chin and cheeks just outside the head box are covered


def test_text_margin_is_capped():
    assert pad_text((0, 0, 50, 10)) == (-3, -3, 53, 13)  # 30% of the height for a small plate
    assert pad_text((0, 0, 200, 100)) == (-8, -8, 208, 108)  # at most 8 px for a large box


def test_neighbouring_frames_share_their_regions():
    per_frame = [[(0, 0, 1, 1)], [], [], [(5, 5, 6, 6)], []]
    u = neighbours_union(per_frame, 1)
    assert u[1] == [(0, 0, 1, 1)]  # a miss in frame 1 is covered by frame 0
    assert u[2] == [(5, 5, 6, 6)] and u[4] == [(5, 5, 6, 6)]
    assert neighbours_union(per_frame, 0) == per_frame


def text_like(h=40, w=160):
    """Black-on-white stripes: high detail, like the characters on a plate."""
    img = np.full((h, w, 3), 255, np.uint8)
    img[:, ::4] = 0
    return img


def test_blur_removes_detail_only_inside_the_boxes():
    img = np.tile(text_like(), (2, 1, 1))  # 80 x 160
    out = blur_boxes(img.copy(), [(0, 0, 160, 40)])
    blurred, untouched = out[:40].astype(float), out[40:].astype(float)
    assert np.abs(np.diff(blurred, axis=1)).mean() < 0.1 * np.abs(np.diff(untouched, axis=1)).mean()
    assert np.array_equal(out[40:], img[40:])


def test_boxes_outside_or_too_small_are_ignored():
    img = text_like()
    assert np.array_equal(blur_boxes(img.copy(), [(-50, -50, -10, -10), (5, 5, 6, 6), (500, 0, 600, 40)]), img)


def scooter_with_plate(color=(235, 235, 235)):
    """A dark 'vehicle' with a bright two-line plate (characters as dark strokes) and a plain bright patch."""
    img = np.full((400, 300, 3), 40, np.uint8)
    cv2.rectangle(img, (110, 200), (190, 245), color, -1)  # plate, 80 x 45: two-line proportions
    for x in range(118, 185, 12):
        cv2.line(img, (x, 206), (x, 220), (20, 20, 20), 2)
        cv2.line(img, (x, 226), (x + 5, 240), (20, 20, 20), 2)
    cv2.rectangle(img, (20, 40), (70, 70), (240, 240, 240), -1)  # plain bright area: no characters
    return img


def test_plate_shape_finds_plates_but_not_plain_bright_areas():
    [box] = plate_shaped(scooter_with_plate(), (0, 0, 300, 400))
    assert box == (110, 200, 191, 246)
    assert len(plate_shaped(scooter_with_plate(color=(40, 200, 230)), (0, 0, 300, 400))) == 1  # yellow plate
    assert plate_shaped(scooter_with_plate(), (0, 0, 300, 100)) == []  # the plate is outside this vehicle
