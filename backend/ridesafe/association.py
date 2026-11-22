"""Head-to-vehicle association: is this head on a bicycle or motorcycle?

The helmet dataset only labels riders, so the helmet model has no idea who is *riding*: pedestrians
and bystanders without a helmet become "violations" (the biggest false-alarm source in
reports/error_analysis.md). A COCO-pretrained detector finds the bicycles and motorcycles,
and a head only counts if it sits where a rider's head would be on one of them.

Geometry, in units of the vehicle box (measured on the train split, see scripts/eval_association.py):
    dx = (head centre x - vehicle centre x) / vehicle width      riders: median 0.0
    dy = (head centre y - vehicle top y)    / vehicle height     riders: median -0.3 (just above the top)
A head is "on" a vehicle if |dx| <= max_dx, min_dy <= dy <= max_dy, and the head is not wider than
max_head_ratio x the vehicle (a big close-up head over a small far-away bike is not its rider).
"""

from dataclasses import dataclass

VEHICLE_CLASSES = ("bicycle", "motorcycle")


@dataclass(frozen=True)
class AssociationRule:
    max_dx: float = 0.75
    min_dy: float = -1.5
    max_dy: float = 0.5
    max_head_ratio: float = 1.2
    vehicle_conf: float = 0.25


def offsets(head, vehicle):
    """(dx, dy, head_ratio) of a head box relative to a vehicle box (both xyxy)."""
    hx1, hy1, hx2, hy2 = head
    vx1, vy1, vx2, vy2 = vehicle
    vw, vh = max(vx2 - vx1, 1e-6), max(vy2 - vy1, 1e-6)
    dx = ((hx1 + hx2) / 2 - (vx1 + vx2) / 2) / vw
    dy = ((hy1 + hy2) / 2 - vy1) / vh
    return dx, dy, (hx2 - hx1) / vw


def on_vehicle(head, vehicle, rule=AssociationRule()):
    dx, dy, ratio = offsets(head, vehicle)
    return abs(dx) <= rule.max_dx and rule.min_dy <= dy <= rule.max_dy and ratio <= rule.max_head_ratio


def associate(heads, vehicles, rule=AssociationRule()):
    """For each head, the index of the vehicle it rides (closest match) or -1.

    heads: list of xyxy boxes. vehicles: list of (xyxy box, confidence); vehicles below
    rule.vehicle_conf are ignored. Several heads may share one vehicle (rider + pillion).
    """
    out = []
    for head in heads:
        best, best_d = -1, None
        for j, (vbox, vconf) in enumerate(vehicles):
            if vconf < rule.vehicle_conf or not on_vehicle(head, vbox, rule):
                continue
            dx, dy, _ = offsets(head, vbox)
            d = abs(dx) + abs(dy - (rule.min_dy + rule.max_dy) / 2)
            if best_d is None or d < best_d:
                best, best_d = j, d
        out.append(best)
    return out
