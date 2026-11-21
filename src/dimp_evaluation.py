"""Small, explicit SOT metrics; annotations never enter the tracking step."""
from __future__ import annotations

import math


def valid_box(box):
    return (box is not None and len(box) == 4
            and all(math.isfinite(float(x)) for x in box)
            and box[2] > 0 and box[3] > 0)


def overlap_and_distance(prediction, target):
    if not valid_box(target):
        raise ValueError('Scored annotations must be finite positive boxes')
    if not valid_box(prediction):
        return 0.0, float('inf')
    x, y, w, h = prediction
    a, b, c, d = target
    area = max(0., min(x+w, a+c)-max(x, a)) * max(0., min(y+h, b+d)-max(y, b))
    union = w*h+c*d-area
    distance = math.hypot(x+w/2-a-c/2, y+h/2-b-d/2)
    return area/union, distance


def sot_metrics(rows):
    scored = [row for row in rows if row['score']]
    if not scored:
        raise ValueError('No scored post-initialization frames')
    n = len(scored)
    return dict(total=n, ao=100*sum(row['iou'] for row in scored)/n,
                sr50=100*sum(row['iou'] >= .5 for row in scored)/n,
                sr75=100*sum(row['iou'] >= .75 for row in scored)/n,
                precision20=100*sum(row['center_error'] <= 20 for row in scored)/n)
