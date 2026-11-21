"""Differentiable ViStream/UniTrack reconstruction-dot association.

Adapted from the author core/association/matching.py reconsdot_distance.
The author MIT notice is retained in third_party/vistream_LICENSE.txt.
It performs no backbone inference, matching, state update, or GT lookup.
"""
import torch
from torch.nn import functional as F


def reconstruction_similarity(track_roi, detection_roi, temperature=100):
    """Return [tracks,detections], exactly one minus author reconsdot cost."""
    track = F.normalize(track_roi.flatten(2), dim=1)
    detection = F.normalize(detection_roi.flatten(2), dim=1)
    nt, channels, st = track.shape
    nd, _, sd = detection.shape
    if nt == 0 or nd == 0:
        return track_roi.new_empty((nt, nd))
    ft = track.permute(0, 2, 1).reshape(-1, channels)
    fd = detection.permute(0, 2, 1).reshape(-1, channels)
    affinity = ft @ fd.T
    td = F.softmax(temperature * affinity, dim=1)
    dt = F.softmax(temperature * affinity, dim=0).T
    rt = torch.einsum('tds,dsm->tdm', td.reshape(nt*st, nd, sd), fd.reshape(nd, sd, channels))
    rd = torch.einsum('dts,tsm->dtm', dt.reshape(nd*sd, nt, st), ft.reshape(nt, st, channels))
    rt = F.normalize(rt.permute(0, 2, 1).reshape(nt, st*channels, nd), dim=1)
    rd = F.normalize(rd.permute(0, 2, 1).reshape(nd, sd*channels, nt), dim=1)
    dot_td = torch.einsum('tad,ta->td', rt, F.normalize(ft.reshape(nt, st*channels), dim=1))
    dot_dt = torch.einsum('dat,da->dt', rd, F.normalize(fd.reshape(nd, sd*channels), dim=1))
    return .5 * (dot_td + dot_dt.T)
