"""Prepare channelwise QCFS-L2 from an existing ReLU ResNet.

This reuses the author's QCFS forward/backward unchanged, with a positive
threshold per channel instead of per layer. Warm starts and channelwise
quantization are established techniques, not the proposed paper contribution.
Cached native-n1 units equal twice the ANN quantization step; they are already
the required QCFS clipping threshold and the subsequent ECL event magnitude.
"""
from __future__ import annotations

import torch
from torch import nn

from src.conversion.activation_sites import activation_sites
from src.conversion.qcfs_ecl_resnet18 import AuthorQCFS

MIN_THRESHOLD = 1.e-6


def install_channel_qcfs(source, units):
    names = [name for name, _ in activation_sites(source)]
    if set(names) != set(units):
        raise ValueError('Calibration must supply each activation exactly once')
    for name in names:
        threshold = units[name].detach().clone().reshape(1,-1,1,1)
        if not bool(torch.isfinite(threshold).all() and (threshold > 0).all()):
            raise ValueError('QCFS thresholds must be finite and positive')
        node = AuthorQCFS(T=0,L=2,thresh=1.).to(source.conv1.weight)
        node.thresh = nn.Parameter(threshold.to(source.conv1.weight).clamp_min(MIN_THRESHOLD))
        parent,_,child = name.rpartition('.')
        setattr(source.get_submodule(parent),child,node)
    return source


def threshold_parameters(model):
    return [node.thresh for node in model.modules() if isinstance(node,AuthorQCFS)]


@torch.no_grad()
def project_thresholds(model):
    for threshold in threshold_parameters(model):
        threshold.clamp_min_(MIN_THRESHOLD)
