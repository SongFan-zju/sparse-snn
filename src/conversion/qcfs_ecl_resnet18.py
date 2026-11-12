"""QCFS author activation training and paper-defined ECL conversion on ResNet18.

ECL's optimizer/data details are not recovered from author code. The surrounding
training recipe is local and explicitly recorded by the experiment script.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path

import torch
from torch import nn

from src.conversion.activation_sites import activation_sites


def _author_qcfs():
    path = Path(__file__).resolve().parents[1] / "third_party/qcfs_layer.py"
    ns = dict(torch=torch, nn=nn, __name__="qcfs_pinned_author")
    nodes = ast.parse(path.read_text()).body
    wanted = {"MergeTemporalDim", "ExpandTemporalDim", "ZIF", "GradFloor", "IF"}
    selected = [n for n in nodes if isinstance(n, ast.ClassDef) and n.name in wanted]
    assert {n.name for n in selected} == wanted
    for node in selected:
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
        if node.name == "GradFloor":
            ns["myfloor"] = ns["GradFloor"].apply
    return ns["IF"]


AuthorQCFS = _author_qcfs()


def install_qcfs(source, levels=2):
    for name, _ in list(activation_sites(source)):
        parent, _, child = name.rpartition(".")
        setattr(source.get_submodule(parent), child,
                AuthorQCFS(T=0, L=levels, thresh=8.).to(source.conv1.weight))
    return source


class DualThresholdSequence(nn.Module):
    """ECL Eq4/5: strict positive event, negative event after a positive count."""
    def __init__(self, threshold, timesteps=2, signed=True):
        super().__init__()
        if not bool(torch.isfinite(threshold).all() and (threshold > 0).all()):
            raise ValueError("Converted thresholds must be finite and positive")
        self.register_buffer("threshold", threshold.detach().clone())
        self.timesteps, self.signed = timesteps, signed

    def forward(self, x):
        values = x.unflatten(0, (self.timesteps, -1))
        membrane = .5 * self.threshold
        count = torch.zeros_like(values[0])
        result = []
        for current in values:
            membrane = membrane + current
            # Unsigned reference preserves original QCFS >= convention.
            event = (membrane > self.threshold if self.signed else
                     membrane >= self.threshold).to(current.dtype)
            if self.signed:
                event = event - ((membrane <= -1e-3) & (count >= 1)).to(current.dtype)
            output = event * self.threshold
            membrane = membrane - output
            count = count + event
            result.append(output)
        return torch.stack(result).flatten(0, 1)


class ECLResNet18(nn.Module):
    def __init__(self, source, timesteps=2, signed=True):
        super().__init__()
        self.source = copy.deepcopy(source).eval().requires_grad_(False)
        self.timesteps = timesteps
        for name, _ in list(activation_sites(source)):
            previous = self.source.get_submodule(name)
            if not isinstance(previous, AuthorQCFS) or previous.T != 0:
                raise ValueError("Conversion requires a QCFS ANN source")
            parent, _, child = name.rpartition(".")
            setattr(self.source.get_submodule(parent), child,
                    DualThresholdSequence(previous.thresh, timesteps, signed))
        self.eval().requires_grad_(False)

    def forward(self, images):
        net = self.source
        x = images.unsqueeze(0).expand(self.timesteps, *images.shape).flatten(0, 1)
        x = net.relu(net.bn1(net.conv1(x)))
        x = net.layer4(net.layer3(net.layer2(net.layer1(x))))
        features = net.avgpool(x).flatten(1).unflatten(0, (self.timesteps, len(images)))
        return net.fc(features.mean(0))
