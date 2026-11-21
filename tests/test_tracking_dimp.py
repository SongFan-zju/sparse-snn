import copy
from collections import OrderedDict
import unittest

import torch
from torch import nn

from src.tracking_dimp import (DimpFeatureBackbone, _CumulativeMaxPool,
                               calibrate_dimp_thresholds, install_dimp_qcfs,
                               split_dimp18_activations)


class _NativeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 2, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(2, 2, 3, padding=1, bias=False)
        self.bn1, self.bn2 = nn.BatchNorm2d(2), nn.BatchNorm2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample, self.stride, self.use_bn = None, 1, True

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(out)) + x)


class _NativeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 2, 3, padding=1, bias=False)
        self.bn1, self.relu = nn.BatchNorm2d(2), nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1, self.layer2, self.layer3 = [nn.Sequential(_NativeBlock()) for _ in range(3)]

    def forward(self, x, output_layers):
        results = OrderedDict()
        x = self.relu(self.bn1(self.conv1(x)))
        if 'conv1' in output_layers:
            results['conv1'] = x
        x = self.maxpool(x)
        for name in ('layer1', 'layer2', 'layer3'):
            x = getattr(self, name)(x)
            if name in output_layers:
                results[name] = x
        return results


class DimpConversionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(2026100321)
        self.native = _NativeBackbone().eval()
        self.images = torch.randn(2, 3, 8, 8)

    def source(self):
        split = split_dimp18_activations(self.native)
        calibration = calibrate_dimp_thresholds(split, [self.images])
        return install_dimp_qcfs(split, 4, calibration['thresholds'])

    def test_split_retains_ann_and_separates_each_activation_call(self):
        split = split_dimp18_activations(self.native)
        expected = self.native(self.images, ['layer2', 'layer3'])
        actual = split(self.images, ['layer2', 'layer3'])
        for name in expected:
            self.assertTrue(torch.equal(expected[name], actual[name]))
        self.assertEqual(set(self.native.state_dict()), set(split.state_dict()))
        self.assertIsNot(split.layer1[0].relu1, split.layer1[0].relu2)

    def test_pool_conserves_decoded_max_and_keeps_images_independent(self):
        ticks = torch.randn(4, 2, 3, 5, 5)
        pool = nn.MaxPool2d(3, stride=2, padding=1)
        temporal = _CumulativeMaxPool(copy.deepcopy(pool), 4)
        result = temporal(ticks.flatten(0, 1)).unflatten(0, (4, 2))
        self.assertTrue(torch.allclose(result.sum(0), pool(ticks.sum(0)), atol=1e-6))
        self.assertTrue(torch.equal(result, temporal(ticks.flatten(0, 1)).unflatten(0, (4, 2))))

    def test_methods_share_hard_features_and_do_not_leak_template_state(self):
        source = self.source()
        models = [DimpFeatureBackbone(source, 4, method) for method in ('hard', 'triangular', 'density')]
        reference = models[0](self.images)
        for model in models[1:]:
            output = model(self.images)
            for name in reference:
                self.assertTrue(torch.equal(reference[name], output[name]))
                self.assertTrue(torch.allclose(output[name][:1], model(self.images[:1])[name], atol=1e-6))
            model(self.images * -1)
            repeated = model(self.images)
            self.assertTrue(torch.equal(output['layer3'], repeated['layer3']))

    def test_both_backward_paths_update_active_weights_and_freeze_buffers(self):
        source = self.source()
        for method in ('triangular', 'density'):
            model = DimpFeatureBackbone(source, 4, method).train()
            buffers = {name: value.clone() for name, value in model.named_buffers()}
            output = model(self.images)
            sum(value.square().mean() for value in output.values()).backward()
            grad = model.source.conv1.weight.grad
            self.assertTrue(bool(torch.isfinite(grad).all()))
            self.assertGreater(float(grad.abs().sum()), 0)
            with torch.no_grad():
                model.source.conv1.weight.add_(grad, alpha=-.001)
            for name, value in model.named_buffers():
                self.assertTrue(torch.equal(buffers[name], value), name)
            self.assertTrue(all(not node.training for node in model.modules() if isinstance(node, nn.BatchNorm2d)))


if __name__ == '__main__':
    unittest.main()
