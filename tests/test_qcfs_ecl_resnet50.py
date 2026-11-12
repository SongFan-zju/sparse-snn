import unittest

import torch
from torch import nn

from src.conversion.joint_phase_adjoint import JointPhaseSequence
from src.conversion.qcfs_ecl_resnet18 import AuthorQCFS, DualThresholdSequence
from src.conversion.qcfs_ecl_resnet50 import (
    ECLResNet50, JointPhaseResNet50, activation_sites, install_channel_qcfs,
    project_thresholds, threshold_parameters,
)
from src.models.resnet50_ann import ResNet50Cifar


SEED = 2026100321


class ResNet50ConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(SEED)

    def source(self):
        source = ResNet50Cifar(num_classes=7, base_width=2, zero_init_residual=False)
        units = {name: torch.linspace(.5, 1., channels)
                 for name, channels in activation_sites(source)}
        return install_channel_qcfs(source, units).eval()

    def test_all49_sites_and_qcfs_trainable_thresholds(self):
        source = self.source()
        sites = list(activation_sites(source))
        self.assertEqual(len(sites), 49)
        self.assertEqual(sum(name.endswith('relu3') for name, _ in sites), 16)
        self.assertEqual(len(threshold_parameters(source)), 49)
        for name, channels in sites:
            node = source.get_submodule(name)
            self.assertIsInstance(node, AuthorQCFS)
            self.assertEqual((node.T, node.L), (0, 2))
            self.assertEqual(tuple(node.thresh.shape), (1, channels, 1, 1))
            self.assertIsInstance(node.thresh, nn.Parameter)
            self.assertTrue(node.thresh.requires_grad)
        logits = source(torch.randn(2, 3, 16, 16))
        nn.functional.cross_entropy(logits, torch.tensor([1, 4])).backward()
        thresholds = threshold_parameters(source)
        self.assertTrue(all(p.grad is not None and bool(p.grad.isfinite().all()) for p in thresholds))
        self.assertGreater(sum(p.grad.abs().sum().item() for p in thresholds), 0.)
        with torch.no_grad():
            thresholds[0].flatten()[0] = -.1
        project_thresholds(source)
        self.assertGreaterEqual(thresholds[0].min().item(), .999e-6)

    def test_ordinary_phase_hard_forward_state_and_source_gradients(self):
        source = self.source()
        ordinary = ECLResNet50(source).eval()
        phase = JointPhaseResNet50(source).eval()
        self.assertEqual(len([m for m in ordinary.modules() if isinstance(m, DualThresholdSequence)]), 49)
        self.assertEqual(len([m for m in phase.modules() if isinstance(m, JointPhaseSequence)]), 49)
        for name, _ in activation_sites(source):
            self.assertIsInstance(phase.source.get_submodule(name), JointPhaseSequence)
            self.assertIsInstance(source.get_submodule(name), AuthorQCFS)
        self.assertEqual(set(ordinary.state_dict()), set(phase.state_dict()))
        ordinary.load_state_dict(phase.state_dict(), strict=True)
        phase.load_state_dict(ordinary.state_dict(), strict=True)
        self.assertFalse(any(name.endswith('threshold') for name, _ in phase.named_parameters()))
        self.assertEqual(sum(name.endswith('threshold') for name, _ in phase.named_buffers()), 49)
        images = torch.randn(2, 3, 16, 16)
        expected_events, actual_events = {}, {}
        handles = []
        for name, _ in activation_sites(source):
            handles.append(ordinary.source.get_submodule(name).register_forward_hook(
                lambda module, inputs, output, key=name: expected_events.__setitem__(key, output.detach().clone())))
            handles.append(phase.source.get_submodule(name).register_forward_hook(
                lambda module, inputs, output, key=name: actual_events.__setitem__(key, output.detach().clone())))
        try:
            with torch.no_grad():
                expected = ordinary(images)
                torch.testing.assert_close(phase(images), expected, rtol=0, atol=0)
            actual = phase(images)  # Exercise the custom autograd hard path.
            self.assertTrue(actual.requires_grad)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(set(expected_events), set(actual_events))
            for name in expected_events:
                torch.testing.assert_close(actual_events[name], expected_events[name], rtol=0, atol=0)
            nn.functional.cross_entropy(actual, torch.tensor([1, 4])).backward()
        finally:
            for handle in handles:
                handle.remove()
        self.assertTrue(all(p.requires_grad and p.grad is not None and bool(p.grad.isfinite().all())
                            for p in phase.parameters()))
        for parameter in (phase.source.conv1.weight, phase.source.bn1.weight,
                          phase.source.layer4[-1].conv3.weight,
                          phase.source.layer4[-1].bn3.weight, phase.source.fc.weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0.)

    def test_calibration_channel_count_validation(self):
        source = ResNet50Cifar(num_classes=7, base_width=2)
        units = {name: torch.ones(channels) for name, channels in activation_sites(source)}
        units['layer1.0.relu3'] = torch.ones(2)  # Expanded Bottleneck has8 channels.
        with self.assertRaisesRegex(ValueError, 'requires 8 channel'):
            install_channel_qcfs(source, units)


if __name__ == '__main__':
    unittest.main()
