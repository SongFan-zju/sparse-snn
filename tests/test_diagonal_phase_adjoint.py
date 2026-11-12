import unittest

import torch

from src.conversion.activation_sites import activation_sites
from src.conversion.channel_qcfs_source import install_channel_qcfs
from src.conversion.diagonal_phase_adjoint import (
    DiagonalPhaseEvent, DiagonalPhaseResNet18, DiagonalPhaseSequence,
)
from src.conversion.joint_phase_adjoint import JointPhaseEvent, JointPhaseResNet18, JointPhaseSequence
from src.conversion.qcfs_ecl_resnet18 import ECLResNet18
from src.models.resnet18_cifar import ResNet18Cifar


SEED = 2026100321


def vjp(event, first, second, threshold, weights):
    first = first.detach().clone().requires_grad_(True)
    second = second.detach().clone().requires_grad_(True)
    return torch.autograd.grad(event.apply(first, second, threshold),
                               (first, second), grad_outputs=weights)


class DiagonalPhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(SEED)

    def test_hard_forward_including_boundaries_and_retraction(self):
        first = torch.tensor([.5, .5001, 1.5, 2., -2., .8, 1.6], dtype=torch.float64)
        second = torch.tensor([0., -.0011, -1., -3., 4., -.9, -.4], dtype=torch.float64)
        packed = torch.stack((first, second)).reshape(-1, 1).requires_grad_(True)
        threshold = torch.tensor(1., dtype=packed.dtype)
        expected = JointPhaseSequence(threshold)(packed)
        actual = DiagonalPhaseSequence(threshold)(packed)
        self.assertTrue(actual.requires_grad)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with torch.no_grad():
            torch.testing.assert_close(DiagonalPhaseSequence(threshold)(packed), expected, rtol=0, atol=0)

    def test_diagonal_blocks_match_full_phase_at_ordinary_points_and_ties(self):
        threshold = torch.tensor([.7, 1., 1.3], dtype=torch.float64).reshape(1, 3, 1, 1)
        first = torch.tensor([-.2, 0., .23, .5, .81, 1., 1.27, 1.6, 2.], dtype=torch.float64)
        second = torch.tensor([.2, 0., .31, .5, -.71, 0., .33, -.4, -1.], dtype=torch.float64)
        first = first.reshape(1, 1, 3, 3).expand(2, 3, 3, 3) * threshold
        second = second.reshape(1, 1, 3, 3).expand_as(first) * threshold
        # Include a retraction boundary at total=-eps in every channel.
        second[:, :, 0, 0] = -first[:, :, 0, 0] - 1.e-3
        weights = (torch.randn_like(first), torch.randn_like(second))
        zero = torch.zeros_like(first)
        first_only = vjp(JointPhaseEvent, first, second, threshold, (weights[0], zero))
        second_only = vjp(JointPhaseEvent, first, second, threshold, (zero, weights[1]))
        actual = vjp(DiagonalPhaseEvent, first, second, threshold, weights)
        torch.testing.assert_close(actual[0], first_only[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], second_only[1], rtol=0, atol=0)
        self.assertTrue(all(bool(value.isfinite().all()) for value in actual))

    def test_positive_cross_removed_and_retraction_cancellation_removed(self):
        first = torch.tensor([1.6, .4], dtype=torch.float64)
        second = torch.tensor([-.4, -2.], dtype=torch.float64)
        threshold = torch.tensor(1., dtype=torch.float64)
        weights = (torch.full_like(first, 2.), torch.full_like(first, 3.))
        full = vjp(JointPhaseEvent, first, second, threshold, weights)
        diagonal = vjp(DiagonalPhaseEvent, first, second, threshold, weights)
        torch.testing.assert_close(full[0], torch.tensor([3., -1.], dtype=first.dtype), rtol=0, atol=0)
        torch.testing.assert_close(full[1], torch.tensor([3., 0.], dtype=first.dtype), rtol=0, atol=0)
        torch.testing.assert_close(diagonal[0], torch.tensor([0., 2.], dtype=first.dtype), rtol=0, atol=0)
        torch.testing.assert_close(diagonal[1], full[1], rtol=0, atol=0)
        equal_weights = (torch.ones_like(first[1:]), torch.ones_like(first[1:]))
        full_retraction = vjp(JointPhaseEvent, first[1:], second[1:], threshold, equal_weights)
        diagonal_retraction = vjp(DiagonalPhaseEvent, first[1:], second[1:], threshold, equal_weights)
        self.assertEqual(full_retraction[0].item(), 0.)
        self.assertEqual(diagonal_retraction[0].item(), 1.)

    def test_resnet_strict_deployment_loading_and_finite_source_gradients(self):
        source = ResNet18Cifar(num_classes=7, base_width=2)
        units = {name: torch.linspace(.5, 1., channels) for name, channels in activation_sites(source)}
        source = install_channel_qcfs(source, units).eval()
        model = DiagonalPhaseResNet18(source).eval()
        phase = JointPhaseResNet18(source).eval()
        ordinary = ECLResNet18(source).eval()
        self.assertEqual(sum(isinstance(node, DiagonalPhaseSequence) for node in model.modules()), 17)
        self.assertEqual(set(model.state_dict()), set(ordinary.state_dict()))
        ordinary.load_state_dict(model.state_dict(), strict=True)
        model.load_state_dict(phase.state_dict(), strict=True)
        self.assertFalse(any(name.endswith('threshold') for name, _ in model.named_parameters()))
        images = torch.randn(2, 3, 16, 16)
        actual = model(images)
        torch.testing.assert_close(actual, phase(images), rtol=0, atol=0)
        with torch.no_grad():
            torch.testing.assert_close(actual, ordinary(images), rtol=0, atol=0)
        torch.nn.functional.cross_entropy(actual, torch.tensor([1, 4])).backward()
        self.assertTrue(all(p.requires_grad and p.grad is not None and bool(p.grad.isfinite().all())
                            for p in model.parameters()))
        self.assertGreater(model.source.conv1.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(model.source.fc.weight.grad.abs().sum().item(), 0.)


if __name__ == '__main__':
    unittest.main()
