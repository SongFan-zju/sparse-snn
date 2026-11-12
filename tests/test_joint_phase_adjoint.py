import unittest

import torch

from src.conversion.phase_moments import conditional_event_moments
from src.conversion.joint_phase_adjoint import JointPhaseEvent, JointPhaseResNet18, JointPhaseSequence
from src.conversion.qcfs_ecl_resnet18 import DualThresholdSequence, ECLResNet18, install_qcfs
from src.models.resnet18_cifar import ResNet18Cifar


class JointPhaseTests(unittest.TestCase):
    def test_hard_events_reset_and_boundaries(self):
        torch.manual_seed(2026100321)
        first = torch.cat((torch.randn(1000), torch.tensor([.5, .5001, 1.5, 2., -2.])))
        second = torch.cat((torch.randn(1000), torch.tensor([0., -.0011, -1., -3., 4.])))
        packed = torch.stack((first, second)).reshape(-1, 1)
        threshold = torch.tensor(1.)
        expected = DualThresholdSequence(threshold)(packed)
        actual = JointPhaseSequence(threshold)(packed.requires_grad_(True))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual[-2].item(), -1.)

    def test_joint_vjp_matches_conditional_mean_finite_difference(self):
        # Stay away from phase-cell intersections where the local derivative
        # is not unique. This tests the stated surrogate, not a hard derivative.
        first = torch.tensor([.23, .81, 1.27, .67], dtype=torch.float64, requires_grad=True)
        second = torch.tensor([.31, -.71, .33, 1.12], dtype=torch.float64, requires_grad=True)
        threshold = torch.tensor(1., dtype=torch.float64)
        weights = (torch.tensor([.4, -.9, .2, 1.1], dtype=torch.float64),
                   torch.tensor([-.3, .6, 1.2, -.7], dtype=torch.float64))
        out = JointPhaseEvent.apply(first, second, threshold)
        actual = torch.autograd.grad(out, (first, second), grad_outputs=weights)
        def objective(a, b):
            m0, m1, *_ = conditional_event_moments(a, b, threshold)
            return (weights[0]*m0 + weights[1]*m1).sum()*threshold
        epsilon = 1e-6
        for arg in range(2):
            expected = torch.zeros_like(first)
            for index in range(len(first)):
                plus = [first.detach().clone(), second.detach().clone()]
                minus = [x.clone() for x in plus]
                plus[arg][index] += epsilon
                minus[arg][index] -= epsilon
                expected[index] = (objective(*plus)-objective(*minus))/(2*epsilon)
            torch.testing.assert_close(actual[arg], expected, rtol=1e-7, atol=1e-8)

    def test_network_forward_and_parameter_gradient(self):
        torch.set_num_threads(2)
        torch.manual_seed(2026100321)
        source = install_qcfs(ResNet18Cifar(num_classes=100), levels=2).eval()
        for node in source.modules():
            if hasattr(node, 'thresh'):
                node.thresh.data.fill_(1.)
        candidate = JointPhaseResNet18(source).eval()
        baseline = ECLResNet18(source).eval()
        images = torch.randn(2, 3, 32, 32)
        actual = candidate(images)
        with torch.no_grad():
            expected = baseline(images)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in candidate.parameters()))
        self.assertGreater(candidate.source.conv1.weight.grad.abs().sum().item(), 0.)
        self.assertFalse(any('threshold' in name for name, _ in candidate.named_parameters()))


if __name__ == '__main__':
    unittest.main()
