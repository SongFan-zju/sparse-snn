import unittest

import torch

from src.conversion.qcfs_ecl_resnet18 import (
    DualThresholdSequence, ECLResNet18, install_qcfs,
)
from src.conversion.triangular_density_phase import (
    TriangularDensityPhaseEvent, TriangularDensityPhaseResNet18,
    TriangularDensityPhaseSequence, triangular_density_cdf,
    triangular_density_charge_means,
)
from src.models.resnet20_cifar import ResNet20Cifar


class TriangularDensityPhaseTests(unittest.TestCase):
    def test_cdf_and_first_step_match_triangular_charge_slope(self):
        values = torch.tensor([-3., -.5, 0., .5, 1., 1.5, 3.], dtype=torch.float64)
        torch.testing.assert_close(triangular_density_cdf(values),
            torch.tensor([0., 0., .125, .5, .875, 1., 1.], dtype=torch.float64))
        theta = torch.tensor(.7, dtype=torch.float64)
        first = (theta * torch.tensor([-.7, -.25, 0., .23, .5, .81, 1.25, 1.7],
                                     dtype=torch.float64)).requires_grad_()
        mean0, _ = triangular_density_charge_means(first, torch.zeros_like(first), theta)
        slope, = torch.autograd.grad(mean0.sum(), first)
        expected = (1 - (first / theta - .5).abs()).clamp_min(0)
        torch.testing.assert_close(slope, expected, rtol=1e-14, atol=1e-14)

    def test_shared_phase_means_include_reset_and_epsilon_retraction(self):
        theta = torch.tensor(1., dtype=torch.float64)
        first = torch.tensor([.4, .8, 1.2], dtype=torch.float64)
        second = torch.tensor([-.6, -1.2, -.4], dtype=torch.float64)
        means = triangular_density_charge_means(first, second, theta)
        torch.testing.assert_close(means[0], torch.tensor([.405, .755, .955], dtype=torch.float64))
        torch.testing.assert_close(means[1], torch.tensor(
            [-.3596995, -.7498995, -.1543005], dtype=torch.float64))
        # Integrate actual hard event paths over the hypothetical phase prior.
        phase = torch.linspace(-.5 + 1e-5, 1.5 - 1e-5, 100000,
                               dtype=torch.float64)[:, None]
        density = (1 - (phase - .5).abs()).clamp_min(0)
        membrane = phase + first
        event0 = (membrane > theta).to(first)
        membrane = membrane - theta * event0 + second
        event1 = (membrane > theta).to(first) - (
            (membrane <= -1e-3) & (event0 >= 1)).to(first)
        for mean, event in zip(means, (event0, event1)):
            torch.testing.assert_close(mean, (theta * density * event).sum(0) / density.sum(0),
                                       rtol=0, atol=1e-8)

    def test_custom_vjp_matches_charge_mean_finite_difference(self):
        first = torch.tensor([.23, .81, 1.27, .67], dtype=torch.float64, requires_grad=True)
        second = torch.tensor([.31, -.71, .33, 1.12], dtype=torch.float64, requires_grad=True)
        theta = torch.tensor([.7, 1., 1.3, .9], dtype=torch.float64)
        weights = (torch.tensor([.4, -.9, .2, 1.1], dtype=torch.float64),
                   torch.tensor([-.3, .6, 1.2, -.7], dtype=torch.float64))
        outputs = TriangularDensityPhaseEvent.apply(first, second, theta)
        actual = torch.autograd.grad(outputs, (first, second), grad_outputs=weights)

        def objective(a, b):
            means = triangular_density_charge_means(a, b, theta)
            return sum((weight * mean).sum() for weight, mean in zip(weights, means))

        delta = 1e-6
        for arg in range(2):
            expected = torch.zeros_like(first)
            for index in range(len(first)):
                plus = [first.detach().clone(), second.detach().clone()]
                minus = [value.clone() for value in plus]
                plus[arg][index] += delta
                minus[arg][index] -= delta
                expected[index] = (objective(*plus) - objective(*minus)) / (2 * delta)
            torch.testing.assert_close(actual[arg], expected, rtol=1e-7, atol=1e-8)
        # This keeps a negative temporal term but fills the old zero VJP example.
        currents = torch.tensor([.4, -.6], dtype=torch.float64, requires_grad=True)
        jacobian = torch.autograd.functional.jacobian(
            lambda x: torch.stack(triangular_density_charge_means(x[0], x[1], theta[1])),
            currents)
        torch.testing.assert_close(jacobian, torch.tensor(
            [[.9, 0.], [-.599, .301]], dtype=torch.float64), rtol=0, atol=1e-12)

    def test_hard_forward_boundaries_and_no_grad_deployment_match(self):
        theta = torch.tensor(1.)
        first = torch.tensor([.5, .5001, 1.5, 2., -2., .8])
        second = torch.tensor([0., -.0011, -1., -3., 4., -1.2])
        packed = torch.stack((first, second)).reshape(-1, 1).requires_grad_()
        candidate = TriangularDensityPhaseSequence(theta)
        expected = DualThresholdSequence(theta)(packed)
        torch.testing.assert_close(candidate(packed), expected, rtol=0, atol=0)
        with torch.no_grad():
            torch.testing.assert_close(candidate(packed), expected, rtol=0, atol=0)
        self.assertEqual(set(candidate.state_dict()), {'threshold'})

    def test_standard_resnet20_forward_gradients_and_state_compatibility(self):
        torch.set_num_threads(2)
        torch.manual_seed(2026100321)
        source = install_qcfs(ResNet20Cifar(num_classes=100), levels=2).eval()
        for module in source.modules():
            if hasattr(module, 'thresh'):
                module.thresh.data.fill_(1.)
        candidate = TriangularDensityPhaseResNet18(source).eval()
        baseline = ECLResNet18(source).eval()
        self.assertEqual(set(candidate.state_dict()), set(baseline.state_dict()))
        baseline.load_state_dict(candidate.state_dict(), strict=True)
        self.assertEqual(sum(isinstance(node, TriangularDensityPhaseSequence)
                             for node in candidate.modules()), 19)
        images = torch.randn(2, 3, 32, 32)
        outputs = candidate(images)
        with torch.no_grad():
            expected = baseline(images)
            torch.testing.assert_close(candidate(images), expected, rtol=0, atol=0)
        torch.testing.assert_close(outputs, expected, rtol=0, atol=0)
        outputs.square().mean().backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in candidate.parameters()))
        self.assertGreater(candidate.source.conv1.weight.grad.abs().sum().item(), 0.)
        self.assertFalse(any('threshold' in name for name, _ in candidate.named_parameters()))


if __name__ == '__main__':
    unittest.main()
