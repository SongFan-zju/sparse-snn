import unittest
from unittest.mock import patch

import torch

from src.conversion.qcfs_ecl_resnet18 import (
    DualThresholdSequence, ECLResNet18, install_qcfs,
)
from src.conversion.triangular_density_horizon import (
    TriangularDensityHorizonResNet18, TriangularDensityHorizonSequence,
    triangle_density_partition_mean, triangle_density_partition_vjp,
)
from src.conversion.triangular_density_phase import triangular_density_charge_means
from src.models.resnet20_cifar import ResNet20Cifar
from tests.triangular_density_partition_reference import (
    triangle_density_partition_mean as cell_reference_mean,
    triangle_density_partition_vjp as cell_reference_vjp,
)


SEED = 2026100321


class TriangularDensityHorizonTests(unittest.TestCase):
    def test_fast_recurrence_matches_retained_phase_cell_reference(self):
        theta = torch.tensor([.7, 1.3, 1.], dtype=torch.float64)
        generator = torch.Generator().manual_seed(SEED)
        for timesteps in (1, 3, 4, 8):
            currents = torch.randn(timesteps, 3, generator=generator, dtype=torch.float64) * theta
            gradient = torch.randn(currents.shape, generator=generator, dtype=torch.float64)
            torch.testing.assert_close(triangle_density_partition_mean(currents, theta),
                                       cell_reference_mean(currents, theta), rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(triangle_density_partition_vjp(currents, theta, gradient),
                                       cell_reference_vjp(currents, theta, gradient), rtol=1e-12, atol=1e-12)
        # Exact ties, retraction, and saturated trajectories retain the same
        # deterministic common-positive-direction Jacobian as the cell engine.
        currents = torch.tensor([[.3, 0., 4.], [0., 0., -5.],
                                 [-.3, 0., 3.], [0., 0., -4.]], dtype=torch.float64)
        gradient = torch.arange(12, dtype=currents.dtype).reshape_as(currents)
        torch.testing.assert_close(triangle_density_partition_mean(currents, theta),
                                   cell_reference_mean(currents, theta), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(triangle_density_partition_vjp(currents, theta, gradient),
                                   cell_reference_vjp(currents, theta, gradient), rtol=1e-12, atol=1e-12)

    def test_t2_closed_mean_and_vjp_away_from_collisions(self):
        theta = torch.tensor([.7, 1.3, 1., .9], dtype=torch.float64)
        currents = (torch.tensor([[.23, .81, 1.27, .67], [.31, -.71, .33, 1.12]],
                                 dtype=torch.float64) * theta).requires_grad_()
        gradient = torch.tensor([[.4, -.9, .2, 1.1], [-.3, .6, 1.2, -.7]],
                                dtype=torch.float64)
        expected = torch.stack(triangular_density_charge_means(
            currents[0], currents[1], theta))
        expected_vjp, = torch.autograd.grad(expected, currents, gradient)
        torch.testing.assert_close(triangle_density_partition_mean(currents, theta),
                                   expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(triangle_density_partition_vjp(currents, theta, gradient),
                                   expected_vjp, rtol=1e-12, atol=1e-12)

    def test_t4_vjp_finite_difference_and_chunk_broadcast(self):
        generator = torch.Generator().manual_seed(SEED)
        theta = torch.tensor([.7, 1.2, 1.8], dtype=torch.float64)
        currents = (torch.rand(4, 2, 3, generator=generator,
                               dtype=torch.float64) * 3 - 1.4) * theta
        gradient = torch.randn(currents.shape, generator=generator, dtype=torch.float64)
        actual = triangle_density_partition_vjp(currents, theta, gradient)
        expected = torch.empty_like(currents)
        step = 1e-6
        for index in range(currents.numel()):
            positive, negative = currents.clone(), currents.clone()
            positive.reshape(-1)[index] += step
            negative.reshape(-1)[index] -= step
            difference = (triangle_density_partition_mean(positive, theta)
                          - triangle_density_partition_mean(negative, theta))
            expected.reshape(-1)[index] = (gradient * difference).sum() / (2 * step)
        torch.testing.assert_close(actual, expected, rtol=1e-7, atol=2e-8)
        means = triangle_density_partition_mean(currents, theta)
        with patch('src.conversion.triangular_density_horizon._CHUNK_NEURONS', 2):
            torch.testing.assert_close(triangle_density_partition_mean(currents, theta),
                                       means, rtol=0, atol=1e-14)
            torch.testing.assert_close(triangle_density_partition_vjp(currents, theta, gradient),
                                       actual, rtol=0, atol=1e-14)
        node_input = currents.clone().requires_grad_()
        output = TriangularDensityHorizonSequence(theta, 4)(node_input.flatten(0, 1))
        vjp, = torch.autograd.grad(output, node_input, gradient.flatten(0, 1))
        torch.testing.assert_close(vjp, actual, rtol=0, atol=0)

    def test_t4_means_match_independent_weighted_hard_trajectories(self):
        theta = torch.tensor([.7, 1.3], dtype=torch.float64)
        currents = torch.tensor([[.4, 1.2], [-.6, -.4], [1.7, -1.8], [-.9, 2.2]],
                                dtype=torch.float64) * theta
        # Independent midpoint quadrature: use direct membrane additions/resets,
        # not prefix boundaries or the partition implementation's charge identity.
        count_points = 200000
        phase = -.5 + (torch.arange(count_points, dtype=torch.float64) + .5) * (2 / count_points)
        density = (1 - (phase - .5).abs()).clamp_min(0)[:, None]
        membrane = phase[:, None] * theta
        count = torch.zeros_like(membrane)
        means = []
        for current in currents:
            membrane = membrane + current
            event = (membrane > theta).to(currents.dtype)
            event = event - ((membrane <= -1e-3) & (count >= 1)).to(currents.dtype)
            membrane = membrane - theta * event
            count = count + event
            means.append((theta * event * density).sum(0) * (2 / count_points))
        torch.testing.assert_close(triangle_density_partition_mean(currents, theta),
                                   torch.stack(means), rtol=0, atol=2e-5)

    def test_hard_forward_and_collision_convention_are_finite(self):
        theta = torch.tensor([.4, 1., 2.], dtype=torch.float64)
        currents = torch.tensor([[.5, 1.5, -.5], [-3., 0., 1.],
                                 [2., -1., 0.], [0., 1., -.5]], dtype=torch.float64) * theta
        packed = currents.requires_grad_()
        candidate = TriangularDensityHorizonSequence(theta, 4)
        reference = DualThresholdSequence(theta, 4, signed=True)
        expected = reference(packed)
        torch.testing.assert_close(candidate(packed), expected, rtol=0, atol=0)
        with torch.no_grad():
            torch.testing.assert_close(candidate(packed), expected, rtol=0, atol=0)
        candidate(packed).sum().backward()
        self.assertTrue(bool(packed.grad.isfinite().all()))
        zero = torch.zeros(4, 3, dtype=torch.float64)
        self.assertTrue(bool(triangle_density_partition_mean(zero, theta).isfinite().all()))
        gradient = torch.arange(12, dtype=zero.dtype).reshape_as(zero)
        # Uniform positive-current perturbation selects a diagonal .5 limit,
        # preserving the first output's unique nonzero derivative at zero input.
        torch.testing.assert_close(triangle_density_partition_vjp(zero, theta, gradient),
                                   .5 * gradient, rtol=0, atol=1e-12)
        repeated = torch.tensor([[.3, 0., .5], [0., 0., -.5],
                                 [-.3, 0., .5], [0., 0., -.5]], dtype=torch.float64)
        tied_vjp = triangle_density_partition_vjp(repeated, theta, gradient)
        one_sided_vjp = triangle_density_partition_vjp(repeated + 1e-7, theta, gradient)
        torch.testing.assert_close(tied_vjp, one_sided_vjp, rtol=0, atol=1e-5)

    def test_standard_r20_hard_logits_gradients_and_state_compatibility(self):
        torch.set_num_threads(2)
        torch.manual_seed(SEED)
        source = install_qcfs(ResNet20Cifar(num_classes=100), levels=2).eval()
        for node in source.modules():
            if hasattr(node, 'thresh'):
                node.thresh.data.fill_(1.)
        model = TriangularDensityHorizonResNet18(source, 4).eval()
        reference = ECLResNet18(source, 4).eval()
        self.assertEqual(set(model.state_dict()), set(reference.state_dict()))
        model.load_state_dict(reference.state_dict(), strict=True)
        reference.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(sum(isinstance(node, TriangularDensityHorizonSequence)
                             for node in model.modules()), 19)
        images = torch.randn(2, 3, 32, 32)
        logits = model(images)
        with torch.no_grad():
            expected = reference(images)
            torch.testing.assert_close(model(images), expected, rtol=0, atol=0)
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)
        torch.nn.functional.cross_entropy(logits, torch.tensor([1, 4])).backward()
        self.assertTrue(all(p.grad is not None and bool(p.grad.isfinite().all())
                            for p in model.parameters()))
        self.assertGreater(model.source.conv1.weight.grad.abs().sum().item(), 0.)
        self.assertFalse(any('threshold' in name for name, _ in model.named_parameters()))


if __name__ == '__main__':
    unittest.main()
