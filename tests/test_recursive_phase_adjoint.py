import unittest

import torch

from src.conversion.activation_sites import activation_sites
from src.conversion.channel_qcfs_source import install_channel_qcfs
from src.conversion.phase_moments import conditional_event_moments
from src.conversion.general_phase_adjoint import phase_partition_mean, phase_partition_vjp
from src.conversion.qcfs_ecl_resnet18 import DualThresholdSequence, ECLResNet18
from src.conversion.recursive_phase_adjoint import (
    RecursivePhaseResNet18, RecursivePhaseSequence, recursive_phase_coefficients,
    recursive_phase_mean, recursive_phase_vjp,
)
from src.models.resnet18_cifar import ResNet18Cifar


SEED = 2026100321


class RecursivePhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_general_mean_and_vjp_match_partition_away_from_ties(self):
        generator = torch.Generator().manual_seed(SEED)
        theta = torch.tensor([.31, .7, 1.2, 1.8], dtype=torch.float64).reshape(1, 4, 1, 1)
        for timesteps in (1, 2, 4, 8):
            currents = torch.randn(timesteps, 3, 4, 2, 2, generator=generator,
                                   dtype=torch.float64) * 2 * theta
            gradient = torch.randn(currents.shape, generator=generator, dtype=currents.dtype)
            torch.testing.assert_close(recursive_phase_mean(currents, theta),
                                       phase_partition_mean(currents, theta, 17),
                                       rtol=1e-12, atol=2e-14)
            torch.testing.assert_close(recursive_phase_vjp(currents, theta, gradient),
                                       phase_partition_vjp(currents, theta, gradient, 17),
                                       rtol=1e-12, atol=2e-14)
            dense = theta.expand_as(currents[0]).clone()
            torch.testing.assert_close(recursive_phase_vjp(currents, dense, gradient),
                                       recursive_phase_vjp(currents, theta, gradient), rtol=0, atol=0)
        scalar = torch.tensor(1., dtype=torch.float64)
        torch.testing.assert_close(recursive_phase_mean(currents, scalar),
                                   phase_partition_mean(currents, scalar, 17), rtol=1e-12, atol=2e-14)

    def test_t2_matches_existing_formula_away_from_ties(self):
        theta = torch.tensor([.7, 1.3], dtype=torch.float64).reshape(1, 2, 1, 1)
        ratios = torch.tensor([[.23, .81, 1.27, .67], [.31, -.71, .33, 1.12]],
                              dtype=torch.float64).reshape(2, 2, 2, 1, 1)
        currents = (ratios * theta).requires_grad_()
        gradient = torch.tensor([[.4, -.9, .2, 1.1], [-.3, .6, 1.2, -.7]],
                                dtype=torch.float64).reshape_as(currents)
        first, second, *_ = conditional_event_moments(currents[0], currents[1], theta)
        means = torch.stack((theta * first, theta * second))
        expected, = torch.autograd.grad(means, currents, gradient)
        torch.testing.assert_close(recursive_phase_mean(currents, theta), means,
                                   rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(recursive_phase_vjp(currents, theta, gradient), expected,
                                   rtol=1e-12, atol=1e-12)

    def test_vjp_matches_central_differences(self):
        generator = torch.Generator().manual_seed(SEED)
        theta = torch.tensor([.7, 1.2, 1.8], dtype=torch.float64)
        for timesteps in (1, 4, 8):
            currents = (torch.rand(timesteps, 3, generator=generator,
                                   dtype=torch.float64) * 3 - 1.4) * theta
            gradient = torch.randn(currents.shape, generator=generator, dtype=torch.float64)
            expected = torch.empty_like(currents)
            step = 1e-6
            for t in range(timesteps):
                for channel in range(3):
                    positive, negative = currents.clone(), currents.clone()
                    positive[t, channel] += step
                    negative[t, channel] -= step
                    delta = recursive_phase_mean(positive, theta) - recursive_phase_mean(negative, theta)
                    expected[t, channel] = (gradient * delta).sum() / (2 * step)
            torch.testing.assert_close(recursive_phase_vjp(currents, theta, gradient),
                                       expected, rtol=1e-7, atol=2e-8)

    def test_smooth_modulo_endpoint_retains_derivative(self):
        # S_4 == theta is a modulo endpoint, but each step remains strictly
        # inside the input-following branch. Locally mu_t == current_t.
        theta = torch.tensor(1., dtype=torch.float64)
        currents = torch.full((4, 1), .25, dtype=torch.float64)
        gradient = torch.tensor([[.4], [-.9], [1.1], [2.]], dtype=torch.float64)
        torch.testing.assert_close(recursive_phase_mean(currents, theta), currents, rtol=0, atol=0)
        actual = recursive_phase_vjp(currents, theta, gradient)
        torch.testing.assert_close(actual, gradient, rtol=0, atol=1e-15)
        generic = phase_partition_vjp(currents, theta, gradient)
        self.assertGreater((generic - actual).abs().max().item(), .1)
        step = 1e-6
        for t in range(4):
            positive, negative = currents.clone(), currents.clone()
            positive[t] += step
            negative[t] -= step
            # The integral itself is smooth, including at the modulo endpoint.
            delta = phase_partition_mean(positive, theta) - phase_partition_mean(negative, theta)
            finite_difference = (gradient * delta).sum() / (2 * step)
            torch.testing.assert_close(finite_difference, gradient[t, 0], rtol=1e-7, atol=1e-8)

    def test_boundary_convention_is_finite_and_explicit(self):
        theta = torch.tensor(1., dtype=torch.float64)
        currents = torch.tensor([[0., 1., 1.1, -1e-3, -2e-3]], dtype=torch.float64)
        a, b = recursive_phase_coefficients(currents, theta)
        torch.testing.assert_close(a, torch.tensor([[True, True, True, True, False]]))
        torch.testing.assert_close(b, torch.zeros_like(b))
        repeated = torch.tensor([[1., -1., .3], [0., 0., -.3],
                                 [-1., 1., .3], [0., 0., -.3]], dtype=torch.float64)
        gradient = torch.arange(12, dtype=repeated.dtype).reshape_as(repeated)
        self.assertTrue(bool(recursive_phase_mean(repeated, theta).isfinite().all()))
        self.assertTrue(bool(recursive_phase_vjp(repeated, theta, gradient).isfinite().all()))
        zeros = torch.zeros_like(repeated)
        torch.testing.assert_close(recursive_phase_vjp(zeros, theta, gradient), zeros, rtol=0, atol=0)

    def test_hard_forward_saved_booleans_and_sequence_vjp(self):
        generator = torch.Generator().manual_seed(SEED)
        for dtype in (torch.float32, torch.float64):
            theta = torch.tensor([.4, 1., 2.], dtype=dtype).reshape(1, 3, 1, 1)
            for timesteps in (1, 2, 4, 8):
                currents = torch.randn(timesteps, 3, 3, 2, 2, generator=generator,
                                       dtype=dtype) * theta * 2
                currents[0, 0] = .5 * theta[0]
                currents[0, 1] = 1.5 * theta[0]
                if timesteps > 1:
                    currents[1, 1] = -3 * theta[0]
                packed = currents.flatten(0, 1).requires_grad_()
                node = RecursivePhaseSequence(theta, timesteps)
                saved = []
                def pack(tensor):
                    saved.append((tensor.dtype, tuple(tensor.shape)))
                    return tensor
                with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                    output = node(packed)
                self.assertEqual(saved, [(torch.bool, tuple(currents.shape))] * 2)
                with torch.no_grad():
                    reference = DualThresholdSequence(theta, timesteps, True)(packed)
                    torch.testing.assert_close(node(packed), reference, rtol=0, atol=0)
                torch.testing.assert_close(output, reference, rtol=0, atol=0)
                gradient = torch.randn(currents.shape, generator=generator, dtype=dtype)
                actual, = torch.autograd.grad(output, packed, gradient.flatten(0, 1))
                expected = recursive_phase_vjp(currents, theta, gradient).flatten(0, 1)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_resnet_state_loading_and_source_gradients(self):
        torch.manual_seed(SEED)
        source = ResNet18Cifar(num_classes=7, base_width=4)
        units = {name: torch.linspace(.5, 1., channels)
                 for name, channels in activation_sites(source)}
        source = install_channel_qcfs(source, units).eval()
        reference = ECLResNet18(source, timesteps=4).eval()
        model = RecursivePhaseResNet18(source, timesteps=4).eval()
        self.assertEqual(set(reference.state_dict()), set(model.state_dict()))
        model.load_state_dict(reference.state_dict(), strict=True)
        reference.load_state_dict(model.state_dict(), strict=True)
        images = torch.randn(2, 3, 16, 16)
        logits = model(images)
        with torch.no_grad():
            torch.testing.assert_close(logits, reference(images), rtol=0, atol=0)
        torch.nn.functional.cross_entropy(logits, torch.tensor([1, 4])).backward()
        self.assertTrue(all(p.requires_grad and p.grad is not None and bool(p.grad.isfinite().all())
                            for p in model.parameters()))
        self.assertGreater(model.source.conv1.weight.grad.abs().sum().item(), 0.)
        self.assertFalse(any('threshold' in name for name, _ in model.named_parameters()))
        self.assertTrue(all(not value.requires_grad for name, value in model.named_buffers()
                            if name.endswith('threshold')))


if __name__ == '__main__':
    unittest.main()
