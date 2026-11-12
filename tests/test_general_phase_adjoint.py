import unittest

import torch

from src.conversion.activation_sites import activation_sites
from src.conversion.channel_qcfs_source import install_channel_qcfs
from src.conversion.phase_moments import conditional_event_moments
from src.conversion.general_phase_adjoint import (
    PhasePartitionResNet18, PhasePartitionSequence,
    TriangularHorizonResNet18, TriangularHorizonSequence,
    hard_sequence, phase_partition_mean, phase_partition_vjp,
)
from src.conversion.qcfs_ecl_resnet18 import DualThresholdSequence, ECLResNet18
from src.models.resnet18_cifar import ResNet18Cifar


SEED = 2026100321


class GeneralPhaseTests(unittest.TestCase):
    def test_hard_forward_is_exact_for_all_horizons(self):
        for dtype in (torch.float32, torch.float64):
            generator = torch.Generator().manual_seed(SEED)
            threshold = torch.tensor([.4, 1., 2.], dtype=dtype).reshape(1, 3, 1, 1)
            for timesteps in (1, 2, 4, 8):
                currents = torch.randn(timesteps, 5, 3, 2, 2, dtype=dtype,
                                       generator=generator) * threshold * 2
                currents[0, 0] = .5 * threshold[0]  # exact positive boundary
                currents[0, 1] = 1.5 * threshold[0]
                if timesteps > 1:
                    currents[1, 1] = -3 * threshold[0]  # eligible retraction
                packed = currents.flatten(0, 1).requires_grad_()
                reference = DualThresholdSequence(threshold, timesteps, signed=True)
                with torch.no_grad():
                    expected = reference(packed)
                torch.testing.assert_close(hard_sequence(currents, threshold).flatten(0, 1),
                                           expected, rtol=0, atol=0)
                for node in (PhasePartitionSequence(threshold, timesteps, 7),
                             TriangularHorizonSequence(threshold, timesteps)):
                    torch.testing.assert_close(node(packed), expected, rtol=0, atol=0)
                    with torch.no_grad():
                        torch.testing.assert_close(node(packed), expected, rtol=0, atol=0)

    def test_t2_matches_existing_mean_and_vjp_away_from_collisions(self):
        threshold = torch.tensor([.7, 1.3], dtype=torch.float64).reshape(1, 2, 1, 1)
        ratios = torch.tensor([[.23, .81, 1.27, .67], [.31, -.71, .33, 1.12]],
                              dtype=torch.float64).reshape(2, 2, 2, 1, 1)
        currents = (ratios * threshold).requires_grad_()
        gradient = torch.tensor([[.4, -.9, .2, 1.1], [-.3, .6, 1.2, -.7]],
                                dtype=torch.float64).reshape_as(currents)
        first, second, *_ = conditional_event_moments(currents[0], currents[1], threshold)
        expected_mean = torch.stack((threshold * first, threshold * second))
        expected_vjp, = torch.autograd.grad(expected_mean, currents, gradient)
        torch.testing.assert_close(phase_partition_mean(currents, threshold, 3),
                                   expected_mean, rtol=1e-12, atol=1e-12)
        expected = phase_partition_vjp(currents, threshold, gradient, 3)
        torch.testing.assert_close(expected, expected_vjp, rtol=1e-12, atol=1e-12)
        output = PhasePartitionSequence(threshold, 2, 3)(currents.flatten(0, 1))
        actual, = torch.autograd.grad(output, currents, gradient.flatten(0, 1))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_general_vjp_matches_central_difference(self):
        generator = torch.Generator().manual_seed(SEED)
        threshold = torch.tensor([.7, 1.2, 1.8], dtype=torch.float64)
        for timesteps in (1, 4, 8):
            currents = (torch.rand(timesteps, 3, generator=generator,
                                   dtype=torch.float64) * 3 - 1.4) * threshold
            gradient = torch.randn(currents.shape, generator=generator, dtype=torch.float64)
            actual = phase_partition_vjp(currents, threshold, gradient, 2)
            # Fixed random cases stay away from collisions; h is below the
            # epsilon-separated positive/negative boundaries by three orders.
            step = 1e-6
            expected = torch.empty_like(currents)
            for t in range(timesteps):
                for channel in range(3):
                    positive, negative = currents.clone(), currents.clone()
                    positive[t, channel] += step
                    negative[t, channel] -= step
                    delta = (phase_partition_mean(positive, threshold, 2)
                             - phase_partition_mean(negative, threshold, 2))
                    expected[t, channel] = (gradient * delta).sum() / (2 * step)
            torch.testing.assert_close(actual, expected, rtol=1e-7, atol=2e-8)

    def test_collision_and_endpoint_convention_is_finite(self):
        threshold = torch.tensor(1., dtype=torch.float64)
        # Every positive boundary is an endpoint, and all negative boundaries
        # coincide. The declared zero-jump convention yields a zero VJP.
        currents = torch.zeros(4, 3, dtype=torch.float64, requires_grad=True)
        gradient = torch.arange(12, dtype=currents.dtype).reshape_as(currents)
        actual = phase_partition_vjp(currents, threshold, gradient, 2)
        torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)
        PhasePartitionSequence(threshold, 4, 2)(currents.reshape(4, 3)).sum().backward()
        self.assertTrue(bool(currents.grad.isfinite().all()))
        repeated = torch.tensor([[1., -1., .3], [0., 0., -.3],
                                 [-1., 1., .3], [0., 0., -.3]], dtype=torch.float64)
        self.assertTrue(bool(phase_partition_mean(repeated, threshold, 2).isfinite().all()))
        self.assertTrue(bool(phase_partition_vjp(repeated, threshold, gradient, 2).isfinite().all()))

    def test_chunking_and_channel_threshold_broadcast(self):
        generator = torch.Generator().manual_seed(SEED)
        currents = torch.randn(4, 2, 3, 2, 2, generator=generator, dtype=torch.float64)
        gradient = torch.randn(currents.shape, generator=generator, dtype=torch.float64)
        threshold = torch.tensor([.4, 1., 2.], dtype=torch.float64).reshape(1, 3, 1, 1)
        dense_threshold = threshold.expand_as(currents[0]).clone()
        expected_mean = phase_partition_mean(currents, dense_threshold)
        expected_vjp = phase_partition_vjp(currents, dense_threshold, gradient)
        for chunk in (1, 7):
            torch.testing.assert_close(phase_partition_mean(currents, threshold, chunk),
                                       expected_mean, rtol=0, atol=1e-14)
            torch.testing.assert_close(phase_partition_vjp(currents, threshold, gradient, chunk),
                                       expected_vjp, rtol=0, atol=1e-14)

    def test_resnet_state_loading_and_source_gradients(self):
        torch.set_num_threads(2)
        torch.manual_seed(SEED)
        source = ResNet18Cifar(num_classes=7, base_width=4)
        units = {name: torch.linspace(.5, 1., channels)
                 for name, channels in activation_sites(source)}
        source = install_channel_qcfs(source, units).eval()
        reference = ECLResNet18(source, timesteps=4).eval()
        images = torch.randn(2, 3, 16, 16)
        with torch.no_grad():
            expected = reference(images)
        for model in (PhasePartitionResNet18(source, 4, 4096).eval(),
                      TriangularHorizonResNet18(source, 4).eval()):
            self.assertEqual(set(reference.state_dict()), set(model.state_dict()))
            model.load_state_dict(reference.state_dict(), strict=True)
            reference.load_state_dict(model.state_dict(), strict=True)
            logits = model(images)
            torch.testing.assert_close(logits, expected, rtol=0, atol=0)
            torch.nn.functional.cross_entropy(logits, torch.tensor([1, 4])).backward()
            self.assertTrue(all(p.requires_grad and p.grad is not None
                                and bool(p.grad.isfinite().all()) for p in model.parameters()))
            self.assertGreater(model.source.conv1.weight.grad.abs().sum().item(), 0.)
            self.assertFalse(any('threshold' in name for name, _ in model.named_parameters()))
            self.assertTrue(all(not tensor.requires_grad for name, tensor in model.named_buffers()
                                if name.endswith('threshold')))


if __name__ == '__main__':
    unittest.main()
