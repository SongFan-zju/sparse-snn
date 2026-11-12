import unittest

import torch

from src.conversion.activation_sites import activation_sites
from src.conversion.channel_qcfs_source import install_channel_qcfs
from src.conversion.ecl_triangular_surrogate import (
    _TriangularEvent, TriangularResNet18, TriangularSequence,
)
from src.conversion.qcfs_ecl_resnet18 import DualThresholdSequence, ECLResNet18
from src.models.resnet18_cifar import ResNet18Cifar


class TriangularECLTests(unittest.TestCase):
    def test_hard_forward_boundaries_retraction_and_reset(self):
        for dtype in (torch.float32, torch.float64):
            generator = torch.Generator().manual_seed(2026100321)
            threshold = torch.tensor([.4, 1., 2.], dtype=dtype).reshape(1, 3, 1, 1)
            random = torch.randn(2, 24, 3, 1, 1, generator=generator,
                                 dtype=dtype) * threshold * 2
            # Exact positive threshold, near-threshold inputs, and inhibition
            # with/without a preceding positive event.
            first = torch.tensor([.5, .5001, 1.25, -2., 3., -.5],
                                 dtype=dtype).reshape(-1, 1, 1, 1) * threshold
            second = torch.tensor([0., -2., -3., -1., 4., .8],
                                  dtype=dtype).reshape(-1, 1, 1, 1) * threshold
            values = torch.cat((random, torch.stack((first, second))), dim=1)
            packed = values.flatten(0, 1).requires_grad_()
            node = TriangularSequence(threshold)
            reference = DualThresholdSequence(threshold, timesteps=2, signed=True)
            with torch.no_grad():
                expected = reference(packed)
            actual = node(packed)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertTrue(bool((actual < 0).any()))
            explicit = actual.unflatten(0, (2, -1))[:, 24:]
            torch.testing.assert_close(explicit[0, 0], torch.zeros_like(threshold[0]),
                                       rtol=0, atol=0)
            torch.testing.assert_close(explicit[1, 2], -threshold[0], rtol=0, atol=0)
            torch.testing.assert_close(explicit[1, 3], torch.zeros_like(threshold[0]),
                                       rtol=0, atol=0)
            # The module has no persistent membrane between calls.
            torch.testing.assert_close(node(packed), actual, rtol=0, atol=0)
            with torch.no_grad():
                torch.testing.assert_close(node(packed), expected, rtol=0, atol=0)

    def test_normalized_triangular_kernel_and_strict_positive_boundary(self):
        threshold = torch.tensor(2., dtype=torch.float64)
        membrane = torch.tensor([-2., 0., 1., 2., 3., 4., 6.],
                                dtype=torch.float64, requires_grad=True)
        event = _TriangularEvent.apply(membrane, threshold,
                                       torch.zeros_like(membrane, dtype=torch.bool))
        torch.testing.assert_close(event, membrane.new_tensor([0, 0, 0, 0, 1, 1, 1]),
                                   rtol=0, atol=0)
        event.sum().backward()
        # Event VJP includes 1/theta. Multiplication by event charge theta
        # outside the Function cancels it for the emitted-charge derivative.
        expected = membrane.new_tensor([0., 0., .25, .5, .25, 0., 0.])
        torch.testing.assert_close(membrane.grad, expected, rtol=0, atol=0)

    def test_negative_boundary_sign_and_hard_eligibility(self):
        threshold = torch.tensor(2., dtype=torch.float64)
        boundary = torch.tensor(-1e-3, dtype=torch.float64)
        above = torch.nextafter(boundary, torch.tensor(float('inf'), dtype=boundary.dtype))
        positions = torch.stack((boundary-4, boundary-2, boundary-1, boundary, above))
        for eligible in (False, True):
            membrane = positions.clone().requires_grad_()
            gate = torch.full_like(membrane, eligible, dtype=torch.bool)
            event = _TriangularEvent.apply(membrane, threshold, gate)
            expected = membrane.new_tensor([-1., -1., -1., -1., 0.]) if eligible else torch.zeros_like(membrane)
            torch.testing.assert_close(event, expected, rtol=0, atol=0)
            event.sum().backward()
            expected_gradient = (membrane.new_tensor([0., 0., .25, .5, .5])
                                 if eligible else torch.zeros_like(membrane))
            torch.testing.assert_close(membrane.grad, expected_gradient, rtol=0, atol=1e-15)

    def test_two_frame_hand_computed_reset_chain(self):
        # theta=2: incoming membranes are 2.5 and -.501, giving charges[2,-2].
        # Each charge derivative is .75. The first reset contributes1-.75=.25.
        # Thus for L=2*y0+3*y1: dL/dI0=2*.75+3*.75*.25; dL/dI1=3*.75.
        current = torch.tensor([1.5, -1.001], dtype=torch.float64).reshape(2, 1, 1, 1)
        current.requires_grad_()
        node = TriangularSequence(torch.tensor(2., dtype=torch.float64))
        output = node(current)
        torch.testing.assert_close(output.flatten(), current.new_tensor([2., -2.]),
                                   rtol=0, atol=0)
        (output.flatten() * current.new_tensor([2., 3.])).sum().backward()
        torch.testing.assert_close(current.grad.flatten(), current.new_tensor([2.0625, 2.25]),
                                   rtol=0, atol=1e-12)

    def test_resnet_deployment_state_dict_and_source_gradients(self):
        torch.set_num_threads(2)
        torch.manual_seed(2026100321)
        source = ResNet18Cifar(num_classes=7, base_width=4)
        units = {name: torch.linspace(.5, 1., channels)
                 for name, channels in activation_sites(source)}
        source = install_channel_qcfs(source, units).eval()
        model = TriangularResNet18(source).eval()
        deployment = ECLResNet18(source).eval()
        self.assertEqual(set(model.state_dict()), set(deployment.state_dict()))
        deployment.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(sum(isinstance(m, TriangularSequence) for m in model.modules()), 17)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertFalse(any('threshold' in name for name, _ in model.named_parameters()))
        self.assertTrue(all(not value.requires_grad for name, value in model.named_buffers()
                            if name.endswith('threshold')))
        images = torch.randn(2, 3, 16, 16)
        logits = model(images)
        with torch.no_grad():
            expected = deployment(images)
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)
        torch.nn.functional.cross_entropy(logits, torch.tensor([1, 4])).backward()
        self.assertTrue(all(p.grad is not None and bool(p.grad.isfinite().all())
                            for p in model.parameters()))
        self.assertGreater(model.source.conv1.weight.grad.abs().sum().item(), 0.)


if __name__ == '__main__':
    unittest.main()
