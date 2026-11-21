import copy
import unittest

import torch
from torch import nn

from src.tracking_clip import ClipViStreamBackbone
from src.tracking_vistream import QuantizedActivation


class ToyAppearance(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer3 = nn.Sequential(nn.Conv2d(3, 4, 1, bias=False),
                                    nn.BatchNorm2d(4), QuantizedActivation(7))
        nn.init.constant_(self.layer3[0].weight, .2)

    def forward(self, x):
        return self.layer3(x)


class ViStreamClipTrainingTests(unittest.TestCase):
    def test_context_matches_streaming_and_preserves_prefix_bias_gradients(self):
        torch.set_num_threads(2)
        context = torch.full((4,3,3,5),1.1,dtype=torch.float64,requires_grad=True)
        frames = context.detach().clone().requires_grad_(True)
        for method in ('density','triangular'):
            model = ClipViStreamBackbone(ToyAppearance().double().eval(),4,method)
            with torch.no_grad():
                model.reset()
                for frame in context:
                    model(frame[None])
                expected = [model(frame[None]) for frame in frames]
            emissions = []
            def keep_emissions(module,args,value):
                value.retain_grad()
                emissions.append(value)
            hook = model.source.layer3[2].register_forward_hook(keep_emissions)
            try:
                output = model.forward_clip_with_context(context,frames)
            finally:
                hook.remove()
            for actual,reference in zip(output,expected):
                torch.testing.assert_close(actual,reference,atol=0,rtol=0)
            output[-1].square().mean().backward()
            bias = model.source.layer3[1].bn.bias
            self.assertIsNotNone(bias.grad)
            self.assertGreater(float(bias.grad.abs().sum()),0.)
            self.assertIsNotNone(context.grad)
            self.assertTrue(bool(context.grad.isfinite().all()))
            # Input deltas may telescope to zero for this static linear toy;
            # prefix event and first-tick bias paths must still be connected.
            self.assertGreater(float(emissions[0].grad[:16].abs().sum()),0.)
            # Recompute after an optimizer step, without using stale states.
            torch.optim.SGD(model.parameters(),lr=.001).step()
            actual = model.forward_clip_with_context(context.detach(),frames.detach())
            reference = model.forward_clip(torch.cat([context.detach(),frames.detach()]))[4:]
            for a,b in zip(actual,reference):
                torch.testing.assert_close(a,b,atol=0,rtol=0)
            context.grad = None
            frames.grad = None

    def test_fixed_weight_forward_and_two_gradient_methods_match(self):
        torch.set_num_threads(2)
        torch.manual_seed(2026100321)
        source = ToyAppearance().double().eval()
        phase = ClipViStreamBackbone(copy.deepcopy(source), 4, 'density')
        triangle = ClipViStreamBackbone(copy.deepcopy(source), 4, 'triangular')
        frames = torch.randn(4, 3, 3, 5, dtype=torch.float64)
        with torch.no_grad():
            streamed = [phase(frame[None]) for frame in frames]
            clipped = phase.forward_clip(frames)
            other = triangle.forward_clip(frames)
        for expected, actual, baseline in zip(streamed, clipped, other):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(actual, baseline, atol=0, rtol=0)

    def test_static_frame_parameter_and_first_tick_bias_paths_are_restored(self):
        frame = torch.full((1, 3, 3, 5), 1.1, dtype=torch.float64)
        for method in ('density', 'triangular'):
            old = ClipViStreamBackbone(ToyAppearance().double().eval(), 4, method)
            with torch.no_grad():
                for _ in range(16):
                    old(frame)
            old(frame).square().mean().backward()
            old_weight = old.source.layer3[0].conv.weight
            old_bias = old.source.layer3[1].bn.bias
            self.assertEqual(float(old_weight.grad.abs().sum()), 0.)
            self.assertIsNone(old_bias.grad)

            repaired = ClipViStreamBackbone(ToyAppearance().double().eval(), 4, method)
            repaired.forward_clip(frame.expand(4, -1, -1, -1))[-1].square().mean().backward()
            weight = repaired.source.layer3[0].conv.weight
            bias = repaired.source.layer3[1].bn.bias
            self.assertTrue(bool(weight.grad.isfinite().all()))
            self.assertGreater(float(weight.grad.abs().sum()), 0.)
            self.assertIsNotNone(bias.grad)
            self.assertTrue(bool(bias.grad.isfinite().all()))
            self.assertGreater(float(bias.grad.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
