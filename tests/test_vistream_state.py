import importlib.util
import unittest
from pathlib import Path

import torch
from torch import nn

from src.conversion.bounded_phase import bounded_phase_events
from src.tracking_vistream import (QuantizedActivation, StatefulNeuron,
                                  TemporalBatchNorm, TemporalMaxPool, hard_events)


class ViStreamStateTests(unittest.TestCase):
    def test_hard_boundaries_and_carried_state(self):
        torch.manual_seed(2026100321)
        x = torch.randn(8, 29, dtype=torch.float64) * 3
        q, c = torch.randn(29, dtype=x.dtype), torch.randint(0, 4, (29,)).to(x)
        theta = x.new_tensor(.7)
        expected, q1, c1 = hard_events(x, q, c, theta, 3)
        got = bounded_phase_events(x.requires_grad_(), q, c, theta, 3)
        self.assertTrue(torch.equal(expected, got))
        first, qm, cm = hard_events(x[:3], q, c, theta, 3)
        rest, q2, c2 = hard_events(x[3:], qm, cm, theta, 3)
        self.assertTrue(torch.equal(expected, torch.cat([first, rest])))
        self.assertTrue(torch.equal(q1, q2) and torch.equal(c1, c2))
        self.assertTrue(bool((c1 >= 0).all() and (c1 <= 3).all()))

    def test_phase_vjp_against_independent_reference(self):
        path = Path(__file__).resolve().parents[1] / 'artifacts/vistream_gradient_20261117/bounded_phase_reference.py'
        if not path.exists():
            self.skipTest('Independent research reference not distributed')
        spec = importlib.util.spec_from_file_location('bounded_reference', path)
        ref = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref)
        torch.manual_seed(2026100321)
        for L in (2, 7):
            for T in (1, 2, 4, 8):
                x = torch.randn(T, 51, dtype=torch.float64).requires_grad_()
                q = torch.randn(51, dtype=x.dtype) * 2
                c = torch.randint(0, L+1, (51,)).to(x)
                g = torch.randn_like(x)
                y = bounded_phase_events(x, q, c, x.new_tensor(.83), L)
                got, = torch.autograd.grad(y, x, g)
                expected = ref.vjp(x, g, q, c, L, theta=.83)
                torch.testing.assert_close(got, expected, rtol=1e-10, atol=1e-10)

    def test_temporal_pool_and_bn_telescope_across_frames(self):
        torch.manual_seed(2026100321)
        x = torch.randn(6, 1, 3, 9, 11)
        bn = nn.BatchNorm2d(3).eval()
        bn.running_mean.copy_(torch.randn(3))
        layer = TemporalBatchNorm(bn)
        y = torch.cat([layer(x[:2]), layer(x[2:])])
        torch.testing.assert_close(y.sum(0), bn(x.sum(0)), atol=2e-6, rtol=2e-6)
        pool = TemporalMaxPool(nn.MaxPool2d(3, 2, 1))
        z = torch.cat([pool(y[:2]), pool(y[2:])])
        torch.testing.assert_close(z.sum(0), pool.pool(y.sum(0)), atol=2e-6, rtol=2e-6)


if __name__ == '__main__':
    unittest.main()
