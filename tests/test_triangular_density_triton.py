"""Dispatch/fallback contracts independent of optional Triton and GPU access."""
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import torch
from src.conversion.triangular_density_horizon import (
    _partition, triangle_density_partition_vjp,
)
from src.conversion import triangular_density_triton as backend


class TriangularDensityTritonDispatchTests(unittest.TestCase):
    def test_cpu_fallback_retains_old_result(self):
        rng=torch.Generator().manual_seed(2026100321)
        for t in (4,7,15):
            x=torch.randn(t,3,generator=rng)
            g=torch.randn(x.shape,generator=rng)
            theta=torch.tensor(.7)
            with patch.object(backend,'triton_density_vjp',side_effect=AssertionError('CPU must not launch')):
                actual=triangle_density_partition_vjp(x,theta,g)
            torch.testing.assert_close(actual,_partition(x,theta,g),rtol=0,atol=0)

    def test_missing_optional_dependency_selects_fallback(self):
        # Only metadata are needed for availability; no CUDA context is created.
        fake=SimpleNamespace(device=SimpleNamespace(type='cuda'),dtype=torch.float32,shape=(7,3))
        with patch.dict(sys.modules,{'triton':None}):
            self.assertFalse(backend.is_supported(fake,torch.tensor(.7)))

    def test_real_backend_failure_is_not_silently_caught(self):
        x=torch.ones(7,3);g=torch.ones_like(x);theta=torch.tensor(.7)
        with patch.object(backend,'is_supported',return_value=True), \
             patch.object(backend,'triton_density_vjp',side_effect=RuntimeError('kernel failed')):
            with self.assertRaisesRegex(RuntimeError,'kernel failed'):
                triangle_density_partition_vjp(x,theta,g)


if __name__=='__main__':
    unittest.main()
