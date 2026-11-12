import unittest

import torch

from src.models.resnet18_cifar import resnet18_cifar_ann
from src.conversion.qcfs_ecl_resnet18 import (
    AuthorQCFS, DualThresholdSequence, ECLResNet18, install_qcfs,
)


class QCFSConversionTests(unittest.TestCase):
    def test_author_quantization_and_threshold_gradient(self):
        node = AuthorQCFS(T=0, L=2, thresh=2.)
        x = torch.tensor([-.4, .3, .6, 1.2, 2.5], requires_grad=True)
        y = node(x)
        torch.testing.assert_close(y, torch.tensor([0., 0., 1., 1., 2.]), atol=0, rtol=0)
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([0., 1., 1., 1., 0.]))
        self.assertTrue(torch.isfinite(node.thresh.grad).all())

    def test_signed_retraction_and_positive_boundary(self):
        node = DualThresholdSequence(torch.tensor([1.]), 2, True)
        # One positive then retraction; never retract without previous count;
        # equality at +theta does not fire in the paper's signed rule.
        x = torch.tensor([[.6, -1., .5], [-1.2, 0., 0.]]).view(2, 3, 1, 1)
        actual = node(x).flatten()
        torch.testing.assert_close(actual, torch.tensor([1., 0., 0., -1., 0., 0.]), atol=0, rtol=0)
        self.assertTrue(torch.equal(node(x), node(x)))

    def test_model_unsigned_matches_author_neurons(self):
        torch.manual_seed(2026100321)
        source = install_qcfs(resnet18_cifar_ann(base_width=4), 2).eval()
        converted = ECLResNet18(source, 2, False)
        literal = install_qcfs(resnet18_cifar_ann(base_width=4), 2).eval()
        literal.load_state_dict(source.state_dict())
        for node in literal.modules():
            if isinstance(node, AuthorQCFS):
                node.T = node.expand.T = 2
        x = torch.randn(3, 3, 32, 32)
        with torch.no_grad():
            expected = literal(x.repeat(2, 1, 1, 1)).unflatten(0, (2, 3)).mean(0)
            actual = converted(x)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
            self.assertTrue(torch.equal(actual, converted(x)))
        self.assertEqual(sum(isinstance(n, AuthorQCFS) for n in source.modules()), 17)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
