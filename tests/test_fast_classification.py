import unittest
from pathlib import Path

import torch

from src.classification_fast_snn import author_events, author_runtime, make_ann, FastClassifier


AUTHOR = Path(__file__).resolve().parents[1] / 'artifacts/fast_snn_detection_20261117/upstream/image classification/CIFAR10_resnet18'


class FastClassificationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.manual_seed(2026100321)

    def test_native_threshold_tie_and_eligible_withdrawal(self):
        currents = torch.tensor([[.5], [-2.], [0.]])
        expected = torch.tensor([[1.], [-1.], [0.]])
        self.assertTrue(torch.equal(author_events(currents, torch.ones(1)), expected))

    @unittest.skipUnless(AUTHOR.is_dir(), 'Optional author checkout absent')
    def test_complete_native_classifier_forward(self):
        runtime = author_runtime(AUTHOR)
        source = make_ann(runtime, 2).eval()
        native = runtime.model.resnet18(spike=True, bit=2, num_classes=100).eval()
        native.load_state_dict(source.state_dict(), strict=True)
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            expected = native(x.clone())
            for method in ['hard','triangular','density']:
                model = FastClassifier(source, runtime, 3, method).eval()
                self.assertTrue(torch.equal(expected, model(x)), method)

    @unittest.skipUnless(AUTHOR.is_dir(), 'Optional author checkout absent')
    def test_both_gradients_update_convolution_and_preserve_source_buffers(self):
        runtime = author_runtime(AUTHOR)
        source = make_ann(runtime, 2).eval()
        x = torch.randn(2, 3, 32, 32)
        expected = None
        for method in ['triangular','density']:
            model = FastClassifier(source, runtime, 3, method).train()
            buffers = {n:v.clone() for n,v in model.named_buffers()}
            first = next(p for p in model.parameters() if p.ndim == 4)
            initial = first.detach().clone()
            out = model(x)
            if expected is None:
                expected = out.detach()
            else:
                self.assertTrue(torch.equal(out, expected))
            torch.nn.functional.cross_entropy(out, torch.tensor([1,2])).backward()
            self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()))
            self.assertGreater(float(first.grad.abs().sum()),0)
            optimizer = torch.optim.SGD(model.parameters(), lr=.01)
            optimizer.step()
            self.assertFalse(torch.equal(first,initial))
            self.assertTrue(all(torch.equal(v,buffers[n]) for n,v in model.named_buffers()))


if __name__ == '__main__':
    unittest.main()
