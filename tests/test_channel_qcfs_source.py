import unittest

import torch
from torch import nn

from src.conversion.qcfs_ecl_resnet18 import AuthorQCFS, DualThresholdSequence
from src.conversion.joint_phase_adjoint import JointPhaseSequence


class ChannelQCFSTests(unittest.TestCase):
    def test_threshold_mapping_and_stationary_conversion(self):
        threshold = torch.tensor([.4,1.2]).reshape(1,2,1,1)
        node = AuthorQCFS(T=0,L=2,thresh=1.)
        node.thresh = nn.Parameter(threshold.clone())
        # Avoid exact ties: ECL's strict > and QCFS's half-up floor differ there.
        ratios = torch.tensor([-.2,.1,.3,.6,.8,1.3]).reshape(-1,1,1,1)
        values = ratios*threshold
        expected = (2*(values/threshold).clamp(0,1)+.5).floor()*threshold/2
        torch.testing.assert_close(node(values),expected)
        converter = DualThresholdSequence(threshold,2,True)
        emitted = converter(torch.cat((values,values))).unflatten(0,(2,-1)).mean(0)
        torch.testing.assert_close(emitted,expected)
        self.assertEqual(tuple(converter.threshold.shape),(1,2,1,1))

    def test_channelwise_joint_adjoint_has_same_hard_output(self):
        threshold = torch.tensor([.4,1.2]).reshape(1,2,1,1)
        values = torch.tensor([.3,1.1,-.5,-2.]).reshape(2,2,1,1).requires_grad_()
        proposed = JointPhaseSequence(threshold)
        reference = DualThresholdSequence(threshold,2,True)
        output = proposed(values)
        with torch.no_grad():
            torch.testing.assert_close(output,reference(values),rtol=0,atol=0)
        output.square().sum().backward()
        self.assertTrue(bool(values.grad.isfinite().all()))


if __name__ == '__main__':
    unittest.main()
